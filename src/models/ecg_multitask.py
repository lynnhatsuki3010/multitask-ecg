"""
Clean multitask ECG model built on top of interchangeable backbones.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.backbones import AttentionPooling, MultiBranchTransformerBackbone


class MLPHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskTokenPooling(nn.Module):
    """Task-specific attention pooling over transformer/CNN token sequences."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.score_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.score_proj(tokens))
        scores = torch.matmul(projected, self.query)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(tokens * weights, dim=1)
        return self.dropout(pooled)


class LeadGroupEncoder(nn.Module):
    """Small dedicated encoder for clinically relevant MI lead groups."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        h = self.net(x)
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        pooled = self.proj(torch.cat([avg, mx], dim=-1))
        if return_sequence:
            return pooled, h
        return pooled


class MIHead(nn.Module):
    """Anatomy-aware MI head supporting 4 MI labels:
      IMI  (inferior):       II, III, aVF  + reciprocal I, aVL
      ASMI (anteroseptal):   V1-V4
      ILMI (inferolateral):  II, III, aVF  + I, aVL + V5, V6
      AMI  (anterior):       V1-V6
    """

    INFERIOR_LEADS   = [1, 2, 5]         # II, III, aVF
    RECIPROCAL_LEADS = [0, 4]            # I, aVL
    ANTERIOR_LEADS   = [6, 7, 8, 9]     # V1, V2, V3, V4
    LATERAL_LEADS    = [8, 9, 10, 11]   # V5, V6 (+ I, aVL for ILMI)
    ANTERIOR6_LEADS  = [6, 7, 8, 9, 10, 11]  # V1-V6 for AMI

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float, use_cross_attention: bool = False) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        # Per-label task-token pooling
        self.imi_token_pool  = TaskTokenPooling(shared_dim, dropout)
        self.asmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ilmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ami_token_pool  = TaskTokenPooling(shared_dim, dropout)

        # Lead-group CNN encoders
        self.inferior_encoder   = LeadGroupEncoder(len(self.INFERIOR_LEADS),   branch_dim,      dropout)
        self.reciprocal_encoder = LeadGroupEncoder(len(self.RECIPROCAL_LEADS), branch_dim // 2, dropout)
        self.anterior_encoder   = LeadGroupEncoder(len(self.ANTERIOR_LEADS),   branch_dim,      dropout)
        self.lateral_encoder    = LeadGroupEncoder(len(self.LATERAL_LEADS),    branch_dim,      dropout)
        self.anterior6_encoder  = LeadGroupEncoder(len(self.ANTERIOR6_LEADS),  branch_dim,      dropout)

        if use_cross_attention:
            self.cross_attn_inferior   = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_reciprocal = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_anterior   = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_lateral    = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_anterior6  = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.proj_k_inf  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_rec  = nn.Conv1d(branch_dim // 2, shared_dim, kernel_size=1)
            self.proj_k_ant  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_lat  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_ant6 = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            imi_in  = shared_dim * 2 + shared_dim * 2   # global + token + inf_attn + rec_attn
            asmi_in = shared_dim * 2 + shared_dim       # global + token + ant_attn
            ilmi_in = shared_dim * 2 + shared_dim * 3   # global + token + inf+rec+lat
            ami_in  = shared_dim * 2 + shared_dim       # global + token + ant6_attn
        else:
            imi_in  = shared_dim * 2 + branch_dim + branch_dim // 2
            asmi_in = shared_dim * 2 + branch_dim
            ilmi_in = shared_dim * 2 + branch_dim + branch_dim // 2 + branch_dim
            ami_in  = shared_dim * 2 + branch_dim

        self.imi_head  = MLPHead(imi_in,  1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(asmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(ilmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(ami_in,  1, hidden_dim=branch_dim, dropout=dropout)

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        imi_tokens  = self.imi_token_pool(sequence_features)
        asmi_tokens = self.asmi_token_pool(sequence_features)
        ilmi_tokens = self.ilmi_token_pool(sequence_features)
        ami_tokens  = self.ami_token_pool(sequence_features)

        if self.use_cross_attention:
            _, inf_seq  = self.inferior_encoder(signal[:, self.INFERIOR_LEADS, :],   return_sequence=True)
            _, rec_seq  = self.reciprocal_encoder(signal[:, self.RECIPROCAL_LEADS, :], return_sequence=True)
            _, ant_seq  = self.anterior_encoder(signal[:, self.ANTERIOR_LEADS, :],   return_sequence=True)
            _, lat_seq  = self.lateral_encoder(signal[:, self.LATERAL_LEADS, :],     return_sequence=True)
            _, ant6_seq = self.anterior6_encoder(signal[:, self.ANTERIOR6_LEADS, :], return_sequence=True)
            inf_seq  = self.proj_k_inf(inf_seq).transpose(1, 2)
            rec_seq  = self.proj_k_rec(rec_seq).transpose(1, 2)
            ant_seq  = self.proj_k_ant(ant_seq).transpose(1, 2)
            lat_seq  = self.proj_k_lat(lat_seq).transpose(1, 2)
            ant6_seq = self.proj_k_ant6(ant6_seq).transpose(1, 2)
            q = shared_features.unsqueeze(1)
            inferior,   _ = self.cross_attn_inferior(q, inf_seq, inf_seq)
            reciprocal, _ = self.cross_attn_reciprocal(q, rec_seq, rec_seq)
            anterior,   _ = self.cross_attn_anterior(q, ant_seq, ant_seq)
            lateral,    _ = self.cross_attn_lateral(q, lat_seq, lat_seq)
            anterior6,  _ = self.cross_attn_anterior6(q, ant6_seq, ant6_seq)
            inferior   = inferior.squeeze(1)
            reciprocal = reciprocal.squeeze(1)
            anterior   = anterior.squeeze(1)
            lateral    = lateral.squeeze(1)
            anterior6  = anterior6.squeeze(1)
        else:
            inferior   = self.inferior_encoder(signal[:, self.INFERIOR_LEADS, :])
            reciprocal = self.reciprocal_encoder(signal[:, self.RECIPROCAL_LEADS, :])
            anterior   = self.anterior_encoder(signal[:, self.ANTERIOR_LEADS, :])
            lateral    = self.lateral_encoder(signal[:, self.LATERAL_LEADS, :])
            anterior6  = self.anterior6_encoder(signal[:, self.ANTERIOR6_LEADS, :])

        imi  = self.imi_head(torch.cat([shared_features, imi_tokens,  inferior, reciprocal],          dim=-1))
        asmi = self.asmi_head(torch.cat([shared_features, asmi_tokens, anterior],                     dim=-1))
        ilmi = self.ilmi_head(torch.cat([shared_features, ilmi_tokens, inferior, reciprocal, lateral], dim=-1))
        ami  = self.ami_head(torch.cat([shared_features,  ami_tokens,  anterior6],                    dim=-1))
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)  # (B, 4)


class ConductionHead(nn.Module):
    """MLP head for Conduction Disturbance labels (LBBB, RBBB, IRBBB, 1AVB).

    Conduction abnormalities primarily manifest as QRS widening / morphology
    changes. This head receives features from its own dedicated Expert branch
    AND from the ConductionLeadGroupEncoder (V1-V3 for RBBB, V5+V6+I+aVL
    for LBBB), so its attention patterns are anatomy-aware.
    """

    def __init__(
        self,
        shared_dim: int,
        num_conduction_labels: int,
        hidden_dim: int,
        dropout: float,
        cd_lead_dim: int = 0,   # set to cd_lead_out_dim when using lead encoder
    ) -> None:
        super().__init__()
        self.token_pool = TaskTokenPooling(shared_dim, dropout)
        in_features = shared_dim * 2 + cd_lead_dim
        self.head = MLPHead(
            in_features=in_features,
            out_features=num_conduction_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.cd_lead_dim = cd_lead_dim

    def forward(
        self,
        global_features: torch.Tensor,
        sequence_features: torch.Tensor,
        cd_lead_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.token_pool(sequence_features)
        parts = [global_features, tokens]
        if self.cd_lead_dim > 0 and cd_lead_feats is not None:
            parts.append(cd_lead_feats)
        return self.head(torch.cat(parts, dim=-1))



class ECGMultiTaskModel(nn.Module):
    """
    Unified multitask model:
      - shared backbone for rhythm/global context
      - anatomy-aware MI branch from raw signal + shared context
      - optional HRV regression branch

    mi_gradient_scale (0.0-1.0): Controls how much MI loss gradient flows back
    into the shared backbone. Lower values reduce gradient competition between
    arrhythmia and MI tasks. MI head still receives full-resolution features
    from its own LeadGroupEncoder (raw signal → inferior/reciprocal/anterior).
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_arrhythmia_labels: int,
        num_mi_labels: int,
        num_conduction_labels: int,
        hrv_enabled: bool,
        num_hrv_targets: int,
        head_hidden_dim: int,
        mi_branch_dim: int,
        dropout: float,
        hrv_detach: bool = False,
        mi_gradient_scale: float = 1.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.hrv_enabled = hrv_enabled
        self.hrv_detach = hrv_detach
        self.mi_gradient_scale = mi_gradient_scale
        self._is_multi_branch = isinstance(backbone, MultiBranchTransformerBackbone)
        shared_dim = backbone.output_dim

        self.sequence_pool = AttentionPooling(shared_dim)
        self.arrhythmia_pool = TaskTokenPooling(shared_dim, dropout)

        self.arrhythmia_head = MLPHead(
            shared_dim * 2,
            num_arrhythmia_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        cd_lead_dim = 0
        if isinstance(backbone, MultiBranchTransformerBackbone):
            cd_lead_dim = backbone.cd_lead_out_dim

        self.mi_head = MIHead(shared_dim=shared_dim, branch_dim=mi_branch_dim, dropout=dropout, use_cross_attention=use_cross_attention)
        self.conduction_head = ConductionHead(
            shared_dim=shared_dim,
            num_conduction_labels=num_conduction_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            cd_lead_dim=cd_lead_dim,
        )

        if hrv_enabled:
            self.hrv_head = MLPHead(
                shared_dim * 2,
                num_hrv_targets,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )

    @staticmethod
    def _scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale gradient flowing through tensor without affecting forward value.

        During forward: returns tensor unchanged.
        During backward: gradient is multiplied by `scale`.
        """
        return tensor * scale + tensor.detach() * (1.0 - scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)

        if self._is_multi_branch:
            # Each expert branch provides its own global + sequence features
            arrhy_global = backbone_out["arrhy_global"]
            arrhy_seq    = backbone_out["arrhy_seq"]
            mi_global    = backbone_out["mi_global"]
            mi_seq       = backbone_out["mi_seq"]
            cond_global  = backbone_out["cond_global"]
            cond_seq     = backbone_out["cond_seq"]

            arrhythmia_tokens = self.arrhythmia_pool(arrhy_seq)
            rhythm_features   = torch.cat([arrhy_global, arrhythmia_tokens], dim=-1)
            mi_shared         = mi_global + self.sequence_pool(mi_seq)
            mi_seq_feats      = mi_seq
        else:
            # Legacy single-branch backbone (HybridTransformerBackbone)
            global_features   = backbone_out["global_features"]
            sequence_features = backbone_out["sequence_features"]
            pooled_sequence   = self.sequence_pool(sequence_features)
            arrhythmia_tokens = self.arrhythmia_pool(sequence_features)
            rhythm_features   = torch.cat([global_features, arrhythmia_tokens], dim=-1)
            shared_features   = global_features + pooled_sequence

            if self.mi_gradient_scale < 1.0:
                mi_shared    = self._scale_gradient(shared_features, self.mi_gradient_scale)
                mi_seq_feats = self._scale_gradient(sequence_features, self.mi_gradient_scale)
            else:
                mi_shared    = shared_features
                mi_seq_feats = sequence_features

            cond_global = shared_features
            cond_seq    = sequence_features

        outputs = {
            "arrhythmia": self.arrhythmia_head(rhythm_features),
            "mi":         self.mi_head(x, mi_shared, mi_seq_feats),
            "conduction": self.conduction_head(
                cond_global, cond_seq,
                backbone_out.get("cond_lead_feats") if self._is_multi_branch else None,
            ),
        }

        if self.hrv_enabled:
            hrv_src = backbone_out.get("global_features", mi_global if self._is_multi_branch else global_features)
            if self._is_multi_branch:
                hrv_features = torch.cat([backbone_out["arrhy_global"], backbone_out["mi_global"]], dim=-1)
                # project down from 2*d to d for hrv head
                hrv_features = hrv_features  # let hrv_head handle size
            else:
                hrv_features = torch.cat([global_features, pooled_sequence], dim=-1)
            if self.hrv_detach:
                hrv_features = hrv_features.detach()
            outputs["hrv"] = self.hrv_head(hrv_features)

        return outputs

    def freeze_for_phase2(self) -> None:
        """
        Freeze the backbone and arrhythmia head for Phase 2 fine-tuning.
        Also disables MI gradient isolation since only MI receives gradients.
        """
        self._phase2_active = True
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.arrhythmia_pool.parameters():
            param.requires_grad = False
            
        for param in self.arrhythmia_head.parameters():
            param.requires_grad = False
            
        for param in self.sequence_pool.parameters():
            param.requires_grad = False
            
        if self.hrv_enabled:
            for param in self.hrv_head.parameters():
                param.requires_grad = False
                
        # Disable gradient isolation
        self.mi_gradient_scale = 1.0
        
        # Set them to eval mode immediately
        self._apply_phase2_eval()

    def _apply_phase2_eval(self) -> None:
        """Helper to force frozen modules into eval mode (to freeze BatchNorm stats and Dropout)."""
        self.backbone.eval()
        self.arrhythmia_pool.eval()
        self.arrhythmia_head.eval()
        self.sequence_pool.eval()
        if self.hrv_enabled:
            self.hrv_head.eval()

    def train(self, mode: bool = True):
        """Override train() to ensure frozen modules stay in eval() mode during Phase 2."""
        super().train(mode)
        if mode and getattr(self, "_phase2_active", False):
            self._apply_phase2_eval()
        return self
