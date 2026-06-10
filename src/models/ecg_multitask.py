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


class PerLeadEncoder(nn.Module):
    """Encodes each lead independently using a 1D CNN to produce a single feature vector per lead."""
    def __init__(self, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7, bias=False),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*12, 1, T)
        h = self.net(x)
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        pooled = self.proj(torch.cat([avg, mx], dim=-1))
        return pooled


class MIHead(nn.Module):
    """Hard-Routing Graph Transformer for MI classification.
    Takes RAW signal (B, 12, T), processes with PerLeadEncoder and GraphTransformer.
    4 Labels: IMI, ASMI, ILMI, AMI.
    """
    INFERIOR_LEADS   = [1, 2, 5]         # II, III, aVF
    RECIPROCAL_LEADS = [0, 4]            # I, aVL
    ANTERIOR_LEADS   = [6, 7, 8, 9]      # V1, V2, V3, V4
    LATERAL_LEADS    = [8, 9, 10, 11]    # V5, V6 (+ I, aVL for ILMI)
    ANTERIOR6_LEADS  = [6, 7, 8, 9, 10, 11]  # V1-V6 for AMI
    NUM_LEADS = 12

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float, use_cross_attention: bool = False) -> None:
        super().__init__()
        # shared_dim is ignored in hard-routing
        self.per_lead_encoder = PerLeadEncoder(out_dim=branch_dim, dropout=dropout)
        
        self.lead_embeddings = nn.Parameter(torch.randn(1, self.NUM_LEADS, branch_dim))
        nn.init.normal_(self.lead_embeddings, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=4,
            dim_feedforward=branch_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.graph_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.imi_node_pool = TaskTokenPooling(branch_dim, dropout)
        self.asmi_node_pool = TaskTokenPooling(branch_dim, dropout)
        self.ilmi_node_pool = TaskTokenPooling(branch_dim, dropout)
        self.ami_node_pool = TaskTokenPooling(branch_dim, dropout)

        self.imi_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        # Hard-routing: Ignore shared_features and sequence_features entirely
        B, L, T = signal.size()
        
        x_reshaped = signal.view(B * L, 1, T)
        node_features = self.per_lead_encoder(x_reshaped)  # (B*12, branch_dim)
        
        nodes = node_features.view(B, L, -1)               # (B, 12, branch_dim)
        nodes = nodes + self.lead_embeddings
        
        graph_nodes = self.graph_transformer(nodes)        # (B, 12, branch_dim)
        
        imi_nodes = graph_nodes[:, self.INFERIOR_LEADS + self.RECIPROCAL_LEADS, :]
        asmi_nodes = graph_nodes[:, self.ANTERIOR_LEADS, :]
        ilmi_nodes = graph_nodes[:, self.INFERIOR_LEADS + self.RECIPROCAL_LEADS + self.LATERAL_LEADS, :]
        ami_nodes = graph_nodes[:, self.ANTERIOR6_LEADS, :]

        imi_pooled = self.imi_node_pool(imi_nodes)
        asmi_pooled = self.asmi_node_pool(asmi_nodes)
        ilmi_pooled = self.ilmi_node_pool(ilmi_nodes)
        ami_pooled = self.ami_node_pool(ami_nodes)
        
        imi  = self.imi_head(imi_pooled)
        asmi = self.asmi_head(asmi_pooled)
        ilmi = self.ilmi_head(ilmi_pooled)
        ami  = self.ami_head(ami_pooled)
        
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)


class ConductionHead(nn.Module):
    """Hard-Routing Graph Transformer for Conduction Disturbance classification.
    Takes RAW signal (B, 12, T), processes with PerLeadEncoder and GraphTransformer.
    4 Labels: LBBB, RBBB, IRBBB, 1AVB.
    """
    NUM_LEADS = 12

    def __init__(
        self,
        shared_dim: int,
        num_conduction_labels: int,
        hidden_dim: int,
        dropout: float,
        cd_lead_dim: int = 0,
    ) -> None:
        super().__init__()
        # We use hidden_dim as the branch dimensionality
        self.per_lead_encoder = PerLeadEncoder(out_dim=hidden_dim, dropout=dropout)
        
        self.lead_embeddings = nn.Parameter(torch.randn(1, self.NUM_LEADS, hidden_dim))
        nn.init.normal_(self.lead_embeddings, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.graph_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.node_pool = TaskTokenPooling(hidden_dim, dropout)

        self.head = MLPHead(
            in_features=hidden_dim,
            out_features=num_conduction_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        signal: torch.Tensor,
        global_features: torch.Tensor,
        sequence_features: torch.Tensor,
        cd_lead_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Hard-routing: Ignore global_features and sequence_features
        B, L, T = signal.size()
        
        x_reshaped = signal.view(B * L, 1, T)
        node_features = self.per_lead_encoder(x_reshaped)
        
        nodes = node_features.view(B, L, -1)
        nodes = nodes + self.lead_embeddings
        
        graph_nodes = self.graph_transformer(nodes)
        pooled = self.node_pool(graph_nodes)
        
        return self.head(pooled)




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
                x, cond_global, cond_seq,
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
