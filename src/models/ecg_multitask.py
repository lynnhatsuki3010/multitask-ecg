"""
Clean multitask ECG model built on top of interchangeable backbones.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import AttentionPooling


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


class DynamicThresholdHead(nn.Module):
    """Predicts per-sample, per-class classification thresholds.

    Given a patient's global feature representation, this lightweight
    head outputs a threshold value tau_c in (0, 1) for each class c.
    During inference, instead of comparing sigmoid(logit) >= 0.5,
    we compare sigmoid(logit) >= tau_c (patient-specific).

    The head is trained with a consistency loss: it is penalized
    whenever the predicted threshold disagrees with the label
    (i.e., output should be low when the class is positive, high otherwise).
    This creates a soft auxiliary signal without requiring extra annotations.
    """

    def __init__(self, in_dim: int, num_classes: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, num_classes),
        )
        # Initialize biases to predict 0.5 initially (sigmoid(0) = 0.5)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted thresholds in (0, 1) for each class."""
        return torch.sigmoid(self.net(x))


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
        self.last_weights = weights.detach()  # Saved for XAI
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        return self.proj(torch.cat([avg, mx], dim=-1))


class MIHead(nn.Module):
    """Transformer-centric MI head with anatomy-aware CNN support.

    IMI uses two complementary lead groups:
      - INFERIOR_LEADS   [II, III, aVF]: direct leads (Q waves, ST elevation)
      - RECIPROCAL_LEADS [I, aVL]:       reciprocal ST depression equally diagnostic
    ASMI uses ANTERIOR_LEADS [V1-V4].
    """

    INFERIOR_LEADS = [1, 2, 5]      # II, III, aVF
    RECIPROCAL_LEADS = [0, 4]       # I, aVL
    ANTERIOR_LEADS = [6, 7, 8, 9]   # V1, V2, V3, V4

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float, use_lead_group: bool = True, use_task_token: bool = True) -> None:
        super().__init__()
        self.use_lead_group = use_lead_group
        self.use_task_token = use_task_token

        imi_in_dim = shared_dim
        asmi_in_dim = shared_dim

        if use_task_token:
            self.imi_token_pool = TaskTokenPooling(shared_dim, dropout)
            self.asmi_token_pool = TaskTokenPooling(shared_dim, dropout)
            imi_in_dim += shared_dim
            asmi_in_dim += shared_dim

        if use_lead_group:
            self.inferior_encoder = LeadGroupEncoder(len(self.INFERIOR_LEADS), branch_dim, dropout)
            self.reciprocal_encoder = LeadGroupEncoder(len(self.RECIPROCAL_LEADS), branch_dim // 2, dropout)
            self.anterior_encoder = LeadGroupEncoder(len(self.ANTERIOR_LEADS), branch_dim, dropout)
            imi_in_dim += branch_dim + branch_dim // 2
            asmi_in_dim += branch_dim

        self.imi_head = MLPHead(
            imi_in_dim,
            1,
            hidden_dim=branch_dim,
            dropout=dropout,
        )
        self.asmi_head = MLPHead(
            asmi_in_dim,
            1,
            hidden_dim=branch_dim,
            dropout=dropout,
        )

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        imi_feats = [shared_features]
        asmi_feats = [shared_features]

        if self.use_task_token:
            imi_feats.append(self.imi_token_pool(sequence_features))
            asmi_feats.append(self.asmi_token_pool(sequence_features))

        if self.use_lead_group:
            imi_feats.append(self.inferior_encoder(signal[:, self.INFERIOR_LEADS, :]))
            imi_feats.append(self.reciprocal_encoder(signal[:, self.RECIPROCAL_LEADS, :]))
            asmi_feats.append(self.anterior_encoder(signal[:, self.ANTERIOR_LEADS, :]))

        imi = self.imi_head(torch.cat(imi_feats, dim=-1))
        asmi = self.asmi_head(torch.cat(asmi_feats, dim=-1))
        return torch.cat([imi, asmi], dim=-1)


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
        hrv_enabled: bool,
        num_hrv_targets: int,
        head_hidden_dim: int,
        mi_branch_dim: int,
        dropout: float,
        hrv_detach: bool = False,
        mi_gradient_scale: float = 1.0,
        use_lead_group: bool = True,
        use_task_token: bool = True,
        use_dynamic_threshold: bool = False,
    ) -> None:
        super().__init__()
        if num_mi_labels != 2:
            raise ValueError("The rebuilt MI head currently expects exactly 2 labels: IMI and ASMI.")

        self.backbone = backbone
        self.hrv_enabled = hrv_enabled
        self.hrv_detach = hrv_detach
        self.mi_gradient_scale = mi_gradient_scale
        self.use_task_token = use_task_token
        self.use_dynamic_threshold = use_dynamic_threshold
        shared_dim = backbone.output_dim

        self.sequence_pool = AttentionPooling(shared_dim)

        arrhy_in_dim = shared_dim
        if use_task_token:
            self.arrhythmia_pool = TaskTokenPooling(shared_dim, dropout)
            arrhy_in_dim += shared_dim

        self.arrhythmia_head = MLPHead(
            arrhy_in_dim,
            num_arrhythmia_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.mi_head = MIHead(
            shared_dim=shared_dim, 
            branch_dim=mi_branch_dim, 
            dropout=dropout,
            use_lead_group=use_lead_group,
            use_task_token=use_task_token,
        )

        # --- Dynamic Threshold Head (optional) ---
        # Predicts per-sample thresholds for each class from global shared features.
        # Total num_classes = num_arrhythmia_labels + num_mi_labels
        if use_dynamic_threshold:
            total_classes = num_arrhythmia_labels + num_mi_labels
            self.threshold_head = DynamicThresholdHead(
                in_dim=shared_dim,
                num_classes=total_classes,
                bottleneck=max(32, shared_dim // 4),
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
        global_features = backbone_out["global_features"]
        sequence_features = backbone_out["sequence_features"]
        pooled_sequence = self.sequence_pool(sequence_features)
        
        shared_features = global_features + pooled_sequence

        rhythm_feats = [global_features]
        if self.use_task_token:
            rhythm_feats.append(self.arrhythmia_pool(sequence_features))
        rhythm_features = torch.cat(rhythm_feats, dim=-1)

        # Gradient isolation: scale MI gradients flowing back into shared backbone
        # MI head still gets full-resolution features from LeadGroupEncoder (raw signal)
        if self.mi_gradient_scale < 1.0:
            mi_shared = self._scale_gradient(shared_features, self.mi_gradient_scale)
            mi_seq = self._scale_gradient(sequence_features, self.mi_gradient_scale)
        else:
            mi_shared = shared_features
            mi_seq = sequence_features

        outputs = {
            "arrhythmia": self.arrhythmia_head(rhythm_features),
            "mi": self.mi_head(x, mi_shared, mi_seq),
        }

        # Dynamic threshold prediction (patient-specific thresholds)
        if self.use_dynamic_threshold:
            # shared_features is detached so the threshold head doesn't
            # affect the gradients of classification logits.
            predicted_thresholds = self.threshold_head(shared_features.detach())
            outputs["predicted_thresholds"] = predicted_thresholds

        if self.hrv_enabled:
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
        
        if self.use_task_token:
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
        if self.use_task_token:
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
