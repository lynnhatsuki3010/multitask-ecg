"""
multitask_head.py
Multi-task output heads for ECG Transformer.

Tasks:
  1. Arrhythmia classification  → multi-label BCE (5 labels: NORM, AFIB, STACH, SBRAD, AFLT)
  2. MI classification          → multi-label BCE (2 labels: IMI, ASMI)
     [Phase 2] MI head extended with lead-specific features from clinically
     relevant lead groups (IMI: II/III/aVF, ASMI: V1-V4)
  3. HRV regression (optional)  → MSE (3 targets: rmssd, sdnn, mean_hr)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class TaskHead(nn.Module):
    """Generic classification/regression head with optional dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers = []
        if hidden_dim:
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features),
            ]
        else:
            layers += [
                nn.Dropout(dropout),
                nn.Linear(in_features, out_features),
            ]
        if activation is not None:
            layers.append(activation)
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MILeadExtractor(nn.Module):
    """
    [Phase 2] Extracts lead-specific features from clinically relevant lead groups.

    PTB-XL standard 12-lead order (0-indexed):
      I(0), II(1), III(2), aVR(3), aVL(4), aVF(5),
      V1(6), V2(7), V3(8), V4(9), V5(10), V6(11)

    Lead groups:
      IMI  (Inferior MI):  II, III, aVF  → indices [1, 2, 5]
      ASMI (Anterior MI):  V1-V4         → indices [6, 7, 8, 9]

    A shared CNN is applied to each lead independently (weight-tied),
    then mean-pooled within each group to give:
      inferior_feats : (B, lead_dim)
      anterior_feats : (B, lead_dim)
      output         : (B, lead_dim * 2)  [inferior || anterior]
    """
    INFERIOR_LEADS = [1, 2, 5]        # II, III, aVF
    ANTERIOR_LEADS = [6, 7, 8, 9]    # V1, V2, V3, V4

    def __init__(self, lead_dim: int = 64, dropout: float = 0.1, fs: int = 100):
        super().__init__()
        self.lead_dim = lead_dim
        scale = max(1, fs // 100)
        # Shared CNN: processes one lead (1, T) → (lead_dim,)
        # Two conv stages: coarse features + pooling
        # [Fix] Concat Pooling = AvgPool || MaxPool
        #   - AvgPool: captures global rhythm/baseline context
        #   - MaxPool: captures local morphological peaks (Q-wave, ST elevation)
        self.lead_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25*scale, stride=10*scale, padding=12*scale),  # ↓ 10x
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, lead_dim, kernel_size=5*scale, stride=2*scale, padding=2*scale),  # ↓ 2x
            nn.GELU(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)   # global context
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)   # local transient peak detector
        # Project concat(avg, max) → lead_dim  (2*lead_dim → lead_dim)
        self.pool_proj = nn.Linear(2 * lead_dim, lead_dim)
        self.dropout = nn.Dropout(dropout)

    def _pool_group(self, x: torch.Tensor, indices: list) -> torch.Tensor:
        """x: (B, 12, T). Returns concat-pooled lead features: (B, lead_dim)."""
        feats = []
        for idx in indices:
            lead = x[:, idx:idx+1, :]                # (B, 1, T)
            h    = self.lead_cnn(lead)               # (B, lead_dim, T')
            avg  = self.global_avg_pool(h).squeeze(-1)  # (B, lead_dim)
            mx   = self.global_max_pool(h).squeeze(-1)  # (B, lead_dim)
            feat = self.pool_proj(torch.cat([avg, mx], dim=-1))  # (B, lead_dim)
            feats.append(feat)
        return torch.stack(feats, dim=1).mean(dim=1)  # (B, lead_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, T) raw ECG signal
        Returns:
            (B, 2 * lead_dim)  [inferior_feats || anterior_feats]
        """
        inferior = self._pool_group(x, self.INFERIOR_LEADS)  # (B, lead_dim)
        anterior = self._pool_group(x, self.ANTERIOR_LEADS)  # (B, lead_dim)
        return self.dropout(torch.cat([inferior, anterior], dim=-1))  # (B, 2*lead_dim)

class MultiTaskECGModel(nn.Module):
    """
    Full model = Backbone (ECGTransformer) + Multi-task heads.

    Outputs raw logits (no sigmoid/softmax) — loss functions handle activation.
    For inference, apply torch.sigmoid() on classification outputs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 192,
        num_arrhythmia_labels: int = 5,   # NORM, AFIB, STACH, SBRAD, AFLT
        num_mi_labels: int = 2,            # IMI, ASMI
        hrv_enabled: bool = True,
        num_hrv_targets: int = 3,          # rmssd, sdnn, mean_hr
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        mi_lead_dim: int = 64,             # [Phase 2] output dim per lead group
        fs: int = 100,
    ):
        super().__init__()
        self.backbone = backbone
        self.hrv_enabled = hrv_enabled

        # [Phase 2] Lead-specific extractor for MI subtypes
        # Processes inferior leads (II,III,aVF) and anterior leads (V1-V4) independently
        self.mi_lead_extractor = MILeadExtractor(lead_dim=mi_lead_dim, dropout=dropout, fs=fs)
        mi_in_features = d_model + 2 * mi_lead_dim  # CLS + inferior + anterior

        # Arrhythmia classification head (uses CLS only — rhythm is global)
        self.arrhythmia_head = TaskHead(
            in_features=d_model,
            out_features=num_arrhythmia_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

        # MI classification head (uses CLS + lead-specific features)
        self.mi_head = TaskHead(
            in_features=mi_in_features,
            out_features=num_mi_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

        # HRV regression head (optional)
        if hrv_enabled:
            self.hrv_head = TaskHead(
                in_features=d_model,
                out_features=num_hrv_targets,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )
        else:
            self.hrv_head = None

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 12, T) ECG signal tensor.

        Returns:
            dict with keys:
              'arrhythmia': (B, num_arrhythmia_labels)  raw logits
              'mi':         (B, num_mi_labels)           raw logits
              'hrv':        (B, num_hrv_targets)         regression output
                            (only if hrv_enabled)
              'features':   (B, d_model)                 backbone CLS embeddings
        """
        features = self.backbone(x)            # (B, d_model)  — CLS token output

        # [Phase 2] Extract lead-specific features for MI
        lead_feats = self.mi_lead_extractor(x) # (B, 2*lead_dim) [inferior || anterior]
        mi_input   = torch.cat([features, lead_feats], dim=-1)  # (B, d_model + 2*lead_dim)

        out = {
            "arrhythmia": self.arrhythmia_head(features),   # global CLS only
            "mi":         self.mi_head(mi_input),            # CLS + lead-specific
            "features":   features,
        }

        if self.hrv_enabled and self.hrv_head is not None:
            out["hrv"] = self.hrv_head(features)

        return out

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Inference-mode forward: returns sigmoid-thresholded binary predictions.
        """
        self.eval()
        with torch.no_grad():
            raw = self.forward(x)
        return {
            "arrhythmia": (torch.sigmoid(raw["arrhythmia"]) >= threshold).float(),
            "mi":         (torch.sigmoid(raw["mi"]) >= threshold).float(),
            "hrv":        raw.get("hrv"),
            "features":   raw["features"],
        }


def build_model(cfg: dict, backbone: nn.Module) -> MultiTaskECGModel:
    """
    Build MultiTaskECGModel from config dict.

    Expected cfg keys:
        model.d_model, model.dropout, model.mi_lead_dim
        hrv.enabled, hrv.features
        label_groups.arrhythmia, label_groups.mi
    """
    arrhythmia_labels = cfg.get("label_groups", {}).get("arrhythmia", [0, 1, 2, 3, 4])
    mi_labels = cfg.get("label_groups", {}).get("mi", [5, 6])

    return MultiTaskECGModel(
        backbone=backbone,
        d_model=cfg["model"]["d_model"],
        num_arrhythmia_labels=len(arrhythmia_labels),
        num_mi_labels=len(mi_labels),
        hrv_enabled=cfg.get("hrv", {}).get("enabled", True),
        num_hrv_targets=len(cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])),
        head_hidden_dim=cfg["model"].get("dim_feedforward", 256) // 4,
        dropout=cfg["model"]["dropout"],
        mi_lead_dim=cfg["model"].get("mi_lead_dim", 64),  # [Phase 2]
    )


if __name__ == "__main__":
    from src.models.ecg_transformer import ECGTransformer

    backbone = ECGTransformer(
        num_leads=12, signal_length=1000, patch_size=25,
        d_model=192, nhead=8, num_encoder_layers=5,
    )
    model = MultiTaskECGModel(backbone=backbone, d_model=192, hrv_enabled=True, mi_lead_dim=64)

    x = torch.randn(4, 12, 1000)
    out = model(x)
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total:,}")
