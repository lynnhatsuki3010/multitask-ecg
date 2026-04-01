"""
multitask_head.py
Multi-task output heads for ECG Transformer.

Tasks:
  1. Arrhythmia classification  → multi-label BCE (5 labels: NORM, AFIB, STACH, SBRAD, AFLT)
  2. MI classification          → multi-label BCE (2 labels: IMI, ASMI)
  3. HRV regression (optional)  → MSE (3 targets: rmssd, sdnn, mean_hr)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


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


class MultiTaskECGModel(nn.Module):
    """
    Full model = Backbone (ECGTransformer) + Multi-task heads.

    Outputs raw logits (no sigmoid/softmax) — loss functions handle activation.
    For inference, apply torch.sigmoid() on classification outputs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int = 128,
        num_arrhythmia_labels: int = 5,   # NORM, AFIB, STACH, SBRAD, AFLT
        num_mi_labels: int = 2,            # IMI, ASMI
        hrv_enabled: bool = True,
        num_hrv_targets: int = 3,          # rmssd, sdnn, mean_hr
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.hrv_enabled = hrv_enabled

        # Arrhythmia classification head
        self.arrhythmia_head = TaskHead(
            in_features=d_model,
            out_features=num_arrhythmia_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

        # MI classification head
        self.mi_head = TaskHead(
            in_features=d_model,
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
              'features':   (B, d_model)                 backbone embeddings
        """
        features = self.backbone(x)  # (B, d_model)

        out = {
            "arrhythmia": self.arrhythmia_head(features),
            "mi":         self.mi_head(features),
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
        model.d_model, hrv.enabled, model.dropout
        plus label_groups.arrhythmia, label_groups.mi
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
    )


if __name__ == "__main__":
    from src.models.ecg_transformer import ECGTransformer

    backbone = ECGTransformer(
        num_leads=12, signal_length=1000, patch_size=25,
        d_model=128, nhead=4, num_encoder_layers=4,
    )
    model = MultiTaskECGModel(backbone=backbone, d_model=128, hrv_enabled=True)

    x = torch.randn(4, 12, 1000)
    out = model(x)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total:,}")
