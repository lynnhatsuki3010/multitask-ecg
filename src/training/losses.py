"""
losses.py
Multi-task loss combining arrhythmia BCE, MI BCE, and HRV MSE.

Baseline version — no pos_weight, no label smoothing, no masked HRV.
Add tricks incrementally on top of this baseline.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Weighted sum of:
      - Arrhythmia BCE loss (multi-label, plain)
      - MI BCE loss         (multi-label, plain)
      - HRV MSE loss        (regression on all samples, optional)
    """

    def __init__(
        self,
        arrhythmia_weight: float = 1.0,
        mi_weight: float = 1.0,
        hrv_weight: float = 0.1,
        hrv_enabled: bool = True,
    ):
        super().__init__()
        self.arrhythmia_weight = arrhythmia_weight
        self.mi_weight = mi_weight
        self.hrv_weight = hrv_weight
        self.hrv_enabled = hrv_enabled

        # Plain BCE — no pos_weight, no label smoothing
        self.arrhythmia_loss_fn = nn.BCEWithLogitsLoss()
        self.mi_loss_fn = nn.BCEWithLogitsLoss()
        # Plain MSE — all samples, no masking
        self.hrv_loss_fn = nn.MSELoss()

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds:   dict with keys 'arrhythmia', 'mi', optionally 'hrv'.
            targets: dict with keys 'arrhythmia', 'mi', optionally 'hrv'.

        Returns:
            dict with keys: 'total', 'arrhythmia', 'mi', 'hrv'.
        """
        losses = {}

        # Arrhythmia loss (plain BCE)
        losses["arrhythmia"] = self.arrhythmia_loss_fn(
            preds["arrhythmia"], targets["arrhythmia"]
        )

        # MI loss (plain BCE)
        losses["mi"] = self.mi_loss_fn(
            preds["mi"], targets["mi"]
        )

        # HRV loss (plain MSE, all samples)
        if self.hrv_enabled and "hrv" in preds and "hrv" in targets:
            losses["hrv"] = self.hrv_loss_fn(preds["hrv"], targets["hrv"])
        else:
            losses["hrv"] = torch.tensor(0.0, device=preds["arrhythmia"].device)

        # Weighted total
        losses["total"] = (
            self.arrhythmia_weight * losses["arrhythmia"]
            + self.mi_weight       * losses["mi"]
            + self.hrv_weight      * losses["hrv"]
        )

        return losses


def build_loss(cfg: dict) -> MultiTaskLoss:
    """Build MultiTaskLoss from config dict (baseline, no pos_weight)."""
    lw      = cfg.get("training", {}).get("loss_weights", {})
    hrv_cfg = cfg.get("hrv", {})

    return MultiTaskLoss(
        arrhythmia_weight = lw.get("arrhythmia", 1.0),
        mi_weight         = lw.get("mi",         1.0),
        hrv_weight        = lw.get("hrv",        0.1),
        hrv_enabled       = hrv_cfg.get("enabled", True),
    )
