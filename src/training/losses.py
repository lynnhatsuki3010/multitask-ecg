"""
losses.py
Multi-task loss combining arrhythmia BCE, MI BCE, and HRV MSE.

Baseline: plain BCE, no pos_weight, no label smoothing.
Incremental tricks (enable one at a time to measure impact):
  - use_focal    : replace BCE with Focal Loss (handles class imbalance)
  - pos_weight   : per-class positive weights for BCE
  - label_smoothing : soft labels (positive only — negatives stay 0 for pos_weight compat)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional



# ─── Focal Loss ───────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0  → standard BCE.
    Recommended defaults: gamma=2.0, alpha=0.25 for rare positive class.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ─── Multi-task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):

    """
    Weighted sum of:
      - Arrhythmia BCE loss (multi-label, with optional pos_weight)
      - MI BCE loss         (multi-label, with optional pos_weight)
      - HRV MSE loss        (regression on all samples, optional)
    """

    def __init__(
        self,
        arrhythmia_weight: float = 1.0,
        mi_weight: float = 1.0,
        imi_weight: float = 1.0,
        asmi_weight: float = 1.0,
        hrv_weight: float = 0.1,
        hrv_enabled: bool = True,
        # ── Class imbalance tricks ──────────────────────────────────────────
        arrhythmia_pos_weight: Optional[torch.Tensor] = None,   # BCE pos_weight
        mi_pos_weight: Optional[torch.Tensor] = None,
        use_focal: bool = False,     # replace BCE with Focal Loss
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        # ── Regularization tricks ───────────────────────────────────────────
        label_smoothing: float = 0.0,
        hrv_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.arrhythmia_weight = arrhythmia_weight
        self.mi_weight = mi_weight
        self.imi_weight = imi_weight
        self.asmi_weight = asmi_weight
        self.hrv_weight = hrv_weight
        self.hrv_enabled = hrv_enabled
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.arrhythmia_pos_weight = arrhythmia_pos_weight
        self.mi_pos_weight = mi_pos_weight

        if use_focal:
            self.arrhythmia_loss_fn = BinaryFocalLoss(
                gamma=focal_gamma, alpha=focal_alpha,
                pos_weight=arrhythmia_pos_weight,
            )
            self.mi_loss_fn = BinaryFocalLoss(
                gamma=focal_gamma, alpha=focal_alpha,
                pos_weight=mi_pos_weight,
            )
            # Pre-built per-label focal losses for IMI/ASMI split path
            imi_pw = mi_pos_weight[0:1] if mi_pos_weight is not None else None
            asmi_pw = mi_pos_weight[1:2] if mi_pos_weight is not None else None
            self.imi_loss_fn = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha, pos_weight=imi_pw)
            self.asmi_loss_fn = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha, pos_weight=asmi_pw)
        else:
            self.arrhythmia_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=arrhythmia_pos_weight
            )
            self.mi_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=mi_pos_weight
            )
            # Pre-built per-label BCE for IMI/ASMI split path
            imi_pw = mi_pos_weight[0:1] if mi_pos_weight is not None else None
            asmi_pw = mi_pos_weight[1:2] if mi_pos_weight is not None else None
            self.imi_loss_fn = nn.BCEWithLogitsLoss(pos_weight=imi_pw)
            self.asmi_loss_fn = nn.BCEWithLogitsLoss(pos_weight=asmi_pw)

        if hrv_loss_type == "mse":
            self.hrv_loss_fn = nn.MSELoss()
        else:
            self.hrv_loss_fn = nn.SmoothL1Loss(beta=0.5)

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

        # Label smoothing: only smooth positive labels (1→1-ε), keep negatives at 0.
        # This preserves pos_weight effectiveness — pos_weight relies on negative
        # targets being exactly 0 to correctly re-weight the positive class.
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            arrhy_targets = targets["arrhythmia"] * (1.0 - eps)
            mi_targets    = targets["mi"]          * (1.0 - eps)
        else:
            arrhy_targets = targets["arrhythmia"]
            mi_targets    = targets["mi"]

        # Arrhythmia loss
        losses["arrhythmia"] = self.arrhythmia_loss_fn(
            preds["arrhythmia"], arrhy_targets
        )

        # MI loss: keep IMI and ASMI separate so we can prioritize IMI
        if preds["mi"].shape[1] == 2:
            losses["imi"] = self.imi_loss_fn(
                preds["mi"][:, 0],
                mi_targets[:, 0],
            )
            losses["asmi"] = self.asmi_loss_fn(
                preds["mi"][:, 1],
                mi_targets[:, 1],
            )

            mi_weight_sum = max(self.imi_weight + self.asmi_weight, 1e-8)
            losses["mi"] = (
                self.imi_weight * losses["imi"]
                + self.asmi_weight * losses["asmi"]
            ) / mi_weight_sum
        else:
            losses["mi"] = self.mi_loss_fn(
                preds["mi"], mi_targets
            )
            losses["imi"] = losses["mi"]
            losses["asmi"] = losses["mi"]

        # HRV loss — masked MSE (only valid samples, i.e. R-peaks >= 3)
        if self.hrv_enabled and "hrv" in preds and "hrv" in targets:
            mask = targets.get("hrv_valid")   # (B,) bool or None
            if mask is not None and mask.any():
                losses["hrv"] = self.hrv_loss_fn(
                    preds["hrv"][mask], targets["hrv"][mask]
                )
            elif mask is None:
                losses["hrv"] = self.hrv_loss_fn(preds["hrv"], targets["hrv"])
            else:
                # All samples invalid in this batch
                losses["hrv"] = torch.tensor(0.0, device=preds["arrhythmia"].device)
        else:
            losses["hrv"] = torch.tensor(0.0, device=preds["arrhythmia"].device)

        # Weighted total
        losses["total"] = (
            self.arrhythmia_weight * losses["arrhythmia"]
            + self.mi_weight       * losses["mi"]
            + self.hrv_weight      * losses["hrv"]
        )

        return losses
