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


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    ASL(p_t) = - (1-p_t)^gamma_pos * log(p_t) for y=1
               - (p_m)^gamma_neg * log(1-p_m) for y=0
    where p_m = max(p - m, 0)
    """
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        p_pos = probs
        p_neg = 1 - probs

        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
            
        loss_pos = -targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = -(1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        
        if self.gamma_pos > 0:
            loss_pos = loss_pos * (1 - p_pos) ** self.gamma_pos
            
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (1 - p_neg) ** self.gamma_neg
            
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        return (loss_pos + loss_neg).mean()


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
        conduction_weight: float = 1.0,
        imi_weight: float = 1.0,
        asmi_weight: float = 1.0,
        ilmi_weight: float = 1.0,
        ami_weight: float = 1.0,
        hrv_weight: float = 0.1,
        hrv_enabled: bool = True,
        # ── Class imbalance tricks ──────────────────────────────────────────
        arrhythmia_pos_weight: Optional[torch.Tensor] = None,   # BCE pos_weight
        mi_pos_weight: Optional[torch.Tensor] = None,
        conduction_pos_weight: Optional[torch.Tensor] = None,
        use_focal: bool = False,     # replace BCE with Focal Loss
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_asl: bool = False,       # replace BCE with Asymmetric Loss
        asl_gamma_pos: float = 1.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        # ── Regularization tricks ───────────────────────────────────────────
        label_smoothing: float = 0.0,
        hrv_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.arrhythmia_weight = arrhythmia_weight
        self.mi_weight = mi_weight
        self.conduction_weight = conduction_weight
        self.imi_weight = imi_weight
        self.asmi_weight = asmi_weight
        self.ilmi_weight = ilmi_weight
        self.ami_weight = ami_weight
        self.hrv_weight = hrv_weight
        self.hrv_enabled = hrv_enabled
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.use_asl = use_asl
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.arrhythmia_pos_weight = arrhythmia_pos_weight
        self.mi_pos_weight = mi_pos_weight
        self.conduction_pos_weight = conduction_pos_weight

        def _make_loss(pos_weight: Optional[torch.Tensor]) -> nn.Module:
            if use_asl:
                return AsymmetricLoss(
                    gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip,
                    pos_weight=pos_weight,
                )
            if use_focal:
                return BinaryFocalLoss(
                    gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight,
                )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.arrhythmia_loss_fn = _make_loss(arrhythmia_pos_weight)
        self.mi_loss_fn = _make_loss(mi_pos_weight)
        self.conduction_loss_fn = _make_loss(conduction_pos_weight)

        # Pre-built per-label losses so each MI sub-label keeps its own pos_weight
        # and can be re-weighted independently (split path for 2- and 4-label heads).
        # MI column order is fixed as [IMI, ASMI, ILMI, AMI].
        def _slice_pw(idx: int) -> Optional[torch.Tensor]:
            if mi_pos_weight is None or idx >= mi_pos_weight.shape[0]:
                return None
            return mi_pos_weight[idx:idx + 1]

        self.imi_loss_fn = _make_loss(_slice_pw(0))
        self.asmi_loss_fn = _make_loss(_slice_pw(1))
        self.ilmi_loss_fn = _make_loss(_slice_pw(2))
        self.ami_loss_fn = _make_loss(_slice_pw(3))

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
            arrhy_targets = targets.get("arrhythmia")
            mi_targets    = targets.get("mi")
            cond_targets  = targets.get("conduction", None)
            if arrhy_targets is not None:
                arrhy_targets = arrhy_targets * (1.0 - eps)
            if mi_targets is not None:
                mi_targets = mi_targets * (1.0 - eps)
            if cond_targets is not None:
                cond_targets = cond_targets * (1.0 - eps)
        else:
            arrhy_targets = targets.get("arrhythmia")
            mi_targets    = targets.get("mi")
            cond_targets  = targets.get("conduction", None)

        device = next(iter(preds.values())).device
        if arrhy_targets is None:
            arrhy_targets = torch.zeros(0, device=device)
        if mi_targets is None:
            mi_targets = torch.zeros(0, device=device)

        # Arrhythmia loss
        if "arrhythmia" in preds and arrhy_targets.numel() > 0:
            losses["arrhythmia"] = self.arrhythmia_loss_fn(
                preds["arrhythmia"], arrhy_targets
            )
        else:
            losses["arrhythmia"] = torch.tensor(0.0, device=device)

        # MI loss: keep IMI and ASMI separate so we can prioritize IMI
        if "mi" not in preds:
            losses["mi"] = torch.tensor(0.0, device=device)
            losses["imi"] = losses["mi"]
            losses["asmi"] = losses["mi"]
        elif preds["mi"].shape[1] == 2:
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
        elif preds["mi"].shape[1] == 4:
            # 4-label MI head: [IMI, ASMI, ILMI, AMI]. Compute per-label losses so
            # each sub-label keeps its own pos_weight and can be re-weighted via
            # loss_weights.{imi,asmi,ilmi,ami}.
            losses["imi"]  = self.imi_loss_fn(preds["mi"][:, 0], mi_targets[:, 0])
            losses["asmi"] = self.asmi_loss_fn(preds["mi"][:, 1], mi_targets[:, 1])
            losses["ilmi"] = self.ilmi_loss_fn(preds["mi"][:, 2], mi_targets[:, 2])
            losses["ami"]  = self.ami_loss_fn(preds["mi"][:, 3], mi_targets[:, 3])

            mi_weight_sum = max(
                self.imi_weight + self.asmi_weight
                + self.ilmi_weight + self.ami_weight,
                1e-8,
            )
            losses["mi"] = (
                self.imi_weight * losses["imi"]
                + self.asmi_weight * losses["asmi"]
                + self.ilmi_weight * losses["ilmi"]
                + self.ami_weight * losses["ami"]
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
                losses["hrv"] = torch.tensor(0.0, device=device)
        else:
            losses["hrv"] = torch.tensor(0.0, device=device)

        # Conduction loss
        if cond_targets is not None and "conduction" in preds:
            losses["conduction"] = self.conduction_loss_fn(
                preds["conduction"], cond_targets
            )
        else:
            losses["conduction"] = torch.tensor(0.0, device=device)

        # Weighted total
        total = torch.tensor(0.0, device=device)
        if self.arrhythmia_weight > 0 and "arrhythmia" in preds:
            total = total + self.arrhythmia_weight * losses["arrhythmia"]
        if self.mi_weight > 0 and "mi" in preds:
            total = total + self.mi_weight * losses["mi"]
        if self.conduction_weight > 0 and "conduction" in preds:
            total = total + self.conduction_weight * losses["conduction"]
        if self.hrv_weight > 0:
            total = total + self.hrv_weight * losses["hrv"]
        losses["total"] = total

        return losses
