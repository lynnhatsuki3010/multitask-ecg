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


# ─── Supervised Contrastive Loss ──────────────────────────────────────────────

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    Adapted for multi-label by defining positive pairs as samples that have exactly the same label combination.
    Features should be L2 normalized before passing into this loss.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] L2 normalized embeddings
        labels: [B, C] Multi-label binary targets
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Compute exact match mask
        # labels: [B, C] -> mask: [B, B]
        labels_eq = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=-1)
        mask = labels_eq.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # mask sum
        mask_sum = mask.sum(1)
        valid_rows = (mask_sum > 0).float()
        mask_sum = mask_sum.clamp(min=1.0)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = (loss * valid_rows).sum() / (valid_rows.sum() + 1e-12)

        return loss

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
        mi_contrastive_weight: float = 0.0,
        hrv_enabled: bool = True,
        # ── Class imbalance tricks ──────────────────────────────────────────
        arrhythmia_pos_weight: Optional[torch.Tensor] = None,   # BCE pos_weight
        mi_pos_weight: Optional[torch.Tensor] = None,
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
        self.imi_weight = imi_weight
        self.asmi_weight = asmi_weight
        self.hrv_weight = hrv_weight
        self.mi_contrastive_weight = mi_contrastive_weight
        self.hrv_enabled = hrv_enabled
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.use_asl = use_asl
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.arrhythmia_pos_weight = arrhythmia_pos_weight
        self.mi_pos_weight = mi_pos_weight
        
        if self.mi_contrastive_weight > 0:
            self.supcon_loss_fn = SupervisedContrastiveLoss()
        else:
            self.supcon_loss_fn = None

        if use_asl:
            self.arrhythmia_loss_fn = AsymmetricLoss(
                gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip,
                pos_weight=arrhythmia_pos_weight,
            )
            self.mi_loss_fn = AsymmetricLoss(
                gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip,
                pos_weight=mi_pos_weight,
            )
            imi_pw = mi_pos_weight[0:1] if mi_pos_weight is not None else None
            asmi_pw = mi_pos_weight[1:2] if mi_pos_weight is not None else None
            self.imi_loss_fn = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip, pos_weight=imi_pw)
            self.asmi_loss_fn = AsymmetricLoss(gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip, pos_weight=asmi_pw)
        elif use_focal:
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

        if self.mi_contrastive_weight > 0 and "mi_proj" in preds:
            losses["mi_contrastive"] = self.supcon_loss_fn(preds["mi_proj"], targets["mi"])
            losses["total"] = losses["total"] + self.mi_contrastive_weight * losses["mi_contrastive"]
        else:
            losses["mi_contrastive"] = torch.tensor(0.0, device=preds["arrhythmia"].device)

        return losses
