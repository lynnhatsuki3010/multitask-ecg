"""
metrics.py
Evaluation metrics: AUROC, F1, Accuracy per task and label.
"""
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, 
    f1_score,
    fbeta_score,
    average_precision_score,
    precision_score,
    recall_score,
    classification_report
)
from typing import Dict, List, Optional, Union

from src.data.label_builder import DEFAULT_LABELS

# Single source of truth with label_builder.DEFAULT_LABELS (PVC, not SBRAD).
MI_LABELS = ["IMI", "ASMI"]
ARRHYTHMIA_LABELS = [lbl for lbl in DEFAULT_LABELS if lbl not in MI_LABELS]
LABEL_NAMES = list(DEFAULT_LABELS)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    prefix: str = "",
) -> Dict[str, Union[float, str]]:
    """
    Compute per-label and macro-averaged AUROC, AUPRC, F1.

    Args:
        y_true:  (N, K) binary ground truth.
        y_score: (N, K) sigmoid probabilities.
        y_pred:  (N, K) binary predictions (after threshold).
        label_names: list of K strings.
        prefix: string prepended to all metric keys.

    Returns:
        Flat dict of metric_name → float.
    """
    metrics = {}
    n_labels = y_true.shape[1]

    auroc_list, auprc_list, f1_list = [], [], []
    precision_list, recall_list = [], []

    for i, name in enumerate(label_names):
        col_true  = y_true[:, i]
        col_score = y_score[:, i]
        col_pred  = y_pred[:, i]

        n_pos = col_true.sum()
        if n_pos == 0 or n_pos == len(col_true):
            # Skip degenerate columns
            continue

        auroc = roc_auc_score(col_true, col_score)
        auprc = average_precision_score(col_true, col_score)
        f1    = f1_score(col_true, col_pred, zero_division=0)
        prec  = precision_score(col_true, col_pred, zero_division=0)
        rec   = recall_score(col_true, col_pred, zero_division=0)

        key = f"{prefix}{name}" if prefix else name
        metrics[f"auroc/{key}"] = auroc
        metrics[f"auprc/{key}"] = auprc
        metrics[f"f1/{key}"]    = f1
        metrics[f"prec/{key}"]  = prec
        metrics[f"rec/{key}"]   = rec

        auroc_list.append(auroc)
        auprc_list.append(auprc)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    # Macro averages
    p = f"{prefix}macro" if prefix else "macro"
    if auroc_list:
        metrics[f"auroc/{p}"]  = float(np.mean(auroc_list))
        metrics[f"auprc/{p}"]  = float(np.mean(auprc_list))
        metrics[f"f1/{p}"]     = float(np.mean(f1_list))
        metrics[f"prec/{p}"]   = float(np.mean(precision_list))
        metrics[f"rec/{p}"]    = float(np.mean(recall_list))

    # Add classification report string
    report_key = f"report/{prefix[:-1]}" if prefix else "report"
    metrics[report_key] = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )

    return metrics


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: List[str],
    candidates: np.ndarray = None,
    min_pos_count: int = 20,
    max_rare_threshold: float = 0.4,
    class_beta: Optional[Dict[str, float]] = None,
    min_threshold: float = 0.1,
    class_min_threshold: Optional[Dict[str, float]] = None,
    class_min_precision: Optional[Dict[str, float]] = None,
    class_max_threshold: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    For each class, sweep threshold candidates and pick the one
    that maximises F_beta on the given set (should be val set only).

    - Rare classes (< min_pos_count positives): threshold capped at max_rare_threshold
      to avoid degenerate precision=1/recall~0 outcomes (e.g. AFLT).
    - class_beta: per-class beta for F_beta score.
        beta=1.0  → standard F1 (default)
        beta=0.5  → precision-weighted (good for MI: reduces false positives)
        beta=2.0  → recall-weighted (good for critical missed diagnoses)

    Args:
        y_true:       (N, K) binary ground truth.
        y_score:      (N, K) sigmoid probabilities.
        label_names:  list of K strings.
        candidates:   1-D array of thresholds to try. Defaults to 0.05…0.95.
        min_pos_count: classes with fewer positive val samples get capped threshold.
        max_rare_threshold: maximum threshold for rare classes.
        class_beta:   dict of label_name → beta. Missing labels default to 1.0.
        min_threshold: global lower bound on searched thresholds.
        class_min_threshold: per-label lower bounds overriding min_threshold.
        class_min_precision: optional per-label minimum precision constraints.
        class_max_threshold: per-label upper bounds for ALL classes (rare or not).
            Prevents high-variance thresholds on small val sets (e.g. SBRAD=0.90).

    Returns:
        Dict  label_name → optimal threshold (float).
    """
    if candidates is None:
        candidates = np.arange(0.05, 0.96, 0.05)
    if class_beta is None:
        class_beta = {}
    if class_min_threshold is None:
        class_min_threshold = {}
    if class_min_precision is None:
        class_min_precision = {}
    if class_max_threshold is None:
        class_max_threshold = {}

    thresholds = {}
    for i, name in enumerate(label_names):
        col_true  = y_true[:, i]
        col_score = y_score[:, i]

        if col_true.sum() == 0 or col_true.sum() == len(col_true):
            thresholds[name] = 0.5
            continue

        n_pos    = int(col_true.sum())
        is_rare  = n_pos < min_pos_count
        beta     = class_beta.get(name, 1.0)
        threshold_floor = class_min_threshold.get(name, min_threshold)
        min_prec = class_min_precision.get(name, None)

        # For rare classes, restrict search to below the rare cap
        search_candidates = candidates[candidates >= threshold_floor]
        if len(search_candidates) == 0:
            search_candidates = np.array([threshold_floor])
        if is_rare:
            search_candidates = search_candidates[search_candidates <= max_rare_threshold]
            if len(search_candidates) == 0:
                search_candidates = np.array([max(threshold_floor, min(max_rare_threshold, 0.5))])
        # Apply per-class upper bound (prevents overfitting to small val sets)
        if name in class_max_threshold:
            ceil = class_max_threshold[name]
            search_candidates = search_candidates[search_candidates <= ceil]
            if len(search_candidates) == 0:
                search_candidates = np.array([min(ceil, threshold_floor)])

        best_score, best_t = -1.0, 0.5
        for t in search_candidates:
            col_pred = (col_score >= t).astype(float)
            if min_prec is not None:
                prec = precision_score(col_true, col_pred, zero_division=0)
                if prec < min_prec:
                    continue
            score = fbeta_score(col_true, col_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score, best_t = score, float(t)

        thresholds[name] = best_t

    return thresholds


def apply_thresholds(
    y_score: np.ndarray,
    label_names: List[str],
    thresholds: Dict[str, float],
    default: float = 0.5,
) -> np.ndarray:
    """Apply per-class thresholds to score matrix → binary prediction matrix."""
    y_pred = np.zeros_like(y_score)
    for i, name in enumerate(label_names):
        t = thresholds.get(name, default)
        y_pred[:, i] = (y_score[:, i] >= t).astype(float)
    return y_pred


def compute_hrv_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str] = ["rmssd", "sdnn", "mean_hr"],
) -> Dict[str, float]:
    """
    Compute MAE and RMSE for HRV regression.
    Only considers samples where ground truth is non-zero (valid HRV).
    """
    metrics = {}
    # valid mask: samples where ground truth is non-zero
    valid = (np.abs(y_true).sum(axis=-1) > 0)
    if valid.sum() == 0:
        return metrics

    for i, name in enumerate(feature_names):
        t = y_true[valid, i]
        p = y_pred[valid, i]
        mae  = float(np.mean(np.abs(t - p)))
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        metrics[f"hrv_mae/{name}"]  = mae
        metrics[f"hrv_rmse/{name}"] = rmse

    return metrics


class MetricsAccumulator:
    """
    Accumulates model outputs across batches for epoch-level metric computation.
    """

    def __init__(self, arrhythmia_labels: List[str], mi_labels: List[str], hrv_enabled: bool = True):
        self.arrhythmia_labels = arrhythmia_labels
        self.mi_labels = mi_labels
        self.hrv_enabled = hrv_enabled
        self.reset()

    def reset(self):
        self.arrhythmia_true,  self.arrhythmia_score,  self.arrhythmia_pred  = [], [], []
        self.mi_true,          self.mi_score,           self.mi_pred          = [], [], []
        self.hrv_true,         self.hrv_pred                                  = [], []
        self.losses = {"total": [], "arrhythmia": [], "mi": [], "imi": [], "asmi": [], "hrv": []}

    def update(
        self,
        preds_raw: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Optional[Dict[str, torch.Tensor]] = None,
        threshold: float = 0.5,
        dynamic_thresholds: Optional[Dict[str, float]] = None,
    ):
        """Accumulate a batch.
        
        Args:
            dynamic_thresholds: Optional per-class threshold dict from DynamicThresholdHead.
                                 If provided, overrides the scalar `threshold` for binarization.
        """
        with torch.no_grad():
            # Arrhythmia
            a_score = torch.sigmoid(preds_raw["arrhythmia"]).cpu().numpy()
            if dynamic_thresholds is not None:
                a_pred = apply_thresholds(a_score, self.arrhythmia_labels, dynamic_thresholds, default=threshold)
            else:
                a_pred = (a_score >= threshold).astype(float)
            a_true  = targets["arrhythmia"].cpu().numpy()
            self.arrhythmia_score.append(a_score)
            self.arrhythmia_pred.append(a_pred)
            self.arrhythmia_true.append(a_true)

            # MI
            mi_score = torch.sigmoid(preds_raw["mi"]).cpu().numpy()
            if dynamic_thresholds is not None:
                mi_pred = apply_thresholds(mi_score, self.mi_labels, dynamic_thresholds, default=threshold)
            else:
                mi_pred = (mi_score >= threshold).astype(float)
            mi_true  = targets["mi"].cpu().numpy()
            self.mi_score.append(mi_score)
            self.mi_pred.append(mi_pred)
            self.mi_true.append(mi_true)

            # HRV
            if self.hrv_enabled and "hrv" in preds_raw:
                self.hrv_pred.append(preds_raw["hrv"].cpu().numpy())
                self.hrv_true.append(targets["hrv"].cpu().numpy())

            # Losses
            if loss_dict:
                for k in self.losses:
                    if k in loss_dict:
                        self.losses[k].append(float(loss_dict[k].item()))

    def compute(self, threshold: float = 0.5, hrv_feature_names: List[str] = ["rmssd", "sdnn", "mean_hr"]) -> Dict[str, Union[float, str]]:
        """Return all metrics as a flat dict."""
        metrics: Dict[str, Union[float, str]] = {}

        # Loss averages
        for k, v in self.losses.items():
            if v:
                metrics[f"loss/{k}"] = float(np.mean(v))

        # Arrhythmia metrics
        if self.arrhythmia_true:
            y_true  = np.concatenate(self.arrhythmia_true, axis=0)
            y_score = np.concatenate(self.arrhythmia_score, axis=0)
            y_pred  = np.concatenate(self.arrhythmia_pred,  axis=0)
            metrics.update(compute_classification_metrics(y_true, y_score, y_pred, self.arrhythmia_labels, prefix="arrhy/"))

        # MI metrics
        if self.mi_true:
            y_true  = np.concatenate(self.mi_true, axis=0)
            y_score = np.concatenate(self.mi_score, axis=0)
            y_pred  = np.concatenate(self.mi_pred,  axis=0)
            metrics.update(compute_classification_metrics(y_true, y_score, y_pred, self.mi_labels, prefix="mi/"))

        # HRV metrics
        if self.hrv_enabled and self.hrv_pred:
            y_true = np.concatenate(self.hrv_true, axis=0)
            y_pred = np.concatenate(self.hrv_pred,  axis=0)
            metrics.update(compute_hrv_metrics(y_true, y_pred, hrv_feature_names))

        return metrics
