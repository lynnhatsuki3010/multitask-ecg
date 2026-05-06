"""
02_train.py
─────────────────────────────────────────────────────────────────
Step 2: Train the ECG multi-task model.

Requires 01_build_metadata.py to have been run first.

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --config configs/config.yaml
    python scripts/02_train.py --no-hrv          # disable HRV task
    python scripts/02_train.py --epochs 10       # override epochs
    python scripts/02_train.py --batch-size 32
    python scripts/02_train.py --debug           # 200 samples only, 3 epochs
"""
import os
import sys
import argparse
import random
import json
import time
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import PTBXLDataset, collate_fn
from src.data.policy import audit_dataset_policy
from src.models.factory import build_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer


# ─── Plot helper ─────────────────────────────────────────────────────────────────────

def plot_training_history(history: dict, run_dir: str, freeze_epoch: int = -1):
    """
    Save training curves as a 2×3 PNG grid:
      Row 1: Total Loss | Arrhy AUROC | MI AUROC
      Row 2: IMI F1     | IMI AUPRC   | Gradient Norm ratio (MI/Backbone)

    A vertical dashed line marks the Phase 2 boundary if freeze_epoch > 0.
    """
    train_h = history.get("train", [])
    val_h   = history.get("val",   [])
    if not val_h:
        print("  No history to plot.")
        return

    epochs = list(range(1, len(val_h) + 1))

    def extract(records, key):
        return [r.get(key, float("nan")) for r in records]

    def add_phase_line(ax):
        """Draw vertical dashed line at phase 2 boundary."""
        if freeze_epoch > 0 and freeze_epoch <= len(epochs):
            ax.axvline(x=freeze_epoch, color="darkorange", linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"Phase 2 start (ep {freeze_epoch})")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # ── (0,0) Total Loss ─────────────────────────────────────────
    ax = axes[0, 0]
    if train_h:
        ax.plot(epochs[:len(train_h)], extract(train_h, "loss/total"),
                label="Train Loss", color="steelblue")
    ax.plot(epochs, extract(val_h, "loss/total"),
            label="Val Loss", color="coral", linestyle="--")
    add_phase_line(ax)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (0,1) Arrhythmia AUROC ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, extract(val_h, "auroc/arrhy/macro"),
            label="Val Arrhy AUROC", color="mediumseagreen")
    ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=0.8, label="0.90 target")
    add_phase_line(ax)
    ax.set_title("Arrhythmia AUROC (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (0,2) MI AUROC ───────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(epochs, extract(val_h, "auroc/mi/macro"),
            label="Val MI AUROC", color="mediumpurple")
    ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=0.8, label="0.90 target")
    add_phase_line(ax)
    ax.set_title("MI AUROC (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (1,0) IMI F1 ─────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, extract(val_h, "f1/mi/IMI"),
            label="Val IMI F1", color="tomato")
    add_phase_line(ax)
    ax.set_title("IMI F1 (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (1,1) IMI AUPRC ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, extract(val_h, "auprc/mi/IMI"),
            label="Val IMI AUPRC", color="darkorange")
    add_phase_line(ax)
    ax.set_title("IMI AUPRC (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── (1,2) Gradient Norm ratio MI/Backbone ────────────────────
    ax = axes[1, 2]
    gn_mi = extract(train_h, "grad_norm/mi") if train_h else []
    gn_bb = extract(train_h, "grad_norm/backbone") if train_h else []
    if gn_mi and any(not np.isnan(v) for v in gn_mi):
        ratios = [m / max(b, 1e-8) for m, b in zip(gn_mi, gn_bb)]
        ax.plot(epochs[:len(gn_mi)], ratios,
                label="GradNorm MI/Backbone", color="slategray")
        add_phase_line(ax)
        ax.set_title("Gradient Norm Ratio (Train)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Ratio")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No gradient norm data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")


def plot_mi_error_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict,
    all_label_names: list,
    all_label_indices: list,
    mi_label_names: list,
    mi_indices: list,
    run_dir: str,
):
    """
    For each MI label, show what other labels co-occur in FN and FP cases.

    FN analysis (missed MI):  GT=1 but Pred=0 → what DID the model predict?
    FP analysis (false MI):   GT=0 but Pred=1 → what was the actual ground truth?

    Saved to run_dir/mi_error_analysis.png
    """
    n_mi = len(mi_label_names)
    fig, axes = plt.subplots(n_mi, 2, figsize=(14, 5 * n_mi))
    if n_mi == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("MI Error Analysis — Co-label Distribution in FP and FN Cases",
                 fontsize=13, fontweight="bold")

    for row_i, (mi_name, mi_col) in enumerate(zip(mi_label_names, mi_indices)):
        thr   = thresholds.get(mi_name, 0.5)
        gt    = y_true[:, mi_col].astype(int)
        pred  = (y_score[:, mi_col] >= thr).astype(int)

        fn_mask = (gt == 1) & (pred == 0)   # missed positives
        fp_mask = (gt == 0) & (pred == 1)   # false alarms

        # Collect all OTHER label names (everything except the MI label itself)
        other_names  = [n for n, c in zip(all_label_names, all_label_indices) if n != mi_name]
        other_cols   = [c for n, c in zip(all_label_names, all_label_indices) if n != mi_name]

        def co_freq(mask, label_col):
            """Fraction of samples in mask where another label is predicted/true positive."""
            if mask.sum() == 0:
                return 0.0
            return float((y_score[:, label_col][mask] >= thresholds.get(
                all_label_names[all_label_indices.index(label_col)]
                if label_col in all_label_indices else 0, 0.5
            )).mean())

        def gt_freq(mask, label_col):
            """Fraction of samples in mask where another label is ground-truth positive."""
            if mask.sum() == 0:
                return 0.0
            return float(y_true[:, label_col][mask].mean())

        fn_pred_rates = [co_freq(fn_mask, c) for c in other_cols]
        fp_gt_rates   = [gt_freq(fp_mask, c) for c in other_cols]

        colors_fn = ["tomato" if r > 0.1 else "lightcoral" for r in fn_pred_rates]
        colors_fp = ["steelblue" if r > 0.1 else "lightblue" for r in fp_gt_rates]

        # FN panel
        ax = axes[row_i, 0]
        bars = ax.bar(other_names, fn_pred_rates, color=colors_fn, edgecolor="white")
        ax.set_title(f"{mi_name} — False Negatives (n={fn_mask.sum()})\n"
                     f"What did model predict instead?", fontsize=10)
        ax.set_ylabel("Co-prediction rate")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.8)
        for bar, val in zip(bars, fn_pred_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # FP panel
        ax = axes[row_i, 1]
        bars = ax.bar(other_names, fp_gt_rates, color=colors_fp, edgecolor="white")
        ax.set_title(f"{mi_name} — False Positives (n={fp_mask.sum()})\n"
                     f"What was the true label?", fontsize=10)
        ax.set_ylabel("Ground-truth positive rate")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.8)
        for bar, val in zip(bars, fp_gt_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "mi_error_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  MI error analysis saved → {out_path}")


def plot_confusion_matrices(
    model,
    loader,
    thresholds: dict,
    arrhy_label_names: list,
    mi_label_names: list,
    arrhy_indices: list,
    mi_indices: list,
    run_dir: str,
    device,
):
    """
    Run inference on `loader`, apply per-class thresholds, then plot
    one binary 2×2 confusion matrix per label.

    Layout:
      Row 1: Arrhythmia labels (NORM, AFIB, STACH, SBRAD, AFLT)
      Row 2: MI labels         (IMI, ASMI) + empty cells

    Saved to run_dir/confusion_matrices.png
    """
    import torch
    model.eval()
    all_true  = []   # list of (B, K) tensors
    all_score = []   # list of (B, K) tensors

    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            labels = batch["labels"]               # (B, K_total)
            preds  = model(signal)

            # Collect arrhythmia scores
            a_score = torch.sigmoid(preds["arrhythmia"]).cpu()  # (B, K_arrhy)
            mi_score = torch.sigmoid(preds["mi"]).cpu()          # (B, K_mi)

            # Full score vector in label order
            K_total = labels.shape[1]
            scores = torch.zeros(labels.shape[0], K_total)
            for j, idx in enumerate(arrhy_indices):
                scores[:, idx] = a_score[:, j]
            for j, idx in enumerate(mi_indices):
                scores[:, idx] = mi_score[:, j]

            all_true.append(labels)
            all_score.append(scores)

    y_true  = torch.cat(all_true,  dim=0).numpy()   # (N, K_total)
    y_score = torch.cat(all_score, dim=0).numpy()

    # Build ordered label info: (name, col_idx_in_full)
    label_info = [
        (nm, idx, "arrhy") for nm, idx in zip(arrhy_label_names, arrhy_indices)
    ] + [
        (nm, idx, "mi")    for nm, idx in zip(mi_label_names,    mi_indices)
    ]

    n_arrhy = len(arrhy_label_names)
    n_mi    = len(mi_label_names)
    n_cols  = max(n_arrhy, n_mi)
    n_rows  = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 7))
    fig.suptitle("Confusion Matrices (Test Set, per-class threshold)",
                 fontsize=13, fontweight="bold", y=1.01)

    row_titles = ["Arrhythmia", "MI"]
    groups     = [
        [(nm, idx) for nm, idx, g in label_info if g == "arrhy"],
        [(nm, idx) for nm, idx, g in label_info if g == "mi"],
    ]

    for row_i, (row_title, group) in enumerate(zip(row_titles, groups)):
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            if col_i >= len(group):
                ax.axis("off")
                continue

            label_name, col_idx = group[col_i]
            thr   = thresholds.get(label_name, 0.5)
            true  = y_true[:, col_idx].astype(int)
            pred  = (y_score[:, col_idx] >= thr).astype(int)

            # 2×2: rows=actual, cols=predicted
            TP = int(((true == 1) & (pred == 1)).sum())
            TN = int(((true == 0) & (pred == 0)).sum())
            FP = int(((true == 0) & (pred == 1)).sum())
            FN = int(((true == 1) & (pred == 0)).sum())
            cm = [[TN, FP], [FN, TP]]

            # Color map: greens for correct (TN,TP), reds for errors (FP,FN)
            colors = [["#b7e4c7", "#f4b8b8"],
                      ["#f4b8b8", "#b7e4c7"]]

            total = len(true)
            n_pos = true.sum()
            for r in range(2):
                for c in range(2):
                    ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1,
                                               color=colors[r][c], zorder=0))
                    ax.text(c + 0.5, 1 - r + 0.5, str(cm[r][c]),
                            ha="center", va="center",
                            fontsize=14, fontweight="bold")

            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_yticks([0.5, 1.5])
            ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
            ax.set_yticklabels(["Act 1", "Act 0"], fontsize=9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

            f1  = (2 * TP) / (2 * TP + FP + FN + 1e-9)
            sup = n_pos
            ax.set_title(
                f"{label_name}\nthr={thr:.2f} | F1={f1:.2f} | n={sup}",
                fontsize=9, pad=4
            )

        # Row label on the left
        axes[row_i, 0].set_ylabel(row_title, fontsize=11,
                                   fontweight="bold", labelpad=8)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrices saved → {out_path}")


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ECG multi-task model")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--architecture", choices=["cnn", "hybrid_transformer"], default=None, help="Override model architecture from config")
    p.add_argument("--no-hrv",     action="store_true", help="Disable HRV regression task")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--seed",       type=int,   default=None, help="Override random seed")
    p.add_argument("--debug",      action="store_true", help="Quick debug run (200 samples, 3 epochs)")
    p.add_argument("--resume",     type=str,   default=None, help="Path to checkpoint .pth to resume from")
    # ── Augmentation overrides (override config.yaml) ─────────────────────
    p.add_argument("--aug-lead-dropout",    action="store_true", default=None, help="[Phase A1] Enable lead dropout aug")
    p.add_argument("--aug-baseline-wander", action="store_true", default=None, help="[Phase A2] Enable baseline wander aug")
    p.add_argument("--aug-random-crop",     action="store_true", default=None, help="[Phase A3] Enable random crop aug")
    p.add_argument("--aug-time-warp",       action="store_true", default=None, help="[Phase A4] Enable time warp aug")
    p.add_argument("--aug-mixup",           action="store_true", default=None, help="[Phase A] Enable mixup aug")
    p.add_argument("--aug-all",             action="store_true", help="[Phase A5] Enable all augmentations")
    # ── Imbalance overrides ───────────────────────────────────────────────
    p.add_argument("--use-focal",         action="store_true", default=None, help="[Phase B] Use Focal Loss instead of BCE")
    p.add_argument("--no-pos-weight",     action="store_true", help="Disable pos_weight (use with --use-focal)")
    p.add_argument("--focal-gamma",       type=float, default=None, help="Focal Loss gamma (default: 2.0)")
    p.add_argument("--weighted-sampler",  action="store_true", default=None, help="[Phase B2] WeightedRandomSampler — oversample minority classes in each batch")
    p.add_argument("--split-method",      default=None, help="Override dataset.split_method from config for audit/reproducibility")
    p.add_argument("--split-seed",        type=int, default=None, help="Override dataset.split_seed from config for audit/reproducibility")
    p.add_argument("--split-group-key",   default=None, help="Override dataset.split_group_key from config for audit/reproducibility")
    p.add_argument("--split-train-ratio", type=float, default=None, help="Override dataset.split_train_ratio from config for audit/reproducibility")
    p.add_argument("--split-val-ratio",   type=float, default=None, help="Override dataset.split_val_ratio from config for audit/reproducibility")
    p.add_argument("--split-test-ratio",  type=float, default=None, help="Override dataset.split_test_ratio from config for audit/reproducibility")
    p.add_argument("--test-fold",         type=int, default=None, help="Override dataset.test_fold from config for audit/reproducibility")
    p.add_argument("--val-fold",          type=int, default=None, help="Override dataset.val_fold from config for audit/reproducibility")
    p.add_argument("--split-num-folds",   type=int, default=None, help="Override dataset.split_num_folds from config for audit/reproducibility")
    p.add_argument("--split-test-fold-index", type=int, default=None, help="Override dataset.split_test_fold_index from config for audit/reproducibility")
    p.add_argument("--split-val-fold-index", type=int, default=None, help="Override dataset.split_val_fold_index from config for audit/reproducibility")
    p.add_argument("--split-stratify-label", default=None, help="Override dataset.split_stratify_label from config for audit/reproducibility")
    return p.parse_args()


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_dataset_cli_overrides(cfg: dict, args) -> dict:
    dataset_cfg = cfg.setdefault("dataset", {})
    overrides = {
        "split_method": args.split_method,
        "split_seed": args.split_seed,
        "split_group_key": args.split_group_key,
        "split_train_ratio": args.split_train_ratio,
        "split_val_ratio": args.split_val_ratio,
        "split_test_ratio": args.split_test_ratio,
        "test_fold": args.test_fold,
        "val_fold": args.val_fold,
        "split_num_folds": args.split_num_folds,
        "split_test_fold_index": args.split_test_fold_index,
        "split_val_fold_index": args.split_val_fold_index,
        "split_stratify_label": args.split_stratify_label,
    }
    for key, value in overrides.items():
        if value is not None:
            dataset_cfg[key] = value
    return cfg


def load_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Using CPU (no CUDA available)")
    return dev


def load_splits_and_data(cfg: dict, args) -> dict:
    fs        = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits    = f"{cfg['paths']['splits']}_{fs}hz"

    meta_path = os.path.join(processed, "metadata.csv")
    lm_path   = os.path.join(processed, "label_matrix.npy")
    hrv_path  = os.path.join(processed, "hrv_matrix.npy")
    pw_path   = os.path.join(processed, "pos_weights.json")

    for p in [meta_path, lm_path]:
        if not os.path.exists(p):
            print(f"\nMissing: {p}")
            print("   Please run:  python scripts/01_build_metadata.py  first.\n")
            sys.exit(1)

    df           = pd.read_csv(meta_path, index_col="ecg_id")
    label_matrix = np.load(lm_path)
    hrv_matrix   = np.load(hrv_path) if os.path.exists(hrv_path) else None
    pos_weights  = json.load(open(pw_path)) if os.path.exists(pw_path) else {}

    split_indices = {}
    for split in ["train", "val", "test"]:
        idx_path = os.path.join(splits, f"{split}_indices.npy")
        if os.path.exists(idx_path):
            split_indices[split] = np.load(idx_path)
        else:
            print(f"Missing split file: {idx_path}  — will derive from strat_fold")
            test_fold = cfg["dataset"]["test_fold"]
            val_fold  = cfg["dataset"]["val_fold"]
            fold_arr  = df["strat_fold"].values
            all_idx   = np.arange(len(df))
            split_indices["test"]  = all_idx[fold_arr == test_fold]
            split_indices["val"]   = all_idx[fold_arr == val_fold]
            split_indices["train"] = all_idx[(fold_arr != test_fold) & (fold_arr != val_fold)]

    return {
        "df":           df,
        "label_matrix": label_matrix,
        "hrv_matrix":   hrv_matrix,
        "pos_weights":  pos_weights,
        "splits":       split_indices,
        "processed_path": processed,
    }


def build_datasets(cfg: dict, data: dict, args) -> dict:
    """Build PTBXLDataset objects for each split."""
    raw_path   = cfg["paths"]["raw_data"]
    fs         = cfg["dataset"]["sampling_rate"]
    length     = cfg["dataset"]["signal_length"]
    prep_cfg   = cfg.get("preprocessing", {})
    bandpass   = (prep_cfg.get("bandpass_low", 0.5), prep_cfg.get("bandpass_high", 40.0))
    notch      = prep_cfg.get("notch_freq", 50.0)
    normalize  = prep_cfg.get("normalize", "zscore")

    df           = data["df"]
    label_matrix = data["label_matrix"]
    hrv_matrix   = data["hrv_matrix"]
    splits       = data["splits"]

    datasets = {}
    for split, indices in splits.items():
        if args.debug and split == "train":
            indices = indices[:200]
        elif args.debug and split == "val":
            indices = indices[:50]

        sub_df  = df.iloc[indices]
        sub_lm  = label_matrix[indices]
        sub_hrv = hrv_matrix[indices] if hrv_matrix is not None else None

        ds = PTBXLDataset(
            metadata     = sub_df,
            label_matrix = sub_lm,
            hrv_matrix   = sub_hrv,
            base_path    = raw_path,
            sampling_rate= fs,
            target_length= length,
            bandpass     = bandpass,
            notch        = notch,
            normalize    = normalize,
            augment      = (split == "train"),
        )

        # Apply augmentation flags from config (only for train split)
        if split == "train":
            aug_cfg = cfg.get("augmentation", {})
            ds.aug_lead_dropout    = aug_cfg.get("aug_lead_dropout",    False)
            ds.aug_random_crop     = aug_cfg.get("aug_random_crop",     False)
            ds.aug_baseline_wander = aug_cfg.get("aug_baseline_wander", False)
            ds.aug_time_warp       = aug_cfg.get("aug_time_warp",       False)

        datasets[split] = ds
        print(f"  {split:6s}: {len(datasets[split]):6d} samples")

    return datasets


def build_weighted_sampler(
    dataset,
    max_ratio: float = 5.0,
) -> torch.utils.data.WeightedRandomSampler:
    """
    WeightedRandomSampler for multi-label imbalance.
    Strategy: weight each sample by the inverse frequency of its RAREST label,
    then cap the maximum ratio to `max_ratio` times the median weight.

    Capping prevents extremely rare labels (e.g. AFLT with 73 samples)
    from completely dominating the training distribution.
    Compatible with Focal Loss — sampler works in data space,
    focal works in loss space; they complement rather than conflict
    when the sampler ratio is capped.
    """
    from torch.utils.data import WeightedRandomSampler

    label_matrix = dataset.label_matrix          # (N, K) float32
    n_samples    = len(label_matrix)
    class_counts = label_matrix.sum(axis=0)      # (K,) positive count per class
    class_counts  = np.where(class_counts == 0, 1, class_counts)  # avoid /0
    class_weights = 1.0 / class_counts           # rare class → high weight

    # Each sample's weight = max class_weight among its positive labels
    sample_weights = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        pos_mask = label_matrix[i] > 0
        if pos_mask.any():
            sample_weights[i] = class_weights[pos_mask].max()
        else:
            sample_weights[i] = class_weights.min()

    # Cap max weight at max_ratio × median to avoid over-sampling rare classes
    median_w = float(np.median(sample_weights[sample_weights > 0]))
    cap      = median_w * max_ratio
    sample_weights = np.clip(sample_weights, 0, cap)

    raw_ratio   = sample_weights.max() / (sample_weights.min() + 1e-9)
    print(f"  WeightedRandomSampler: median_w={median_w:.4f}  "
          f"cap={cap:.4f}  effective_ratio={raw_ratio:.1f}x  "
          f"(max_ratio={max_ratio}x)")

    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights),
        num_samples = n_samples,
        replacement = True,
    )


def build_loaders(datasets: dict, cfg: dict, args) -> dict:
    """Build DataLoaders for each split."""
    train_cfg        = cfg.get("training", {})
    batch_size       = args.batch_size or train_cfg.get("batch_size", 64)
    num_workers      = 0 if args.debug else train_cfg.get("num_workers", 4)
    pin_memory       = torch.cuda.is_available() and train_cfg.get("pin_memory", True)
    use_wrs          = train_cfg.get("weighted_sampler", False)

    if use_wrs:
        print("\n► WeightedRandomSampler enabled (oversampling minority classes)")

    loaders = {}
    for split, ds in datasets.items():
        if split == "train" and use_wrs:
            sampler = build_weighted_sampler(ds)
            loaders[split] = DataLoader(
                ds,
                batch_size  = batch_size,
                sampler     = sampler,       # sampler replaces shuffle
                num_workers = num_workers,
                pin_memory  = pin_memory,
                collate_fn  = collate_fn,
                drop_last   = True,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size  = batch_size,
                shuffle     = (split == "train"),
                num_workers = num_workers,
                pin_memory  = pin_memory,
                collate_fn  = collate_fn,
                drop_last   = (split == "train"),
            )
    return loaders


def build_pos_weight_tensors(
    pos_weights: dict,
    arrhy_names: list,
    mi_names: list,
    device: torch.device,
    max_weight: float = 50.0,
    power: float = 1.0,
) -> tuple:
    """
    Build pos_weight tensors for BCEWithLogitsLoss from pos_weights.json.
    Clamps each weight to max_weight to prevent gradient explosion
    (e.g. AFLT raw=298 → clamped to 50).

    Returns:
        (arrhy_pw_tensor, mi_pw_tensor)  — both on `device`
    """
    def _build(names):
        weights = []
        for n in names:
            raw_weight = float(pos_weights.get(n, 1.0))
            adjusted_weight = min(max(raw_weight, 1.0) ** power, max_weight)
            weights.append(adjusted_weight)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    arrhy_pw = _build(arrhy_names)
    mi_pw    = _build(mi_names)

    print(f"  Arrhythmia pos_weights (clamped ≤{max_weight}):")
    for n, w in zip(arrhy_names, arrhy_pw.tolist()):
        print(f"    {n}: {w:.2f}")
    print(f"  MI pos_weights (clamped ≤{max_weight}):")
    for n, w in zip(mi_names, mi_pw.tolist()):
        print(f"    {n}: {w:.2f}")

    return arrhy_pw, mi_pw


def normalize_hrv(
    hrv_matrix: np.ndarray,
    train_indices: np.ndarray,
    processed_path: str,
    feature_names: list,
) -> tuple:
    """
    Z-score normalize HRV targets using train-split statistics only.
    NaN values are excluded from stat computation and left as NaN
    (they will be zero-filled in PTBXLDataset.__getitem__).

    Returns:
        hrv_matrix_norm: normalized array (same shape)
        stats: dict with 'mean' and 'std' lists
    """
    train_hrv = hrv_matrix[train_indices]          # (N_train, 3)
    hrv_mean  = np.nanmean(train_hrv, axis=0)      # (3,)
    hrv_std   = np.nanstd(train_hrv,  axis=0)      # (3,)
    hrv_std   = np.where(hrv_std < 1e-6, 1.0, hrv_std)

    hrv_matrix_norm = (hrv_matrix - hrv_mean) / hrv_std

    stats = {"mean": hrv_mean.tolist(), "std": hrv_std.tolist(), "features": feature_names}
    stats_path = os.path.join(processed_path, "hrv_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  HRV z-score normalized (train stats)")
    for i, name in enumerate(feature_names):
        print(f"    {name:10s}: mean={hrv_mean[i]:.2f}  std={hrv_std[i]:.2f}")
    print(f"  Stats saved → {stats_path}")

    return hrv_matrix_norm, stats


def audit_and_save_dataset_policy(cfg: dict, data: dict, run_dir: str) -> None:
    report = audit_dataset_policy(
        cfg,
        data["processed_path"],
        metadata_columns=list(data["df"].columns),
    )
    report_path = os.path.join(run_dir, "dataset_policy_audit.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n► Dataset policy audit ...")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"  [warn] {warning}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  [error] {error}")
        raise ValueError(
            "Dataset policy audit failed. Rebuild metadata or align config before training."
        )
    print(f"  Audit passed → {report_path}")



# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_dataset_cli_overrides(cfg, args)

    # Overrides from CLI
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.architecture:
        cfg.setdefault("model", {})["architecture"] = args.architecture
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.no_hrv:
        cfg["hrv"]["enabled"] = False
    if args.debug:
        cfg["training"]["epochs"] = 3
        print("[DEBUG] 200 train / 50 val samples, 3 epochs")
    # ── Augmentation CLI overrides ────────────────────────────────────────
    aug_cfg = cfg.setdefault("augmentation", {})
    if args.aug_all:
        aug_cfg["aug_lead_dropout"]    = True
        aug_cfg["aug_baseline_wander"] = True
        aug_cfg["aug_random_crop"]     = True
        aug_cfg["aug_time_warp"]       = True
    else:
        if args.aug_lead_dropout:    aug_cfg["aug_lead_dropout"]    = True
        if args.aug_baseline_wander: aug_cfg["aug_baseline_wander"] = True
        if args.aug_random_crop:     aug_cfg["aug_random_crop"]     = True
        if args.aug_time_warp:       aug_cfg["aug_time_warp"]       = True
        if args.aug_mixup:           aug_cfg["aug_mixup"]           = True
    # ── Imbalance CLI overrides ───────────────────────────────────────────
    if args.use_focal:          cfg["training"]["use_focal"]          = True
    if args.no_pos_weight:      cfg["training"]["use_pos_weight"]      = False
    if args.focal_gamma is not None:
        cfg["training"]["focal_gamma"] = args.focal_gamma
    if args.weighted_sampler:   cfg["training"]["weighted_sampler"]   = True

    # ── Focal + WeightedSampler: complementary when capped ───────────────
    # Capped sampler (5x) + focal loss are complementary, not conflicting:
    # - Sampler (data space):  AFLT appears ~5x more often per epoch
    # - Focal  (loss space):   up-weights hard/misclassified examples
    # Both active is now the recommended strategy for extreme imbalance.
    if cfg["training"].get("use_focal") and cfg["training"].get("weighted_sampler"):
        print("  use_focal=True + weighted_sampler=True: capped sampler (5x) + focal active.")

    set_seed(cfg["training"].get("seed", 42))
    device = get_device()
    fs = cfg["dataset"]["sampling_rate"]

    print(f"\n{'='*60}")
    print("  ECG Multi-Task Training")
    print(f"{'='*60}\n")

    # ── Create timestamped run directory ──────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Encode active flags in the folder name for quick identification
    tag_parts = []
    architecture = cfg.get("model", {}).get("architecture", "hybrid_transformer")
    tag_parts.append(architecture.replace("_transformer", "-tf"))
    if cfg["training"].get("use_focal"):        tag_parts.append("focal")
    if cfg["training"].get("weighted_sampler"): tag_parts.append("wrs")
    aug_cfg = cfg.get("augmentation", {})
    if any([v for k, v in aug_cfg.items() if k.startswith("aug_") and v]):
        tag_parts.append("aug")
        if aug_cfg.get("aug_mixup"):
            tag_parts.append("mxp")
    if args.debug:                              tag_parts.append("debug")
    tag = ("_" + "-".join(tag_parts)) if tag_parts else ""
    run_name = f"run_{ts}{tag}"
    run_dir  = os.path.join(cfg["paths"]["checkpoints"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Override checkpoint path so Trainer saves into run_dir
    cfg["paths"]["checkpoints"] = run_dir

    # Save a snapshot of the active config into run_dir for reproducibility
    cfg_snapshot_path = os.path.join(run_dir, "config_snapshot.yaml")
    with open(cfg_snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"  Run directory : {run_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("► Loading preprocessed data ...")
    data = load_splits_and_data(cfg, args)
    audit_and_save_dataset_policy(cfg, data, run_dir)

    # ── Build label index lists ───────────────────────────────────────────────
    label_cfgs     = cfg["labels"]
    label_names    = [l["name"] for l in label_cfgs]
    arrhy_indices  = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_indices     = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    arrhy_names    = [label_names[i] for i in arrhy_indices]
    mi_names       = [label_names[i] for i in mi_indices]
    hrv_features   = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    hrv_enabled    = cfg.get("hrv", {}).get("enabled", True)

    print(f"  Arrhythmia labels [{len(arrhy_names)}]: {arrhy_names}")
    print(f"  MI labels         [{len(mi_names)}]: {mi_names}")
    print(f"  HRV enabled: {hrv_enabled}")

    # ── Z-score normalize HRV targets (train stats only) ────────────────────
    if hrv_enabled and data["hrv_matrix"] is not None:
        print("\n► Normalizing HRV targets ...")
        data["hrv_matrix"], _ = normalize_hrv(
            data["hrv_matrix"],
            data["splits"]["train"],
            f"{cfg['paths']['processed']}_{fs}hz",
            hrv_features,
        )

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("\n► Building datasets ...")
    datasets = build_datasets(cfg, data, args)
    loaders  = build_loaders(datasets, cfg, args)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n► Building model ...")
    model_cfg = cfg["model"]
    model = build_model(
        cfg=cfg,
        num_arrhythmia_labels=len(arrhy_indices),
        num_mi_labels=len(mi_indices),
        num_hrv_targets=len(hrv_features),
    )
    print(f"  Architecture: {model_cfg.get('architecture', 'hybrid_transformer')}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    if args.resume:
        print(f"\n► Resuming from: {args.resume}")
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    lw = cfg["training"].get("loss_weights", {})

    use_pos_weight = cfg["training"].get("use_pos_weight", True)
    use_focal      = cfg["training"].get("use_focal",      False)
    focal_gamma    = cfg["training"].get("focal_gamma",    2.0)
    focal_alpha    = cfg["training"].get("focal_alpha",    0.25)
    smoothing      = cfg["training"].get("label_smoothing", 0.0)
    hrv_loss       = cfg["training"].get("hrv_loss", "smooth_l1")
    max_pos_weight = float(cfg["training"].get("max_pos_weight", 50.0))
    pos_weight_power = float(cfg["training"].get("pos_weight_power", 1.0))

    arrhy_pw, mi_pw = None, None
    if use_pos_weight:
        print("\n► Building pos_weight tensors (from train split only) ...")
        # Compute pos_weight from TRAIN SPLIT only (not entire dataset)
        # This correctly reflects class imbalance in the actual training data.
        from src.data.label_builder import compute_pos_weights
        train_label_matrix = data["label_matrix"][data["splits"]["train"]]
        raw_pw = compute_pos_weights(train_label_matrix)
        train_pos_weights = {name: float(w) for name, w in zip(label_names, raw_pw)}
        print(f"  Train split pos_weights (raw):")
        for name, w in train_pos_weights.items():
            print(f"    {name}: {w:.3f}")

        arrhy_pw, mi_pw = build_pos_weight_tensors(
            train_pos_weights,
            arrhy_names,
            mi_names,
            device,
            max_weight=max_pos_weight,
            power=pos_weight_power,
        )
    else:
        print("\n► pos_weight disabled (use_pos_weight=false)")

    print(f"  use_focal: {use_focal}  focal_gamma: {focal_gamma}  focal_alpha: {focal_alpha}")
    print(
        f"  label_smoothing: {smoothing}  hrv_loss: {hrv_loss}  "
        f"max_pos_weight: {max_pos_weight}  pos_weight_power: {pos_weight_power}"
    )

    loss_fn = MultiTaskLoss(
        arrhythmia_weight     = lw.get("arrhythmia", 1.0),
        mi_weight             = lw.get("mi",         1.0),
        imi_weight            = lw.get("imi",        1.0),
        asmi_weight           = lw.get("asmi",       1.0),
        hrv_weight            = lw.get("hrv",        0.1),
        hrv_enabled           = hrv_enabled,
        arrhythmia_pos_weight = arrhy_pw,
        mi_pos_weight         = mi_pw,
        use_focal             = use_focal,
        focal_gamma           = focal_gamma,
        focal_alpha           = focal_alpha,
        label_smoothing       = smoothing,
        hrv_loss_type         = hrv_loss,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model                    = model,
        loss_fn                  = loss_fn,
        train_loader             = loaders["train"],
        val_loader               = loaders["val"],
        cfg                      = cfg,
        device                   = device,
        arrhythmia_label_indices = arrhy_indices,
        mi_label_indices         = mi_indices,
        arrhythmia_label_names   = arrhy_names,
        mi_label_names           = mi_names,
        hrv_feature_names        = hrv_features,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n► Loading best model for test evaluation ...")
    best_ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = load_checkpoint(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("  Loaded best_model.pth")
    else:
        print("  best_model.pth not found — using last model state")

    # ── Per-class threshold tuning on val set ─────────────────────────────────
    print("\n► Finding optimal thresholds on val set ...")
    thresholds = trainer.find_thresholds(loaders["val"])

    # Save thresholds
    thr_path = os.path.join(cfg["paths"]["checkpoints"], "thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Thresholds saved → {thr_path}")

    print("\n► Test set evaluation (per-class thresholds):")
    test_metrics = trainer.evaluate(loaders["test"], thresholds=thresholds)

    # Save test metrics
    test_metrics_path = os.path.join(cfg["paths"]["checkpoints"], "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Test metrics saved → {test_metrics_path}")

    # ── Plot training curves ───────────────────────────────────────────────
    print("\n► Saving training curves ...")
    freeze_epoch = cfg.get("training", {}).get("freeze_backbone_at_epoch", -1)
    plot_training_history(trainer.history, cfg["paths"]["checkpoints"],
                          freeze_epoch=int(freeze_epoch) if freeze_epoch else -1)

    # ── Plot confusion matrices ─────────────────────────────────────────
    print("\n► Saving confusion matrices ...")
    # Collect full-score matrix for MI error analysis
    _all_true, _all_scores = [], []
    model.eval()
    with torch.no_grad():
        for _batch in loaders["test"]:
            _sig = _batch["signal"].to(device)
            _lbl = _batch["labels"]
            _preds = model(_sig)
            K = _lbl.shape[1]
            _scores = torch.zeros(_lbl.shape[0], K)
            for j, idx in enumerate(arrhy_indices):
                _scores[:, idx] = torch.sigmoid(_preds["arrhythmia"]).cpu()[:, j]
            for j, idx in enumerate(mi_indices):
                _scores[:, idx] = torch.sigmoid(_preds["mi"]).cpu()[:, j]
            _all_true.append(_lbl)
            _all_scores.append(_scores)
    _y_true  = torch.cat(_all_true,  dim=0).numpy()
    _y_score = torch.cat(_all_scores, dim=0).numpy()

    plot_confusion_matrices(
        model        = model,
        loader       = loaders["test"],
        thresholds   = thresholds,
        arrhy_label_names = trainer.arrhythmia_label_names,
        mi_label_names    = trainer.mi_label_names,
        arrhy_indices     = trainer.arrhythmia_idx,
        mi_indices        = trainer.mi_idx,
        run_dir      = cfg["paths"]["checkpoints"],
        device       = device,
    )

    # ── MI Error Analysis ───────────────────────────────────────────────
    print("\n► Saving MI error analysis ...")
    _all_label_names   = trainer.arrhythmia_label_names + trainer.mi_label_names
    _all_label_indices = list(trainer.arrhythmia_idx)   + list(trainer.mi_idx)
    plot_mi_error_analysis(
        y_true           = _y_true,
        y_score          = _y_score,
        thresholds       = thresholds,
        all_label_names  = _all_label_names,
        all_label_indices = _all_label_indices,
        mi_label_names   = trainer.mi_label_names,
        mi_indices       = list(trainer.mi_idx),
        run_dir          = cfg["paths"]["checkpoints"],
    )


if __name__ == "__main__":
    main()
