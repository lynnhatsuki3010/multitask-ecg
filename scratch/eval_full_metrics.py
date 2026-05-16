"""
scratch/eval_full_metrics.py
─────────────────────────────────────────────────────────────────
Load fine-tuned checkpoints for Georgia and PTB, run inference on
their respective test sets, and report a full clinical metric table:
  AUROC | AUPRC | Sensitivity (Recall) | Specificity | F1 | Precision

Usage:
    python scratch/eval_full_metrics.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model


# ── Generic Dataset (works for both Georgia and PTB) ──────────────
class ECGDataset(Dataset):
    def __init__(self, features_path, labels_path, indices=None):
        self.X  = np.load(features_path)
        self.df = pd.read_csv(labels_path)
        self.arrhy_names = ["NORM", "AFIB", "STACH", "PVC", "AFLT"]
        self.mi_names    = ["IMI", "ASMI"]
        self.indices = indices if indices is not None else np.arange(len(self.df))
        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32)
        self.Y_mi    = self.df[self.mi_names].values.astype(np.float32)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy()
        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)   # per-lead z-score
        sig = torch.from_numpy(sig.T).float()               # (12, 5000)
        return {
            "signal":           sig,
            "arrhythmia_labels": torch.from_numpy(self.Y_arrhy[idx]),
            "mi_labels":         torch.from_numpy(self.Y_mi[idx]),
        }


# ── Inference ─────────────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    a_true, a_pred, m_true, m_pred = [], [], [], []
    with torch.no_grad():
        for b in loader:
            out  = model(b["signal"].to(device))
            a_true.append(b["arrhythmia_labels"].numpy())
            m_true.append(b["mi_labels"].numpy())
            a_pred.append(torch.sigmoid(out["arrhythmia"]).cpu().numpy())
            m_pred.append(torch.sigmoid(out["mi"]).cpu().numpy())
    return (np.vstack(a_true), np.vstack(a_pred),
            np.vstack(m_true), np.vstack(m_pred))


# ── Compute full clinical metrics for one class ───────────────────
def class_metrics(y_true_col, y_score_col, threshold=0.5):
    """Returns dict with all clinical metrics for a single binary label."""
    y_pred = (y_score_col >= threshold).astype(int)
    support = int(y_true_col.sum())

    if support == 0:
        return None   # skip labels with no positives

    tn, fp, fn, tp = confusion_matrix(y_true_col, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)
    auroc  = roc_auc_score(y_true_col, y_score_col)
    auprc  = average_precision_score(y_true_col, y_score_col)

    return {
        "Support":     support,
        "AUROC":       auroc,
        "AUPRC":       auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision":   precision,
        "F1":          f1,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ── Find best threshold per class on val set ──────────────────────
def find_threshold(y_true, y_score, beta=1.0, steps=50):
    """Grid search for best F-beta threshold."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, steps):
        y_p = (y_score >= t).astype(int)
        p = precision_score(y_true, y_p, zero_division=0)
        r = f1_score(y_true, y_p, zero_division=0)
        f = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)
        if f > best_f:
            best_f, best_t = f, t
    return best_t


# ── Pretty print table ────────────────────────────────────────────
def print_table(results_dict, title):
    cols = ["Label", "Support", "AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1"]
    widths = [8, 8, 7, 7, 12, 12, 10, 7]
    
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * 80)
    
    for label, m in results_dict.items():
        if m is None:
            print(f"  {label:<8}  (no positive samples in test set)")
            continue
        vals = [
            label,
            str(m["Support"]),
            f"{m['AUROC']:.3f}",
            f"{m['AUPRC']:.3f}",
            f"{m['Sensitivity']:.3f}",
            f"{m['Specificity']:.3f}",
            f"{m['Precision']:.3f}",
            f"{m['F1']:.3f}",
        ]
        print("  ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))

    # Macro averages (only over labels with positives)
    valid = {k: v for k, v in results_dict.items() if v is not None}
    if valid:
        print("-" * 80)
        macro = {}
        for metric in ["AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1"]:
            macro[metric] = np.mean([v[metric] for v in valid.values()])
        total_support = sum(v["Support"] for v in valid.values())
        vals = ["MACRO", str(total_support)] + [f"{macro[m]:.3f}" for m in
               ["AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1"]]
        print("  ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))


# ── Load model from checkpoint folder ─────────────────────────────
def load_model(ckpt_folder, device):
    cfg_path = None
    # Try config snapshot from original training run first
    for candidate in [
        "configs/experiments/cross_validation/xval_georgia_finetune.yaml",
        "configs/experiments/cross_validation/xval_ptb_finetune.yaml",
    ]:
        if os.path.exists(candidate):
            cfg_path = candidate

    # Use the finetune config (both share same model architecture)
    with open("configs/experiments/cross_validation/xval_ptb_finetune.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["hrv"] = {"enabled": False}

    model = build_model(cfg, num_arrhythmia_labels=5, num_mi_labels=2, num_hrv_targets=3)
    ckpt = torch.load(os.path.join(ckpt_folder, "best_model.pth"),
                      map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    return model.to(device)


# ── Main ──────────────────────────────────────────────────────────
def evaluate_dataset(name, ckpt_folder, processed_dir, splits_dir, device):
    print(f"\n{'#'*80}")
    print(f"#  DATASET: {name}")
    print(f"#  Checkpoint: {ckpt_folder}")
    print(f"{'#'*80}")

    test_idx = np.load(os.path.join(splits_dir, "test_indices.npy"))
    ds = ECGDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        indices=test_idx,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    model  = load_model(ckpt_folder, device)

    print(f"Test set: {len(test_idx)} records")
    print("Running inference...")
    y_true_a, y_score_a, y_true_m, y_score_m = run_inference(model, loader, device)

    # Fixed threshold 0.5 for all (clinical default)
    # ── Arrhythmia ──
    results_a = {}
    for i, lbl in enumerate(ds.arrhy_names):
        results_a[lbl] = class_metrics(y_true_a[:, i], y_score_a[:, i], threshold=0.5)
    print_table(results_a, f"{name} — ARRHYTHMIA HEAD")

    # ── MI ──
    results_m = {}
    for i, lbl in enumerate(ds.mi_names):
        results_m[lbl] = class_metrics(y_true_m[:, i], y_score_m[:, i], threshold=0.5)
    print_table(results_m, f"{name} — MI HEAD")

    return results_a, results_m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Georgia
    g_arrhy, g_mi = evaluate_dataset(
        name          = "Georgia (Proxy: Ischaemia)",
        ckpt_folder   = "checkpoints/finetune_georgia",
        processed_dir = "data/processed_georgia",
        splits_dir    = "data/splits_georgia",
        device        = device,
    )

    # PTB
    p_arrhy, p_mi = evaluate_dataset(
        name          = "PTB (Direct: Inferior/Anterior MI)",
        ckpt_folder   = "checkpoints/finetune_ptb",
        processed_dir = "data/processed_ptb",
        splits_dir    = "data/splits_ptb",
        device        = device,
    )

    # ── Side-by-side MI comparison ──
    print(f"\n{'='*80}")
    print("  SIDE-BY-SIDE MI HEAD COMPARISON (Fine-Tuned, threshold=0.5)")
    print(f"{'='*80}")
    print(f"{'Label':<8} | {'Metric':<12} | {'Georgia':>10} | {'PTB':>10}")
    print("-" * 50)
    for lbl in ["IMI", "ASMI"]:
        gm = g_mi.get(lbl)
        pm = p_mi.get(lbl)
        for metric in ["AUROC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1"]:
            gv = f"{gm[metric]:.3f}" if gm else "N/A"
            pv = f"{pm[metric]:.3f}" if pm else "N/A"
            print(f"{lbl:<8} | {metric:<12} | {gv:>10} | {pv:>10}")
        print(f"{'':8} | {'':12} | {'':>10} | {'':>10}")


if __name__ == "__main__":
    main()
