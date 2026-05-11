"""
14_eval_ptb_zeroshot.py
─────────────────────────────────────────────────────────────────
Zero-shot evaluation of PTB-XL trained model on the PTB dataset.

Unlike Georgia (which used Ischaemia as proxy), PTB has the EXACT same
label ontology (IMI = Inferior MI, ASMI = Anterior MI).
This is a direct, clean cross-dataset evaluation.

Usage:
    python scripts/14_eval_ptb_zeroshot.py \\
        --checkpoint checkpoints/run_20260508_185741_hybrid-tf-focal-aug/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model


class PTBDataset(Dataset):
    """
    Loads PTB processed data. Only evaluates MI labels (IMI, ASMI)
    since PTB has no fine-grained Arrhythmia labels.
    NORM is included for Arrhythmia head reference.
    """
    def __init__(self, features_path, labels_path, indices=None):
        self.X  = np.load(features_path)   # (N, 5000, 12)
        self.df = pd.read_csv(labels_path)

        self.arrhy_names = ["NORM", "AFIB", "STACH", "PVC", "AFLT"]
        self.mi_names    = ["IMI", "ASMI"]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.df))

        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32)
        self.Y_mi    = self.df[self.mi_names].values.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy()   # (5000, 12)

        # Per-lead Z-score normalization (same as training pipeline)
        sig_mean = np.mean(sig, axis=0)
        sig_std  = np.std(sig, axis=0) + 1e-8
        sig = (sig - sig_mean) / sig_std

        # (5000, 12) → (12, 5000)
        sig = torch.from_numpy(sig.T).float()
        y_a = torch.from_numpy(self.Y_arrhy[idx])
        y_m = torch.from_numpy(self.Y_mi[idx])
        return {"signal": sig, "arrhythmia_labels": y_a, "mi_labels": y_m}


def evaluate(model, loader, device):
    model.eval()
    all_a_true, all_a_pred = [], []
    all_m_true, all_m_pred = [], []

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            sig = batch["signal"].to(device)
            out = model(sig)

            all_a_true.append(batch["arrhythmia_labels"].numpy())
            all_m_true.append(batch["mi_labels"].numpy())
            all_a_pred.append(torch.sigmoid(out["arrhythmia"]).cpu().numpy())
            all_m_pred.append(torch.sigmoid(out["mi"]).cpu().numpy())

    return (
        np.vstack(all_a_true), np.vstack(all_a_pred),
        np.vstack(all_m_true), np.vstack(all_m_pred),
    )


def print_metrics(y_true, y_score, label_names, group_name, skip_zero_support=True):
    print(f"\n{'='*50}")
    print(f"  {group_name}")
    print(f"{'='*50}")
    print(f"{'Label':<8} | {'AUROC':>6} | {'AUPRC':>6} | {'F1@0.5':>7} | {'Support':>8}")
    print("-" * 50)

    y_pred = (y_score >= 0.5).astype(int)
    for i, name in enumerate(label_names):
        support = int(y_true[:, i].sum())
        if skip_zero_support and support == 0:
            print(f"{name:<8} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {support:>8}  (no positives)")
            continue
        try:
            auroc = roc_auc_score(y_true[:, i], y_score[:, i])
            auprc = average_precision_score(y_true[:, i], y_score[:, i])
            f1    = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            print(f"{name:<8} | {auroc:>6.3f} | {auprc:>6.3f} | {f1:>7.3f} | {support:>8}")
        except Exception as e:
            print(f"{name:<8} | ERROR: {e}")

    print(f"\n{group_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth of the PTB-XL trained model")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    # Load config from checkpoint snapshot
    ckpt_dir    = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config_snapshot.yaml")
    print(f"Loading config snapshot from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load splits
    processed_dir = "data/processed_ptb"
    splits_dir    = "data/splits_ptb"

    if args.split == "all":
        indices = None   # use full dataset
    else:
        idx_file = os.path.join(splits_dir, f"{args.split}_indices.npy")
        indices  = np.load(idx_file)
        print(f"Loaded {args.split} split: {len(indices)} records")

    ds = PTBDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        indices=indices,
    )
    batch_size = cfg.get("training", {}).get("batch_size", 64)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build & load model
    cfg["hrv"] = {"enabled": False}
    model = build_model(cfg, num_arrhythmia_labels=5, num_mi_labels=2, num_hrv_targets=3).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    # Evaluate
    y_true_a, y_score_a, y_true_m, y_score_m = evaluate(model, loader, device)

    # NOTE: Arrhythmia metrics on PTB will only be meaningful for NORM.
    # AFIB/STACH/PVC/AFLT will all have 0 support (they are not labeled in PTB).
    print_metrics(y_true_a, y_score_a, ds.arrhy_names,
                  "ARRHYTHMIA HEAD (NORM only meaningful)")
    print_metrics(y_true_m, y_score_m, ds.mi_names,
                  "MI HEAD — DIRECT label match (IMI & ASMI)")

    print("\nDone. Zero-Shot evaluation on PTB complete.")


if __name__ == "__main__":
    main()
