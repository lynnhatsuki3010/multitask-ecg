"""
16_eval_zeroshot_decoupled.py
─────────────────────────────────────────────────────────────────
Zero-shot (external) evaluation of a DECOUPLED multi-task model
(trained on PTB-XL) on an external dataset (PTB or Georgia).

Unlike the legacy scripts (09 / 14) which assumed the old
`hybrid_transformer` backbone with exactly 2 MI labels [IMI, ASMI],
this script supports the DIR5 decoupled model whose MI head emits
4 labels [IMI, ASMI, ILMI, AMI]. It:

  1. Rebuilds the model with the EXACT label counts from the
     checkpoint's config_snapshot.yaml (arrhythmia / mi / conduction).
  2. Applies the SAME preprocessing chain used during training
     (bandpass + notch + the configured normalize, e.g. robust),
     because the external feature files are stored as raw mV.
  3. Scores only the MI labels that exist in the external dataset
     (IMI / ASMI for PTB) by slicing the matching output columns.

Usage:
    python scripts/16_eval_zeroshot_decoupled.py \
        --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth \
        --dataset ptb --split test
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model
from src.data.preprocessing import bandpass_filter, notch_filter, normalize_signal


DATASET_PATHS = {
    "ptb":     {"processed": "data/processed_ptb",     "splits": "data/splits_ptb"},
    "georgia": {"processed": "data/processed_georgia", "splits": "data/splits_georgia"},
}


class ExternalDataset(Dataset):
    """
    Loads an external dataset's pre-built raw features (N, 5000, 12) in mV
    and applies the training preprocessing chain on the fly so that the
    input distribution matches what the PTB-XL model saw.

    Only label columns that are physically present in the external
    `labels.csv` are returned for each task.
    """

    def __init__(self, processed_dir, mi_label_pool, arrhy_label_pool,
                 preproc, sampling_rate, indices=None):
        self.X  = np.load(os.path.join(processed_dir, "features.npy"))  # (N, 5000, 12)
        self.df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))

        # Keep only labels that exist as columns in this external dataset
        self.mi_names    = [c for c in mi_label_pool    if c in self.df.columns]
        self.arrhy_names = [c for c in arrhy_label_pool  if c in self.df.columns]

        self.indices = np.arange(len(self.df)) if indices is None else indices
        self.Y_mi    = self.df[self.mi_names].values.astype(np.float32) if self.mi_names else None
        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32) if self.arrhy_names else None

        self.bandpass = (preproc.get("bandpass_low", 0.5), preproc.get("bandpass_high", 40.0))
        self.notch    = preproc.get("notch_freq", 50.0)
        self.normalize_method = preproc.get("normalize", "zscore")
        self.fs = float(sampling_rate)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy().astype(np.float32)  # (5000, 12) raw mV, axis0 = time

        # Match training preprocessing: bandpass + notch on (T, 12)
        if self.bandpass is not None:
            sig = bandpass_filter(sig, self.fs, low=self.bandpass[0], high=self.bandpass[1])
        if self.notch is not None:
            sig = notch_filter(sig, self.fs, freq=self.notch)

        sig = np.ascontiguousarray(sig.T)  # (12, 5000)
        sig = normalize_signal(sig, method=self.normalize_method)  # per-lead over time axis

        out = {"signal": torch.from_numpy(sig.astype(np.float32))}
        if self.Y_mi is not None:
            out["mi_labels"] = torch.from_numpy(self.Y_mi[idx])
        if self.Y_arrhy is not None:
            out["arrhythmia_labels"] = torch.from_numpy(self.Y_arrhy[idx])
        return out


def group_label_names(cfg, group):
    """Names for a task group, in label_groups index order (includes NORM
    in the arrhythmia group, matching the trained head dimensions)."""
    idx_to_name = {l["index"]: l["name"] for l in cfg["labels"]}
    indices = cfg.get("label_groups", {}).get(group, [])
    return [idx_to_name[i] for i in indices]


def evaluate(model, loader, device, mi_col_idx, has_arrhy):
    model.eval()
    m_true, m_score = [], []
    a_true, a_score = [], []
    with torch.no_grad():
        for batch in loader:
            sig = batch["signal"].to(device)
            out = model(sig)
            mi_logits = out["mi"][:, mi_col_idx]          # slice to external MI labels
            m_score.append(torch.sigmoid(mi_logits).cpu().numpy())
            m_true.append(batch["mi_labels"].numpy())
            if has_arrhy:
                a_score.append(torch.sigmoid(out["arrhythmia"]).cpu().numpy())
                a_true.append(batch["arrhythmia_labels"].numpy())
    res = {
        "mi_true":  np.vstack(m_true),
        "mi_score": np.vstack(m_score),
    }
    if has_arrhy:
        res["a_true"]  = np.vstack(a_true)
        res["a_score"] = np.vstack(a_score)
    return res


def report_block(y_true, y_score, label_names, group_name):
    print(f"\n{'='*56}")
    print(f"  {group_name}")
    print(f"{'='*56}")
    print(f"{'Label':<8} | {'AUROC':>6} | {'AUPRC':>6} | {'F1@0.5':>7} | {'Support':>8}")
    print("-" * 56)
    y_pred = (y_score >= 0.5).astype(int)
    rows = {}
    for i, name in enumerate(label_names):
        support = int(y_true[:, i].sum())
        if support == 0:
            print(f"{name:<8} | {'N/A':>6} | {'N/A':>6} | {'N/A':>7} | {support:>8}  (no pos)")
            continue
        auroc = roc_auc_score(y_true[:, i], y_score[:, i])
        auprc = average_precision_score(y_true[:, i], y_score[:, i])
        f1    = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        print(f"{name:<8} | {auroc:>6.3f} | {auprc:>6.3f} | {f1:>7.3f} | {support:>8}")
        rows[name] = {"auroc": float(auroc), "auprc": float(auprc),
                      "f1": float(f1), "support": support}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth of the PTB-XL trained decoupled model")
    parser.add_argument("--dataset", default="ptb", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    ckpt_dir    = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config_snapshot.yaml")
    print(f"Loading config snapshot from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arrhy_names = group_label_names(cfg, "arrhythmia")
    mi_names    = group_label_names(cfg, "mi")
    cond_names  = group_label_names(cfg, "conduction")
    print(f"Model labels -> arrhythmia={arrhy_names}  mi={mi_names}  conduction={cond_names}")

    paths = DATASET_PATHS[args.dataset]
    processed_dir, splits_dir = paths["processed"], paths["splits"]

    if args.split == "all":
        indices = None
    else:
        indices = np.load(os.path.join(splits_dir, f"{args.split}_indices.npy"))
        print(f"Loaded {args.split} split: {len(indices)} records")

    ds = ExternalDataset(
        processed_dir=processed_dir,
        mi_label_pool=mi_names,
        arrhy_label_pool=arrhy_names,
        preproc=cfg.get("preprocessing", {}),
        sampling_rate=cfg.get("dataset", {}).get("sampling_rate", 500),
        indices=indices,
    )
    print(f"External MI labels present: {ds.mi_names}")
    print(f"External arrhythmia labels present: {ds.arrhy_names}")

    batch_size = cfg.get("training", {}).get("batch_size", 64)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Map external MI columns -> model MI output columns
    mi_col_idx = [mi_names.index(n) for n in ds.mi_names]

    cfg["hrv"] = {"enabled": False}
    model = build_model(
        cfg,
        num_arrhythmia_labels=len(arrhy_names),
        num_mi_labels=len(mi_names),
        num_hrv_targets=3,
        num_conduction_labels=len(cond_names),
    ).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    has_arrhy = ds.Y_arrhy is not None and len(ds.arrhy_names) > 0
    print("Running inference...")
    res = evaluate(model, loader, device, mi_col_idx, has_arrhy)

    summary = {}
    summary["mi"] = report_block(
        res["mi_true"], res["mi_score"], ds.mi_names,
        f"MI HEAD — zero-shot on {args.dataset.upper()} (direct label match)")
    print("\nMI classification report:")
    print(classification_report(
        res["mi_true"], (res["mi_score"] >= 0.5).astype(int),
        target_names=ds.mi_names, zero_division=0))

    if has_arrhy:
        summary["arrhythmia"] = report_block(
            res["a_true"], res["a_score"], ds.arrhy_names,
            f"ARRHYTHMIA HEAD — zero-shot on {args.dataset.upper()} (NORM only meaningful)")

    out_path = os.path.join(ckpt_dir, f"zeroshot_{args.dataset}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved zero-shot results -> {out_path}")
    print(f"Done. Zero-Shot evaluation on {args.dataset.upper()} complete.")


if __name__ == "__main__":
    main()
