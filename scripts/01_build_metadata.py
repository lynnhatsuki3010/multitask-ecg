"""
01_build_metadata.py
─────────────────────────────────────────────────────────────────
Step 1: Load PTB-XL database, build multi-label matrix,
        compute HRV features, and save metadata.csv + splits.

Usage:
    python scripts/01_build_metadata.py
    python scripts/01_build_metadata.py --config configs/config.yaml
    python scripts/01_build_metadata.py --no-hrv   # skip HRV computation (faster)
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Make src importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.label_builder import (
    DEFAULT_LABELS, build_label_matrix, get_label_statistics, compute_pos_weights
)
from src.data.hrv_features import compute_hrv_batch
import wfdb


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Build PTB-XL metadata CSV with labels and HRV features.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--no-hrv", action="store_true", help="Skip HRV computation (much faster).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick testing).")
    return parser.parse_args()


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── Signal loader (for HRV only) ────────────────────────────────────────────

def load_signals_for_hrv(df: pd.DataFrame, base_path: str, fs: int, max_samples=None) -> np.ndarray:
    """Load lead II signals for HRV computation (memory efficient)."""
    n = min(len(df), max_samples) if max_samples else len(df)
    T = int(fs * 10)  # 10 seconds
    signals = np.zeros((n, T), dtype=np.float32)

    filename_col = "filename_lr" if fs == 100 else "filename_hr"

    for i in tqdm(range(n), desc="Loading signals for HRV", unit="rec"):
        row = df.iloc[i]
        path = os.path.join(base_path, row[filename_col])
        try:
            record, _ = wfdb.rdsamp(path)
            lead_ii = record[:, 1].astype(np.float32)  # Lead II = index 1
            T_actual = min(len(lead_ii), T)
            signals[i, :T_actual] = lead_ii[:T_actual]
        except Exception as e:
            pass  # leave as zeros → will yield nan HRV

    return signals


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    raw_path       = cfg["paths"]["raw_data"]
    processed_path = cfg["paths"]["processed"]
    splits_path    = cfg["paths"]["splits"]
    fs             = cfg["dataset"]["sampling_rate"]
    threshold      = cfg["dataset"]["label_threshold"]
    test_fold      = cfg["dataset"]["test_fold"]
    val_fold       = cfg["dataset"]["val_fold"]
    selected_labels = [l["name"] for l in cfg["labels"]]

    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(splits_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PTB-XL Metadata Builder")
    print(f"  Raw path:  {raw_path}")
    print(f"  Labels:    {selected_labels}")
    print(f"  Threshold: {threshold}")
    print(f"  FS:        {fs} Hz")
    print(f"{'='*60}\n")

    # ── 1. Load database CSV ──────────────────────────────────────────────────
    print("► Loading ptbxl_database.csv ...")
    db_path = os.path.join(raw_path, "ptbxl_database.csv")
    df = pd.read_csv(db_path, index_col="ecg_id")

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"  ⚠ Limiting to {args.max_samples} samples for testing.")

    print(f"  Total records: {len(df)}")

    # ── 2. Build label matrix ─────────────────────────────────────────────────
    print("\n► Building label matrix ...")
    label_matrix, labels = build_label_matrix(df, selected_labels, threshold=threshold)

    stats = get_label_statistics(label_matrix, labels)
    print("\n  Label distribution:")
    print(stats.to_string(index=False))

    covered = (label_matrix.sum(axis=1) > 0).sum()
    multi   = (label_matrix.sum(axis=1) > 1).sum()
    print(f"\n  Samples with ≥1 label:  {covered} ({covered/len(df)*100:.1f}%)")
    print(f"  Multi-label samples:     {multi} ({multi/len(df)*100:.1f}%)")

    # ── 3. Compute HRV features ───────────────────────────────────────────────
    hrv_enabled = cfg.get("hrv", {}).get("enabled", True) and not args.no_hrv

    if hrv_enabled:
        print(f"\n► Computing HRV features (Lead II, fs={fs}Hz) ...")
        t0 = time.time()
        signals_leadII = load_signals_for_hrv(df, raw_path, fs, args.max_samples)
        hrv_results = compute_hrv_batch(signals_leadII, fs=fs, lead_idx=0)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        valid_hrv = (~np.isnan(hrv_results["rmssd"])).sum()
        print(f"  Valid HRV samples: {valid_hrv}/{len(df)} ({valid_hrv/len(df)*100:.1f}%)")
        print(f"  RMSSD  — mean: {np.nanmean(hrv_results['rmssd']):.2f} ms")
        print(f"  SDNN   — mean: {np.nanmean(hrv_results['sdnn']):.2f} ms")
        print(f"  Mean HR — mean: {np.nanmean(hrv_results['mean_hr']):.2f} bpm")

        df["hrv_rmssd"]   = hrv_results["rmssd"]
        df["hrv_sdnn"]    = hrv_results["sdnn"]
        df["hrv_mean_hr"] = hrv_results["mean_hr"]
        df["hrv_npeaks"]  = hrv_results["num_rpeaks"]
    else:
        print("\n► Skipping HRV computation (--no-hrv)")
        df["hrv_rmssd"]   = np.nan
        df["hrv_sdnn"]    = np.nan
        df["hrv_mean_hr"] = np.nan
        df["hrv_npeaks"]  = 0

    # ── 4. Add label columns to df ────────────────────────────────────────────
    for i, lbl in enumerate(labels):
        df[f"label_{lbl}"] = label_matrix[:, i]

    # ── 5. Compute and save pos_weights ───────────────────────────────────────
    pos_weights = compute_pos_weights(label_matrix)
    pw_dict = {lbl: float(w) for lbl, w in zip(labels, pos_weights)}
    print("\n  Positive class weights (for BCEWithLogitsLoss):")
    for lbl, w in pw_dict.items():
        print(f"    {lbl}: {w:.3f}")

    import json
    pw_path = os.path.join(processed_path, "pos_weights.json")
    with open(pw_path, "w") as f:
        json.dump(pw_dict, f, indent=2)
    print(f"\n  Saved pos_weights → {pw_path}")

    # ── 6. Save full metadata ─────────────────────────────────────────────────
    meta_path = os.path.join(processed_path, "metadata.csv")
    df.to_csv(meta_path)
    print(f"  Saved metadata  → {meta_path}")

    # ── 7. Create train / val / test splits ───────────────────────────────────
    print(f"\n► Creating splits (test_fold={test_fold}, val_fold={val_fold}) ...")

    label_matrix_full = label_matrix
    idx_all = np.arange(len(df))

    is_test = df["strat_fold"].values == test_fold
    is_val  = df["strat_fold"].values == val_fold
    is_train = ~is_test & ~is_val

    # Save split index files
    splits = {
        "train": idx_all[is_train],
        "val":   idx_all[is_val],
        "test":  idx_all[is_test],
    }
    for split_name, indices in splits.items():
        path = os.path.join(splits_path, f"{split_name}_indices.npy")
        np.save(path, indices)
        print(f"  {split_name:6s}: {len(indices):5d} samples → {path}")

    # Save label matrix as npy
    lm_path = os.path.join(processed_path, "label_matrix.npy")
    np.save(lm_path, label_matrix_full)
    print(f"\n  Label matrix saved → {lm_path}  shape: {label_matrix_full.shape}")

    # Save HRV matrix
    hrv_matrix = np.stack([
        df["hrv_rmssd"].values,
        df["hrv_sdnn"].values,
        df["hrv_mean_hr"].values,
    ], axis=1).astype(np.float32)
    hrv_path = os.path.join(processed_path, "hrv_matrix.npy")
    np.save(hrv_path, hrv_matrix)
    print(f"  HRV matrix saved   → {hrv_path}  shape: {hrv_matrix.shape}")

    print(f"\n✅ Preprocessing complete!\n")


if __name__ == "__main__":
    main()
