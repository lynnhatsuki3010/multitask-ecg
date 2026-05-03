"""
07_audit_imi_splits.py
─────────────────────────────────────────────────────────────────
Audit IMI (and optionally ASMI) at record level and patient level per split.

Helps interpret threshold_search.min_pos_count and val_tune_thresholds_every:
if val has very few IMI-positive patients, F_beta threshold sweeps are noisy.

Usage:
  python scripts/07_audit_imi_splits.py --config configs/hybrid_transformer_imi_randomgrouped.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _processed_and_splits_paths(cfg: dict) -> tuple:
    fs = int(cfg["dataset"]["sampling_rate"])
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits = f"{cfg['paths']['splits']}_{fs}hz"
    return processed, splits


def main():
    parser = argparse.ArgumentParser(description="Audit MI labels per split (records + patients).")
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config (paths + labels).")
    parser.add_argument(
        "--label",
        default="IMI",
        choices=("IMI", "ASMI"),
        help="Which MI column to audit (must appear in cfg labels).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed, splits_dir = _processed_and_splits_paths(cfg)
    meta_path = os.path.join(processed, "metadata.csv")
    lm_path = os.path.join(processed, "label_matrix.npy")

    if not os.path.isfile(meta_path) or not os.path.isfile(lm_path):
        print(f"Missing {meta_path} or {lm_path}. Run scripts/01_build_metadata.py first.")
        sys.exit(1)

    label_cfgs = cfg.get("labels", [])
    name_to_idx = {entry["name"]: int(entry["index"]) for entry in label_cfgs}
    if args.label not in name_to_idx:
        print(f"Label {args.label!r} not in config labels.")
        sys.exit(1)
    mi_col = name_to_idx[args.label]

    df = pd.read_csv(meta_path, index_col="ecg_id")
    lm = np.load(lm_path)
    if lm.shape[0] != len(df):
        print(f"Warning: label_matrix rows ({lm.shape[0]}) != metadata rows ({len(df)}).")

    group_key = cfg.get("dataset", {}).get("split_group_key", "patient_id")
    if group_key not in df.columns:
        print(f"Column {group_key!r} not in metadata; patient-level counts unavailable.")
        group_key = None

    print(f"\n{'='*60}")
    print(f"  Split audit — {args.label} @ {processed}")
    print(f"  Splits dir: {splits_dir}")
    if group_key:
        print(f"  Patient key: {group_key}")
    print(f"{'='*60}\n")

    thr_cfg = cfg.get("eval", {}).get("threshold_search", {})
    min_pos = int(thr_cfg.get("min_pos_count", 20))

    for split in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split}_indices.npy")
        if not os.path.isfile(p):
            print(f"  [{split}] missing {p}")
            continue
        idx = np.load(p)
        y = lm[idx, mi_col] > 0.5
        n_pos = int(y.sum())
        n_tot = len(idx)
        prev = 100.0 * n_pos / max(n_tot, 1)

        line = f"  {split:5s}: records={n_tot:5d}  {args.label}+={n_pos:4d}  prevalence={prev:5.2f}%"
        if group_key is not None:
            sub = df.iloc[idx]
            pos_mask = y
            patients_pos = sub.loc[pos_mask, group_key].nunique(dropna=True)
            patients_all = sub[group_key].nunique(dropna=True)
            line += f"  | patients+={int(patients_pos):4d} / {int(patients_all):4d} unique"
        print(line)

        if split == "val" and n_pos < min_pos:
            print(
                f"    → Val has fewer than threshold_search.min_pos_count={min_pos} "
                f"positive **records** for {args.label}; rare-class threshold cap applies. "
                "If patient-level positives are also tiny, consider stratified_group_kfold "
                "or lowering min_pos_count with care."
            )

    policy_path = os.path.join(processed, "dataset_policy.json")
    if os.path.isfile(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            pol = json.load(f)
        print("\n  dataset_policy.json (split summary):")
        for k in ("split_method", "split_seed", "split_group_key", "label_threshold", "label_threshold_overrides"):
            if k in pol:
                print(f"    {k}: {pol[k]}")

    print()


if __name__ == "__main__":
    main()
