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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Make src importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.label_builder import (
    DEFAULT_LABELS, build_label_matrix, get_label_statistics, compute_pos_weights
)
from src.data.policy import build_dataset_policy, save_dataset_policy
from src.data.hrv_features import compute_hrv_batch
import wfdb


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Build PTB-XL metadata CSV with labels and HRV features.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--no-hrv", action="store_true", help="Skip HRV computation (much faster).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick testing).")
    parser.add_argument("--split-method", default=None, help="Override dataset.split_method from config.")
    parser.add_argument("--split-seed", type=int, default=None, help="Override dataset.split_seed from config.")
    parser.add_argument("--split-group-key", default=None, help="Override dataset.split_group_key from config.")
    parser.add_argument("--split-train-ratio", type=float, default=None, help="Override dataset.split_train_ratio from config.")
    parser.add_argument("--split-val-ratio", type=float, default=None, help="Override dataset.split_val_ratio from config.")
    parser.add_argument("--split-test-ratio", type=float, default=None, help="Override dataset.split_test_ratio from config.")
    parser.add_argument("--test-fold", type=int, default=None, help="Override dataset.test_fold from config.")
    parser.add_argument("--val-fold", type=int, default=None, help="Override dataset.val_fold from config.")
    parser.add_argument("--split-num-folds", type=int, default=None, help="Override dataset.split_num_folds from config.")
    parser.add_argument("--split-test-fold-index", type=int, default=None, help="Override dataset.split_test_fold_index from config.")
    parser.add_argument("--split-val-fold-index", type=int, default=None, help="Override dataset.split_val_fold_index from config.")
    parser.add_argument("--split-stratify-label", default=None, help="Override dataset.split_stratify_label from config.")
    return parser.parse_args()


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_dataset_overrides(cfg: dict, args) -> dict:
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


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train+val+test={total:.6f}."
        )


def _build_random_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val

    return {
        "train": np.sort(indices[:n_train]),
        "val": np.sort(indices[n_train:n_train + n_val]),
        "test": np.sort(indices[n_train + n_val:]),
    }


def _build_grouped_random_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: str,
) -> dict:
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    if group_key not in df.columns:
        raise ValueError(f"group_key='{group_key}' not found in dataframe columns.")

    group_series = df[group_key].copy()
    fallback = df.index.to_series()
    group_series = group_series.where(group_series.notna(), fallback)
    group_counts = group_series.value_counts()
    groups = group_counts.index.to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    total_samples = len(df)
    target_train = total_samples * train_ratio
    target_val = total_samples * val_ratio

    group_to_indices = group_series.groupby(group_series).indices
    split_indices = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for group in groups:
        member_idx = np.asarray(group_to_indices[group], dtype=np.int64)
        if counts["train"] < target_train:
            target_split = "train"
        elif counts["val"] < target_val:
            target_split = "val"
        else:
            target_split = "test"
        split_indices[target_split].append(member_idx)
        counts[target_split] += len(member_idx)

    finalized = {}
    for split_name, chunks in split_indices.items():
        finalized[split_name] = (
            np.sort(np.concatenate(chunks)) if chunks else np.array([], dtype=np.int64)
        )
    return finalized


def _get_group_series(df: pd.DataFrame, group_key: str) -> pd.Series:
    if group_key not in df.columns:
        raise ValueError(f"group_key='{group_key}' not found in dataframe columns.")
    group_series = df[group_key].copy()
    fallback = df.index.to_series()
    return group_series.where(group_series.notna(), fallback)


def _build_group_kfold_splits(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    labels: list[str],
    split_method: str,
    num_folds: int,
    test_fold_index: int,
    val_fold_index: int | None,
    group_key: str,
    seed: int,
    stratify_label: str | None,
) -> tuple[dict, dict]:
    if num_folds < 3:
        raise ValueError("split_num_folds must be at least 3 to create train/val/test splits.")

    test_fold_index = int(test_fold_index)
    if not 0 <= test_fold_index < num_folds:
        raise ValueError(
            f"split_test_fold_index must be in [0, {num_folds - 1}], got {test_fold_index}."
        )

    if val_fold_index is None:
        val_fold_index = (test_fold_index + 1) % num_folds
    val_fold_index = int(val_fold_index)
    if not 0 <= val_fold_index < num_folds:
        raise ValueError(
            f"split_val_fold_index must be in [0, {num_folds - 1}], got {val_fold_index}."
        )
    if val_fold_index == test_fold_index:
        raise ValueError("split_val_fold_index must differ from split_test_fold_index.")

    group_series = _get_group_series(df, group_key)
    groups = group_series.to_numpy()
    dummy_x = np.zeros(len(df), dtype=np.int8)

    if split_method == "group_kfold":
        splitter = GroupKFold(n_splits=num_folds)
        split_iter = splitter.split(dummy_x, groups=groups)
    else:
        if label_matrix is None or labels is None:
            raise ValueError(
                "label_matrix and labels are required for stratified_group_kfold."
            )
        if not stratify_label:
            raise ValueError(
                "dataset.split_stratify_label is required for stratified_group_kfold."
            )
        if stratify_label not in labels:
            raise ValueError(
                f"split_stratify_label='{stratify_label}' not found in labels {labels}."
            )
        stratify_idx = labels.index(stratify_label)
        y = label_matrix[:, stratify_idx].astype(int)
        splitter = StratifiedGroupKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(dummy_x, y=y, groups=groups)

    fold_assignments = np.full(len(df), -1, dtype=np.int64)
    fold_positive_counts = []
    for fold_idx, (_, fold_indices) in enumerate(split_iter):
        fold_assignments[fold_indices] = fold_idx
        if label_matrix is not None and labels is not None and stratify_label in labels:
            stratify_col = labels.index(stratify_label)
            fold_positive_counts.append(int(label_matrix[fold_indices, stratify_col].sum()))

    if np.any(fold_assignments < 0):
        raise RuntimeError("Failed to assign every sample to a CV fold.")

    idx_all = np.arange(len(df))
    splits = {
        "train": idx_all[(fold_assignments != test_fold_index) & (fold_assignments != val_fold_index)],
        "val": idx_all[fold_assignments == val_fold_index],
        "test": idx_all[fold_assignments == test_fold_index],
    }

    summary = {
        "split_method": split_method,
        "split_seed": seed,
        "split_group_key": group_key,
        "split_num_folds": num_folds,
        "split_test_fold_index": test_fold_index,
        "split_val_fold_index": val_fold_index,
        "unique_groups": int(group_series.nunique(dropna=True)),
    }
    if stratify_label:
        summary["split_stratify_label"] = stratify_label
    if fold_positive_counts:
        summary["split_fold_positive_counts"] = fold_positive_counts

    return splits, summary


def build_splits(
    df: pd.DataFrame,
    cfg: dict,
    label_matrix: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> tuple[dict, dict]:
    dataset_cfg = cfg.get("dataset", {})
    split_method = dataset_cfg.get("split_method", "strat_fold")
    split_seed = int(dataset_cfg.get("split_seed", 42))

    if split_method == "strat_fold":
        test_fold = dataset_cfg["test_fold"]
        val_fold = dataset_cfg["val_fold"]
        fold_arr = df["strat_fold"].values
        idx_all = np.arange(len(df))
        splits = {
            "train": idx_all[(fold_arr != test_fold) & (fold_arr != val_fold)],
            "val": idx_all[fold_arr == val_fold],
            "test": idx_all[fold_arr == test_fold],
        }
        summary = {
            "split_method": split_method,
            "test_fold": test_fold,
            "val_fold": val_fold,
            "split_seed": split_seed,
        }
        return splits, summary

    train_ratio = float(dataset_cfg.get("split_train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("split_val_ratio", 0.15))
    test_ratio = float(dataset_cfg.get("split_test_ratio", 0.15))

    if split_method == "random":
        splits = _build_random_splits(df, train_ratio, val_ratio, test_ratio, split_seed)
        summary = {
            "split_method": split_method,
            "split_seed": split_seed,
            "split_train_ratio": train_ratio,
            "split_val_ratio": val_ratio,
            "split_test_ratio": test_ratio,
        }
        return splits, summary

    if split_method == "random_grouped":
        group_key = dataset_cfg.get("split_group_key", "patient_id")
        splits = _build_grouped_random_splits(
            df,
            train_ratio,
            val_ratio,
            test_ratio,
            split_seed,
            group_key,
        )
        summary = {
            "split_method": split_method,
            "split_seed": split_seed,
            "split_group_key": group_key,
            "split_train_ratio": train_ratio,
            "split_val_ratio": val_ratio,
            "split_test_ratio": test_ratio,
            "unique_groups": int(df[group_key].nunique(dropna=True)) if group_key in df.columns else 0,
        }
        return splits, summary

    if split_method in {"group_kfold", "stratified_group_kfold"}:
        num_folds = int(dataset_cfg.get("split_num_folds", 5))
        test_fold_index = int(dataset_cfg.get("split_test_fold_index", 0))
        val_fold_index = dataset_cfg.get("split_val_fold_index")
        if val_fold_index is not None:
            val_fold_index = int(val_fold_index)
        group_key = dataset_cfg.get("split_group_key", "patient_id")
        stratify_label = dataset_cfg.get("split_stratify_label")
        return _build_group_kfold_splits(
            df=df,
            label_matrix=label_matrix,
            labels=labels,
            split_method=split_method,
            num_folds=num_folds,
            test_fold_index=test_fold_index,
            val_fold_index=val_fold_index,
            group_key=group_key,
            seed=split_seed,
            stratify_label=stratify_label,
        )

    raise ValueError(
        "Unsupported split_method='"
        f"{split_method}'. Use 'strat_fold', 'random', 'random_grouped', "
        "'group_kfold', or 'stratified_group_kfold'."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_dataset_overrides(cfg, args)

    raw_path       = cfg["paths"]["raw_data"]
    fs             = cfg["dataset"]["sampling_rate"]
    processed_path = f"{cfg['paths']['processed']}_{fs}hz"
    splits_path    = f"{cfg['paths']['splits']}_{fs}hz"
    threshold      = cfg["dataset"]["label_threshold"]
    label_threshold_overrides = cfg["dataset"].get("label_threshold_overrides", {})
    normal_label   = cfg["dataset"].get("normal_label", "NORM")
    normal_mode    = cfg["dataset"].get("normal_mode", "exclusive")
    selected_labels = [l["name"] for l in cfg["labels"]]

    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(splits_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PTB-XL Metadata Builder")
    print(f"  Raw path:  {raw_path}")
    print(f"  Labels:    {selected_labels}")
    print(f"  Threshold: {threshold}")
    if label_threshold_overrides:
        print(f"  Overrides: {label_threshold_overrides}")
    print(f"  Normal:    {normal_label} ({normal_mode})")
    print(f"  FS:        {fs} Hz")
    print(f"  Split:     {cfg['dataset'].get('split_method', 'strat_fold')}")
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
    label_groups = cfg.get("label_groups_override", None)  # optional per-run override
    label_matrix, labels = build_label_matrix(
        df,
        selected_labels,
        threshold=threshold,
        label_threshold_overrides=label_threshold_overrides,
        normal_label=normal_label,
        normal_mode=normal_mode,
        label_groups=label_groups,   # None → uses DEFAULT_LABEL_GROUPS from label_builder
    )

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
    print(f"\n► Creating splits ...")
    label_matrix_full = label_matrix
    splits, split_summary = build_splits(df, cfg, label_matrix=label_matrix_full, labels=labels)
    for key, value in split_summary.items():
        print(f"  {key}: {value}")

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

    policy = build_dataset_policy(cfg, processed_path)
    policy_path = save_dataset_policy(processed_path, policy)
    print(f"  Dataset policy saved → {policy_path}")

    print(f"\n✅ Preprocessing complete!\n")


if __name__ == "__main__":
    main()
