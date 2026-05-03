"""
06_sweep_splits.py
─────────────────────────────────────────────────────────────────
Automate multi-seed or multi-fold evaluation to measure how much
the train/val/test split assignment affects reported metrics.

Supports two strategies (auto-detected from config, or forced):
  1. random / random_grouped  →  sweep over --seeds
  2. stratified_group_kfold   →  sweep over all (test, val) fold pairs

For each iteration the script:
  1. Runs 01_build_metadata.py  (skips HRV after the first build)
  2. Runs the chosen training script
  3. Reads test_metrics.json from the new run directory
  4. Aggregates mean ± std across all iterations

Usage — random_grouped seeds:
    python scripts/06_sweep_splits.py \\
        --config configs/hybrid_transformer_imi_randomgrouped.yaml \\
        --seeds 42 123 456 789 1234

    # Use phased training script instead of default 02_train.py:
    python scripts/06_sweep_splits.py \\
        --config configs/hybrid_transformer_imi_randomgrouped.yaml \\
        --seeds 42 123 456 --train-script scripts/05_train_phased_multitask.py

Usage — GroupKFold (5 folds → 5 iterations):
    python scripts/06_sweep_splits.py \\
        --config configs/hybrid_transformer_imi_groupkfold.yaml \\
        --strategy kfold

    # Custom number of folds:
    python scripts/06_sweep_splits.py \\
        --config configs/hybrid_transformer_imi_groupkfold.yaml \\
        --strategy kfold --num-folds 5

Flags:
    --skip-build   Skip 01_build_metadata for every iteration
                   (use when processed data already exists and
                    only the training seeds should vary)
    --dry-run      Print commands without executing them
    --metrics      Comma-separated list of metric keys to report
                   (default: f1/mi/IMI,auroc/mi/IMI,auprc/mi/IMI,
                             f1/mi/ASMI,f1/arrhy/macro)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_SEEDS   = [42, 123, 456, 789, 1234]
DEFAULT_METRICS = [
    "f1/mi/IMI",
    "auroc/mi/IMI",
    "auprc/mi/IMI",
    "f1/mi/ASMI",
    "auroc/mi/ASMI",
    "f1/arrhy/macro",
    "auroc/arrhy/macro",
]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep split seeds / folds and aggregate test metrics."
    )
    p.add_argument("--config", required=True,
                   help="Base config YAML (e.g. configs/hybrid_transformer_imi_randomgrouped.yaml)")
    p.add_argument("--strategy", default=None, choices=["random", "kfold"],
                   help="Override strategy detection from config. "
                        "'random' sweeps seeds; 'kfold' sweeps fold pairs.")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seed list for random/random_grouped strategy. "
                        f"Default: {DEFAULT_SEEDS}")
    p.add_argument("--num-folds", type=int, default=None,
                   help="Number of folds for kfold strategy (default: from config).")
    p.add_argument("--train-script", default="scripts/02_train.py",
                   help="Training script to use. Default: scripts/02_train.py. "
                        "Alternative: scripts/05_train_phased_multitask.py")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip 01_build_metadata for every iteration.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated metric keys to report. "
                        f"Default: {','.join(DEFAULT_METRICS)}")
    p.add_argument("--output", default=None,
                   help="Path to save aggregated JSON. "
                        "Default: checkpoints/sweep_<timestamp>.json")
    return p.parse_args()


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_strategy(cfg: dict) -> str:
    method = cfg.get("dataset", {}).get("split_method", "random")
    if "kfold" in method.lower():
        return "kfold"
    return "random"


def get_num_folds(cfg: dict, override: Optional[int]) -> int:
    if override is not None:
        return override
    return int(cfg.get("dataset", {}).get("split_num_folds", 5))


def processed_data_exists(cfg: dict) -> bool:
    """True if label_matrix.npy already exists → skip HRV next time."""
    fs = cfg.get("dataset", {}).get("sampling_rate", 500)
    base = cfg["paths"]["processed"]
    lm_path = Path(f"{base}_{fs}hz") / "label_matrix.npy"
    return lm_path.exists()


# ─── Command runners ──────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], dry_run: bool, label: str = "") -> int:
    display = " ".join(cmd)
    print(f"\n{'─'*60}")
    if label:
        print(f"  [{label}]")
    print(f"  $ {display}")
    print(f"{'─'*60}")
    if dry_run:
        print("  (dry-run — skipped)")
        return 0
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode


def make_temp_config(
    base_config: str,
    split_seed: Optional[int] = None,
    test_fold: Optional[int] = None,
    val_fold: Optional[int] = None,
    train_seed: Optional[int] = None,
) -> str:
    """Clone base config, bake in split params, write to tmp/ and return path.

    This ensures the policy audit in 02_train.py sees the same split params
    that were passed to 01_build_metadata.py during the sweep iteration.
    """
    cfg = load_config(base_config)
    dataset_cfg = cfg.setdefault("dataset", {})
    if split_seed is not None:
        dataset_cfg["split_seed"] = split_seed
    if test_fold is not None:
        dataset_cfg["split_test_fold_index"] = test_fold
    if val_fold is not None:
        dataset_cfg["split_val_fold_index"] = val_fold
    if train_seed is not None:
        cfg.setdefault("training", {})["seed"] = train_seed

    tmp_dir = Path("tmp_sweep_configs")
    tmp_dir.mkdir(exist_ok=True)
    stem = Path(base_config).stem
    parts = []
    if split_seed is not None:
        parts.append(f"seed{split_seed}")
    if test_fold is not None:
        parts.append(f"tf{test_fold}_vf{val_fold}")
    suffix = "_".join(parts) or "iter"
    tmp_path = str(tmp_dir / f"{stem}_{suffix}.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


def build_data(
    config: str,
    seed: Optional[int] = None,
    test_fold: Optional[int] = None,
    val_fold: Optional[int] = None,
    skip_hrv: bool = False,
    dry_run: bool = False,
    label: str = "build",
) -> int:
    cmd = [sys.executable, "scripts/01_build_metadata.py", "--config", config]
    if seed is not None:
        cmd += ["--split-seed", str(seed)]
    if test_fold is not None:
        cmd += ["--split-test-fold-index", str(test_fold)]
    if val_fold is not None:
        cmd += ["--split-val-fold-index", str(val_fold)]
    if skip_hrv:
        cmd.append("--no-hrv")
    return run_cmd(cmd, dry_run, label)


def train_model(
    train_script: str,
    config: str,
    dry_run: bool = False,
    label: str = "train",
) -> int:
    """Run training with config (pass a temp config that has baked-in split params)."""
    cmd = [sys.executable, train_script, "--config", config]
    return run_cmd(cmd, dry_run, label)


# ─── Results collection ───────────────────────────────────────────────────────

def find_latest_test_metrics(checkpoints_dir: str = "checkpoints") -> Optional[Dict[str, Any]]:
    """Return test_metrics.json from the most-recently-modified run directory."""
    ckpt_root = Path(checkpoints_dir)
    if not ckpt_root.exists():
        return None
    run_dirs = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        metrics_file = run_dir / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_run_dir"] = str(run_dir)
            return data
    return None


def extract_metrics(
    test_metrics: Dict[str, Any],
    keys: List[str],
) -> Dict[str, float]:
    out = {}
    for k in keys:
        v = test_metrics.get(k)
        if v is not None and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(all_metrics: List[Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    import statistics
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        if not vals:
            continue
        summary[k] = {
            "mean": statistics.mean(vals),
            "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min":  min(vals),
            "max":  max(vals),
            "runs": vals,
        }
    return summary


def print_summary(summary: Dict[str, Dict[str, float]], run_labels: List[str]) -> None:
    print(f"\n{'='*70}")
    print("  SWEEP SUMMARY")
    print(f"{'='*70}")
    col_w = max(len(k) for k in summary) + 2 if summary else 30

    # Header
    print(f"  {'Metric':<{col_w}}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*col_w}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for k, v in summary.items():
        print(
            f"  {k:<{col_w}}  {v['mean']:8.4f}  {v['std']:8.4f}  "
            f"{v['min']:8.4f}  {v['max']:8.4f}"
        )

    # Per-run breakdown
    if run_labels:
        print(f"\n  Per-run breakdown:")
        print(f"  {'Run':<20}", end="")
        for k in summary:
            short = k.split("/")[-1][:10]
            print(f"  {short:>10}", end="")
        print()
        for i, label in enumerate(run_labels):
            print(f"  {label:<20}", end="")
            for k, v in summary.items():
                val = v["runs"][i] if i < len(v["runs"]) else float("nan")
                print(f"  {val:10.4f}", end="")
            print()
    print(f"{'='*70}\n")


# ─── Main sweep loops ─────────────────────────────────────────────────────────

def sweep_random_seeds(
    args: argparse.Namespace,
    cfg: dict,
    metric_keys: List[str],
) -> Tuple[List[Dict[str, float]], List[str]]:
    all_metrics: List[Dict[str, float]] = []
    run_labels: List[str] = []

    for i, seed in enumerate(args.seeds):
        label = f"seed={seed}  ({i+1}/{len(args.seeds)})"
        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{len(args.seeds)} — {label}")
        print(f"{'#'*60}")

        # Build data
        if not args.skip_build:
            skip_hrv = (i > 0) and processed_data_exists(cfg)
            if skip_hrv:
                print(f"  [INFO] Processed data found → skipping HRV recomputation.")
            rc = build_data(
                config=args.config,
                seed=seed,
                skip_hrv=skip_hrv,
                dry_run=args.dry_run,
                label=f"build {label}",
            )
            if rc != 0:
                print(f"  [ERROR] build_metadata failed (exit {rc}). Skipping this iteration.")
                continue

        # Write temp config with baked-in split_seed so policy audit passes
        tmp_cfg = make_temp_config(args.config, split_seed=seed, train_seed=seed)
        print(f"  [INFO] Temp config: {tmp_cfg}")

        # Train
        rc = train_model(
            train_script=args.train_script,
            config=tmp_cfg,
            dry_run=args.dry_run,
            label=f"train {label}",
        )
        if rc != 0:
            print(f"  [ERROR] Training failed (exit {rc}). Skipping this iteration.")
            continue

        # Collect results
        if not args.dry_run:
            test_metrics = find_latest_test_metrics()
            if test_metrics is None:
                print(f"  [WARN] No test_metrics.json found for seed={seed}.")
            else:
                run_dir = test_metrics.pop("_run_dir", "?")
                metrics = extract_metrics(test_metrics, metric_keys)
                all_metrics.append(metrics)
                run_labels.append(f"seed={seed}")
                print(f"\n  Results from: {run_dir}")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

    return all_metrics, run_labels


def sweep_kfold(
    args: argparse.Namespace,
    cfg: dict,
    metric_keys: List[str],
) -> Tuple[List[Dict[str, float]], List[str]]:
    num_folds = get_num_folds(cfg, args.num_folds)
    # Standard adjacent-fold pairs: (0,1), (1,2), ..., (4,0)
    fold_pairs = [(i, (i + 1) % num_folds) for i in range(num_folds)]

    all_metrics: List[Dict[str, float]] = []
    run_labels: List[str] = []

    train_seed = cfg.get("training", {}).get("seed", 42)

    for i, (test_fold, val_fold) in enumerate(fold_pairs):
        label = f"test_fold={test_fold}, val_fold={val_fold}  ({i+1}/{num_folds})"
        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{num_folds} — {label}")
        print(f"{'#'*60}")

        # Build data — for kfold we always rebuild splits, but skip HRV after first
        if not args.skip_build:
            skip_hrv = (i > 0) and processed_data_exists(cfg)
            if skip_hrv:
                print(f"  [INFO] Processed data found → skipping HRV recomputation.")
            rc = build_data(
                config=args.config,
                test_fold=test_fold,
                val_fold=val_fold,
                skip_hrv=skip_hrv,
                dry_run=args.dry_run,
                label=f"build {label}",
            )
            if rc != 0:
                print(f"  [ERROR] build_metadata failed (exit {rc}). Skipping.")
                continue

        # Write temp config with baked-in fold indices so policy audit passes
        tmp_cfg = make_temp_config(
            args.config,
            test_fold=test_fold,
            val_fold=val_fold,
            train_seed=train_seed,
        )
        print(f"  [INFO] Temp config: {tmp_cfg}")

        # Train
        rc = train_model(
            train_script=args.train_script,
            config=tmp_cfg,
            dry_run=args.dry_run,
            label=f"train {label}",
        )
        if rc != 0:
            print(f"  [ERROR] Training failed (exit {rc}). Skipping.")
            continue

        # Collect results
        if not args.dry_run:
            test_metrics = find_latest_test_metrics()
            if test_metrics is None:
                print(f"  [WARN] No test_metrics.json found.")
            else:
                run_dir = test_metrics.pop("_run_dir", "?")
                metrics = extract_metrics(test_metrics, metric_keys)
                all_metrics.append(metrics)
                run_labels.append(f"fold({test_fold},{val_fold})")
                print(f"\n  Results from: {run_dir}")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

    return all_metrics, run_labels


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # Determine strategy
    strategy = args.strategy or detect_strategy(cfg)
    print(f"\n  Config   : {args.config}")
    print(f"  Strategy : {strategy}")
    print(f"  Train    : {args.train_script}")
    if strategy == "random":
        print(f"  Seeds    : {args.seeds}")
    else:
        print(f"  Folds    : {get_num_folds(cfg, args.num_folds)}")
    if args.skip_build:
        print(f"  [INFO] --skip-build: skipping all 01_build_metadata runs.")
    if args.dry_run:
        print(f"  [DRY-RUN] No commands will be executed.")

    # Metric keys to collect
    metric_keys = (
        [m.strip() for m in args.metrics.split(",")]
        if args.metrics else DEFAULT_METRICS
    )

    start = time.time()

    if strategy == "random":
        all_metrics, run_labels = sweep_random_seeds(args, cfg, metric_keys)
    else:
        all_metrics, run_labels = sweep_kfold(args, cfg, metric_keys)

    elapsed = time.time() - start

    if not all_metrics:
        print("\n  No results collected. Exiting.")
        return

    # Aggregate and display
    summary = aggregate(all_metrics, metric_keys)
    print_summary(summary, run_labels)
    print(f"  Total wall time: {elapsed/60:.1f} min\n")

    # Save results
    out_path = args.output or f"checkpoints/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    result_doc = {
        "config":    args.config,
        "strategy":  strategy,
        "seeds":     args.seeds if strategy == "random" else None,
        "num_folds": get_num_folds(cfg, args.num_folds) if strategy == "kfold" else None,
        "run_labels": run_labels,
        "per_run":   all_metrics,
        "summary":   {k: {sk: sv for sk, sv in v.items() if sk != "runs"}
                      for k, v in summary.items()},
        "elapsed_min": elapsed / 60,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2)
    print(f"  Aggregated results saved → {out_path}")


if __name__ == "__main__":
    main()
