"""
compare_val_runs.py
-------------------
Compare runs on VALIDATION metrics, so stage winners are picked without ever
looking at the test set.

Stage selection must run on validation: choosing a config because it scored well
on test turns the test set into part of the training loop, and every number
reported from it afterwards is optimistic. Test is read here only for the
val->test drift table, which is a diagnostic, never a selection signal.

For each run this reads history.json, finds the epoch with the best validation
`auprc/mi/IMI` (the `imi_auprc` monitor metric that also decides which epoch
best_model.pth comes from), and reports that epoch's validation metrics.

Usage:
    python scripts/compare_val_runs.py --glob "checkpoints/run_*gp_s*"
    python scripts/compare_val_runs.py --glob "checkpoints/run_*gp_s1_*" --drift
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MONITOR_KEY = "auprc/mi/IMI"          # fallback when a run records no monitor_metric

# Mirrors Trainer.MONITOR_ALIASES. The epoch reported here has to be the epoch that
# produced best_model.pth, so it is resolved per run from its own config_snapshot -
# runs in this project were trained under different monitors (imi_auprc early,
# f1/cond/macro during the conduction-tuning branch), and assuming one of them would
# report an epoch the checkpoint never came from.
MONITOR_ALIASES = {
    "task_auprc": ["auprc/arrhy/macro", "auprc/mi/macro", "auprc/cond/macro"],
    "mean_auroc": ["auroc/arrhy/macro", "auroc/mi/macro"],
    "macro_f1": ["f1/arrhy/macro", "f1/mi/macro"],
    "arrhythmia_f1": ["f1/arrhy/macro"],
    "mi_f1": ["f1/mi/macro"],
    "imi_f1": ["f1/mi/IMI"],
    "imi_auroc": ["auroc/mi/IMI"],
    "imi_auprc": ["auprc/mi/IMI"],
    "asmi_f1": ["f1/mi/ASMI"],
    "asmi_auroc": ["auroc/mi/ASMI"],
    "asmi_auprc": ["auprc/mi/ASMI"],
}

# Stage winners are a different decision from checkpoint selection, so they get a
# different metric. Picking the recipe by one label's AUPRC would quietly optimise
# the pipeline for IMI while the paper claims balanced 13-label multitask; mixup
# shows exactly that failure mode (best arrhythmia, worst conduction).
#
# S averages the three task macro AUPRCs: threshold-free, one weight per task, and
# far steadier than an F1 composite - across GPU architectures macro AUPRC moves by
# ~0.01 while rare-label F1 moves by ~0.16.
COMPOSITE_KEY = "S/composite"
COMPOSITE_PARTS = ["auprc/arrhy/macro", "auprc/mi/macro", "auprc/cond/macro"]

PRIMARY_METRICS = [
    ("S (mean task AUPRC)", COMPOSITE_KEY),
    ("MI macro F1", "f1/mi/macro"),
    ("IMI F1", "f1/mi/IMI"),
    ("ILMI F1", "f1/mi/ILMI"),
    ("AMI F1", "f1/mi/AMI"),
    ("ASMI F1", "f1/mi/ASMI"),
    ("IMI AUPRC", "auprc/mi/IMI"),
    ("Cond macro F1", "f1/cond/macro"),
    ("Arrhy macro F1", "f1/arrhy/macro"),
]

SECONDARY_METRICS = [
    ("MI macro AUROC", "auroc/mi/macro"),
    ("Cond macro AUROC", "auroc/cond/macro"),
    ("Arrhy macro AUROC", "auroc/arrhy/macro"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare runs on validation metrics")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Explicit run directories (supports globs if quoted)")
    p.add_argument("--glob", default=None,
                   help='Glob pattern, e.g. "checkpoints/run_*gp_s*"')
    p.add_argument("--drift", action="store_true",
                   help="Also print the val->test drift table (diagnostic only)")
    p.add_argument("--aggregate", action="store_true",
                   help="Group runs by experiment id and report mean+/-SD across seeds")
    p.add_argument("--out", default=None, help="Optional path to save markdown")
    return p.parse_args()


def resolve_run_dirs(args: argparse.Namespace) -> List[Path]:
    dirs: List[Path] = []
    for pattern in (args.runs or []):
        matched = glob.glob(pattern)
        dirs.extend(Path(p) for p in matched) if matched else None
        if not matched and Path(pattern).is_dir():
            dirs.append(Path(pattern))
    if args.glob:
        dirs.extend(Path(p) for p in glob.glob(args.glob))
    unique = {d.resolve(): d for d in dirs if d.is_dir()}
    return sorted(unique.values(), key=lambda p: p.name)


def load_experiment_id(run_dir: Path) -> str:
    snap = run_dir / "config_snapshot.yaml"
    if snap.exists():
        cfg = yaml.safe_load(snap.read_text(encoding="utf-8"))
        exp_id = (cfg.get("experiment") or {}).get("id")
        if exp_id:
            return exp_id
    return run_dir.name


def add_composite(metrics: Dict) -> Dict:
    """Attach S = mean of the three task macro AUPRCs, when all three are present."""
    parts = [metrics.get(k) for k in COMPOSITE_PARTS]
    if all(p is not None for p in parts):
        metrics[COMPOSITE_KEY] = sum(parts) / len(parts)
    return metrics


def monitor_keys(run_dir: Path) -> Tuple[str, List[str]]:
    """Return (monitor name, metric keys) this run actually selected its epoch by."""
    snap = run_dir / "config_snapshot.yaml"
    name = MONITOR_KEY
    if snap.exists():
        cfg = yaml.safe_load(snap.read_text(encoding="utf-8"))
        name = (cfg.get("training") or {}).get("monitor_metric") or MONITOR_KEY
    return name, MONITOR_ALIASES.get(name, [name])


def best_val_epoch(run_dir: Path) -> Optional[Tuple[int, Dict, str]]:
    """Return (epoch_index, val metrics, monitor name) for the checkpoint's epoch.

    The epoch is chosen by the run's own monitor so it matches best_model.pth; S is
    computed on top afterwards, purely for comparing runs against runs.
    """
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    val = history.get("val") or []
    name, keys = monitor_keys(run_dir)

    def score(entry: Dict) -> float:
        vals = [entry.get(k) for k in keys]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    scored = [(i, e) for i, e in enumerate(val) if not math.isnan(score(e))]
    if not scored:
        return None
    idx, metrics = max(scored, key=lambda pair: score(pair[1]))
    return idx, add_composite(dict(metrics)), name


def load_seed(run_dir: Path) -> Optional[int]:
    snap = run_dir / "config_snapshot.yaml"
    if not snap.exists():
        return None
    cfg = yaml.safe_load(snap.read_text(encoding="utf-8"))
    return (cfg.get("training") or {}).get("seed")


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{value:.3f}"


def aggregate_by_experiment(
    columns: List[str],
    rows: Dict[str, Dict[str, float]],
    exp_ids: Dict[str, str],
    metrics,
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, int]]:
    """Collapse per-run columns into one column per experiment id.

    Each cell becomes "mean+/-sd" over the seeds available for that experiment.
    A single-seed experiment prints the bare value, so it stays visibly weaker
    evidence than a cell carrying a spread.
    """
    groups: Dict[str, List[str]] = {}
    for col in columns:
        groups.setdefault(exp_ids[col], []).append(col)

    agg_cols = sorted(groups)
    agg_rows: Dict[str, Dict[str, str]] = {}
    counts = {exp: len(cols) for exp, cols in groups.items()}

    for exp, cols in groups.items():
        cell: Dict[str, str] = {}
        for _, key in metrics:
            values = [rows[c][key] for c in cols if rows[c].get(key) is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            if len(values) == 1:
                cell[key] = f"{mean:.3f}"
            else:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                cell[key] = f"{mean:.3f}+/-{var ** 0.5:.3f}"
        agg_rows[exp] = cell
    return agg_cols, agg_rows, counts


def render_table(title: str, metrics, columns: List[str],
                 rows: Dict[str, Dict[str, float]]) -> List[str]:
    width = max(24, max(len(c) for c in columns) + 2)
    out = [f"## {title}", ""]
    out.append("| Metric | " + " | ".join(columns) + " |")
    out.append("|--------| " + " | ".join("-" * width for _ in columns) + " |")
    for label, key in metrics:
        cells = [fmt(rows[c].get(key)) for c in columns]
        out.append(f"| {label} | " + " | ".join(f"{c:^{width}}" for c in cells) + " |")
    out.append("")
    return out


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        print("No run directories matched.")
        return

    columns: List[str] = []
    val_rows: Dict[str, Dict[str, float]] = {}
    drift_rows: Dict[str, Dict[str, float]] = {}
    epochs: Dict[str, int] = {}
    paths: Dict[str, Path] = {}
    exp_ids: Dict[str, str] = {}
    monitors: Dict[str, str] = {}

    for run_dir in run_dirs:
        picked = best_val_epoch(run_dir)
        if picked is None:
            print(f"[skip] no usable history.json in {run_dir}")
            continue
        epoch_idx, val_metrics, monitor_name = picked

        exp_id = load_experiment_id(run_dir)
        seed = load_seed(run_dir)
        name = f"{exp_id}@s{seed}" if seed is not None else exp_id
        while name in val_rows:                     # keep duplicate ids distinct
            name += "'"
        columns.append(name)
        exp_ids[name] = exp_id
        val_rows[name] = val_metrics
        epochs[name] = epoch_idx + 1
        monitors[name] = monitor_name
        paths[name] = run_dir

        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            test_metrics = add_composite(json.loads(test_path.read_text(encoding="utf-8")))
            drift_rows[name] = {
                key: (test_metrics[key] - val_metrics[key])
                for _, key in PRIMARY_METRICS + SECONDARY_METRICS
                if key in test_metrics and key in val_metrics
            }

    if not columns:
        print("No runs with a validation history.")
        return

    lines = ["# Validation Comparison", "",
             f"Winner picked on validation `{MONITOR_KEY}`; test set untouched.", ""]

    if args.aggregate:
        all_metrics = PRIMARY_METRICS + SECONDARY_METRICS
        agg_cols, agg_rows, counts = aggregate_by_experiment(
            columns, val_rows, exp_ids, all_metrics)
        lines += [f"Mean+/-SD across seeds: "
                  + ", ".join(f"{e} (n={counts[e]})" for e in agg_cols), ""]
        lines += render_table("Primary metrics (validation, best epoch)",
                              PRIMARY_METRICS, agg_cols, agg_rows)
        lines += render_table("Secondary metrics (validation, best epoch)",
                              SECONDARY_METRICS, agg_cols, agg_rows)
    else:
        lines += render_table("Primary metrics (validation, best epoch)",
                              PRIMARY_METRICS, columns, val_rows)
        lines += render_table("Secondary metrics (validation, best epoch)",
                              SECONDARY_METRICS, columns, val_rows)

    if args.drift and drift_rows:
        drift_cols = [c for c in columns if c in drift_rows]
        lines += ["## Val -> test drift (diagnostic, NOT a selection signal)", "",
                  "Large per-run swings on rare labels mean the test ranking is",
                  "sampling noise, not evidence about the technique.", ""]
        lines += render_table("test - val", PRIMARY_METRICS + SECONDARY_METRICS,
                              drift_cols, drift_rows)[1:]

    lines += ["## Best validation epoch", ""]
    for name in columns:
        lines.append(f"- **{name}**: epoch {epochs[name]} by `{monitors[name]}`  "
                     f"(val S={fmt(val_rows[name].get(COMPOSITE_KEY))}, "
                     f"IMI AUPRC={fmt(val_rows[name].get(MONITOR_KEY))})  `{paths[name]}`")
    used = sorted(set(monitors.values()))
    if len(used) > 1:
        lines += ["", f"> Runs were trained under different monitors ({', '.join(used)}), "
                      "so their checkpoints come from epochs chosen by different rules. "
                      "Validation metrics above are still comparable; test metrics are not."]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
