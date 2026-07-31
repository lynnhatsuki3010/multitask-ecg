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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MONITOR_KEY = "auprc/mi/IMI"          # training.monitor_metric = imi_auprc

PRIMARY_METRICS = [
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


def best_val_epoch(run_dir: Path) -> Optional[Tuple[int, Dict]]:
    """Return (epoch_index, val metrics) for the best validation monitor score."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    val = history.get("val") or []
    scored = [(i, e) for i, e in enumerate(val) if e.get(MONITOR_KEY) is not None]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[1][MONITOR_KEY])


def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


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

    for run_dir in run_dirs:
        picked = best_val_epoch(run_dir)
        if picked is None:
            print(f"[skip] no usable history.json in {run_dir}")
            continue
        epoch_idx, val_metrics = picked

        name = load_experiment_id(run_dir)
        while name in val_rows:                     # keep duplicate ids distinct
            name += "'"
        columns.append(name)
        val_rows[name] = val_metrics
        epochs[name] = epoch_idx + 1
        paths[name] = run_dir

        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            test_metrics = json.loads(test_path.read_text(encoding="utf-8"))
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
        score = val_rows[name].get(MONITOR_KEY)
        lines.append(f"- **{name}**: epoch {epochs[name]}  "
                     f"(val {MONITOR_KEY}={fmt(score)})  `{paths[name]}`")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
