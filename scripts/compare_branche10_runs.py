"""
compare_branche10_runs.py
-------------------------
Aggregate test metrics from multiple BRANCHE10 experiment runs for side-by-side comparison.

Usage:
    python scripts/compare_branche10_runs.py --runs checkpoints/run_*_branche10_00_baseline checkpoints/run_*_branche10_01_mi_asl

    python scripts/compare_branche10_runs.py --glob "checkpoints/run_*_branche10_*"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Primary metrics per branch (from BRANCHE10_MATRIX.md)
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
    p = argparse.ArgumentParser(description="Compare BRANCHE10 experiment runs")
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Explicit run directories (supports shell globs if quoted)",
    )
    p.add_argument(
        "--glob",
        default=None,
        help='Glob pattern for run dirs, e.g. "checkpoints/run_*_branche10_*"',
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to save markdown table",
    )
    return p.parse_args()


def resolve_run_dirs(args: argparse.Namespace) -> List[Path]:
    dirs: List[Path] = []
    if args.runs:
        for pattern in args.runs:
            matched = glob.glob(pattern)
            if matched:
                dirs.extend(Path(p) for p in matched)
            elif Path(pattern).is_dir():
                dirs.append(Path(pattern))
    if args.glob:
        dirs.extend(Path(p) for p in glob.glob(args.glob))
    # Deduplicate, keep sorted by mtime desc
    unique = {d.resolve(): d for d in dirs if d.is_dir()}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def load_experiment_id(run_dir: Path) -> str:
    snap = run_dir / "config_snapshot.yaml"
    if snap.exists():
        cfg = yaml.safe_load(snap.read_text(encoding="utf-8"))
        exp = cfg.get("experiment", {})
        if exp.get("id"):
            return str(exp["id"])
    return run_dir.name


def load_metrics(run_dir: Path) -> Optional[Dict]:
    path = run_dir / "test_metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def build_table(rows: List[Dict]) -> str:
    if not rows:
        return "No runs with test_metrics.json found."

    exp_ids = [r["exp_id"] for r in rows]
    col_w = max(14, max(len(e) for e in exp_ids) + 2)

    lines = []
    lines.append("# BRANCHE10 Comparison\n")
    lines.append("## Primary metrics (test set)\n")
    header = f"| Metric |" + "".join(f" {e} |" for e in exp_ids)
    sep = f"|{'-' * 8}|" + "".join(f" {'-' * col_w} |" for _ in exp_ids)
    lines.append(header)
    lines.append(sep)

    for label, key in PRIMARY_METRICS:
        cells = [fmt(r["metrics"].get(key)) for r in rows]
        lines.append(f"| {label} |" + "".join(f" {c:^{col_w}} |" for c in cells))

    lines.append("\n## Secondary metrics\n")
    lines.append(header)
    lines.append(sep)
    for label, key in SECONDARY_METRICS:
        cells = [fmt(r["metrics"].get(key)) for r in rows]
        lines.append(f"| {label} |" + "".join(f" {c:^{col_w}} |" for c in cells))

    lines.append("\n## Run directories\n")
    for r in rows:
        lines.append(f"- **{r['exp_id']}**: `{r['run_dir']}`")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        print("No run directories found. Pass --runs or --glob.")
        sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        metrics = load_metrics(run_dir)
        if metrics is None:
            print(f"[skip] No test_metrics.json in {run_dir}")
            continue
        rows.append({
            "exp_id": load_experiment_id(run_dir),
            "run_dir": str(run_dir),
            "metrics": metrics,
        })

    if not rows:
        print("No runs with test_metrics.json.")
        sys.exit(1)

    # Sort by experiment id for stable display
    rows.sort(key=lambda r: r["exp_id"])
    table = build_table(rows)
    print(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table, encoding="utf-8")
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
