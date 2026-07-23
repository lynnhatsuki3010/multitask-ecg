#!/usr/bin/env python
"""Aggregate multi-seed results for one or more experiment ids.

Scans checkpoints/ for run folders, groups them by experiment.id (read from each
run's config_snapshot.yaml), and reports mean +/- std of the key MI / arrhythmia /
conduction metrics across seeds. Prefers calibrated metrics
(calibration_results.json -> calibrated_metrics) when present, else falls back to
test_metrics.json. Dedupes by (exp_id, seed) keeping the most recent run.

Usage:
  python scripts/aggregate_seeds.py dir5_p3a_imi50 dir5_p5_dyn_anat_v2
  python scripts/aggregate_seeds.py dir5_p3a_imi50 dir5_p5_dyn_anat_v2 --ckpt-dir checkpoints
"""
import argparse
import glob
import json
import os
import statistics as stats

import yaml

# Metrics to report (label -> key in the metrics dict).
METRICS = [
    ("MI  Macro F1",    "f1/mi/macro"),
    ("MI  Macro AUPRC", "auprc/mi/macro"),
    ("MI  Macro AUROC", "auroc/mi/macro"),
    ("  IMI  F1",       "f1/mi/IMI"),
    ("  IMI  AUPRC",    "auprc/mi/IMI"),
    ("  ASMI F1",       "f1/mi/ASMI"),
    ("  ASMI AUPRC",    "auprc/mi/ASMI"),
    ("  ILMI F1",       "f1/mi/ILMI"),
    ("  ILMI AUPRC",    "auprc/mi/ILMI"),
    ("  AMI  F1",       "f1/mi/AMI"),
    ("  AMI  AUPRC",    "auprc/mi/AMI"),
    ("Arrhy Macro F1",  "f1/arrhy/macro"),
    ("Cond  Macro F1",  "f1/cond/macro"),
    ("Cond  Macro AUPRC", "auprc/cond/macro"),
]


def load_run(run_dir):
    """Return (exp_id, seed, metrics_dict, source) or None if unusable."""
    cfg_path = os.path.join(run_dir, "config_snapshot.yaml")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp_id = (cfg.get("experiment") or {}).get("id")
    seed = (cfg.get("training") or {}).get("seed")
    if exp_id is None:
        return None

    cal_path = os.path.join(run_dir, "calibration_results.json")
    tm_path = os.path.join(run_dir, "test_metrics.json")
    if os.path.exists(cal_path):
        cal = json.load(open(cal_path, "r", encoding="utf-8"))
        metrics = cal.get("calibrated_metrics") or {}
        source = "calibrated"
    elif os.path.exists(tm_path):
        metrics = json.load(open(tm_path, "r", encoding="utf-8"))
        source = "test(uncal)"
    else:
        return None
    return exp_id, seed, metrics, source


def collect(ckpt_dir, exp_ids):
    """Return {exp_id: {seed: (metrics, source, run_dir, mtime)}} deduped by newest."""
    out = {e: {} for e in exp_ids}
    for run_dir in glob.glob(os.path.join(ckpt_dir, "run_*")):
        if run_dir.endswith("-debug") or not os.path.isdir(run_dir):
            continue
        res = load_run(run_dir)
        if res is None:
            continue
        exp_id, seed, metrics, source = res
        if exp_id not in out:
            continue
        mtime = os.path.getmtime(run_dir)
        prev = out[exp_id].get(seed)
        if prev is None or mtime > prev[3]:
            out[exp_id][seed] = (metrics, source, run_dir, mtime)
    return out


def fmt(vals):
    if not vals:
        return "   n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}       (1 seed)"
    m = stats.mean(vals)
    sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3f} +/- {sd:.3f}  (n={len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_ids", nargs="+", help="experiment.id values to aggregate")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    data = collect(args.ckpt_dir, args.exp_ids)

    for exp_id in args.exp_ids:
        runs = data[exp_id]
        seeds = sorted(runs.keys(), key=lambda s: (s is None, s))
        print(f"\n=== {exp_id} ===")
        if not runs:
            print("  (no runs found)")
            continue
        srcs = {runs[s][1] for s in seeds}
        print(f"  seeds: {seeds}   source: {', '.join(sorted(srcs))}")
        for s in seeds:
            print(f"    seed {s}: {runs[s][2]}")

    # Per-metric table with mean +/- std, and delta if exactly 2 exp_ids.
    print("\n" + "=" * 78)
    header = f"{'metric':22s}" + "".join(f"{e[:24]:>26s}" for e in args.exp_ids)
    print(header)
    print("-" * len(header))
    means = {e: {} for e in args.exp_ids}
    for label, key in METRICS:
        cells = []
        for e in args.exp_ids:
            vals = [m[0][key] for m in data[e].values() if key in m[0]]
            if vals:
                means[e][key] = stats.mean(vals)
            cells.append(fmt(vals))
        print(f"{label:22s}" + "".join(f"{c:>26s}" for c in cells))

    if len(args.exp_ids) == 2:
        a, b = args.exp_ids
        print("\n" + "=" * 78)
        print(f"Delta of means  ({b}  -  {a})   [positive = {b} better]")
        print("-" * 60)
        for label, key in METRICS:
            if key in means[a] and key in means[b]:
                d = means[b][key] - means[a][key]
                mark = "  <-- " if abs(d) >= 0.01 else ""
                print(f"  {label:22s} {d:+.3f}{mark}")


if __name__ == "__main__":
    main()
