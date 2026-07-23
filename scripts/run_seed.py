#!/usr/bin/env python
"""Train + calibrate one config at one seed, in a single command.

Runs scripts/02_train.py with --seed, finds the run folder it just created
(newest folder whose name contains the config's experiment id), then runs
scripts/03_calibrate.py on it. Terminal-agnostic (no shell globbing needed).

Usage:
  python scripts/run_seed.py --config configs/experiments/dir5_p5_dyn_anat_v2.yaml --seed 43
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import yaml


def newest_run_dir(ckpt_dir, exp_id, after_ts):
    """Newest non-debug run folder for exp_id created at/after after_ts."""
    best, best_m = None, -1.0
    for d in glob.glob(os.path.join(ckpt_dir, "run_*")):
        if not os.path.isdir(d) or d.endswith("-debug"):
            continue
        if exp_id not in os.path.basename(d):
            continue
        m = os.path.getmtime(d)
        if m >= after_ts - 5 and m > best_m:   # small slack for clock skew
            best, best_m = d, m
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--no-calibrate", action="store_true", help="Train only, skip calibrate")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp_id = (cfg.get("experiment") or {}).get("id")
    if not exp_id:
        sys.exit(f"config {args.config} has no experiment.id")

    py = sys.executable
    t0 = time.time()

    print(f"\n===== TRAIN  {exp_id}  seed={args.seed} =====", flush=True)
    r = subprocess.run([py, "scripts/02_train.py", "--config", args.config, "--seed", str(args.seed)])
    if r.returncode != 0:
        sys.exit(f"training failed (exit {r.returncode})")

    if args.no_calibrate:
        return

    run_dir = newest_run_dir(args.ckpt_dir, exp_id, t0)
    if run_dir is None:
        sys.exit(f"could not locate the run folder for {exp_id} (seed {args.seed}); "
                 f"run calibrate manually on the folder printed above.")

    print(f"\n===== CALIBRATE  {run_dir} =====", flush=True)
    r = subprocess.run([py, "scripts/03_calibrate.py", "--dir", run_dir])
    if r.returncode != 0:
        sys.exit(f"calibrate failed (exit {r.returncode})")
    print(f"\n[done] {exp_id} seed={args.seed} in {(time.time()-t0)/60:.1f} min -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
