"""
verify_box.py
-------------
Acceptance test for a fresh GPU box before spending money on new seeds.

RUNBOOK_FINAL.md records two facts that make this cheap and necessary:
re-running the same config at the same seed on the same hardware reproduces
every metric bit-for-bit (max |delta| = 0 over 87 metrics), while moving to a
different GPU architecture shifts validation IMI AUPRC by ~0.0099 - about a
quarter of the effect the paper is trying to establish. Seeds from two
different GPU models therefore cannot be pooled into one mean.

So: on a new box, re-run one config at a seed that already exists, point this
script at both run folders, and read the max delta. Zero means the box is the
same machine as far as the numbers are concerned and the new seeds can join the
old ones. Anything else means stop and re-plan before burning the budget.

Compares every validation metric at every epoch, not just the monitor epoch -
a box that diverges only late in training would still poison the pooled mean.

Usage:
    python scripts/verify_box.py --new checkpoints/run_2026...._abl_arch10_contrast \
                                 --ref scratch/ref_n3/run_20260802_182432_abl_arch10_contrast
    python scripts/verify_box.py --new <dir> --ref-root scratch/ref_n3   # auto-match
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Below this, a difference is float noise from a non-deterministic reduction
# order rather than a different machine. Same-hardware reruns measured exactly
# 0.0, so anything at 1e-6 already deserves a look.
EXACT_TOL = 1e-9
WARN_TOL = 1e-6
# Measured cross-GPU-architecture drift on validation IMI AUPRC.
ARCH_DRIFT = 0.0099


def load(run_dir: Path) -> Tuple[Optional[dict], Optional[dict]]:
    hist_path = run_dir / "history.json"
    cfg_path = run_dir / "config_snapshot.yaml"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    return hist, cfg


def ident(cfg: dict) -> Tuple[Optional[str], Optional[int]]:
    exp = (cfg.get("experiment") or {}).get("id")
    seed = (cfg.get("training") or {}).get("seed")
    return exp, seed


def find_reference(ref_root: Path, exp_id: str, seed: int) -> Optional[Path]:
    for candidate in sorted(glob.glob(str(ref_root / f"*{exp_id}*"))):
        path = Path(candidate)
        _, cfg = load(path)
        if cfg and ident(cfg) == (exp_id, seed):
            return path
    return None


def compare(new_hist: dict, ref_hist: dict) -> Dict[str, Tuple[float, int]]:
    """Max |delta| per metric key across epochs, with the epoch it happened on."""
    new_val, ref_val = new_hist.get("val") or [], ref_hist.get("val") or []
    worst: Dict[str, Tuple[float, int]] = {}
    for epoch, (a, b) in enumerate(zip(new_val, ref_val), start=1):
        for key in set(a) & set(b):
            if not isinstance(a[key], (int, float)) or not isinstance(b[key], (int, float)):
                continue
            delta = abs(a[key] - b[key])
            if key not in worst or delta > worst[key][0]:
                worst[key] = (delta, epoch)
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="run folder produced on the new box")
    ap.add_argument("--ref", default=None, help="reference run folder from the original box")
    ap.add_argument("--ref-root", default="scratch/ref_n3",
                    help="folder of reference runs to auto-match by (exp_id, seed)")
    args = ap.parse_args()

    new_dir = Path(args.new)
    new_hist, new_cfg = load(new_dir)
    if new_hist is None or new_cfg is None:
        sys.exit(f"{new_dir}: missing history.json or config_snapshot.yaml")
    exp_id, seed = ident(new_cfg)
    print(f"new run : {new_dir}\n          exp_id={exp_id} seed={seed}")

    if args.ref:
        ref_dir = Path(args.ref)
    else:
        ref_dir = find_reference(Path(args.ref_root), exp_id, seed)
        if ref_dir is None:
            sys.exit(f"no reference run for ({exp_id}, seed {seed}) under {args.ref_root}")
    ref_hist, ref_cfg = load(ref_dir)
    if ref_hist is None or ref_cfg is None:
        sys.exit(f"{ref_dir}: missing history.json or config_snapshot.yaml")
    print(f"ref run : {ref_dir}\n          exp_id={ident(ref_cfg)[0]} seed={ident(ref_cfg)[1]}")

    if ident(ref_cfg) != (exp_id, seed):
        sys.exit("refusing to compare: reference is a different (exp_id, seed)")

    n_new, n_ref = len(new_hist.get("val") or []), len(ref_hist.get("val") or [])
    if n_new != n_ref:
        print(f"\n[warn] epoch counts differ: new={n_new} ref={n_ref}; "
              f"comparing the first {min(n_new, n_ref)}")

    worst = compare(new_hist, ref_hist)
    if not worst:
        sys.exit("no comparable validation metrics found")

    ranked = sorted(worst.items(), key=lambda kv: -kv[1][0])
    top = ranked[0][1][0]
    print(f"\ncompared {len(worst)} validation metrics over "
          f"{min(n_new, n_ref)} epochs")
    print(f"max |delta| = {top:.3e}\n")
    print("worst 10 metrics:")
    for key, (delta, epoch) in ranked[:10]:
        print(f"  {delta:.3e}  epoch {epoch:2d}  {key}")

    print()
    if top <= EXACT_TOL:
        print("VERDICT: bit-identical. Same machine as far as the metrics are "
              "concerned - new seeds may be pooled with the existing ones.")
        return
    if top <= WARN_TOL:
        print(f"VERDICT: not bit-identical but within {WARN_TOL:g} (reduction-order "
              f"noise). Pooling is defensible; record the deviation in the log.")
        return
    print(f"VERDICT: DIVERGED (max |delta| {top:.3e}). The original campaign "
          f"reproduced at exactly 0.0, and cross-GPU drift on this project "
          f"measures ~{ARCH_DRIFT} on validation IMI AUPRC.")
    print("Do NOT pool these seeds with the existing three. Either find a box "
          "matching the original, or budget for re-running all 10 seeds of "
          "every declared arm on this one.")
    sys.exit(1)


if __name__ == "__main__":
    main()
