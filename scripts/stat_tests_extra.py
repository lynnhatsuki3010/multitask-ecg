"""
stat_tests_extra.py
-------------------
Paired tests for the post-hoc contrast arms, ARCH-13, ARCH-14 and ARCH-15.

stat_tests.py hard-codes the pre-registered family and must stay that way. The
two new arms were added after the protocol was frozen, to answer one question -
does the gated sibling contrast itself carry the IMI gain - so they are
reported here, descriptively, with no Holm correction and no claim to be
confirmatory.

Each pair is oriented so that a positive delta means "the arm WITH the
explicit contrast scored higher":

  ARCH-12 - ARCH-14   partial decoupling, contrast on vs off   (main question)
  ARCH-10 - ARCH-13   full decoupling,    contrast on vs off
  ARCH-14 - ARCH-05   what the projections + heads add without the contrast
  ARCH-13 - ARCH-09   what the contrast_v2 token layout adds without the contrast
  ARCH-12 - ARCH-15   partial decoupling, gate on vs off (contrast kept in both)
  ARCH-15 - ARCH-14   partial decoupling, ungated contrast vs no contrast

Same collection, Wilcoxon, bootstrap and seed list as stat_tests.py, imported
rather than copied, so the numbers are comparable line for line.

Usage:
    python scripts/stat_tests_extra.py --out docs/stat_tests_contrast.md
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stat_tests import (  # noqa: E402
    ARCH05, ARCH09, ARCH10, BOOTSTRAP_SEED, SEEDS, collect, compare, describe,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ARCH12 = "abl_arch12_partial"
ARCH13 = "abl_arch13_nogate"
ARCH14 = "abl_arch14_partial_nogate"
ARCH15 = "abl_arch15_partial_ungated"

# (without, with): compare() reports with - without.
PAIRS = (
    ("ARCH-12 - ARCH-14 (partial: contrast on - off)", ARCH14, ARCH12),
    ("ARCH-10 - ARCH-13 (decoupled: contrast on - off)", ARCH13, ARCH10),
    ("ARCH-14 - ARCH-05 (projection + heads, no contrast)", ARCH05, ARCH14),
    ("ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast)", ARCH09, ARCH13),
    ("ARCH-12 - ARCH-15 (partial: gate on - off)", ARCH15, ARCH12),
    ("ARCH-15 - ARCH-14 (partial: ungated contrast - none)", ARCH14, ARCH15),
)
METRICS = ("auprc/mi/IMI", "auprc/mi/ILMI", "auprc/mi/ASMI", "auprc/mi/AMI",
           "auprc/mi/macro", "S/composite")


def collect_all(globs, metric):
    """collect() over several folders. The same (exp_id, seed) in two folders is
    refused: it means one box's run would silently stand in for another's."""
    merged = {}
    for pattern in globs:
        for exp_id, seeds in collect(pattern, metric).items():
            slot = merged.setdefault(exp_id, {})
            for seed, val in seeds.items():
                if seed in slot:
                    sys.exit(f"{exp_id} seed {seed} found in two folders:\n  {slot[seed][1]}\n"
                             f"  {val[1]}\nPass globs that do not overlap.")
                slot[seed] = val
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", action="append", default=None,
                    help="run folders to scan; repeat to combine folders "
                         "(default: checkpoints/run_*abl_arch*)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    globs = args.glob or ["checkpoints/run_*abl_arch*"]

    lines = ["# Contrast arms - exploratory paired tests (validation)", "",
             f"Run folders: {', '.join(f'`{g}`' for g in globs)}", "",
             "Post-hoc: added after the protocol was frozen. No Holm correction; "
             "report as descriptive evidence.",
             "Positive delta = the arm WITH the explicit gated contrast is higher.",
             f"Bootstrap seed {BOOTSTRAP_SEED}.", ""]

    summary = ["## Summary", "",
               "| pair | metric | with | without | delta | 95% CI | wins | p |",
               "|---|---|---|---|---|---|---|---|"]
    detail = ["## Detail", ""]

    for metric in METRICS:
        data = collect_all(globs, metric)
        rng = random.Random(BOOTSTRAP_SEED)
        if metric == METRICS[0]:
            lines += ["## Arms found", ""]
            for exp_id in (ARCH05, ARCH09, ARCH10, ARCH12, ARCH13, ARCH14, ARCH15):
                seeds = sorted(data.get(exp_id, {}))
                missing = [s for s in SEEDS if s not in seeds]
                vals = [data[exp_id][s][0] for s in seeds] if seeds else []
                cell = (f"{statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}"
                        if len(vals) > 1 else (f"{vals[0]:.4f}" if vals else "-"))
                note = f"  [missing seeds: {missing}]" if missing else ""
                lines.append(f"- `{exp_id}` n={len(seeds)}: {cell}{note}")
            lines.append("")
        for label, without, with_ in PAIRS:
            res = compare(data, without, with_, rng)
            if res is None:
                summary.append(f"| {label} | `{metric}` | - | - | no shared seeds | | | |")
                continue
            t = res["test"]
            p = f"{t['p']:.4f}" if t else "n/a"
            lo, hi = res["ci"]
            summary.append(
                f"| {label} | `{metric}` | {res['var_mean']:.4f} | {res['base_mean']:.4f} | "
                f"{res['mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {res['wins']}/{res['n']} | {p} |")
            if metric == METRICS[0]:
                detail += describe(res, None, label)

    report = "\n".join(lines + summary + [""] + detail)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
