"""
stat_tests.py
-------------
Paired significance tests across seeds for the architecture ablation chain.

The numbers tested here are VALIDATION metrics read at each run's monitor epoch -
the same epoch that produced best_model.pth - via compare_val_runs.best_val_epoch.
That is deliberate: the ARCH table in the paper is a validation table, and running
the test on `test_metrics.json` instead would turn the test set into a selection
signal. Test-side numbers for the reported model come from aggregate_seeds.py.

Seeds are the pairing unit. Two arms trained under the same seed share data order
and init RNG stream, so a paired test blocks that variance out and is strictly
more powerful than an unpaired one.

PRE-REGISTERED PROTOCOL (declared before the n=10 runs finished; do not edit to
match an observed result - add a new named protocol instead):

  metric     validation auprc/mi/IMI at the monitor epoch
  primary    ARCH-09 vs ARCH-10   (the gated-contrast claim)
  secondary  ARCH-05 vs ARCH-10, ARCH-10 vs ARCH-10b, ARCH-05 vs ARCH-09
             Holm-corrected within the secondary family only
  test       Wilcoxon signed-rank, two-sided, exact, alpha = 0.05
  reported   mean delta with a 95% percentile bootstrap CI for every declared
             pair, whatever p comes out

ARCH-05 vs ARCH-09 is a null claim ("decoupling on its own does not help"). A
p above alpha does not establish it; only a CI contained in +/-0.01 does. That
floor is the measured GPU-architecture drift on this metric (RUNBOOK_FINAL.md),
i.e. the smallest difference this setup can call real.

Usage:
    python scripts/stat_tests.py
    python scripts/stat_tests.py --metric S/composite          # robustness only
    python scripts/stat_tests.py --glob "checkpoints/run_*abl_arch*" --out stats.md
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import statistics as stats
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_val_runs import (  # noqa: E402  (path shim has to come first)
    COMPOSITE_KEY,
    best_val_epoch,
    load_experiment_id,
    load_seed,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

METRIC = "auprc/mi/IMI"
ALPHA = 0.05

# Declared seed list. Frozen before the extra runs started; a run on any other
# seed is reported but never silently folded into a declared pair.
SEEDS: Tuple[int, ...] = (42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415)

ARCH05 = "abl_arch05_shared_tf"
ARCH09 = "abl_arch09_decoupled"
ARCH10 = "abl_arch10_contrast"
ARCH10B = "abl_arch10b_cond_dim"

# (baseline, variant): delta is reported as variant - baseline, so positive
# always means "the later step in the chain scored higher".
PRIMARY: Tuple[str, str] = (ARCH09, ARCH10)
SECONDARY: Tuple[Tuple[str, str], ...] = (
    (ARCH05, ARCH10),
    (ARCH10, ARCH10B),
    (ARCH05, ARCH09),
)
NULL_CLAIMS = {(ARCH05, ARCH09)}

# Smallest difference this setup can distinguish from hardware/run noise.
PRACTICAL_FLOOR = 0.01

BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 20260810      # fixed so the CI is reproducible


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def collect(pattern: str, metric: str) -> Dict[str, Dict[int, Tuple[float, Path]]]:
    """Return {exp_id: {seed: (value, run_dir)}} from validation monitor epochs.

    Duplicate (exp_id, seed) pairs keep the newest run, matching aggregate_seeds:
    a re-run supersedes the run it repeated.
    """
    out: Dict[str, Dict[int, Tuple[float, Path, float]]] = {}
    for path in glob.glob(pattern):
        run_dir = Path(path)
        if not run_dir.is_dir() or run_dir.name.endswith("-debug"):
            continue
        picked = best_val_epoch(run_dir)
        if picked is None:
            continue
        _, val_metrics, _ = picked
        value = val_metrics.get(metric)
        seed = load_seed(run_dir)
        if value is None or seed is None:
            continue
        exp_id = load_experiment_id(run_dir)
        mtime = os.path.getmtime(run_dir)
        prev = out.setdefault(exp_id, {}).get(seed)
        if prev is None or mtime > prev[2]:
            out[exp_id][seed] = (value, run_dir, mtime)
    return {e: {s: (v, d) for s, (v, d, _) in seeds.items()} for e, seeds in out.items()}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks of |d|, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def wilcoxon(deltas: Sequence[float]) -> Optional[Dict]:
    """Two-sided Wilcoxon signed-rank.

    Exact by enumerating all 2^n sign assignments under the null (n <= 20),
    normal approximation with continuity correction above that. Zero
    differences are dropped, the classic Wilcoxon handling, and counted in the
    result so a run of identical values cannot hide inside a small n.

    Verified against scipy.stats.wilcoxon(method="exact") on 400 random tie-free
    samples: W and p match to 1e-12. When |d| has ties the two disagree by
    design - scipy evaluates the untied null (and warns the exact p is invalid),
    while enumerating here conditions on the observed average ranks, which is
    the right null for tied data. Ties are near-impossible on these float
    metrics anyway; no scipy dependency is added.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n_zero = len(deltas) - len(nonzero)
    n = len(nonzero)
    if n == 0:
        return None

    ranks = average_ranks([abs(d) for d in nonzero])
    total_rank = sum(ranks)
    w_plus = sum(r for d, r in zip(nonzero, ranks) if d > 0)
    stat = min(w_plus, total_rank - w_plus)

    if n <= 20:
        hits = 0
        for mask in range(1 << n):
            wp = 0.0
            for i, r in enumerate(ranks):
                if mask >> i & 1:
                    wp += r
            if min(wp, total_rank - wp) <= stat + 1e-12:
                hits += 1
        p = hits / float(1 << n)
        exact = True
    else:
        mean_w = n * (n + 1) / 4.0
        sd_w = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
        z = (abs(w_plus - mean_w) - 0.5) / sd_w if sd_w else 0.0
        p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(z / (2 ** 0.5))))
        exact = False

    return {"W": stat, "p": min(1.0, p), "n": n, "n_zero": n_zero, "exact": exact}


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def min_achievable_p(n: int) -> float:
    """Smallest two-sided p an exact Wilcoxon can return at this n.

    Printed alongside the result because with n=3 that floor is 0.25: no
    arrangement of three seeds can clear alpha, so a non-significant verdict
    there says nothing about the effect.
    """
    if n == 0:
        return 1.0
    if n <= 20:
        return 2.0 / float(1 << n) * 1.0 if n > 0 else 1.0
    return 0.0


def bootstrap_ci(deltas: Sequence[float], iters: int, rng: random.Random,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference.

    Seeds are resampled as whole pairs, so the pairing survives the resample.
    """
    n = len(deltas)
    means = []
    for _ in range(iters):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return lo, hi


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def compare(data, baseline: str, variant: str, rng: random.Random) -> Optional[Dict]:
    a, b = data.get(baseline, {}), data.get(variant, {})
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    deltas = [b[s][0] - a[s][0] for s in shared]
    res = {
        "baseline": baseline,
        "variant": variant,
        "seeds": shared,
        "n": len(shared),
        "deltas": deltas,
        "mean": stats.mean(deltas),
        "median": stats.median(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "base_mean": stats.mean([a[s][0] for s in shared]),
        "var_mean": stats.mean([b[s][0] for s in shared]),
    }
    if len(deltas) > 1:
        res["sd"] = stats.stdev(deltas)
        res["dz"] = res["mean"] / res["sd"] if res["sd"] else float("inf")
        res["ci"] = bootstrap_ci(deltas, BOOTSTRAP_ITERS, rng)
    else:
        res["sd"], res["dz"], res["ci"] = 0.0, float("nan"), (float("nan"), float("nan"))
    res["test"] = wilcoxon(deltas)
    return res


def verdict(res: Dict, p_used: float) -> str:
    pair = (res["baseline"], res["variant"])
    lo, hi = res["ci"]
    if pair in NULL_CLAIMS:
        if lo == lo and abs(lo) <= PRACTICAL_FLOOR and abs(hi) <= PRACTICAL_FLOOR:
            return f"bounded below the +/-{PRACTICAL_FLOOR} floor"
        if p_used >= ALPHA:
            return "inconclusive (CI includes 0 and is wider than the floor)"
        return "difference detected - the null claim does not hold"
    if p_used < ALPHA:
        return "significant"
    return "not significant"


def describe(res: Dict, p_used: Optional[float], label: str) -> List[str]:
    t = res["test"]
    lo, hi = res["ci"]
    lines = [
        f"### {label}: {res['variant']}  -  {res['baseline']}",
        "",
        f"- seeds (n={res['n']}): {res['seeds']}",
        f"- arm means: {res['baseline']} {res['base_mean']:.4f} | "
        f"{res['variant']} {res['var_mean']:.4f}",
        f"- delta: mean {res['mean']:+.4f}, median {res['median']:+.4f}, "
        f"SD {res['sd']:.4f}",
        f"- 95% CI (percentile bootstrap, {BOOTSTRAP_ITERS} resamples): "
        f"[{lo:+.4f}, {hi:+.4f}]",
        f"- Cohen d_z: {res['dz']:.2f}   variant wins {res['wins']}/{res['n']} seeds",
    ]
    if res["n"] < 6:
        lines.append(f"- **CI unreliable**: a percentile bootstrap resamples "
                     f"{res['n']} distinct values, so the interval understates "
                     f"the true spread. Treat it as descriptive until n=10.")
    if t is None:
        lines.append("- Wilcoxon: not computable (every difference was exactly zero)")
        return lines + [""]
    kind = "exact" if t["exact"] else "normal approx"
    holm_note = ""
    if p_used is not None and abs(p_used - t["p"]) > 1e-12:
        holm_note = f", p_holm={p_used:.4f}"
    lines.append(f"- Wilcoxon signed-rank ({kind}, two-sided): W={t['W']:.1f}, "
                 f"p={t['p']:.4f}{holm_note}")
    if t["n_zero"]:
        lines.append(f"- {t['n_zero']} zero difference(s) dropped before ranking")
    floor = min_achievable_p(t["n"])
    if floor > ALPHA:
        lines.append(f"- **underpowered**: at n={t['n']} the smallest reachable "
                     f"two-sided p is {floor:.3f} > alpha={ALPHA}. No seed "
                     f"arrangement can clear the threshold; read the CI, not p.")
    lines.append(f"- verdict: **{verdict(res, p_used if p_used is not None else t['p'])}**")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="checkpoints/run_*abl_arch*",
                    help="run folders to scan (default: the ARCH ablation chain)")
    ap.add_argument("--metric", default=METRIC,
                    help=f"validation metric key (default {METRIC}). Anything other "
                         f"than the pre-registered metric is exploratory, not confirmatory.")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    data = collect(args.glob, args.metric)
    if not data:
        print(f"No runs with validation '{args.metric}' matched {args.glob!r}.")
        return

    rng = random.Random(BOOTSTRAP_SEED)
    confirmatory = args.metric == METRIC

    lines = [
        "# Paired significance tests (validation)",
        "",
        f"Metric: `{args.metric}` at each run's monitor epoch. "
        f"Test set untouched.",
        f"Bootstrap seed {BOOTSTRAP_SEED}; alpha {ALPHA}.",
        "",
    ]
    if not confirmatory:
        lines += [f"> Exploratory run: `{args.metric}` is not the pre-registered "
                  f"metric (`{METRIC}`). Do not report these p-values as "
                  f"confirmatory evidence.", ""]

    lines += ["## Arms found", ""]
    for exp_id in sorted(data):
        seeds = sorted(data[exp_id])
        extra = [s for s in seeds if s not in SEEDS]
        missing = [s for s in SEEDS if s not in seeds]
        values = [data[exp_id][s][0] for s in seeds]
        cell = (f"{stats.mean(values):.4f} +/- {stats.stdev(values):.4f}"
                if len(values) > 1 else f"{values[0]:.4f}")
        note = ""
        if missing:
            note += f"  [missing declared seeds: {missing}]"
        if extra:
            note += f"  [undeclared seeds present: {extra}]"
        lines.append(f"- `{exp_id}` n={len(seeds)}: {cell}{note}")
    lines.append("")

    # Primary - never Holm-corrected; it is a single declared hypothesis.
    lines += ["## Primary hypothesis", ""]
    primary = compare(data, PRIMARY[0], PRIMARY[1], rng)
    if primary is None:
        lines += ["_no shared seeds between the primary arms_", ""]
    else:
        lines += describe(primary, None, "PRIMARY")

    # Secondary family - Holm within the family only.
    lines += ["## Secondary family (Holm-corrected)", ""]
    results = [compare(data, a, b, rng) for a, b in SECONDARY]
    usable = [(pair, res) for pair, res in zip(SECONDARY, results)
              if res is not None and res["test"] is not None]
    if usable:
        adjusted = holm([res["test"]["p"] for _, res in usable])
        for (pair, res), p_adj in zip(usable, adjusted):
            tag = "SECONDARY (null claim)" if pair in NULL_CLAIMS else "SECONDARY"
            lines += describe(res, p_adj, tag)
    for pair, res in zip(SECONDARY, results):
        if res is None:
            lines += [f"_no shared seeds for {pair[1]} - {pair[0]}_", ""]

    lines += [
        "## How to read this",
        "",
        "- The CI is the reportable effect size; p only says whether 0 is",
        "  excluded. Quote both, and drop the older SE-ratio phrasing.",
        f"- A CI inside +/-{PRACTICAL_FLOOR} is smaller than the measured drift",
        "  between GPU architectures, so it is not a claimable difference.",
        "- 'not significant' is never evidence of no effect unless the CI is",
        "  bounded below that floor.",
        "",
    ]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
