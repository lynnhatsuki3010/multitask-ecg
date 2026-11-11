"""
29_select_lr.py
---------------
Learning-rate selection of the CV study (protocol v2, section 5 and amendment 1).

Every arm was trained on fold 1 (validation = fold 2) once per lr in the grid. For
each arm this keeps the lr whose run reached the lowest total validation loss
(`loss/total`, the same quantity that picks the checkpoint). Ties go to the smaller
lr.

  * the chosen fold-1 run stays in --root and becomes that arm's fold-1 CV run
    (same seed, 3001, as every other fold-1 run would have had)
  * the other grid runs are moved to --grid-dir so the aggregation never sees them;
    their test predictions are not analysed
  * writes --choice (json: every lr's loss, the chosen lr, grid completeness)
  * writes --jobs: the phase-2 job list for scripts/cloud/run_cv.sh
        post <arm> 1 3001 <lr>        calibrate + dump the chosen fold-1 run
        full <arm> k 3000+k <lr>      train + calibrate + dump, k = 2..10

Run folders are named <arm>_f<fold>_lr<lr> (run_cv.sh passes --run-name). Rerunning
is safe: runs already moved to --grid-dir are read from there.

Usage:
  python scripts/29_select_lr.py --root checkpoints/cv_v2 --grid-dir checkpoints/cv_v2_lr_grid \
      --arms cv_layout cv_shared ... --lrs 1e-4 3e-4 1e-3 \
      --choice logs/cv_v2/lr_choice.json --jobs logs/cv_v2/jobs_phase2.txt
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def best_val_loss(run: Path) -> float:
    """Lowest validation loss/total over the run's epochs, NaN if the run did not finish."""
    if not (run / "best_model.pth").exists() or not (run / "test_metrics.json").exists():
        return math.nan
    hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    vals = [e.get("loss/total") for e in hist.get("val") or []]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return min(vals) if vals else math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--lrs", nargs="+", required=True, help="as written in the run folder names")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(2, 11)),
                    help="folds of phase 2 (default 2..10)")
    ap.add_argument("--choice", required=True)
    ap.add_argument("--jobs", required=True)
    args = ap.parse_args()

    root, grid = ROOT / args.root, ROOT / args.grid_dir
    grid.mkdir(parents=True, exist_ok=True)
    choice, jobs, problems = {}, [], []
    for arm in args.arms:
        losses = {}
        for lr in args.lrs:
            name = f"{arm}_f1_lr{lr}"
            run = root / name if (root / name).exists() else grid / name
            losses[lr] = best_val_loss(run) if run.exists() else math.nan
        done = {lr: v for lr, v in losses.items() if not math.isnan(v)}
        if not done:
            problems.append(f"{arm}: no finished grid run, arm skipped in phase 2")
            choice[arm] = {"val_loss": losses, "chosen_lr": None, "complete_grid": False}
            continue
        # lowest loss; ties broken towards the smaller lr
        lr_best = min(done, key=lambda lr: (round(done[lr], 10), float(lr)))
        complete = len(done) == len(args.lrs)
        if not complete:
            problems.append(f"{arm}: grid incomplete, finished lr {sorted(done, key=float)}")
        choice[arm] = {"val_loss": losses, "chosen_lr": lr_best, "complete_grid": complete}
        for lr in args.lrs:
            name = f"{arm}_f1_lr{lr}"
            src_root, src_grid = root / name, grid / name
            if lr == lr_best and src_grid.exists() and not src_root.exists():
                shutil.move(str(src_grid), str(src_root))
            elif lr != lr_best and src_root.exists():
                if src_grid.exists():
                    sys.exit(f"{name} exists in both {root} and {grid}; resolve by hand")
                shutil.move(str(src_root), str(src_grid))
        jobs.append(f"post {arm} 1 3001 {lr_best}")
        jobs += [f"full {arm} {k} {3000 + k} {lr_best}" for k in args.folds]

    # phase 2 ordered by fold, so complete folds accumulate if the box stops early;
    # the fold-1 post jobs are short and go first
    order = {a: i for i, a in enumerate(args.arms)}
    jobs.sort(key=lambda j: (int(j.split()[2]), order[j.split()[1]]))

    Path(args.choice).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choice).write_text(json.dumps({"arms": choice, "problems": problems}, indent=2) + "\n",
                                 encoding="utf-8")
    Path(args.jobs).write_text("\n".join(jobs) + "\n", encoding="utf-8")

    print(f"{'arm':<20}" + "".join(f"{lr:>12}" for lr in args.lrs) + "   chosen")
    for arm, c in choice.items():
        print(f"{arm:<20}" + "".join(f"{c['val_loss'][lr]:>12.5f}" for lr in args.lrs)
              + f"   {c['chosen_lr']}")
    for p in problems:
        print("WARNING", p)
    print(f"phase-2 jobs: {len(jobs)} -> {args.jobs}")


if __name__ == "__main__":
    main()
