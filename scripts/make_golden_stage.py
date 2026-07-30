#!/usr/bin/env python
"""Generate ONE stage of the 13-label golden pipeline at a time (stage-gated).

Every config differs from its stage siblings by exactly one pipeline stage, and
every config in a later stage inherits the winners you pass in explicitly — so
the chain is auditable and no winner is silently hard-coded. One arm covers a
technique rather than a single flag: `noise_warp` turns on baseline wander and
time warping together, since they are two halves of one signal-distortion scheme
and were never meant to be swept apart.

All runs share one dataloader regime (`--num-workers`, default 28) because
num_workers changes the numpy augmentation stream and is therefore NOT neutral:
comparing a num_workers=0 run against a num_workers=28 run confounds the
technique with the augmentation realisation.

Base architecture: multi_branch_transformer (shared baseline, ARCH-5/6).
Dataset: processed_dir5_imi50 (13 labels, threshold 50 uniform).

Usage (run each stage, evaluate, then generate the next with the winner):
  python scripts/make_golden_stage.py --stage norm
  python scripts/make_golden_stage.py --stage aug  --norm zscore
  python scripts/make_golden_stage.py --stage samp --norm zscore --aug lead_dropout
  python scripts/make_golden_stage.py --stage loss --norm zscore --aug lead_dropout --samp uniform
"""
import argparse
import copy
import os
import yaml

CFG_DIR = "configs/experiments"
BASE = os.path.join(CFG_DIR, "arch_e06_multi_branch.yaml")
PROCESSED = "data/processed_dir5_imi50"
SPLITS = "data/splits_dir5_imi50"

AUG_KEYS = ["aug_lead_dropout", "aug_random_crop", "aug_baseline_wander", "aug_time_warp", "aug_mixup"]

# augmentation name -> flags to switch on
AUG_OPTIONS = {
    "none":         [],
    "lead_dropout": ["aug_lead_dropout"],
    "mixup":        ["aug_mixup"],
    "random_crop":  ["aug_random_crop"],
    "noise_warp":   ["aug_baseline_wander", "aug_time_warp"],
}

# loss name -> (use_pos_weight, use_focal, use_asl)
LOSS_OPTIONS = {
    "bce":       (False, False, False),
    "posweight": (True,  False, False),
    "focal":     (False, True,  False),
    "asl":       (False, False, True),
}

SAMP_OPTIONS = {"uniform": False, "weighted": True}

# Which variants each stage sweeps, and which variant is just the inherited
# baseline cell (already trained in the previous stage -> do not re-run).
STAGES = {
    "norm": dict(variants=["zscore", "robust", "minmax"], reuse=None),
    "aug":  dict(variants=list(AUG_OPTIONS),              reuse="none"),
    "samp": dict(variants=list(SAMP_OPTIONS),             reuse="uniform"),
    "loss": dict(variants=list(LOSS_OPTIONS),             reuse="bce"),
}


def build(base, stage, variant, norm, aug, samp, loss, num_workers):
    cfg = copy.deepcopy(base)

    cfg["paths"]["processed"] = PROCESSED
    cfg["paths"]["splits"] = SPLITS
    cfg["dataset"]["label_threshold_overrides"] = {}     # threshold 50 for ALL labels

    # resolve the four knobs, overriding the one this stage sweeps
    norm = variant if stage == "norm" else norm
    aug  = variant if stage == "aug"  else aug
    samp = variant if stage == "samp" else samp
    loss = variant if stage == "loss" else loss

    cfg["preprocessing"]["normalize"] = norm

    aug_cfg = cfg.setdefault("augmentation", {})
    for k in AUG_KEYS:
        aug_cfg[k] = k in AUG_OPTIONS[aug]
    aug_cfg.setdefault("mixup_alpha", 0.2)

    posw, focal, asl = LOSS_OPTIONS[loss]
    tr = cfg["training"]
    tr["use_pos_weight"], tr["use_focal"], tr["use_asl"] = posw, focal, asl
    tr["weighted_sampler"] = SAMP_OPTIONS[samp]
    tr["num_workers"] = num_workers
    # Keep best_model.pth only. The base config saves every 5 epochs, which adds
    # three 147 MB dumps per run (~4.8 GB across the 11-run chain) that nothing
    # downstream reads.
    tr["save_every"] = 10 ** 6

    lw = tr.setdefault("loss_weights", {})
    for k in ["arrhythmia", "mi", "conduction", "imi", "asmi", "ilmi", "ami"]:
        lw.setdefault(k, 1.0)
    lw.setdefault("hrv", 0.0)

    stage_no = {"norm": 1, "aug": 2, "samp": 3, "loss": 4}[stage]
    exp_id = f"gp_s{stage_no}_{stage}_{variant}"
    cfg["experiment"] = {
        "id": exp_id,
        "track": "golden13_imi50_staged",
        "parent": "arch_e06_multi_branch",
        "stage": stage,
        "variant": variant,
        "inherited": {"normalize": norm, "augmentation": aug, "sampler": samp, "loss": loss},
    }
    return exp_id, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--norm", default="zscore", help="inherited NORM winner")
    ap.add_argument("--aug", default="none", choices=list(AUG_OPTIONS), help="inherited AUG winner")
    ap.add_argument("--samp", default="uniform", choices=list(SAMP_OPTIONS), help="inherited SAMP winner")
    ap.add_argument("--loss", default="bce", choices=list(LOSS_OPTIONS), help="inherited LOSS baseline")
    ap.add_argument("--num-workers", type=int, default=28)
    args = ap.parse_args()

    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    spec = STAGES[args.stage]
    made, skipped = [], []
    for variant in spec["variants"]:
        exp_id, cfg = build(base, args.stage, variant, args.norm, args.aug,
                            args.samp, args.loss, args.num_workers)
        # the inherited-baseline arm was already trained in the previous stage
        if variant == spec["reuse"]:
            skipped.append((exp_id, variant))
            continue
        dst = os.path.join(CFG_DIR, f"{exp_id}.yaml")
        with open(dst, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        made.append(dst)

    print(f"STAGE {args.stage.upper()}  (num_workers={args.num_workers})")
    print(f"  inherited: norm={args.norm} aug={args.aug} samp={args.samp} loss={args.loss}")
    print()
    for p in made:
        print(f"  [new] {p}")
    for exp_id, v in skipped:
        print(f"  [reuse] variant '{v}' = inherited baseline cell (already trained; do not re-run)")
    print()
    print("Run:")
    for p in made:
        print(f"  python scripts/02_train.py --config {p.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
