#!/usr/bin/env python
"""Generate the 13-label golden-pipeline configs (norm / aug / sampler / loss) at
label threshold 50, on the SHARED baseline architecture (multi_branch_transformer),
for the unified journal re-run (conduction present from the start).

Base config: arch_e06_multi_branch.yaml (13-label shared baseline, ARCH-5/6).
Dataset: processed_dir5_imi50 (13 labels, threshold 50 uniform — already built).

Stage-gated (winner carries forward), single-variable per config:
  NORM  (aug off, uniform, BCE):        zscore / robust / minmax
  AUG   (robust, uniform, BCE):         lead_dropout / mixup / random_crop / noise_warp
  SAMP  (robust, lead_dropout, BCE):    uniform / weighted
  LOSS  (robust, lead_dropout, uniform):BCE(base) / pos_weight / focal / asl

Usage:
  python scripts/make_golden13_configs.py
"""
import copy
import os
import yaml

CFG_DIR = "configs/experiments"
BASE = os.path.join(CFG_DIR, "arch_e06_multi_branch.yaml")
PROCESSED = "data/processed_dir5_imi50"
SPLITS = "data/splits_dir5_imi50"

AUG_KEYS = ["aug_lead_dropout", "aug_random_crop", "aug_baseline_wander", "aug_time_warp", "aug_mixup"]

# (id, normalize, aug flags on, weighted_sampler, use_pos_weight, use_focal, use_asl)
MATRIX = [
    # NORM stage — no aug, uniform, BCE
    ("g13_norm_e01_zscore", "zscore", [],                                    False, False, False, False),
    ("g13_norm_e02_robust", "robust", [],                                    False, False, False, False),
    ("g13_norm_e03_minmax", "minmax", [],                                    False, False, False, False),
    # AUG stage — robust, uniform, BCE
    ("g13_aug_e04_lead_dropout", "robust", ["aug_lead_dropout"],             False, False, False, False),
    ("g13_aug_e01_mixup",        "robust", ["aug_mixup"],                    False, False, False, False),
    ("g13_aug_e02_random_crop",  "robust", ["aug_random_crop"],              False, False, False, False),
    ("g13_aug_e03_noise_warp",   "robust", ["aug_baseline_wander", "aug_time_warp"], False, False, False, False),
    # SAMP stage — robust, lead_dropout, BCE
    ("g13_samp_e01_uniform",  "robust", ["aug_lead_dropout"],                False, False, False, False),
    ("g13_samp_e02_weighted", "robust", ["aug_lead_dropout"],                True,  False, False, False),
    # LOSS stage — robust, lead_dropout, uniform
    ("g13_loss_e01_posweight", "robust", ["aug_lead_dropout"],               False, True,  False, False),
    ("g13_loss_e02_focal",     "robust", ["aug_lead_dropout"],               False, False, True,  False),
    ("g13_loss_e04_asl",       "robust", ["aug_lead_dropout"],               False, False, False, True),
]


def main():
    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    made = []
    for exp_id, norm, augs_on, wrs, posw, focal, asl in MATRIX:
        cfg = copy.deepcopy(base)
        # dataset -> 13-label, threshold 50 uniform
        cfg["paths"]["processed"] = PROCESSED
        cfg["paths"]["splits"] = SPLITS
        cfg["dataset"]["label_threshold_overrides"] = {}
        # preprocessing
        cfg["preprocessing"]["normalize"] = norm
        # augmentation: all off, then enable the tested ones
        aug = cfg.setdefault("augmentation", {})
        for k in AUG_KEYS:
            aug[k] = (k in augs_on)
        aug.setdefault("mixup_alpha", 0.2)
        # training knobs
        tr = cfg["training"]
        tr["weighted_sampler"] = wrs
        tr["use_pos_weight"] = posw
        tr["use_focal"] = focal
        tr["use_asl"] = asl
        # ensure all 4 MI labels present in loss_weights (DIR5 loss needs ilmi/ami)
        lw = tr.setdefault("loss_weights", {})
        for k in ["arrhythmia", "mi", "conduction", "imi", "asmi", "ilmi", "ami"]:
            lw.setdefault(k, 1.0)
        lw.setdefault("hrv", 0.0)
        # identifiable run folder
        cfg["experiment"] = {"id": exp_id, "track": "golden13_imi50", "parent": "arch_e06_multi_branch"}

        dst = os.path.join(CFG_DIR, f"{exp_id}.yaml")
        with open(dst, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        made.append(dst)
        print(f"  [ok] {dst}")

    print(f"\nGenerated {len(made)} config(s) on 13-label shared baseline (multi_branch_transformer) @ threshold 50.")


if __name__ == "__main__":
    main()
