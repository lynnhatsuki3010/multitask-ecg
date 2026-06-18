# DIR5 — Bottleneck Resolution Matrix

Branch: `EXP-DIR5-BOTTLENECK-RESOLUTION`. Goal: lift MI (especially IMI)
macro-F1 on the E10 decoupled architecture by attacking the diagnosed
bottlenecks (per-label loss bug, recall-biased calibration, rare-class
instability, IMI/ILMI and ASMI/AMI anatomical overlap).

Baseline reference: `branche10_00_baseline.yaml` (== `arch_e10_decoupled`,
BCE, no ASL, monitor `imi_auprc`).

## Phases

| Phase | Scope | Change vs baseline | Config / code |
|-------|-------|--------------------|---------------|
| 0 | Loss bug | 4-label MI head now applies per-label `loss_weights.{imi,asmi,ilmi,ami}` and keeps per-label `pos_weight` | `src/training/losses.py`, `scripts/02_train.py` |
| 1 | Cheap levers | weighted sampler ON; precision-aware tuning (IMI F_beta beta=0.5 + `mi_min_precision.IMI=0.5`); full augmentation; monitor `mi_macro_f1_tuned` (val tuned each epoch) | `dir5_p1_cheap_levers.yaml` |
| 2 | MI head | `model.mi_head_mode: contrast` — region-once encoding + shared region TF (IMI/ASMI also see lateral leads) + sibling contrast refinement to push IMI<->ILMI and ASMI<->AMI apart | `dir5_p2_mi_contrast.yaml`, `src/models/ecg_multitask.py` |
| 2.1 | MI head | `model.mi_head_mode: contrast_v2` — adds a dedicated joint encoder per extended sibling (anterior6 for AMI, inferolateral for ILMI) + gated contrast (identity at init) to fix the ILMI/AMI regression while keeping the IMI gain | `dir5_p2b_mi_contrast_v2.yaml`, `src/models/ecg_multitask.py` |
| 3 | Data/labels | label-threshold ablation (`dir5_p3a_imi50`: IMI 80->50, separate dataset) on the best head (contrast_v2); rare-class merge + PTB/Georgia decoupled finetune still TODO | `dir5_p3a_imi50.yaml` |
| 4 | Eval | split common vs rare report; add micro/weighted-F1 + bootstrap CIs for classes <30 samples | TODO |

## Phase 1 design notes

- `use_pos_weight` stays OFF: weighted_sampler already up-weights positives;
  stacking pos_weight inflates IMI false positives (hurts precision).
- per-label `loss_weights` kept neutral (1.0); IMI re-weighting is a separate
  ablation (Phase 0 made these effective).
- monitor `mi_macro_f1_tuned` requires `eval.val_tune_thresholds_every > 0`.

## Run

```bash
# Phase 1
python scripts/02_train.py --config configs/experiments/dir5_p1_cheap_levers.yaml
python scripts/03_calibrate.py --dir checkpoints/run_*_dir5_p1_cheap_levers*

# Phase 2 (clean ablation vs E10: only mi_head_mode changes)
python scripts/02_train.py --config configs/experiments/dir5_p2_mi_contrast.yaml
python scripts/03_calibrate.py --dir checkpoints/run_*_dir5_p2_mi_contrast*

# Phase 2.1 (fix extended-sibling regression)
python scripts/02_train.py --config configs/experiments/dir5_p2b_mi_contrast_v2.yaml
python scripts/03_calibrate.py --dir checkpoints/run_*_dir5_p2b_mi_contrast_v2*

# Phase 3a (IMI threshold ablation - builds a SEPARATE dataset first)
python scripts/01_build_metadata.py --config configs/experiments/dir5_p3a_imi50.yaml
python scripts/02_train.py --config configs/experiments/dir5_p3a_imi50.yaml
python scripts/03_calibrate.py --dir checkpoints/run_*_dir5_p3a_imi50*
```

## Phase 2.1 outcome (best MI head)

`contrast_v2` recovered MI macro to E10 parity (F1 0.494 vs 0.498, AUPRC 0.476
vs 0.477) while keeping a clean IMI gain over E10 (IMI AUPRC 0.437 -> 0.460,
F1 0.437 -> 0.464) and best-ever AMI (F1 0.205, AUPRC 0.159). ILMI is the only
residual loss (F1 0.605 -> 0.554) because IMI/ILMI share the inferior region
(contrast is zero-sum within the pair). Adopted as the DIR5 MI head; further IMI
gains now require data/label changes (Phase 3), not architecture.

## Phase 2 outcome (for the record)

`contrast` raised IMI separability for the first time (IMI AUPRC 0.437 -> 0.480,
F1 0.437 -> 0.490 with both precision AND recall up) — cheap levers could not do
this. But it degraded the extended siblings (ILMI F1 0.605 -> 0.549, AMI F1
0.188 -> 0.105), because AMI lost its dedicated V1-V6 joint encoding and the hard
symmetric subtraction impoverished the extended-sibling representation. MI macro
roughly flat (F1 0.469, AUPRC 0.468). Phase 2.1 (`contrast_v2`) restores the
joint encoders and uses gated contrast to recover ILMI/AMI.

## Phase 1 outcome (for the record)

Phase 1 did NOT beat E10: MI macro F1 0.454 vs E10 0.498, and MI macro AUPRC
0.450 vs 0.477. Precision-aware tuning lifted IMI precision (0.34 -> 0.49) but
collapsed recall (0.62 -> 0.30); cheap levers move the operating point but do
not raise separability. Conclusion: the ceiling is representational -> Phase 2
attacks it at the architecture level (raise AUPRC), not via calibration.

## Compare

```bash
python scripts/compare_branche10_runs.py
```
