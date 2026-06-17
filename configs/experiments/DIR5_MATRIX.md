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
| 3 | Data/labels | label-threshold + rare-class merge ablation; move PTB/Georgia finetune to decoupled arch | TODO |
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
```

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
