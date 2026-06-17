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
| 2 | MI head | reduce IMI/ILMI and ASMI/AMI overlap in `DecoupledSubsetMIHead` (hierarchy / separate encoders / soft exclusion) | TODO |
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
```

## Compare

```bash
python scripts/compare_branche10_runs.py
```
