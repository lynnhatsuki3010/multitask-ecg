# Experiment Log

## Branch: E-02 — Bug Fixes, Gradient Isolation & Training Improvements

**Date**: 2026-05-03  
**Parent Branch**: E-01  
**Status**: Ready for training

---

### Objective
Fix critical bugs identified in code review, implement gradient isolation for MI task, 
and improve training strategy to resolve IMI performance ceiling.

### Baseline (E-01 best run: run_20260503_074622)

| Metric | Value |
|--------|-------|
| IMI AUROC | 0.940 |
| IMI AUPRC | 0.511 |
| IMI F1 | 0.489 |
| ASMI F1 | 0.743 |
| Arrhy macro F1 | 0.751 |

---

### Changes Made

#### Phase 1: Cleanup
- **Deleted**: `ecg_transformer.py`, `multitask_head.py`, `phased_multitask_transformer.py`
- **Deleted**: `05_train_phased_multitask.py`, `history_stats.py`
- **Deleted**: 7 unused config files (kept `baseline` + `imi_randomgrouped`)
- **Deleted**: `tmp_sweep_configs/` directory
- **Updated**: `.gitignore` to cover all `data/processed_*` and `data/splits_*`

#### Phase 2: Bug Fixes
1. **pos_weight on train split only** (`02_train.py`):
   - Was: computed on entire dataset from `pos_weights.json`
   - Now: computed live from `train_label_matrix` after split

2. **Label smoothing fix** (`losses.py`):
   - Was: `y*(1-ε) + ε/2` — negatives become 0.01, interfering with pos_weight
   - Now: `y*(1-ε)` — positives smoothed (1→0.98), negatives stay at 0

3. **Focal Loss object reuse** (`losses.py`):
   - Was: creating new `BinaryFocalLoss` instance per batch in `_binary_loss()`
   - Now: pre-built `imi_loss_fn` and `asmi_loss_fn` in `__init__`

4. **Label builder** (`label_builder.py`):
   - Was: `confidence == 0.0` treated as "Present" for ALL labels
   - Now: explicit `RHYTHM_STATEMENTS` set — rhythm=always present, diagnostic=threshold

#### Phase 3: Architecture
- **Gradient isolation** (`ecg_multitask.py`):
  - New `mi_gradient_scale=0.3` parameter
  - MI loss gradients scaled to 30% when flowing back into shared backbone
  - MI head LeadGroupEncoder still gets raw signal (unaffected)
  - Reduces gradient competition: backbone can serve both tasks

#### Phase 4: Training Strategy
- **Differential LR** (`trainer.py`):
  - MI head params: 3x base LR (`mi_lr_multiplier=3.0`)
  - Allows MI head to adapt faster despite reduced gradient flow

- **Cosine Warm Restart scheduler**:
  - Was: `OneCycleLR` — LR decays monotonically, no escape from local minima
  - Now: `cosine_restart` with T0=10 — LR resets every 10 epochs

- **Training duration**: 30 → 50 epochs, patience 8 → 12

- **Gradient norm logging**: Every epoch prints `gn_mi/bb` ratio to monitor balance

---

### Config: `hybrid_transformer_imi_randomgrouped.yaml`
Key parameters:
```yaml
model:
  mi_gradient_scale: 0.3

training:
  mi_lr_multiplier: 3.0
  scheduler: cosine_restart
  cosine_T0: 10
  epochs: 50
  early_stopping_patience: 12
```

### Files Modified
- `src/models/ecg_multitask.py` — gradient isolation
- `src/models/factory.py` — removed phased model, added mi_gradient_scale
- `src/training/losses.py` — label smoothing fix, focal loss reuse
- `src/training/trainer.py` — differential LR, gradient norm logging, OneCycleLR fix
- `src/data/label_builder.py` — rhythm vs diagnostic label logic
- `scripts/02_train.py` — pos_weight from train split
- `configs/hybrid_transformer_imi_randomgrouped.yaml` — new training params
- `configs/hybrid_transformer_baseline.yaml` — added mi_gradient_scale default

### Results (After Training)
_To be filled after training run_

| Metric | E-01 | E-02 | Delta |
|--------|------|------|-------|
| IMI AUROC | 0.940 | | |
| IMI AUPRC | 0.511 | | |
| IMI F1 | 0.489 | | |
| ASMI F1 | 0.743 | | |
| Arrhy F1 | 0.751 | | |
