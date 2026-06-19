# ECG Multi-Task Transformer (DIR4 / DIR5)

A deep learning project for 12-lead ECG analysis on PTB-XL using a **Fully Decoupled Multitask** architecture (E10): three independent paths from raw signal — no Shared CNN.

> **Current best (DIR5):** `contrast_v2` MI head + IMI label threshold 50.
> - Config: `configs/experiments/dir5_p3a_imi50.yaml`
> - Result folder: `checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug`
> - Dataset: `data/processed_dir5_imi50_500hz` / `data/splits_dir5_imi50_500hz` (IMI threshold 50, separate from the IMI:80 `processed_dir4`)
> - Headline: **MI Macro F1 0.511**, **IMI F1 0.590**, **IMI AUPRC 0.645** (IMI test support 175).
> - See `EXPERIMENT_LOG.md` (DIR5 section) and `configs/experiments/DIR5_MATRIX.md` for the full phase history.

## Directory Structure
```
ECG_HRV/
├── data/
│   ├── raw/PTB-XL/          ← Raw dataset
│   ├── processed_dir4/      ← Preprocessed metadata and matrices (13 labels)
│   └── splits_dir4/         ← Train/val/test split indices
├── src/
│   ├── data/                ← Data loading, preprocessing, and label extraction
│   ├── models/              ← Decoupled backbone & multi-task heads
│   ├── training/            ← Custom training loop & loss functions
│   └── utils/               ← Evaluation metrics
├── configs/
│   └── experiments/         ← Configuration files (ablation, final)
├── scripts/                 ← Executable scripts for pipeline stages
└── checkpoints/             ← Auto-generated model checkpoints
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing (Required)
```bash
# Full preprocessing (Includes HRV, ~30 mins)
python scripts/01_build_metadata.py

# Without HRV (Faster, ~5 mins)
python scripts/01_build_metadata.py --no-hrv

# Quick test (100 samples)
python scripts/01_build_metadata.py --max-samples 100 --no-hrv
```

### 2. Training

**DIR5 best (recommended): `contrast_v2` MI head + IMI:50**
```bash
# 1) Build the IMI:50 dataset (separate dir; fast, signals read on-the-fly)
python scripts/01_build_metadata.py --config configs/experiments/dir5_p3a_imi50.yaml
# 2) Train
python scripts/02_train.py --config configs/experiments/dir5_p3a_imi50.yaml
# 3) Calibrate
python scripts/03_calibrate.py --dir checkpoints/run_*_dir5_p3a_imi50*
```

**E10 baseline (IMI:80, original primary)**
```bash
python scripts/02_train.py --config configs/experiments/arch_e10_decoupled.yaml

# Debug mode (200 samples, 3 epochs)
python scripts/02_train.py --config configs/experiments/arch_e10_decoupled.yaml --debug

# Calibrate after training
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_decoupled_multitask-aug
```

## Supported Labels (13-class — DIR4)

| Label | Task | Description |
|-------|------|-------------|
| NORM  | normal     | Normal ECG |
| AFIB  | arrhythmia | Atrial Fibrillation |
| STACH | arrhythmia | Sinus Tachycardia |
| PVC   | arrhythmia | Premature Ventricular Contraction |
| AFLT  | arrhythmia | Atrial Flutter |
| IMI   | mi | Inferior MI — II, III, aVF + reciprocal I, aVL |
| ASMI  | mi | Anteroseptal MI — V1–V4 |
| ILMI  | mi | Inferolateral MI — II, III, aVF + I, aVL + V5, V6 |
| AMI   | mi | Anterior MI — V1–V6 |
| LBBB  | conduction | Left Bundle Branch Block |
| RBBB  | conduction | Right Bundle Branch Block |
| IRBBB | conduction | Incomplete Right Bundle Branch Block |
| 1AVB  | conduction | First-degree AV Block |

## Model Architecture — E10 (Primary)

| Item | Value |
|------|-------|
| Config | `configs/experiments/arch_e10_decoupled.yaml` |
| `architecture` | `decoupled_multitask` |
| `routing_mode` | `decoupled` |
| Backbone | `DecoupledMultiTaskBackbone` |
| MI head | `DecoupledSubsetMIHead` |
| Checkpoint | `run_20260614_004857_decoupled_multitask-aug` |

> **For thesis diagrams:** use the diagram below. Code: `src/models/backbones.py` (`DecoupledMultiTaskBackbone`) + `src/models/ecg_multitask.py` (`DecoupledSubsetMIHead`).

```
Input (B, 12, 5000)
  │
  ├── [Arrhythmia CNN (12L)] ──→ Arrhy Expert TF ×2 ──→ Arrhy Head (5 labels)
  │
  ├── [Conduction CNN (12L)] ──→ Cond Expert TF ×2 ──→ Cond Head (4 labels)
  │                               + CDLGE raw [V1–V3 / V5,V6,I,aVL]
  │
  └── [MI — raw signal only]
        Subset LeadGroupEncoder + Group TF ×1 per label ──→ 4 MI heads
```

Three fully independent paths — no Shared CNN, no Shared Transformer, no `mi_expert`. MI gradient updates only subset anatomy encoders on raw signal.

## Training Pipeline

| Component | Setting |
|-----------|---------|
| Data | `processed_dir4` / `splits_dir4`, 13 labels, no MI merging |
| Split | `strat_fold` (fold 10 = test, fold 9 = val) |
| Normalization | Robust (Median/IQR) |
| Augmentation | Lead dropout |
| Loss | Standard BCE, uniform sampler |
| Epochs | 15, monitor `imi_auprc` |

## Post-Hoc Calibration

```bash
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_decoupled_multitask-aug
```

Temperature scaling + per-class threshold tuning on validation set.

## Evaluation Results (Test, Calibrated)

Progression from the original multi-branch baseline (ARCH-E06) through E10 to the
DIR5 best model. ARCH-E06 and E10 use IMI:80 (`processed_dir4`); the DIR5 best
uses IMI:50 (`processed_dir5_imi50`).

| Metric | ARCH-E06 (baseline) | E10 (decoupled) | **DIR5 best (contrast_v2 + IMI:50)** |
|--------|---------------------|-----------------|--------------------------------------|
| **MI Macro F1** | 0.452 | 0.495 | **0.511** |
| MI Macro AUPRC | — | 0.477 | **0.517** |
| IMI F1 | 0.444 | 0.457 | **0.590** |
| IMI AUPRC | 0.459 | 0.437 | **0.645** |
| ASMI F1 | 0.724 | 0.757 | 0.736 |
| ILMI F1 | 0.519 | **0.605** | 0.541 |
| AMI F1 | 0.121 | 0.162 | 0.176 |
| Arrhy Macro F1 | 0.770 | **0.777** | 0.720 |
| Cond Macro F1 | 0.711 | 0.720 | 0.704 |
| IMI test support | 103 | 103 | 175 |

Run folders:
- ARCH-E06: ablation `A` (see `EXPERIMENT_LOG.md`)
- E10: `checkpoints/run_20260614_004857_decoupled_multitask-aug`
- DIR5 best: `checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug`

> **Note on the IMI gain:** DIR5 lifts IMI via two stages — (1) the `contrast_v2`
> MI head reduces IMI/ILMI and ASMI/AMI overlap (raises IMI AUPRC at IMI:80), and
> (2) relaxing the IMI label threshold 80->50 adds borderline IMI positives
> (test support 103->175). The threshold change is a **label-definition change**,
> so IMI AUPRC is not directly comparable across thresholds (AUPRC is sensitive to
> prevalence). Report both thresholds as a label ablation. Full DIR5 phase
> breakdown (P0-P3a) is in `EXPERIMENT_LOG.md`.

Earlier MI ablation (IMI:80, MI Macro F1): E10 (0.495) > B (0.458) > A/E06 (0.452) > C (0.425) > D (0.404).

## Outputs (checkpoints/)
- `best_model.pth`: Best validation checkpoint (monitor metric).
- `epoch_XXX.pth`: Periodic checkpoints.
- `history.json`: Epoch-wise training and validation metrics.
- `test_metrics.json`: Final evaluation results.
- `calibration_results.json`: Temperature values and per-class thresholds.

## Cross-Dataset Validation

| Dataset | Setup | IMI AUROC | ASMI AUROC | IMI AUPRC |
|---------|-------|-----------|------------|-----------|
| Georgia | Zero-shot | 0.509 | — | — |
| Georgia | Fine-tune (15 epochs) | 0.923 | 0.924 | 0.650 |
| PTB     | Zero-shot | 0.501 | 0.628 | — |
| PTB     | Fine-tune (15 epochs) | 0.848 | 0.915 | 0.865 |

## Explainable AI (XAI)

```bash
python scripts/16_xai_explainer.py \
    --checkpoint checkpoints/run_20260614_004857_decoupled_multitask-aug \
    --split test \
    --max-per-class 5 \
    --method gxi \
    --out artifacts/xai_v2
```

Calibration temperatures and per-class thresholds are loaded from `calibration_results.json` if present.
