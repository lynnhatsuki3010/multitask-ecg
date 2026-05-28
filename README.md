# ECG Multi-Task Transformer

A deep learning project for 12-lead ECG analysis on the PTB-XL dataset utilizing a Hybrid Transformer architecture and multi-task learning.

## Directory Structure
```
ECG_HRV/
├── data/
│   ├── raw/PTB-XL/          ← Raw dataset
│   ├── processed/           ← Preprocessed metadata and matrices
│   └── splits/              ← Train/val/test split indices
├── src/
│   ├── data/                ← Data loading, preprocessing, and label extraction
│   ├── models/              ← Hybrid Transformer & Multi-task heads
│   ├── training/            ← Custom training loop & loss functions
│   └── utils/               ← Evaluation metrics
├── configs/                 
│   └── experiments/         ← Configuration files (Ablation, X-Val, Final)
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
```bash
# Train using a specific configuration
python scripts/02_train.py --config configs/experiments/final/final_e01_seed42.yaml

# Debug mode (200 samples, 3 epochs)
python scripts/02_train.py --config configs/experiments/final/final_e01_seed42.yaml --debug

# Override hyperparameters from CLI
python scripts/02_train.py --config configs/experiments/final/final_e01_seed42.yaml --epochs 30 --batch-size 32 --lr 5e-5

# Resume from checkpoint
python scripts/02_train.py --config configs/experiments/final/final_e01_seed42.yaml --resume checkpoints/run_xxx/epoch_010.pth
```

## Supported Labels

| Label | Task | Description |
|-------|------|-------------|
| NORM  | normal | Normal ECG |
| AFIB  | arrhythmia | Atrial Fibrillation |
| STACH | arrhythmia | Sinus Tachycardia |
| AFLT  | arrhythmia | Atrial Flutter |
| PVC   | arrhythmia | Premature Ventricular Contraction |
| IMI   | mi | Inferior Myocardial Infarction |
| ASMI  | mi | Anteroseptal Myocardial Infarction |

## Model Architecture (ARCH-5)

```
Input (B, 12, 5000)   ← 12-lead ECG, 10 seconds @ 500 Hz
  │
  ├── [Shared CNN Front-End]
  │     → DepthwiseConv1d  (12 → 12, per-lead, stride 5)   [ADD] per-lead morphology extraction
  │     → PointwiseConv1d  (12 → stem_dim=96)               [ADD] cross-lead feature mixing
  │     → ResidualConvBlock (96  → 128)
  │     → ResidualConvBlock (128 → 192)
  │     → ResidualConvBlock (192 → 256)
  │     → Conv1d Projection (256 → d_model=256)
  │     → Sequence tokens (B, T, 256)
  │
  ├── [Transformer Encoder]
  │     → Sinusoidal Positional Encoding                     [ADD]
  │     → [CLS] Token prepended                              [ADD]
  │     → 4× TransformerEncoderLayer (8 heads, FFN dim=512, Pre-LN)
  │     → CLS Features   (B, 256)                           [ADD]
  │     → Sequence Features (B, T, 256)
  │     → AttentionPooling [CLS + Seq] → Linear Fuse → global_features (B, 256)  [ADD] multi-scale fusion
  │
  ├── [Arrhythmia Branch]
  │     → TaskTokenPooling over sequence → (B, 256)
  │     → Concat [global_features + task-pooled]  → (B, 512)
  │     → MLPHead (512 → 128 → 5)
  │     → Logits: NORM, AFIB, STACH, PVC, AFLT
  │
  └── [MI Branch]  ← anatomy-aware, gradient-isolated       [ADD] gradient isolation
        │
        ├── Shared context  (B, 256)  [from global_features]
        ├── TaskTokenPooling (IMI)  → (B, 256)
        ├── TaskTokenPooling (ASMI) → (B, 256)
        │
        ├── LeadGroupEncoder — Inferior   [II, III, aVF]  → (B, 128)
        ├── LeadGroupEncoder — Reciprocal [I, aVL]        → (B, 64)
        └── LeadGroupEncoder — Anterior   [V1–V4]         → (B, 128)
              │
              ├── IMI Head:  Concat [shared + task_imi + inferior + reciprocal] → MLPHead → (B, 1)
              └── ASMI Head: Concat [shared + task_asmi + anterior]             → MLPHead → (B, 1)
```

## Training Pipeline (Golden Stack)

The final pipeline was determined through systematic ablation across 5 stages:

| Stage | Component | Winner | Key Finding |
|-------|-----------|--------|-------------|
| Split | Strategy | `strat_fold` (native PTB-XL folds: 10=Test, 9=Val) | Rigorous out-of-distribution evaluation |
| Norm | Normalization | **Robust (Median/IQR)** | +3.47% abs. MI F1 vs Z-Score |
| Aug | Augmentation | **Lead Dropout (1-2 leads)** | +7.83% abs. IMI AUPRC; forces redundant spatial learning |
| Samp | Batch Sampling | **Uniform Sampler** | WRS collapsed Arrhy F1 by -4.5% |
| Loss | Loss Function | **Standard BCE** | Advanced class-balancing losses destabilized optimization |

## Post-Hoc Calibration

After 50-epoch training across 3 seeds, **Temperature Scaling** and **Per-class Threshold Tuning** are applied:

```bash
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_HHMMSS
```

| Seed | Arrhy T | MI T |
|------|---------|------|
| 42   | 1.4155  | 1.3042 |
| 123  | 1.2993  | 1.3986 |
| 2024 | 1.5000  | 2.1475 |
| **Mean** | **1.405** | **1.617** |

> T > 1.0 across all seeds confirms systematic overconfidence. Temperature scaling softens probability peaks without affecting ranking or AUROC.

## Outputs (checkpoints/)
- `best_model.pth`: Best validation AUROC checkpoint.
- `epoch_XXX.pth`: Periodic checkpoints.
- `history.json`: Epoch-wise training and validation metrics.
- `test_metrics.json`: Final evaluation results.
- `calibration_results.json`: Temperature values and per-class thresholds.

## Evaluation Results

Final calibrated performance averaged across 3 seeds (42, 123, 2024):

| Metric | Baseline (τ=0.5) | Calibrated | Δ |
|--------|-----------------|------------|---|
| Arrhy Macro F1 | 0.7670 ± 0.0201 | 0.7584 ± 0.0112 | ↓ 44% variance |
| MI Macro F1 | 0.5849 ± 0.0045 | 0.6176 ± 0.0073 | +3.27% abs. |
| Arrhy Macro AUROC | — | 0.9462 ± 0.0040 | — |
| MI Macro AUROC | — | 0.9547 ± 0.0007 | — |
| MI Macro AUPRC | — | 0.6353 ± 0.0125 | — |
| IMI AUPRC | — | 0.4567 ± 0.0301 | — |
| IMI F1 | — | 0.4937 ± 0.0086 | — |
| ASMI F1 | — | 0.7416 ± 0.0170 | — |

Per-class Arrhythmia F1 (calibrated, mean across seeds):

| Class | Seed 42 | Seed 123 | Seed 2024 | Mean |
|-------|---------|----------|-----------|------|
| NORM  | 0.8511  | 0.8557   | 0.8418    | **0.8495** |
| AFIB  | 0.8462  | 0.8730   | 0.8535    | **0.8576** |
| STACH | 0.8655  | 0.8409   | 0.8471    | **0.8512** |
| PVC   | 0.8494  | 0.8300   | 0.8340    | **0.8378** |
| AFLT  | 0.3333  | 0.4545   | 0.4000    | **0.3959** |

## Cross-Dataset Validation

| Dataset | Setup | IMI AUROC | ASMI AUROC | IMI AUPRC |
|---------|-------|-----------|------------|-----------|
| Georgia | Zero-shot | 0.509 | — | — |
| Georgia | Fine-tune (15 epochs) | 0.923 | 0.924 | 0.650 |
| PTB     | Zero-shot | 0.501 | 0.628 | — |
| PTB     | Fine-tune (15 epochs) | 0.848 | 0.915 | 0.865 |

> Zero-shot AUROC ~0.50 confirms no domain shortcuts. Fine-tuning only the MI head for 15 epochs immediately recovers 0.84–0.92 AUROC, validating the backbone as a robust universal ECG feature extractor.

## Explainable AI (XAI)

The XAI module (v2) uses **Gradient × Input (GxI)** saliency instead of attention-weight visualization. Unlike attention maps — which apply a single identical overlay to all 12 leads — GxI computes `saliency[lead, t] = |∂L/∂input[lead, t] × input[lead, t]|` directly on the raw (12, T) input, giving each lead its own independent saliency signal.

An optional **Integrated Gradients (IG)** mode is also available for smoother, publication-quality figures.

Clinical lead group annotations are shown per lead:
- **MI labels** (IMI, ASMI): INF (inferior), REC (reciprocal), ANT (anterior)
- **Arrhythmia labels**: RHY (rhythm), LAT (lateral), SEP (septal)

```bash
# GxI saliency (fast, default)
python scripts/16_xai_explainer.py \
    --checkpoint checkpoints/run_YYYYMMDD_HHMMSS \
    --split test \
    --max-per-class 5 \
    --max-total 100 \
    --method gxi \
    --out artifacts/xai_v2

# Integrated Gradients (smoother, slower)
python scripts/16_xai_explainer.py \
    --checkpoint checkpoints/run_YYYYMMDD_HHMMSS \
    --split test \
    --max-per-class 5 \
    --method ig \
    --ig-steps 20
```

Heatmaps are saved to `artifacts/xai_v2/` with the following structure:

```
artifacts/xai_v2/
  TP/
    <LABEL>/          ← True Positives: single-label heatmaps
    <LABEL+LABEL>/    ← TP combos: averaged saliency across co-predicted labels
  FP/
    <LABEL>/          ← False Positives: single-label heatmaps
    <LABEL+LABEL>/    ← FP combos: averaged saliency across all predicted labels
```

A summary table of TP / FP / FN counts per class is printed at the end of each run. Calibration temperatures and per-class thresholds are automatically loaded from `calibration_results.json` if present in the checkpoint directory.