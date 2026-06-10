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
# Train DIR4 Hard-Routing Graph (recommended)
python scripts/02_train.py --config configs/experiments/arch_e07_hard_routing_graph.yaml

# Debug mode (200 samples, 3 epochs)
python scripts/02_train.py --config configs/experiments/arch_e07_hard_routing_graph.yaml --debug

# Override hyperparameters from CLI
python scripts/02_train.py --config configs/experiments/arch_e07_hard_routing_graph.yaml --epochs 30 --batch-size 64 --lr 1e-4

# Resume from checkpoint
python scripts/02_train.py --config configs/experiments/arch_e07_hard_routing_graph.yaml --resume checkpoints/run_xxx/epoch_010.pth
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

## Model Architecture (DIR4 — Hard-Routing Graph)

```
Input (B, 12, 5000)   ← 12-lead ECG, 10 seconds @ 500 Hz
  │
  ├─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  [Shared CNN Front-End]        (Arrhythmia Backbone)        │  [Raw Signal Copy]  (Hard-Routed — No Gradient Leakage)
  │    DepthwiseConv1d (12→12, stride 5)                        │
  │    PointwiseConv1d (12→stem_dim=96)                         │
  │    ResidualConvBlock (96→128)                               │
  │    ResidualConvBlock (128→192)                              │
  │    ResidualConvBlock (192→256)                              │
  │    Conv1d Projection (256→d_model=256)                      │
  │    Sequence Tokens (B, T, 256)                              │
  │                                                             │
  │  [Transformer Encoder]                                      │
  │    Sinusoidal Positional Encoding                           │
  │    [CLS] Token prepended                                    │
  │    4× TransformerEncoderLayer (8 heads, FFN=512, Pre-LN)    │
  │    → global_features (B, 256)                               │
  │    → sequence_features (B, T, 256)                          │
  │                                                             │
  ▼                                                             │
  [Arrhythmia Branch]                                           │
    TaskTokenPooling(sequence_features) → (B, 256)             │
    Concat [global + task-pooled]       → (B, 512)             │
    MLPHead (512 → 128 → 5)                                     │
    Logits: NORM, AFIB, STACH, PVC, AFLT                        │
                                                             ▼  ▼
                                      ┌──────────────────────────────────────┐
                                      │     PerLeadEncoder (shared weights)  │
                                      │  for each of 12 leads independently: │
                                      │   Conv1d(1→32, k=15, s=4)            │
                                      │   Conv1d(32→64, k=9, s=2)            │
                                      │   Conv1d(64→dim, k=7, s=2)           │
                                      │   AvgPool + MaxPool → Concat → Linear│
                                      │   → 12 Node vectors (B, 12, dim)     │
                                      └───────────────┬──────────────────────┘
                                                      │ Learnable Lead Embeddings
                                          ┌───────────┴───────────┐
                                          │                       │
                                    [MI Branch]           [Conduction Branch]
                                  GraphTransformer        GraphTransformer
                                  (dim=128, 4 heads,      (dim=128, 4 heads,
                                   2 layers, Pre-LN)       2 layers, Pre-LN)
                                          │                       │
                          ┌───────────────┤           ┌───────────┴──────────────┐
                          │ Node Selection │           │     Global Node Pool     │
                          │               │           │  TaskTokenPooling → (B,128)
                   Inferior [II,III,aVF]  │           │  MLPHead (128→128→4)     │
                   Reciprocal [I,aVL]     │           │  Logits:                 │
                   Anterior [V1-V4]       │           │    LBBB, RBBB,           │
                   Lateral [V5,V6]        │           │    IRBBB, 1AVB           │
                   Anterior6 [V1-V6]      │           └──────────────────────────┘
                          │               │
                   ┌──────┴──────┐
                   │Per-label    │
                   │Node Pooling │
                   │(TaskToken)  │
                   └──────┬──────┘
                          │
              ┌───────────┼───────────┬────────────┐
              │           │           │            │
         IMI Head    ASMI Head   ILMI Head    AMI Head
      (inf+rec)→(B,1) (ant)→(B,1) (inf+rec+lat)→(B,1) (ant6)→(B,1)
              │           │           │            │
              └───────────┴───────────┴────────────┘
                       Logits (B, 4):
                    IMI, ASMI, ILMI, AMI
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

After training, **Temperature Scaling** and **Per-class Threshold Tuning** are applied:

```bash
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_HHMMSS
```

> T > 1.0 across all seeds confirms systematic overconfidence. Temperature scaling softens probability peaks without affecting ranking or AUROC.

## Outputs (checkpoints/)
- `best_model.pth`: Best validation AUROC checkpoint.
- `epoch_XXX.pth`: Periodic checkpoints.
- `history.json`: Epoch-wise training and validation metrics.
- `test_metrics.json`: Final evaluation results.
- `calibration_results.json`: Temperature values and per-class thresholds.

## Evaluation Results (DIR4 — pending)

> Results for the DIR4 Hard-Routing Graph architecture will be updated after training is complete.
> Previous baseline results (ARCH-5, 7-class):
>
> | Metric | Calibrated |
> |--------|------------|
> | Arrhy Macro F1 | 0.7584 ± 0.0112 |
> | MI Macro F1 | 0.6176 ± 0.0073 |
> | Arrhy Macro AUROC | 0.9462 ± 0.0040 |
> | MI Macro AUROC | 0.9547 ± 0.0007 |
> | IMI F1 | 0.4937 ± 0.0086 |
> | ASMI F1 | 0.7416 ± 0.0170 |

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
- **MI labels** (IMI, ILMI): INF (inferior), REC (reciprocal)
- **MI labels** (ASMI, AMI): ANT (anterior), ANT6 (V1–V6)
- **Conduction labels**: V1 (RBBB), I/aVL/V5/V6 (LBBB), global (1AVB/IRBBB)
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