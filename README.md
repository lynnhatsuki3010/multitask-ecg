# ECG Multi-Task Transformer (DIR4 / DIR5)

A deep learning project for 12-lead ECG analysis on PTB-XL using a **Fully Decoupled Multitask** architecture (E10): three independent paths from raw signal — no Shared CNN.

> **Current best (balanced multitask):** `branche10_05_cond_depth` on top of DIR5
> (`contrast_v2` MI head + IMI label threshold 50).
> - Config: `configs/experiments/branche10_05_cond_depth.yaml`
> - Result folder: `checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug`
> - Dataset: `data/processed_dir5_imi50_500hz` / `data/splits_dir5_imi50_500hz` (IMI threshold 50, separate from the IMI:80 `processed_dir4`)
> - Calibrated headline: **MI Macro F1 0.524**, **MI Macro AUPRC 0.536**, **IMI AUPRC 0.671**, **Arrhy Macro F1 0.777**, **Cond Macro AUPRC 0.757**.
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

**Current best (recommended): B10-05 `cond_depth` on DIR5 (`contrast_v2` + IMI:50)**
```bash
# 1) Build the IMI:50 dataset (separate dir; fast, signals read on-the-fly)
python scripts/01_build_metadata.py --config configs/experiments/dir5_p3a_imi50.yaml
# 2) Train
python scripts/02_train.py --config configs/experiments/branche10_05_cond_depth.yaml
# 3) Calibrate
python scripts/03_calibrate.py --dir checkpoints/run_*_branche10_05_cond_depth*
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

Progression from the original multi-branch baseline (ARCH-E06) through E10,
DIR5, and the B10 track sweep. ARCH-E06 and E10 use IMI:80 (`processed_dir4`);
DIR5/B10 use IMI:50 (`processed_dir5_imi50`).

| Metric | ARCH-E06 | E10 | DIR5 best (`p3a`) | **Current best (`B10-05 cond_depth`)** |
|--------|---------:|----:|------------------:|---------------------------------------:|
| **MI Macro F1** | 0.452 | 0.495 | 0.511 | **0.524** |
| MI Macro AUPRC | — | 0.477 | 0.517 | **0.536** |
| IMI F1 | 0.444 | 0.457 | **0.590** | 0.574 |
| IMI AUPRC | 0.459 | 0.437 | 0.645 | **0.671** |
| ASMI F1 | 0.724 | 0.757 | 0.746 | 0.732 |
| ILMI F1 | 0.519 | **0.605** | 0.476 | 0.506 |
| AMI F1 | 0.121 | 0.162 | 0.176 | **0.286** |
| Arrhy Macro F1 | 0.770 | **0.777** | 0.728 | **0.777** |
| Arrhy Macro AUPRC | — | — | 0.768 | **0.787** |
| Cond Macro F1 | 0.711 | 0.720 | **0.708** | 0.697 |
| Cond Macro AUPRC | — | — | 0.725 | **0.757** |
| IMI test support | 103 | 103 | 175 | 175 |

Run folders:
- ARCH-E06: ablation `A` (see `EXPERIMENT_LOG.md`)
- E10: `checkpoints/run_20260614_004857_decoupled_multitask-aug`
- DIR5 best: `checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug`
- Current best: `checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug`

> **Note on the IMI gain:** DIR5 lifts IMI via two stages — (1) the `contrast_v2`
> MI head reduces IMI/ILMI and ASMI/AMI overlap (raises IMI AUPRC at IMI:80), and
> (2) relaxing the IMI label threshold 80->50 adds borderline IMI positives
> (test support 103->175). The threshold change is a **label-definition change**,
> so IMI AUPRC is not directly comparable across thresholds (AUPRC is sensitive to
> prevalence). Report both thresholds as a label ablation. Full DIR5 phase
> breakdown (P0-P3a) is in `EXPERIMENT_LOG.md`.

> **Why B10-05 is current best:** It preserves the DIR5 MI setup while improving
> overall balance: best MI macro F1/AUPRC, best IMI AUPRC, best conduction AUPRC,
> and arrhythmia F1 close to the strongest arrhythmia run. DIR5 P3a remains the
> clean MI-threshold ablation; B10-05 is the recommended multitask checkpoint.

Earlier MI ablation (IMI:80, MI Macro F1): E10 (0.495) > B (0.458) > A/E06 (0.452) > C (0.425) > D (0.404).

## Outputs (checkpoints/)
- `best_model.pth`: Best validation checkpoint (monitor metric).
- `epoch_XXX.pth`: Periodic checkpoints.
- `history.json`: Epoch-wise training and validation metrics.
- `test_metrics.json`: Final evaluation results.
- `calibration_results.json`: Temperature values and per-class thresholds.

## Cross-Dataset Validation (DIR5 best, decoupled)

External validation of the DIR5 best model
(`run_20260618_233545_dir5_p3a_imi50`) on two independent datasets:
**PTB Diagnostic ECG** (direct IMI/ASMI labels) and **Georgia / PhysioNet 2020**
(arrhythmia/conduction labels are real; MI labels are proxy-mapped from SNOMED,
hence noisy).
All external signals are re-preprocessed with the **same chain as training**
(bandpass 0.5–40 Hz + 50 Hz notch + robust normalize). Reproduce with:

```bash
# Zero-shot (external test, whole dataset)
python scripts/16_eval_zeroshot_decoupled.py --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth --dataset ptb     --split all
python scripts/16_eval_zeroshot_decoupled.py --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth --dataset georgia --split all

# Fine-tune MI head only, report on held-out test split
python scripts/17_finetune_decoupled.py --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth --dataset ptb     --epochs 30 --lr 1e-3
python scripts/17_finetune_decoupled.py --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth --dataset georgia --epochs 30 --lr 1e-3

# Fine-tune Georgia non-MI heads (arrhythmia + conduction, after rebuilding Georgia labels)
python scripts/08_build_georgia.py
python scripts/10_split_georgia.py
python scripts/18_finetune_georgia_non_mi.py --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth --epochs 30 --lr 1e-4
```

### 1) Zero-shot (no adaptation, evaluated on the entire external dataset)

**MI head** — threshold-free metrics (AUROC/AUPRC) are the fair comparison:

| Dataset | Label | AUROC | AUPRC | F1@0.5 | Support |
|---------|-------|------:|------:|-------:|--------:|
| PTB     | ASMI  | **0.858** | 0.843 | 0.616 | 190 |
| PTB     | IMI   | 0.499 | 0.452 | 0.000 | 192 |
| Georgia | IMI   | 0.517 | 0.096 | 0.011 | 451 |
| Georgia | ASMI  | 0.482 | 0.059 | 0.061 | 281 |

**Arrhythmia head** — Georgia has real arrhythmia labels (PTB only has NORM):

| Dataset | Label | AUROC | AUPRC | F1@0.5 | Support |
|---------|-------|------:|------:|-------:|--------:|
| Georgia | STACH | **0.989** | 0.977 | 0.911 | 1261 |
| Georgia | NORM  | **0.955** | 0.940 | 0.875 | 1752 |
| Georgia | AFIB  | **0.904** | 0.761 | 0.755 | 570 |
| Georgia | AFLT  | 0.808 | 0.265 | 0.163 | 186 |
| Georgia | PVC   | 0.770 | 0.520 | 0.533 | 395 |
| PTB     | NORM  | 0.865 | 0.538 | 0.595 | 80 |

### 2) Fine-tune (MI head only, ~8% of params; evaluated on held-out test split)

Everything except the MI head is frozen; only IMI/ASMI columns are supervised
(ILMI/AMI untouched). Arrow shows **zero-shot → fine-tuned** on the test split.

| Dataset | Label | AUROC (zs→ft) | AUPRC (zs→ft) | F1 (zs→ft) | Support |
|---------|-------|--------------:|--------------:|-----------:|--------:|
| PTB     | IMI   | 0.520 → **0.868** | 0.495 → **0.887** | 0.000 → **0.764** | 27 |
| PTB     | ASMI  | 0.839 → **0.928** | 0.860 → **0.922** | 0.474 → **0.816** | 28 |
| Georgia | IMI   | 0.484 → **0.917** | 0.098 → **0.580** | 0.026 → **0.553** | 68 |
| Georgia | ASMI  | 0.492 → **0.950** | 0.067 → **0.729** | 0.084 → **0.598** | 42 |

### 3) Fine-tune Georgia non-MI heads (arrhythmia + conduction)

Georgia was rebuilt with conduction SNOMED mappings (`LBBB/RBBB/IRBBB/1AVB`)
and re-split with multilabel stratification. Everything except the task-specific
non-MI heads is frozen; arrow shows **zero-shot → fine-tuned** on the held-out
Georgia test split.

**Macro summary:**

| Task | AUROC (zs→ft) | AUPRC (zs→ft) | F1@0.5 (zs→ft) |
|------|--------------:|--------------:|---------------:|
| Arrhythmia | 0.899 → **0.938** | 0.685 → **0.737** | 0.646 → 0.604 |
| Conduction | 0.931 → **0.943** | 0.638 → **0.680** | 0.591 → **0.598** |

**Per-label result:**

| Task | Label | AUROC (zs→ft) | AUPRC (zs→ft) | F1@0.5 (zs→ft) | Support |
|------|-------|--------------:|--------------:|----------------:|--------:|
| Arrhy | NORM  | 0.952 → **0.969** | 0.909 → **0.927** | 0.848 → **0.862** | 263 |
| Arrhy | AFIB  | 0.923 → **0.946** | 0.728 → **0.761** | 0.740 → 0.619 | 86 |
| Arrhy | STACH | 0.993 → **0.995** | 0.986 → **0.988** | 0.942 → 0.900 | 189 |
| Arrhy | PVC   | 0.762 → **0.838** | 0.430 → **0.484** | 0.466 → 0.398 | 59 |
| Arrhy | AFLT  | 0.865 → **0.942** | 0.371 → **0.523** | 0.235 → **0.239** | 28 |
| Cond  | LBBB  | 0.960 → **0.972** | 0.670 → **0.728** | 0.554 → **0.607** | 62 |
| Cond  | RBBB  | **0.936** → 0.930 | **0.772** → 0.735 | **0.734** → 0.690 | 83 |
| Cond  | IRBBB | 0.896 → **0.921** | 0.463 → **0.504** | **0.480** → 0.444 | 61 |
| Cond  | 1AVB  | 0.931 → **0.952** | 0.646 → **0.752** | 0.598 → **0.650** | 115 |

> Non-MI fine-tuning mostly improves ranking/calibration quality (AUROC/AUPRC).
> Some arrhythmia F1@0.5 drops after adaptation, so threshold tuning/calibration
> should be applied before treating F1 as final. The strongest gains are AFLT
> AUPRC (0.371→0.523), PVC AUROC (0.762→0.838), LBBB AUPRC (0.670→0.728), and
> 1AVB AUPRC (0.646→0.752).

> **Reading the results.** The **arrhythmia branch generalizes zero-shot**
> (Georgia AUROC 0.77–0.99) — strong evidence of robustness across recording
> sites. For **MI**, ASMI transfers when the target labels are clean (PTB ASMI
> AUROC 0.86) but **IMI does not transfer zero-shot** (AUROC ≈ 0.50) and Georgia
> MI is near-chance because its MI labels are proxy-mapped and rare. A light
> **head-only fine-tune recovers MI dramatically on both datasets** (IMI AUROC
> 0.52→0.87 on PTB, 0.48→0.92 on Georgia), showing the representation already
> carries MI-relevant features and mainly needs domain re-calibration (gain /
> acquisition differences). Zero-shot is reported as the primary generalization
> evidence; fine-tune as the adaptation ceiling. Results saved as
> `zeroshot_<ds>_all.json` (source checkpoint dir) and
> `finetune_<ds>_decoupled/finetune_result.json` /
> `finetune_georgia_non_mi_decoupled/finetune_result.json`.

> *Note:* zero-shot numbers in table (2) are on the small per-dataset **test
> split** (used as the fine-tune baseline), while table (1) reports zero-shot on
> the **entire** external dataset — hence minor differences in the same metric.

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
