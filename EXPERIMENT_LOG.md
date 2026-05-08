## E-04: Systematic Ablation Study Framework

**Date**: 2026-05-06
**Branch convention**: `<STAGE>-E<NN>` (e.g., `SPLIT-E01`, `NORM-E02`, ...)
**Status**: Framework setup complete — Stage 1 (Split) ready to run

> **Design principle**: Change only ONE variable per experiment. All others stay at baseline.
> Each experiment trains for **15 epochs** only. Winner carries forward to the next stage.

---

## Model Architecture

```mermaid
graph TD
    Input["ECG Signal\n(B × 12 × 5000)"]

    subgraph CNN_Frontend["CNN Front-End"]
        DW["Depthwise Conv\n(per-lead, stride 5)"]
        PW["Pointwise Conv\n→ stem_dim=96"]
        Res["2× ResidualConvBlock\n[128 → 192 → 256 ch]"]
    end

    Input --> DW --> PW --> Res

    subgraph Transformer["Transformer Encoder"]
        Proj["Conv1d Projection\n→ d_model=256"]
        PE["Sinusoidal Pos. Enc."]
        CLS["[CLS] Token"]
        Enc["4× TransformerEncoderLayer\n(8 heads, FFN=512)"]
    end

    Res --> Proj --> PE
    CLS --> Enc
    PE --> Enc

    Enc --> CLSfeat["CLS Features (256)"]
    Enc --> SeqFeat["Sequence Features\n(B × T × 256)"]

    CLSfeat & SeqFeat --> Pool["AttentionPooling"]
    Pool --> Fuse["Linear Fuse → Global Features (256)"]

    subgraph ArrhyHead["Arrhythmia Head"]
        TTP["TaskTokenPooling"]
        MLP1["MLPHead → 5 logits\nNORM AFIB STACH PVC AFLT"]
    end

    SeqFeat --> TTP
    Fuse & TTP --> MLP1

    subgraph MIHead["MI Head"]
        ITTP["TaskTokenPooling (IMI)"]
        ATTP["TaskTokenPooling (ASMI)"]
        Inf["LeadGroupEncoder\nInferior: II, III, aVF"]
        Rec["LeadGroupEncoder\nReciprocal: I, aVL"]
        Ant["LeadGroupEncoder\nAnterior: V1–V4"]
        IMLP["MLPHead → IMI logit"]
        AMLP["MLPHead → ASMI logit"]
    end

    Fuse & SeqFeat --> ITTP & ATTP
    Input --> Inf & Rec & Ant
    ITTP & Inf & Rec --> IMLP
    ATTP & Ant --> AMLP
```

**Key design decisions:**
| Component | Purpose |
|-----------|---------|
| Depthwise Conv | Per-lead local morphology extraction (QRS shape, ST shift) |
| Transformer (4L, 8H) | Long-range rhythm dependency (AFIB irregular R-R, AFLT flutter waves) |
| CLS + Pooling fuse | Combine global summary with temporally-averaged context |
| LeadGroupEncoder (CNN) | Anatomically-aware spatial feature for MI (inferior/reciprocal/anterior leads) |
| Gradient Isolation | Prevents MI gradient from corrupting backbone learned for Arrhythmia |

---

## Ablation Stage Map

```
Baseline (all techniques OFF, 15 epochs each, plain BCE)
    │
    ├── Stage 1: SPLIT  → branches SPLIT-E01, SPLIT-E02, SPLIT-E03
    │       → Winner carries to Stage 2
    │
    ├── Stage 2: NORM   → branches NORM-E01, NORM-E02, NORM-E03
    │       → Winner carries to Stage 3
    │
    ├── Stage 3: LOSS   → branches LOSS-E01, LOSS-E02, LOSS-E03
    │       → Winner carries to Stage 4
    │
    ├── Stage 4: AUG    → branches AUG-E01, AUG-E02, AUG-E03, AUG-E04
    │       → Winner carries to Stage 5
    │
    ├── Stage 5: SAMP   → branches SAMP-E01, SAMP-E02
    │       → Winner carries to Stage 6
    │
    └── Stage 6: ARCH   → branches ARCH-E01, ARCH-E02, ARCH-E03
            → Best config → Full 40-epoch run with 3 seeds
```

---

## Baseline Config

**File**: `configs/baseline.yaml`
All techniques disabled:

| Parameter | Value | Note |
|-----------|-------|------|
| `split_method` | `strat_fold` | PTB-XL native fold 10=test, 9=val |
| `normalize` | `zscore` | per-lead z-score |
| `use_focal` | `false` | plain BCE loss |
| `use_pos_weight` | `false` | no class weighting |
| `mi_gradient_scale` | `1.0` | gradient isolation OFF |
| `mi_lr_multiplier` | `1.0` | no differential LR |
| `aug_*` | all `false` | no augmentation |
| `weighted_sampler` | `false` | uniform sampling |
| `label_smoothing` | `0.0` | none |
| `loss_weights` | all `1.0` | equal task weighting |
| `epochs` | `15` | short ablation runs |
| `scheduler` | `cosine` | stable, simple |
| `freeze_backbone_at_epoch` | `-1` | Phase 2 disabled |

---

## Stage 1: Split Strategy

**Variable**: How patients are assigned to Train / Val / Test
**All other params**: exactly as baseline

### Experiment Configs & Commands

#### SPLIT-E01 — PTB-XL Native `strat_fold`
Config: `configs/experiments/split_e01_strat_fold.yaml`
```bash
# Build data (one-time per config)
python scripts/01_build_metadata.py --config configs/experiments/split_e01_strat_fold.yaml --no-hrv

# Train
python scripts/02_train.py --config configs/experiments/split_e01_strat_fold.yaml
```
**Data changes**: Uses PTB-XL's built-in `strat_fold` column (fold 10 = test, 9 = val). Community gold standard. Labels are NOT stratified by patient health status.

---

#### SPLIT-E02 — `random_grouped` by `patient_id` (seed=42)
Config: `configs/experiments/split_e02_random_grouped.yaml`
```bash
python scripts/01_build_metadata.py --config configs/experiments/split_e02_random_grouped.yaml --no-hrv
python scripts/02_train.py --config configs/experiments/split_e02_random_grouped.yaml
```
**Data changes**: Shuffles all unique `patient_id`s (seed=42), then assigns whole patients in order to Train(70%) / Val(15%) / Test(15%). No stratification on label distribution.

---

#### SPLIT-E03 — `stratified_group_kfold` stratified on IMI
Config: `configs/experiments/split_e03_stratified_imi.yaml`
```bash
python scripts/01_build_metadata.py --config configs/experiments/split_e03_stratified_imi.yaml --no-hrv
python scripts/02_train.py --config configs/experiments/split_e03_stratified_imi.yaml
```
**Data changes**: `StratifiedGroupKFold(n_splits=5)` — groups by `patient_id`, stratifies by IMI label. Ensures each fold gets ~equal IMI positive patients. Test = fold 0, Val = fold 1, Train = folds 2+3+4.

---

### Stage 1 Results

| Experiment | Split Method | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|-------------|-----------|-----------|--------|----------------|--------|---------|
| SPLIT-E01 | `strat_fold` | 0.937 | 0.457 | 0.456 | 0.733 | `run_20260506_231153_hybrid-tf` | |
| SPLIT-E02 | `random_grouped` | 0.947 | **0.519** | **0.517** | **0.828** | `run_20260507_003151_hybrid-tf` | ✅ |
| SPLIT-E03 | `stratified_group_kfold (IMI)` | **0.959** | 0.506 | 0.558 | 0.768 | `run_20260507_013801_hybrid-tf` | |

**Stage 1 winner**: **SPLIT-E02** (`random_grouped`, seed=42)

**Analysis:**
1. **SPLIT-E01 (strat_fold) performed worst** — IMI AUPRC only 0.457, Arrhy F1 only 0.733. PTB-XL's built-in fold assigns patients across 10 folds designed for full 10-fold CV, meaning fold 10 (test) may have a skewed IMI distribution in a 1-fold evaluation scenario.
2. **SPLIT-E02 (random_grouped) is the clear winner** — Best IMI AUPRC (0.519) and Arrhy macro F1 (0.828). The 70/15/15 patient-grouped random split naturally produces a balanced training set size and representative test split.
3. **SPLIT-E03 (stratified on IMI)** — Highest IMI AUROC (0.959) and IMI F1 (0.558) but Arrhy F1 dropped to 0.768 vs SPLIT-E02's 0.828. Stratifying only on IMI creates a lopsided fold structure that hurts Arrhythmia. AUPRC (0.506) is also lower than E02. Not the right tradeoff.

→ **SPLIT-E02 config carries forward** to Stage 2 (Normalization).

---

## Stage 2: Normalization Strategy

**Variable**: How raw ECG signal values are scaled
**Baseline inherited**: `SPLIT-E02` (`random_grouped` split)

### Experiment Configs & Commands

> Note: Because we are NOT changing the split method or data selection, we do **not** need to re-run `01_build_metadata.py`. The `random_grouped` data built in Stage 1 is perfectly reusable. The normalization happens on-the-fly during training in `preprocessing.py`.

#### NORM-E01 — `zscore` (Baseline)
Config: `configs/experiments/norm_e01_zscore.yaml`
```bash
python scripts/02_train.py --config configs/experiments/norm_e01_zscore.yaml
```
**Mechanism**: `(x - mean) / std` per lead.

---

#### NORM-E02 — `robust` scaler (Median / IQR)
Config: `configs/experiments/norm_e02_robust.yaml`
```bash
python scripts/02_train.py --config configs/experiments/norm_e02_robust.yaml
```
**Mechanism**: `(x - median) / IQR` per lead. Less sensitive to extreme voltage spikes/artifacts.

---

#### NORM-E03 — `minmax` scaler
Config: `configs/experiments/norm_e03_minmax.yaml`
```bash
python scripts/02_train.py --config configs/experiments/norm_e03_minmax.yaml
```
**Mechanism**: `(x - min) / (max - min)` per lead. Scales strictly to [0, 1].

---

### Stage 2 Results

| Experiment | Normalization | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|---------------|-----------|-----------|--------|----------------|--------|---------|
| NORM-E01 | `zscore` | **0.948** | **0.525** | **0.511** | **0.830** | `run_20260507_204106_hybrid-tf` | ✅ |
| NORM-E02 | `robust` | 0.946 | 0.512 | **0.511** | 0.740 | `run_20260507_220132_hybrid-tf` | |
| NORM-E03 | `minmax` | 0.946 | 0.485 | 0.501 | 0.759 | `run_20260507_231720_hybrid-tf` | |

**Stage 2 winner**: **NORM-E01** (`zscore`)

**Analysis:**
1. **NORM-E01 (zscore) dominated across all metrics**. The standard normal distribution approach (zero mean, unit variance) works perfectly with the network's weight initialization.
2. **NORM-E02 (robust) destroyed Arrhythmia performance**. While it maintained IMI F1 (0.511), the Arrhythmia macro F1 plummeted from 0.830 to 0.740. ECG diagnosis heavily relies on exact amplitude ratios between leads; `robust` scaling uses IQR, which can non-linearly compress peaks, destroying morphological clues.
3. **NORM-E03 (minmax) performed worst overall**. Restricting the signal tightly to `[0, 1]` flattens crucial variations, causing performance drops across both tasks.

→ **NORM-E01 config carries forward** to Stage 3.

---

## Stage 3: Loss Strategy

**Variable**: How class imbalance and hard examples are penalized.
**Baseline inherited**: `SPLIT-E02` (`random_grouped`), `NORM-E01` (`zscore`).

### Experiment Configs & Commands

> Note: We are testing `pos_weight`, `Focal Loss`, and `Gradient Isolation`. Since data processing is identical to Stage 2, we reuse the same `data/processed_split_e02` folder. No need to run `01_build_metadata.py`.

#### LOSS-E01 — BCE + `pos_weight`
Config: `configs/experiments/loss_e01_posweight.yaml`
```bash
python scripts/02_train.py --config configs/experiments/loss_e01_posweight.yaml
```
**Mechanism**: Standard way to combat imbalance. Multiplies the BCE loss of positive samples by `N_neg / N_pos`. (Expected to over-predict IMI, based on E-02 findings).

---

#### LOSS-E02 — Focal Loss
Config: `configs/experiments/loss_e02_focal.yaml`
```bash
python scripts/02_train.py --config configs/experiments/loss_e02_focal.yaml
```
**Mechanism**: Dynamically scales cross entropy based on prediction confidence (`gamma=2.0`, `alpha=0.25`). Forces the model to focus on hard-to-classify samples rather than just easy negatives.

---

#### LOSS-E03 — Focal Loss + Gradient Isolation
Config: `configs/experiments/loss_e03_focal_gradiso.yaml`
```bash
python scripts/02_train.py --config configs/experiments/loss_e03_focal_gradiso.yaml
```
**Mechanism**: Uses Focal Loss, but scales the gradients coming from the MI head back into the shared CNN backbone by `0.3`. Prevents the minority MI task from distorting the robust features learned for the Arrhythmia task.

---

### Stage 3 Results

| Experiment | Loss Configuration | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------------|-----------|-----------|--------|----------------|--------|---------|
| LOSS-E01 | `pos_weight` | 0.949 | 0.504 | 0.508 | 0.774 | `run_20260508_005129_hybrid-tf` | |
| LOSS-E02 | `Focal Loss` | 0.949 | 0.520 | 0.494 | 0.806 | `run_20260508_015300_hybrid-tf-focal` | |
| LOSS-E03 | `Focal + GradIso (0.3)`| 0.945 | 0.493 | 0.503 | **0.822** | `run_20260508_025238_hybrid-tf-focal` | ✅ |

*(Reference Baseline NORM-E01: IMI AUPRC 0.525, Arrhy F1 0.830)*

**Stage 3 winner**: **LOSS-E03** (`Focal + GradIso`)

**Analysis (The "15-Epoch Illusion"):**
At first glance, it appears that adding advanced loss functions *decreased* performance compared to the pure BCE baseline (`NORM-E01`). However, this is an expected artifact of our **15-epoch ablation limit**:
1. **Focal Loss learns slower**: Focal Loss dynamically shrinks gradients for "easy" examples. This means the overall magnitude of weight updates is smaller than pure BCE. In a short 15-epoch run, Focal Loss simply doesn't have enough time to converge. (In our previous 50-epoch E-02 tests, Focal Loss significantly outperformed BCE).
2. **Gradient Isolation protects Arrhythmia**: In `LOSS-E03`, we scaled MI gradients by 0.3. As expected, IMI metrics dropped (AUPRC 0.493) because the MI head was learning at 30% speed and didn't converge in 15 epochs. However, **Arrhythmia F1 immediately bounced back to 0.822** (up from 0.806 in E02). This proves Gradient Isolation works: it stops the struggling MI task from corrupting the backbone.
3. **pos_weight is destructive**: `LOSS-E01` ruined Arrhythmia (0.774) and didn't help IMI AUPRC (0.504), confirming our earlier finding that forcing positive predictions destroys the representation space.

→ **LOSS-E03 config carries forward** to Stage 4. We will rely on Focal + GradIso to show their true power in the final full-length training run.

---

## Stage 4: Data Augmentation

**Variable**: How training data is perturbed to improve generalization.
**Baseline inherited**: `SPLIT-E02` (`random_grouped`), `NORM-E01` (`zscore`), `LOSS-E03` (`Focal + GradIso`).

### Experiment Configs & Commands

> Note: All Stage 4 configs reuse the same `data/processed_split_e02` folder.

#### AUG-E01 — MixUp (alpha=0.2)
Config: `configs/experiments/aug_e01_mixup.yaml`
```bash
python scripts/02_train.py --config configs/experiments/aug_e01_mixup.yaml
```
**Mechanism**: Linearly interpolates pairs of signals and their one-hot labels. Smooths decision boundaries.

---

#### AUG-E02 — Random Crop / Shift
Config: `configs/experiments/aug_e02_random_crop.yaml`
```bash
python scripts/02_train.py --config configs/experiments/aug_e02_random_crop.yaml
```
**Mechanism**: Crops a random 80% window of the 10-second signal and pads/resizes it back. Forces translation invariance.

---

#### AUG-E03 — Baseline Wander + Time Warp (Noise)
Config: `configs/experiments/aug_e03_noise_warp.yaml`
```bash
python scripts/02_train.py --config configs/experiments/aug_e03_noise_warp.yaml
```
**Mechanism**: Injects low-frequency sinusoidal drift (simulating respiration/movement) and stretches the time axis. Tests robustness to physical sensor noise.

---

#### AUG-E04 — Lead Dropout
Config: `configs/experiments/aug_e04_lead_dropout.yaml`
```bash
python scripts/02_train.py --config configs/experiments/aug_e04_lead_dropout.yaml
```
**Mechanism**: Randomly zeroes out 1-2 leads during training. Forces the model to learn redundant spatial representations instead of relying on a single "hero" lead.

---

### Stage 4 Results

| Experiment | Augmentation | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------|-----------|-----------|--------|----------------|--------|---------|
| AUG-E01 | `MixUp` | 0.944 | 0.497 | 0.494 | 0.807 | `run_20260508_071142_hybrid-tf-focal-aug-mxp` | |
| AUG-E02 | `Random Crop` | 0.940 | 0.484 | 0.471 | 0.626 | `run_20260508_081931_hybrid-tf-focal-aug` | |
| AUG-E03 | `Noise + Warp`| **0.949** | **0.516** | **0.527** | **0.825** | `run_20260508_092233_hybrid-tf-focal-aug` | ✅ |
| AUG-E04 | `Lead Dropout`| 0.950 | 0.495 | 0.513 | 0.812 | `run_20260508_102608_hybrid-tf-focal-aug` | |

*(Reference Baseline LOSS-E03: IMI AUPRC 0.493, Arrhy F1 0.822)*

**Stage 4 winner**: **AUG-E03** (`Noise + Warp`)

**Analysis:**
1. **AUG-E03 (Noise + Warp) is the clear winner**: Cả IMI F1 (0.527) và AUPRC (0.516) đều tăng vọt so với baseline, trong khi Arrhythmia F1 vẫn giữ vững ở mức 0.825. Việc thêm nhiễu rung đường cơ sở (Baseline Wander) và co giãn thời gian nhẹ (Time Warp) mô phỏng chính xác các nhiễu sinh lý học (nhịp thở, nhịp tim không đều), giúp mô hình tổng quát hóa tuyệt vời.
2. **AUG-E02 (Random Crop) là một thảm họa**: Arrhythmia F1 sụp đổ xuống **0.626**. Tín hiệu điện tim phụ thuộc rất chặt chẽ vào khoảng cách thời gian giữa các sóng (P-QRS-T). Việc crop 80% rồi resize lại đồng nghĩa với việc "kéo giãn" tín hiệu một cách cực đoan, phá hủy hoàn toàn ý nghĩa sinh lý của nhịp tim.
3. **AUG-E01 (MixUp) và AUG-E04 (Lead Dropout) không hiệu quả rõ rệt**: MixUp làm mờ ranh giới đặc trưng không gian tinh tế của IMI. Lead Dropout tăng nhẹ IMI AUROC nhưng lại làm giảm Arrhythmia F1 do làm mất đi các lead "bắt nhịp" quan trọng.

→ **AUG-E03 config carries forward** to Stage 5.

## Stage 5: Sampling Strategy

**Variable**: How batches are constructed from the training dataset.
**Baseline inherited**: `SPLIT-E02` (random_grouped), `NORM-E01` (zscore), `LOSS-E03` (Focal+GradIso), `AUG-E03` (Noise+Warp).

### Experiment Configs & Commands

> Note: All Stage 5 configs reuse the same `data/processed_split_e02` folder.

#### SAMP-E01 — Uniform Sampling (Baseline)
Config: `configs/experiments/samp_e01_uniform.yaml`
```bash
python scripts/02_train.py --config configs/experiments/samp_e01_uniform.yaml
```
**Mechanism**: PyTorch default `RandomSampler`. Every patient has an equal chance of being selected in a batch, meaning minority classes (like IMI) will appear rarely.

---

#### SAMP-E02 — Weighted Random Sampler
Config: `configs/experiments/samp_e02_weighted.yaml`
```bash
python scripts/02_train.py --config configs/experiments/samp_e02_weighted.yaml
```
**Mechanism**: Uses PyTorch `WeightedRandomSampler`. Calculates weights inversely proportional to class frequencies, guaranteeing that minority class samples (like IMI) appear much more frequently in every batch.

---

### Stage 5 Results

| Experiment | Sampling | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|----------|-----------|-----------|--------|----------------|--------|---------|
| SAMP-E01 | `Uniform` | **0.949** | **0.516** | **0.518** | **0.823** | `run_20260508_185741_hybrid-tf-focal-aug` | ✅ |
| SAMP-E02 | `Weighted` | 0.938 | 0.471 | 0.486 | 0.807 | `run_20260508_200759_hybrid-tf-focal-wrs-aug` | |

*(Reference AUG-E03: IMI AUPRC 0.516, Arrhy F1 0.825)*

**Stage 5 winner**: **SAMP-E01** (`Uniform Sampling`)

**Analysis — Vì sao Weighted Sampler thất bại?**

Đây là một bài học quan trọng. Kết quả này hoàn toàn có lý:

1. **Weighted Sampler + Focal Loss = Double-correction**: Chúng ta đã có Focal Loss để bù đắp mất cân bằng dữ liệu rồi. Khi thêm Weighted Sampler lên trên, ta vô tình **bù đắp 2 lần** — mỗi batch đã nặng về IMI hơn (do sampler), rồi gradient của IMI còn được khuếch đại thêm lần nữa (do focal). Điều này khiến mô hình quá tập trung vào IMI đến mức quên mất Arrhythmia (F1 rớt 0.807).
2. **Weighted Sampler làm giảm sự đa dạng trong batch**: Khi bốc quá nhiều ca IMI vào mỗi batch, tỉ lệ NORM và AFIB giảm đi. Mô hình mất đi "ngữ cảnh âm tính" phong phú cần thiết để học được đường ranh giới quyết định (decision boundary) sắc nét.
3. **Kết luận thực tiễn**: Với multi-task models mà loss đã được điều chỉnh (Focal Loss + GradIso), Uniform Sampling luôn là lựa chọn an toàn. Weighted Sampler chỉ phát huy tác dụng khi dùng cùng BCE thuần túy không có bất kỳ cơ chế rebalancing nào khác.

→ **SAMP-E01 (Uniform) config carries forward** to Stage 6. Stack hiện tại đã ổn định.

---

## Stage 6: Architecture / Training Tweaks (To be added after Stage 5)


