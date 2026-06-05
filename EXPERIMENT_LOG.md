# Experiment Log: ARCH-E06 Multi-Branch Transformer

A deep learning project for 12-lead ECG analysis utilizing a Multi-Branch Transformer architecture with heterogeneous expert blocks and anatomy-aware lead group encoders.

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
| ILMI  | mi | Inferolateral Myocardial Infarction |
| AMI   | mi | Anterior Myocardial Infarction |
| LBBB  | conduction | Left Bundle Branch Block (Grouped: CLBBB + LAFB + ILBBB) |
| RBBB  | conduction | Complete Right Bundle Branch Block (CRBBB) |
| IRBBB | conduction | Incomplete Right Bundle Branch Block |
| 1AVB  | conduction | First-degree Atrioventricular Block |

## Model Architecture (ARCH-E06 Multi-Branch)

```text
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
  ├── [Shared Transformer Encoder]
  │     → Sinusoidal Positional Encoding
  │     → 2× TransformerEncoderLayer (8 heads, FFN dim=512, Pre-LN)  [MODIFIED: Increased to 2 layers]
  │     → Shared Sequence (B, T, 256)
  │
  ├── [Arrhythmia Expert Branch]
  │     → 2× TransformerEncoderLayer (8 heads, FFN dim=512)
  │     → AttentionPooling [CLS + Seq] → arrhy_global
  │     → TaskTokenPooling over sequence → arrhy_tokens
  │     → Concat [arrhy_global + arrhy_tokens]
  │     → MLPHead (512 → 128 → 5)
  │     → Logits: NORM, AFIB, STACH, PVC, AFLT
  │
  ├── [MI Expert Branch]  ← anatomy-aware
  │     │
  │     ├── 2× TransformerEncoderLayer (8 heads, FFN dim=512)
  │     ├── AttentionPooling [CLS + Seq] → mi_global
  │     ├── TaskTokenPooling (IMI)
  │     ├── TaskTokenPooling (ASMI)
  │     │
  │     ├── LeadGroupEncoder — Inferior   [II, III, aVF]  → (B, 128)
  │     ├── LeadGroupEncoder — Reciprocal [I, aVL]        → (B, 64)
  │     └── LeadGroupEncoder — Anterior   [V1–V4]         → (B, 128)
  │           │
  │           ├── IMI Head:  Concat [mi_global + task_imi + inferior + reciprocal] → MLPHead → (B, 1)
  │           └── ASMI Head: Concat [mi_global + task_asmi + anterior]             → MLPHead → (B, 1)
  │
  └── [Conduction Expert Branch] ← NEW: anatomy-aware for bundle blocks
        │
        ├── 2× TransformerEncoderLayer (8 heads, FFN dim=512)
        ├── AttentionPooling [CLS + Seq] → cond_global
        ├── TaskTokenPooling over sequence → cond_tokens
        │
        ├── ConductionLeadGroupEncoder — Right Precordial [V1, V2, V3]     → (B, 128)  [RBBB focus]
        └── ConductionLeadGroupEncoder — Left Precordial  [I, aVL, V5, V6] → (B, 128)  [LBBB focus]
              │
              └── Conduction Head: Concat [cond_global + cond_tokens + right_feats + left_feats] 
              └── MLPHead → (B, 4)
              └── Logits: LBBB, RBBB, IRBBB, 1AVB
```

## Post-Hoc Calibration Results

After 15 epochs, **Temperature Scaling** and **Per-class Threshold Tuning** were applied using `03_calibrate.py` to correct systemic overconfidence.

| Branch | Temperature (T) | NLL Improvement (Val) |
|--------|-----------------|------------------------|
| Arrhythmia | 1.4691 | 0.1102 → 0.0980 |
| MI | 1.4747 | 0.0898 → 0.0819 |
| Conduction | 1.3509 | 0.0801 → 0.0752 |

> $T > 1.0$ across all branches confirms overconfidence. Temperature scaling successfully softened probability peaks without altering AUROC or ranking.

### Optimal Thresholds ($\tau$) tuned on Validation

| Label | Task | Optimal Threshold ($\tau$) |
|-------|------|--------------------------|
| NORM  | Arrhythmia | 0.500 |
| AFIB  | Arrhythmia | 0.450 |
| STACH | Arrhythmia | 0.650 |
| PVC   | Arrhythmia | 0.550 |
| AFLT  | Arrhythmia | 0.350 |
| IMI   | MI | 0.400 |
| ASMI  | MI | 0.550 |
| ILMI  | MI | 0.400 |
| AMI   | MI | 0.350 |
| LBBB  | Conduction | **0.300** |
| RBBB  | Conduction | 0.450 |
| IRBBB | Conduction | 0.500 |
| 1AVB  | Conduction | 0.450 |

---

## Evaluation Results (Test Set)

Comparison of baseline metrics ($\tau=0.5, T=1.0$) vs. Calibrated metrics:

| Metric | Baseline (τ=0.5) | Calibrated | Δ |
|--------|-----------------|------------|---|
| **Arrhythmia Macro F1** | 0.8182 | 0.7985 | ↓ 0.0197 |
| **MI Macro F1** | 0.4269 | **0.4704** | **↑ 0.0435** |
| **Conduction Macro F1** | 0.7257 | 0.7243 | ↓ 0.0014 |
| **IMI AUPRC** (Monitor) | 0.4404 | 0.4404 | - |

### Per-class F1 Scores (Baseline, 15 Epochs)

*These are the raw F1 scores (τ=0.5) before calibration.*

**Arrhythmia Branch:**
- NORM: 0.8464
- AFIB: 0.8738
- STACH: 0.8333
- PVC: 0.8706
- AFLT: 0.4762

**MI Branch:**
- ASMI: 0.7363
- IMI: 0.4699
- ILMI: 0.4444
- AMI: 0.1538

**Conduction Branch:**
- LBBB: 0.7974
- RBBB: **0.8174**
- IRBBB: 0.6578
- 1AVB: 0.5667

---

## Analysis

1. **Conduction Branch Validation:** 
   The addition of the `ConductionLeadGroupEncoder` alongside a dedicated Expert Transformer proved highly effective. In just 15 epochs, the model achieved an impressive **~0.80 F1** for both RBBB and LBBB. The anatomy-specific routing (V1-V3 for the right bundle, V5-V6/I/aVL for the left bundle) accurately captured bundle block morphologies.
   
2. **Impact of Calibration:**
   Calibration significantly rescued the **MI Macro F1 (+4.35% absolute)**. Thresholds for difficult classes like IMI and AMI were correctly lowered to `0.35-0.40`. This demonstrates that the model learns the correct ranking of MI probabilities early on, even if its raw output magnitudes are suppressed by dataset imbalance.

3. **Arrhythmia Stability:**
   Despite reducing the shared network capacity (from 4 shared layers in the previous architecture down to 2 shared + 2 expert), the Arrhythmia branch maintained strong performance (~0.80 Macro F1).
