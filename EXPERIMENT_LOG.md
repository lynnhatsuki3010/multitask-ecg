# Experiment Log: Decoupled Multi-Task ECG (DIR4)

A deep learning project for 12-lead ECG analysis on PTB-XL. **Primary architecture: E10 (Fully Decoupled Multitask)** — three independent CNN+TF paths, no Shared CNN.

## Primary Architecture — E10 (Production)

| Item | Value |
|------|-------|
| **Config** | `configs/experiments/arch_e10_decoupled.yaml` |
| **Checkpoint** | `run_20260614_004857_decoupled_multitask-aug` |
| `architecture` | `decoupled_multitask` |
| `routing_mode` | `decoupled` (auto-set by factory) |
| **Backbone class** | `DecoupledMultiTaskBackbone` (`src/models/backbones.py`) |
| **MI head class** | `DecoupledSubsetMIHead` (`src/models/ecg_multitask.py`) |
| **Model wrapper** | `ECGMultiTaskModel` via `build_model()` (`src/models/factory.py`) |

> **For architecture diagrams:** use the simple 3-branch diagram below. Code entry points: `DecoupledMultiTaskBackbone` (Arrhy + Cond paths) + `DecoupledSubsetMIHead` (MI path). Do **not** use `MultiBranchTransformerBackbone` — that is ablation A/B/C/D only.

```text
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

### E10 Results (Test, Calibrated)

Checkpoint: `run_20260614_004857_decoupled_multitask-aug` · `best_model` epoch 11 · monitor `imi_auprc`

| Metric | E10 |
|--------|-----|
| **MI Macro F1** | **0.495** |
| IMI F1 / AUPRC | 0.457 / 0.437 |
| ASMI F1 | 0.757 |
| ILMI F1 | **0.605** |
| AMI F1 | **0.162** |
| Arrhy Macro F1 | **0.777** |
| Cond Macro F1 | 0.720 |

---

## Supported Labels (13-class — no MI merging)

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

> **Ablation A/B/C/D/E10** all use `processed_dir4` with **4 separate MI labels** (IMI, ASMI, ILMI, AMI). No label merging.

---

## DIR4 Ablation Study: Routing Strategies (A / B / C / D / E10)

Fair comparison setup (identical across runs):
- **Data:** `processed_dir4` / `splits_dir4` (13 labels)
- **Preprocessing:** robust normalization
- **Augmentation:** lead dropout
- **Training:** 15 epochs, BCE, uniform sampler, monitor `imi_auprc`
- **Split:** PTB-XL `strat_fold` (fold 10 = test, fold 9 = val)

| Run | Config | Checkpoint | Advisor mapping |
|-----|--------|------------|-----------------|
| **A** | `arch_e06_multi_branch.yaml` | `run_20260613_000218_multi_branch-tf-aug` | Baseline: soft routing + shared TF ×2 |
| **B** | `arch_e08_separated_experts.yaml` | `run_20260612_170141_multi_branch-tf-aug` | **Approach 1:** soft routing + no shared TF |
| **C** | `arch_e07_ablation_hard_graph_13L.yaml` | `run_20260613_081832_multi_branch-tf-aug` | Partial Approach 2 (hard graph 12L) |
| **D** | `arch_e09_subset_lead_13L.yaml` | `run_20260613_182621_multi_branch-tf-aug` | Subset-lead + shared CNN fusion |
| **E10** | `arch_e10_decoupled.yaml` | `run_20260614_004857_decoupled_multitask-aug` | **Fully decoupled:** no Shared CNN for MI |

### What Each Run Actually Separates

> **Important:** A/B/C/D share one **Shared CNN**. **E10 removes it entirely** — three fully independent paths from raw signal.

| Component | A | B | C | D | **E10** |
|-----------|---|---|---|---|---------|
| **Shared CNN** | Yes (all tasks) | Yes | Yes (Arrhy trains it) | Yes (all tasks) | **No** |
| **Shared Transformer** | Yes (×2) | No | No | No | No |
| **MI expert TF** | Yes, fused | Yes, fused | Dead | Yes, fused | **None** |
| **MI path** | LGE + expert fusion | Same as A | 12L hard graph | Subset LGE + expert fusion | **Subset LGE + Group TF only** |
| **Arrhy / Cond CNN** | Shared | Shared | Shared | Shared | **Independent each** |

**D vs E10:** D still fuses `mi_expert` (Shared CNN) into every MI logit. E10 MI head uses **only** raw-signal subset encoders — no shared backbone gradient for MI.

### Key Config Differences

| Setting | A (E06) | B (E08) | C (E07) | D (E09) | **E10** |
|---------|---------|---------|---------|---------|---------|
| `architecture` | `multi_branch_transformer` | same | same | same | **`decoupled_multitask`** |
| `routing_mode` | `soft` | `soft` | `hard_graph` | `subset_lead` | **`decoupled`** |
| `num_shared_layers` | **2** | **0** | **0** | **0** | N/A |
| Shared CNN | Yes | Yes | Yes | Yes | **No** |
| MI head | Expert + LGE fusion | Same as A | PerLead 12L graph | Subset LGE + expert fusion | **DecoupledSubsetMIHead** |
| Cond head | Expert + CDLGE | Same as A | PerLead 12L graph | Expert + CDLGE | Expert + CDLGE (own CNN) |
| MI expert gradient | Yes | Yes | No (dead) | Yes | **N/A (no mi_expert)** |

---

### Architecture A — Soft Routing + Shared Transformer (Baseline)

**Config:** `routing_mode: soft`, `num_shared_layers: 2`

```text
Input (B, 12, 5000)
  │
  ├── [Shared CNN Front-End]  → tokens (B, T, 256)
  │
  ├── [Shared Transformer ×2]  ← pre-mixed sequence before branching
  │
  ├── [Arrhythmia Expert TF ×2] → arrhy_global + arrhy_seq → Arrhy Head (5 labels)
  │
  ├── [MI Expert TF ×2] → mi_global + mi_seq
  │     ├── LeadGroupEncoder — Inferior   [II, III, aVF]     → (B, 128)
  │     ├── LeadGroupEncoder — Reciprocal [I, aVL]           → (B, 64)
  │     ├── LeadGroupEncoder — Anterior   [V1–V4]           → (B, 128)
  │     ├── LeadGroupEncoder — Lateral    [V5, V6]          → (B, 128)
  │     └── LeadGroupEncoder — Anterior6  [V1–V6]           → (B, 128)
  │           → 4 per-label heads: IMI, ASMI, ILMI, AMI
  │
  └── [Conduction Expert TF ×2] → cond_global + cond_seq
        ├── ConductionLeadGroupEncoder — Right [V1–V3] → (B, 128)
        ├── ConductionLeadGroupEncoder — Left  [V5,V6,I,aVL] → (B, 128)
        └── Conduction Head → LBBB, RBBB, IRBBB, 1AVB
```

---

### Architecture B — Soft Routing + Separated Experts (Advisor Approach 1)

**Config:** `routing_mode: soft`, `num_shared_layers: 0`

Same as **A**, except **no Shared Transformer**. Each expert branch (Arrhythmia / MI / Conduction) builds its own attention map directly from CNN tokens.

```text
Input (B, 12, 5000)
  │
  ├── [Shared CNN Front-End]  → tokens (B, T, 256)
  │       (NO shared Transformer)
  │
  ├── [Arrhythmia Expert TF ×2] ──→ Arrhy Head
  ├── [MI Expert TF ×2] + LeadGroupEncoders (raw) ──→ 4 MI heads
  └── [Cond Expert TF ×2] + ConductionLeadGroupEncoder (raw) ──→ Cond Head
```

**Advisor intent:** Remove shared sequence mixing; keep shared CNN morphology front-end and per-branch expert transformers.

---

### Architecture C — Hard Graph Routing (Partial Approach 2)

**Config:** `routing_mode: hard_graph`, `num_shared_layers: 0`

```text
Input (B, 12, 5000)
  │
  ├── [Shared CNN] → [Arrhythmia Expert TF ×2] → Arrhy Head  ← only path using backbone for Arrhy
  │
  ├── [MI Expert TF ×2]  ──×── (dead params, no gradient)
  ├── [Cond Expert TF ×2] ──×── (dead params, no gradient)
  └── [ConductionLeadGroupEncoder] ──×── (computed but ignored)
  │
  └── [Raw Signal Copy — Hard-Routed MI & Conduction]
        PerLeadEncoder (each of 12 leads independently)
              ↓
        GraphTransformer (attention across ALL 12 lead nodes)
              ↓
        Per-label node pooling:
          IMI  ← [II, III, aVF, I, aVL]
          ASMI ← [V1–V4]
          ILMI ← [II, III, aVF, I, aVL, V5, V6]
          AMI  ← [V1–V6]
        Conduction ← pool all 12 nodes → 4 logits
```

**Limitations vs advisor's Approach 2:**
- Still encodes all 12 leads before subset pooling (cross-lead noise not eliminated)
- MI/Cond experts and anatomy encoders from A/B are orphaned (dead parameters)
- Shared CNN only trained by Arrhythmia loss

---

### Architecture D — Subset-Lead Routing (True Advisor Approach 2)

**Config:** `routing_mode: subset_lead`, `num_shared_layers: 0`, `use_cross_attention: false`  
**File:** `configs/experiments/arch_e09_subset_lead_13L.yaml`  
**Checkpoint:** `run_20260613_182621_multi_branch-tf-aug`

```text
Input (B, 12, 5000)
  │
  ├── [Shared CNN Front-End]  ← STILL SHARED (same as A/B)
  │       (NO shared Transformer)
  │
  ├── [Arrhythmia Expert TF ×2] ──→ Arrhy Head
  ├── [MI Expert TF ×2] ──→ mi_global + mi_seq ──┐
  ├── [Cond Expert TF ×2] + ConductionLeadGroupEncoder ──→ Cond Head (soft)
  │
  └── [Raw Signal — Subset LeadGroupEncoders]  ← ONLY this path is "isolated"
        Inferior CNN   [II, III, aVF]  ─┐
        Reciprocal CNN [I, aVL]         ├─→ Group TF, 2 nodes (IMI)
                                          └─→ Group TF, 3 nodes (ILMI: + V5/V6)
        Anterior CNN   [V1–V4]         ─→ Group TF, 1 node (ASMI)
        Anterior6 CNN  [V1–V6]         ─→ Group TF, 1 node (AMI)
              ↓
        Concat [mi_expert (from Shared CNN) + subset group context] → 4 MI heads
```

**What D isolates vs what stays shared:**
- **Isolated:** raw-signal anatomy encoding — attention only within each label's lead subset (not 12-lead graph).
- **Still shared:** `MorphologyConvFrontEnd` CNN → all three expert branches; MI loss gradient still updates shared CNN + `mi_expert`.

**Key difference from C:** Attention is within anatomy subsets only, not across all 12 leads. Unlike C, `mi_expert` and `ConductionLeadGroupEncoder` are actively trained.

**Key difference from B:** Same backbone as B; only the MI head's raw anatomy path uses group-isolated transformers instead of direct LeadGroupEncoder fusion (and cross-attention is disabled).

---

### Architecture E10 — Ablation Reference

Same as **Primary Architecture** section above. Won ablation: MI Macro F1 0.495 vs B 0.458.

---

## Ablation Results (Test Set, Calibrated)

Post-hoc **Temperature Scaling** + **per-class threshold tuning** via `03_calibrate.py`.

| Metric | A (E06) | B (E08) | C (E07) | D (E09) | **E10** | Best |
|--------|---------|---------|---------|---------|---------|------|
| **MI Macro F1** | 0.452 | 0.458 | 0.425 | 0.404 | **0.495** | **E10** |
| IMI F1 | 0.444 | **0.473** | 0.460 | 0.410 | 0.457 | B |
| **IMI AUPRC** | **0.459** | 0.422 | 0.389 | 0.454 | 0.437 | A |
| ASMI F1 | 0.724 | 0.742 | **0.771** | 0.717 | 0.757 | C |
| ILMI F1 | **0.519** | 0.500 | 0.468 | 0.426 | **0.605** | **E10** |
| AMI F1 | 0.121 | 0.118 | 0.000 | 0.063 | **0.162** | **E10** |
| MI Macro AUROC | **0.949** | 0.945 | 0.939 | 0.943 | 0.934 | A |
| **Cond Macro F1** | 0.711 | **0.731** | 0.599 | 0.699 | 0.720 | B |
| Arrhy Macro F1 | 0.770 | 0.754 | **0.771** | 0.736 | **0.777** | **E10** |

### Per-class MI F1 (Calibrated)

| Label | A | B | C | D | **E10** |
|-------|---|---|---|---|---------|
| IMI | 0.444 | **0.473** | 0.460 | 0.410 | 0.457 |
| ASMI | 0.724 | 0.742 | **0.771** | 0.717 | 0.757 |
| ILMI | **0.519** | 0.500 | 0.468 | 0.426 | **0.605** |
| AMI | 0.121 | 0.118 | 0.000 | 0.063 | **0.162** |

### Per-class Conduction F1 (Calibrated)

| Label | A | B | C | D | **E10** |
|-------|---|---|---|---|---------|
| LBBB | **0.820** | 0.794 | 0.774 | 0.792 | 0.813 |
| RBBB | **0.832** | 0.833 | 0.770 | 0.828 | 0.857 |
| IRBBB | 0.645 | **0.693** | 0.658 | 0.644 | 0.680 |
| 1AVB | **0.547** | 0.605 | 0.195 | 0.532 | 0.529 |

---

## Ablation Analysis

1. **Approach 1 (B vs A) — modest improvement, not breakthrough**
   - Removing the shared Transformer (B) improved MI Macro F1 by **+0.6%** (0.458 vs 0.452) and Conduction by **+2.0%**.
   - Trade-off: IMI AUPRC dropped **−3.7%** (0.422 vs 0.459) because training monitors `imi_auprc`.
   - Shared CNN remains common to all tasks — gradient competition at the morphology layer is unresolved.

2. **Partial Approach 2 (C) — architectural regression**
   - Hard graph routing with 12-lead attention before subset pooling performed poorly on MI Macro F1 (0.425) and Conduction (0.599).
   - Orphaned backbone parameters (`mi_expert`, `cond_expert`, `ConductionLeadGroupEncoder`) receive no gradient from MI/Cond losses.
   - ASMI F1 is highest in C (0.771), suggesting anteroseptal graph pooling can work in isolation, but overall multitask performance collapses.

3. **True Approach 2 (D) — subset isolation did not help; shared CNN unchanged**
   - D achieved the **lowest MI Macro F1** (0.404 calibrated) despite correct subset-lead anatomy routing.
   - IMI AUPRC (0.454) is near A and better than B/C, but IMI precision is very low (0.30) — high recall (0.65), many false positives.
   - **D still uses Shared CNN + mi_expert** — only the raw anatomy branch is subset-isolated. The advisor's full vision of "no shared backbone for MI" was not implemented; removing shared CNN was never part of E09.
   - Disabling cross-attention (vs A/B) and using shallow 1-layer group transformers may have reduced capacity.
   - 1AVB F1 (0.532) is the best among rare conduction classes across all runs.

4. **Rare classes — E10 breaks AMI plateau**
   - AMI F1 ≤ 0.12 across A/B/D (n=21 test); C fails completely.
   - **E10 AMI F1 = 0.162** (+37% vs B), AUPRC 0.149 vs 0.104 — first meaningful AMI detection.
   - Still only ~3/21 recall; remains fragile but architecture change clearly helps.

5. **Comparison to earlier E06 run (processed_baseline)**
   - Previous log reported MI Macro F1 **0.470** (calibrated) on `processed_baseline`.
   - Current A/B on `processed_dir4` with corrected lateral lead indices (V5/V6) score 0.452–0.458 — not directly comparable due to data path and code changes.

6. **E10 (decoupled) — current best architecture**
   - Removing Shared CNN for MI eliminates gradient competition — **MI Macro F1 0.495** (+3.7pp vs B).
   - ILMI +10.5pp, AMI +4.4pp vs B; trade-off IMI F1 −1.6pp, Cond −1.1pp.
   - `best_model` saved at epoch 11 (`imi_auprc` monitor); epoch 15 had higher val AMI — room for checkpoint tuning.
   - Validates advisor hypothesis: shared morphology CNN was the MI bottleneck, not just Shared Transformer.

7. **Previous recommendation (B) superseded by E10** for MI-focused thesis.

### Common Pitfall: `hybrid_transformer` ≠ Approach 2

Run `run_20260610_134808_hybrid-tf-aug` used `architecture: hybrid_transformer` with `num_shared_layers: 2` in the yaml — but **`num_shared_layers` is ignored** for hybrid; only `num_encoder_layers` applies. That run is **not** a valid E07 multi-branch test. If `MIHead` was hard-routed at the time, only Arrhythmia used the hybrid backbone; MI used PerLeadEncoder graph on raw signal (similar to C, not D).

**Final ranking (MI Macro F1, calibrated):** **E10 (0.495)** > B (0.458) > A (0.452) > C (0.425) > D (0.404)

---

## E11 — MI Single-Task Ensemble (Pending)

Train MI-only model (`arch_e11_mi_single_task.yaml`) and ensemble with E08/E10 via `scripts/04_ensemble_eval.py` — **not yet run**.

---

## BRANCHE10 — Per-Branch Optimization Matrix (Pending)

Sau E10, tối ưu từng nhánh độc lập (MI / Conduction / Arrhythmia) mà **không đổi kiến trúc decoupled**.

**Hướng dẫn đầy đủ:** `configs/experiments/BRANCHE10_MATRIX.md`

| ID | Config | Track | Thay đổi chính |
|----|--------|-------|-----------------|
| 00 | `branche10_00_baseline.yaml` | baseline | E10 gốc |
| 01 | `branche10_01_mi_asl.yaml` | MI | ASL + pos_weight |
| 02 | `branche10_02_mi_imi_weight.yaml` | MI | + `loss_weights.imi: 1.75` |
| 03 | `branche10_03_mi_macro_select.yaml` | MI | + `mi_macro_f1_tuned` monitor |
| 04 | `branche10_04_cond_lead_dim.yaml` | Conduction | `cd_lead_out_dim: 192` |
| 05 | `branche10_05_cond_depth.yaml` | Conduction | + `cond_layers: 3` |
| 06 | `branche10_06_arr_calib.yaml` | Arrhythmia | val threshold tune + arrhythmia constraints |

Run folders có prefix `branche10_XX_*` (field `experiment.id`).

```powershell
# MI track (chạy tuần tự)
python scripts/02_train.py --config configs/experiments/branche10_01_mi_asl.yaml
python scripts/02_train.py --config configs/experiments/branche10_02_mi_imi_weight.yaml
python scripts/02_train.py --config configs/experiments/branche10_03_mi_macro_select.yaml

# Conduction track
python scripts/02_train.py --config configs/experiments/branche10_04_cond_lead_dim.yaml
python scripts/02_train.py --config configs/experiments/branche10_05_cond_depth.yaml

# Arrhythmia track
python scripts/02_train.py --config configs/experiments/branche10_06_arr_calib.yaml

# So sánh sau khi train xong
python scripts/compare_branche10_runs.py --glob "checkpoints/run_*_branche10_*" --out artifacts/branche10_comparison.md
```

### Results table (fill after runs)

| Run | MI Macro F1 | IMI F1 | AMI F1 | Cond Macro F1 | Arrhy Macro F1 |
|-----|-------------|--------|--------|---------------|----------------|
| 00 baseline | | | | | |
| 01 mi_asl | | | | | |
| 02 mi_imi_weight | | | | | |
| 03 mi_macro_select | | | | | |
| 04 cond_lead_dim | | | | | |
| 05 cond_depth | | | | | |
| 06 arr_calib | | | | | |

---

## DIR5 — Bottleneck Resolution (MI / IMI focus)

Branch `EXP-DIR5-BOTTLENECK-RESOLUTION`. Goal: lift MI (especially IMI) by
attacking the diagnosed bottlenecks in order — per-label loss bug, calibration,
MI-head sibling overlap, then label definition. Full matrix:
`configs/experiments/DIR5_MATRIX.md`. All runs on `processed_dir4` (IMI:80)
unless noted; decoupled E10 architecture.

### Phase results (Test, calibrated)

| Variant | MI Macro F1 | MI Macro AUPRC | IMI F1 | IMI AUPRC | IMI support |
|---------|-------------|----------------|--------|-----------|-------------|
| E10 baseline (IMI:80) | 0.498 | 0.477 | 0.437 | 0.437 | 103 |
| P1 cheap levers (IMI:80) | 0.454 | 0.450 | 0.373 | 0.447 | 103 |
| P2 `contrast` (IMI:80) | 0.469 | 0.468 | 0.490 | 0.480 | 103 |
| P2.1 `contrast_v2` (IMI:80) | 0.494 | 0.476 | 0.464 | 0.460 | 103 |
| **P3a `contrast_v2` + IMI:50** | **0.511** | **0.517** | **0.590** | **0.645** | **175** |

### Findings

1. **Phase 0 — per-label MI loss bug fixed.** `MultiTaskLoss` ignored
   `loss_weights.{imi,asmi,ilmi,ami}` for the 4-label MI head (only the 2-label
   path applied them). Now each MI sub-label keeps its own `pos_weight` and
   weight; verified `loss/imi != loss/asmi`.

2. **Phase 1 — cheap levers do NOT beat E10.** Weighted sampler + precision-aware
   tuning (IMI beta=0.5 + min_precision) + heavy aug + `mi_macro_f1_tuned`
   monitor: IMI precision rose (0.34->0.49) but recall collapsed (0.62->0.30);
   MI macro F1 0.454 < E10 0.498. Crucially **AUPRC stayed flat** -> calibration
   cannot raise separability; the ceiling is representational.

3. **Phase 2 — `contrast` MI head: first real IMI separability gain.** Region-once
   encoding + shared region TF (IMI/ASMI now see lateral leads) + sibling
   contrast. **IMI AUPRC 0.437->0.480** (cheap levers could not). But it degraded
   the extended siblings (ILMI/AMI lost dedicated joint encoding + hard
   subtraction impoverished them) -> MI macro flat.

4. **Phase 2.1 — `contrast_v2`: best MI head.** Restores dedicated joint encoders
   (anterior6 for AMI, inferolateral for ILMI) + gated `SiblingContrast`
   (identity at init). MI macro back to E10 parity (F1 0.494, AUPRC 0.476) while
   keeping a clean IMI gain over E10 (AUPRC 0.460, F1 0.464) and best AMI.
   ILMI is the residual loss (IMI<->ILMI share inferior region -> contrast is
   zero-sum within the pair).

5. **Phase 3a — IMI threshold 80->50: large IMI jump, but a label-definition
   change.** IMI F1 0.464->0.590, AUPRC 0.460->0.645, MI macro F1 0.494->0.511.
   **Caveat:** IMI test support changes 103->175 (adds borderline SCP-confidence
   50-79 cases), so this is NOT the same task — AUPRC is sensitive to prevalence
   and is not directly comparable across thresholds. The genuine signal: with
   more positives IMI is far more learnable (F1 0.59, recall 0.77, precision
   0.48). ASMI/ILMI/AMI are unchanged (same threshold), so the macro gain is
   entirely from the relabelled IMI. Report both thresholds as a label ablation.

**Recommended DIR5 config:** `contrast_v2` MI head + IMI:50
(`dir5_p3a_imi50.yaml`). Configs: `dir5_p1_cheap_levers`, `dir5_p2_mi_contrast`,
`dir5_p2b_mi_contrast_v2`, `dir5_p3a_imi50`.

---

## Legacy: ARCH-E06 on processed_baseline (Historical Reference)

*The section below documents an earlier E06 run on `processed_baseline` before the DIR4 ablation. Results are not directly comparable to A/B/C above.*

### Model Architecture (ARCH-E06 Multi-Branch — original)

```text
Input (B, 12, 5000)   ← 12-lead ECG, 10 seconds @ 500 Hz
  │
  ├── [Shared CNN Front-End]
  │     → DepthwiseConv1d  (12 → 12, per-lead, stride 5)
  │     → PointwiseConv1d  (12 → stem_dim=96)
  │     → ResidualConvBlock (96  → 128)
  │     → ResidualConvBlock (128 → 192)
  │     → ResidualConvBlock (192 → 256)
  │     → Conv1d Projection (256 → d_model=256)
  │     → Sequence tokens (B, T, 256)
  │
  ├── [Shared Transformer Encoder]
  │     → 2× TransformerEncoderLayer (8 heads, FFN dim=512, Pre-LN)
  │     → Shared Sequence (B, T, 256)
  │
  ├── [Arrhythmia Expert Branch] → 5 arrhythmia labels
  ├── [MI Expert Branch] + LeadGroupEncoders → IMI, ASMI, ILMI, AMI
  └── [Conduction Expert Branch] + ConductionLeadGroupEncoder → 4 conduction labels
```

### Historical Calibration (processed_baseline)

| Branch | Temperature (T) |
|--------|-----------------|
| Arrhythmia | 1.4691 |
| MI | 1.4747 |
| Conduction | 1.3509 |

### Historical Test Results (processed_baseline, Calibrated)

| Metric | Baseline (τ=0.5) | Calibrated |
|--------|-----------------|------------|
| Arrhythmia Macro F1 | 0.8182 | 0.7985 |
| **MI Macro F1** | 0.4269 | **0.4704** |
| Conduction Macro F1 | 0.7257 | 0.7243 |
| IMI AUPRC | 0.4404 | 0.4404 |

**Historical per-class MI F1 (raw, τ=0.5):** IMI 0.470, ASMI 0.736, ILMI 0.444, AMI 0.154

---

## Training Commands

```powershell
cd "E:\KLTN\KL Project\ECG_HRV"
$env:PYTHONPATH = "."

# A — Baseline soft + shared TF ×2
python scripts/02_train.py --config configs/experiments/arch_e06_multi_branch.yaml

# B — Approach 1: soft + no shared TF
python scripts/02_train.py --config configs/experiments/arch_e08_separated_experts.yaml

# C — Partial Approach 2: hard graph 12L
python scripts/02_train.py --config configs/experiments/arch_e07_ablation_hard_graph_13L.yaml

# D — True Approach 2: subset-lead routing (E09)
python scripts/02_train.py --config configs/experiments/arch_e09_subset_lead_13L.yaml

# E10 — Fully decoupled CNN (no shared backbone)
python scripts/02_train.py --config configs/experiments/arch_e10_decoupled.yaml

# E11 — MI-only single task
python scripts/02_train.py --config configs/experiments/arch_e11_mi_single_task.yaml

# Calibrate after training (replace run folder name)
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_HHMMSS_decoupled_multitask-aug

# E11 ensemble (after MI-only training)
python scripts/04_ensemble_eval.py `
  --multitask-dir checkpoints/run_20260614_004857_decoupled_multitask-aug `
  --mi-dir checkpoints/run_YYYYMMDD_mi_only-tf-aug
```
