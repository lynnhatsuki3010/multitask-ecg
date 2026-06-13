# Experiment Log: Multi-Branch Transformer (DIR4)

A deep learning project for 12-lead ECG analysis utilizing a Multi-Branch Transformer architecture with heterogeneous expert blocks and anatomy-aware lead group encoders.

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

> **Ablation A/B/C/E09** all use `processed_dir4` with **4 separate MI labels** (IMI, ASMI, ILMI, AMI). No label merging.

---

## DIR4 Ablation Study: Routing Strategies (A / B / C / D)

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
| **D** | `arch_e09_subset_lead_13L.yaml` | `run_20260613_182621_multi_branch-tf-aug` | **True Approach 2:** subset-lead routing |

### What Each Run Actually Separates

> **Important:** None of A/B/C/D remove the **Shared CNN Front-End**. All four use `MultiBranchTransformerBackbone` with a single shared morphology CNN before expert branching.

| Component | A | B | C | D |
|-----------|---|---|---|---|
| **Shared CNN** | Yes (all tasks) | Yes | Yes (Arrhy only trains it) | **Yes (all tasks)** |
| **Shared Transformer** | Yes (×2) | **No** | No | No |
| **MI expert TF** | Yes, fused | Yes, fused | Dead | Yes, fused |
| **Raw anatomy path** | Full LeadGroupEncoder | Full LeadGroupEncoder | PerLead 12L graph | **Subset LeadGroupEncoder + group TF** |

**D does NOT drop shared CNN.** It only changes how the **raw-signal anatomy branch** encodes leads (subset groups with isolated group attention) while still fusing `mi_global + mi_seq` from the shared CNN → MI expert path.

Only **C (hard_graph)** partially isolates MI from the backbone — MI/Cond ignore `mi_expert` output, but even C still **instantiates** the shared CNN (trained mainly by Arrhythmia).

### Key Config Differences

| Setting | A (E06) | B (E08) | C (E07) | D (E09) |
|---------|---------|---------|---------|---------|
| `routing_mode` | `soft` | `soft` | `hard_graph` | `subset_lead` |
| `num_shared_layers` | **2** | **0** | **0** | **0** |
| `use_cross_attention` | true | true | true | **false** |
| Shared CNN | **Yes** | **Yes** | **Yes** | **Yes** |
| MI head | Expert + LeadGroupEncoder fusion | Same as A | PerLeadEncoder + GraphTransformer 12L | Subset LeadGroupEncoder + group TF + expert fusion |
| Cond head | Expert + CDLGE fusion | Same as A | PerLeadEncoder + GraphTransformer 12L | Expert + CDLGE fusion (soft) |
| MI expert receives gradient | Yes | Yes | **No (dead)** | Yes |

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

## Ablation Results (Test Set, Calibrated)

Post-hoc **Temperature Scaling** + **per-class threshold tuning** via `03_calibrate.py`.

| Metric | A (E06) | B (E08) | C (E07 hard) | D (E09 subset) | Best |
|--------|---------|---------|--------------|----------------|------|
| **MI Macro F1** | 0.452 | **0.458** | 0.425 | 0.404 | B |
| IMI F1 | 0.444 | **0.473** | 0.460 | 0.410 | B |
| **IMI AUPRC** | **0.459** | 0.422 | 0.389 | 0.454 | A |
| ASMI F1 | 0.724 | 0.742 | **0.771** | 0.717 | C |
| ILMI F1 | **0.519** | 0.500 | 0.468 | 0.426 | A |
| AMI F1 | **0.121** | 0.118 | 0.000 | 0.063 | A |
| MI Macro AUROC | **0.949** | 0.945 | 0.939 | 0.943 | A |
| **Cond Macro F1** | 0.711 | **0.731** | 0.599 | 0.699 | B |
| Arrhy Macro F1 | 0.770 | 0.754 | **0.771** | 0.736 | C |

### Per-class MI F1 (Calibrated)

| Label | A | B | C | D |
|-------|---|---|---|---|
| IMI | 0.444 | **0.473** | 0.460 | 0.410 |
| ASMI | 0.724 | 0.742 | **0.771** | 0.717 |
| ILMI | **0.519** | 0.500 | 0.468 | 0.426 |
| AMI | **0.121** | 0.118 | 0.000 | 0.063 |

### Per-class Conduction F1 (Calibrated)

| Label | A | B | C | D |
|-------|---|---|---|---|
| LBBB | **0.820** | 0.794 | 0.774 | 0.792 |
| RBBB | **0.832** | 0.833 | 0.770 | 0.828 |
| IRBBB | 0.645 | **0.693** | 0.658 | 0.644 |
| 1AVB | **0.547** | 0.605 | 0.195 | 0.532 |

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

4. **Rare classes remain the bottleneck**
   - AMI F1 ≤ 0.12 across A/B/D; C and D fail completely or near-zero (n=21 test samples).
   - Architecture changes alone do not solve extreme class imbalance for AMI.

5. **Comparison to earlier E06 run (processed_baseline)**
   - Previous log reported MI Macro F1 **0.470** (calibrated) on `processed_baseline`.
   - Current A/B on `processed_dir4` with corrected lateral lead indices (V5/V6) score 0.452–0.458 — not directly comparable due to data path and code changes.

6. **Recommended architecture: B (E08)**
   - Best overall MI Macro F1 (0.458) and Conduction (0.731) in fair 13-label ablation.
   - Approach 2 variants (C hard graph, D subset lead) did not outperform B.

### Common Pitfall: `hybrid_transformer` ≠ Approach 2

Run `run_20260610_134808_hybrid-tf-aug` used `architecture: hybrid_transformer` with `num_shared_layers: 2` in the yaml — but **`num_shared_layers` is ignored** for hybrid; only `num_encoder_layers` applies. That run is **not** a valid E07 multi-branch test. If `MIHead` was hard-routed at the time, only Arrhythmia used the hybrid backbone; MI used PerLeadEncoder graph on raw signal (similar to C, not D).

**Final ranking (MI Macro F1, calibrated):** B (0.458) > A (0.452) > C (0.425) > D (0.404)

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

# Calibrate after training (replace run folder name)
python scripts/03_calibrate.py --dir checkpoints/run_YYYYMMDD_HHMMSS_multi_branch-tf-aug
```
