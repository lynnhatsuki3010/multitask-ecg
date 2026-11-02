# Paired significance tests (validation)

Metric: `auprc/mi/IMI` at each run's monitor epoch. Test set untouched.
Bootstrap seed 20260810; alpha 0.05.

## Arms found

- `abl_arch05_shared_tf` n=10: 0.6281 +/- 0.0149
- `abl_arch09_decoupled` n=10: 0.6174 +/- 0.0123
- `abl_arch10_contrast` n=10: 0.6578 +/- 0.0098
- `abl_arch12_partial` n=10: 0.6609 +/- 0.0118
- `abl_arch13_nogate` n=10: 0.6587 +/- 0.0090
- `abl_arch14_partial_nogate` n=10: 0.6234 +/- 0.0121

## Primary hypothesis

### PRIMARY: abl_arch10_contrast  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6174 | abl_arch10_contrast 0.6578
- delta: mean +0.0404, median +0.0449, SD 0.0176
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0297, +0.0503]
- Cohen d_z: 2.29   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020
- verdict: **significant**

## Secondary family (Holm-corrected)

### SECONDARY: abl_arch10_contrast  -  abl_arch05_shared_tf

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch05_shared_tf 0.6281 | abl_arch10_contrast 0.6578
- delta: mean +0.0297, median +0.0279, SD 0.0198
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0183, +0.0413]
- Cohen d_z: 1.50   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020, p_holm=0.0039
- verdict: **significant**

### SECONDARY (null claim): abl_arch09_decoupled  -  abl_arch05_shared_tf

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch05_shared_tf 0.6281 | abl_arch09_decoupled 0.6174
- delta: mean -0.0107, median -0.0049, SD 0.0156
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0203, -0.0024]
- Cohen d_z: -0.69   variant wins 2/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=9.0, p=0.0645
- verdict: **inconclusive (CI includes 0 and is wider than the floor)**

_no shared seeds for abl_arch10b_cond_dim - abl_arch10_contrast_

## How to read this

- The CI is the reportable effect size; p only says whether 0 is
  excluded. Quote both, and drop the older SE-ratio phrasing.
- A CI inside +/-0.01 is smaller than the measured drift
  between GPU architectures, so it is not a claimable difference.
- 'not significant' is never evidence of no effect unless the CI is
  bounded below that floor.

