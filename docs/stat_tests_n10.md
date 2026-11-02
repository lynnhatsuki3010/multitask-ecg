# Paired significance tests (validation)

Metric: `auprc/mi/IMI` at each run's monitor epoch. Test set untouched.
Bootstrap seed 20260810; alpha 0.05.

## Arms found

- `abl_arch05_shared_tf` n=10: 0.6293 +/- 0.0169
- `abl_arch09_decoupled` n=10: 0.6213 +/- 0.0169
- `abl_arch10_contrast` n=10: 0.6609 +/- 0.0078
- `abl_arch10b_cond_dim` n=10: 0.6539 +/- 0.0065

## Primary hypothesis

### PRIMARY: abl_arch10_contrast  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6213 | abl_arch10_contrast 0.6609
- delta: mean +0.0396, median +0.0394, SD 0.0228
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0252, +0.0518]
- Cohen d_z: 1.74   variant wins 9/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=1.0, p=0.0039
- verdict: **significant**

## Secondary family (Holm-corrected)

### SECONDARY: abl_arch10_contrast  -  abl_arch05_shared_tf

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch05_shared_tf 0.6293 | abl_arch10_contrast 0.6609
- delta: mean +0.0317, median +0.0353, SD 0.0189
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0207, +0.0428]
- Cohen d_z: 1.68   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020, p_holm=0.0059
- verdict: **significant**

### SECONDARY: abl_arch10b_cond_dim  -  abl_arch10_contrast

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch10_contrast 0.6609 | abl_arch10b_cond_dim 0.6539
- delta: mean -0.0070, median -0.0087, SD 0.0107
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0131, -0.0007]
- Cohen d_z: -0.65   variant wins 3/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=10.0, p=0.0840, p_holm=0.1680
- verdict: **not significant**

### SECONDARY (null claim): abl_arch09_decoupled  -  abl_arch05_shared_tf

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch05_shared_tf 0.6293 | abl_arch09_decoupled 0.6213
- delta: mean -0.0079, median -0.0040, SD 0.0251
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0232, +0.0065]
- Cohen d_z: -0.32   variant wins 4/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=21.0, p=0.5566
- verdict: **inconclusive (CI includes 0 and is wider than the floor)**

## How to read this

- The CI is the reportable effect size; p only says whether 0 is
  excluded. Quote both, and drop the older SE-ratio phrasing.
- A CI inside +/-0.01 is smaller than the measured drift
  between GPU architectures, so it is not a claimable difference.
- 'not significant' is never evidence of no effect unless the CI is
  bounded below that floor.

