# Contrast arms - exploratory paired tests (validation)

Run folders: `checkpoints/box_5070ti_b/run_*`

Post-hoc: added after the protocol was frozen. No Holm correction; report as descriptive evidence.
Positive delta = the arm WITH the explicit gated contrast is higher.
Bootstrap seed 20260810.

## Arms found

- `abl_arch05_shared_tf` n=10: 0.6281 +/- 0.0149
- `abl_arch09_decoupled` n=10: 0.6174 +/- 0.0123
- `abl_arch10_contrast` n=10: 0.6578 +/- 0.0098
- `abl_arch12_partial` n=10: 0.6609 +/- 0.0118
- `abl_arch13_nogate` n=10: 0.6587 +/- 0.0090
- `abl_arch14_partial_nogate` n=10: 0.6234 +/- 0.0121

## Summary

| pair | metric | with | without | delta | 95% CI | wins | p |
|---|---|---|---|---|---|---|---|
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/IMI` | 0.6609 | 0.6234 | +0.0375 | [+0.0265, +0.0467] | 9/10 | 0.0039 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/IMI` | 0.6578 | 0.6587 | -0.0009 | [-0.0044, +0.0024] | 4/10 | 0.6250 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/IMI` | 0.6234 | 0.6281 | -0.0047 | [-0.0146, +0.0049] | 4/10 | 0.4316 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/IMI` | 0.6587 | 0.6174 | +0.0414 | [+0.0309, +0.0505] | 10/10 | 0.0020 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ILMI` | 0.4002 | 0.4119 | -0.0117 | [-0.0378, +0.0152] | 4/10 | 0.3223 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ILMI` | 0.4047 | 0.4075 | -0.0029 | [-0.0122, +0.0058] | 4/10 | 0.5566 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ILMI` | 0.4119 | 0.4170 | -0.0050 | [-0.0292, +0.0199] | 4/10 | 0.8457 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ILMI` | 0.4075 | 0.4062 | +0.0014 | [-0.0130, +0.0154] | 5/10 | 0.6953 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ASMI` | 0.8121 | 0.7983 | +0.0138 | [+0.0080, +0.0200] | 9/10 | 0.0059 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ASMI` | 0.8096 | 0.8120 | -0.0024 | [-0.0075, +0.0016] | 4/10 | 0.4922 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ASMI` | 0.7983 | 0.7995 | -0.0012 | [-0.0092, +0.0071] | 5/10 | 0.9219 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ASMI` | 0.8120 | 0.7937 | +0.0182 | [+0.0111, +0.0260] | 10/10 | 0.0020 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/AMI` | 0.1738 | 0.1959 | -0.0220 | [-0.0419, +0.0028] | 2/10 | 0.1602 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/AMI` | 0.1963 | 0.1944 | +0.0020 | [-0.0139, +0.0218] | 4/10 | 0.5566 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/AMI` | 0.1959 | 0.1896 | +0.0062 | [-0.0050, +0.0188] | 6/10 | 0.4316 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/AMI` | 0.1944 | 0.1776 | +0.0168 | [-0.0105, +0.0438] | 6/10 | 0.2754 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/macro` | 0.5118 | 0.5074 | +0.0044 | [-0.0046, +0.0131] | 6/10 | 0.3750 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/macro` | 0.5171 | 0.5181 | -0.0011 | [-0.0074, +0.0056] | 5/10 | 0.6953 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/macro` | 0.5074 | 0.5086 | -0.0012 | [-0.0076, +0.0045] | 4/10 | 0.7695 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/macro` | 0.5181 | 0.4987 | +0.0194 | [+0.0114, +0.0276] | 10/10 | 0.0020 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `S/composite` | 0.7046 | 0.7018 | +0.0028 | [-0.0015, +0.0070] | 7/10 | 0.2324 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `S/composite` | 0.6935 | 0.6941 | -0.0005 | [-0.0051, +0.0035] | 6/10 | 0.9219 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `S/composite` | 0.7018 | 0.7000 | +0.0019 | [-0.0017, +0.0061] | 5/10 | 0.6953 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `S/composite` | 0.6941 | 0.6955 | -0.0014 | [-0.0122, +0.0074] | 6/10 | 0.6250 |

## Detail

### ARCH-12 - ARCH-14 (partial: contrast on - off): abl_arch12_partial  -  abl_arch14_partial_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch14_partial_nogate 0.6234 | abl_arch12_partial 0.6609
- delta: mean +0.0375, median +0.0425, SD 0.0173
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0265, +0.0467]
- Cohen d_z: 2.17   variant wins 9/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=1.0, p=0.0039
- verdict: **significant**

### ARCH-10 - ARCH-13 (decoupled: contrast on - off): abl_arch10_contrast  -  abl_arch13_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch13_nogate 0.6587 | abl_arch10_contrast 0.6578
- delta: mean -0.0009, median -0.0009, SD 0.0058
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0044, +0.0024]
- Cohen d_z: -0.16   variant wins 4/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=22.0, p=0.6250
- verdict: **not significant**

### ARCH-14 - ARCH-05 (projection + heads, no contrast): abl_arch14_partial_nogate  -  abl_arch05_shared_tf

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch05_shared_tf 0.6281 | abl_arch14_partial_nogate 0.6234
- delta: mean -0.0047, median -0.0076, SD 0.0163
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0146, +0.0049]
- Cohen d_z: -0.29   variant wins 4/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=19.0, p=0.4316
- verdict: **not significant**

### ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast): abl_arch13_nogate  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6174 | abl_arch13_nogate 0.6587
- delta: mean +0.0414, median +0.0458, SD 0.0172
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0309, +0.0505]
- Cohen d_z: 2.41   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020
- verdict: **significant**

