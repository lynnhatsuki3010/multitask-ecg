# Contrast arms - exploratory paired tests (validation)

Run folders: `checkpoints/box_5070ti/run_*abl_arch1[34]*`, `checkpoints/run_*abl_arch*`

Post-hoc: added after the protocol was frozen. No Holm correction; report as descriptive evidence.
Positive delta = the arm WITH the explicit gated contrast is higher.
Bootstrap seed 20260810.

## Arms found

- `abl_arch05_shared_tf` n=10: 0.6293 +/- 0.0169
- `abl_arch09_decoupled` n=10: 0.6213 +/- 0.0169
- `abl_arch10_contrast` n=10: 0.6609 +/- 0.0078
- `abl_arch12_partial` n=10: 0.6577 +/- 0.0130
- `abl_arch13_nogate` n=10: 0.6587 +/- 0.0090
- `abl_arch14_partial_nogate` n=9: 0.6246 +/- 0.0125  [missing seeds: [31415]]

## Summary

| pair | metric | with | without | delta | 95% CI | wins | p |
|---|---|---|---|---|---|---|---|
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/IMI` | 0.6587 | 0.6246 | +0.0342 | [+0.0222, +0.0455] | 9/9 | 0.0039 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/IMI` | 0.6609 | 0.6587 | +0.0022 | [-0.0025, +0.0071] | 5/10 | 0.4316 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/IMI` | 0.6246 | 0.6308 | -0.0063 | [-0.0198, +0.0067] | 3/9 | 0.4961 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/IMI` | 0.6587 | 0.6213 | +0.0374 | [+0.0228, +0.0498] | 9/10 | 0.0039 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ILMI` | 0.4069 | 0.4103 | -0.0034 | [-0.0273, +0.0192] | 6/9 | 1.0000 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ILMI` | 0.4048 | 0.4075 | -0.0027 | [-0.0120, +0.0077] | 4/10 | 0.6250 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ILMI` | 0.4103 | 0.4151 | -0.0048 | [-0.0272, +0.0170] | 3/9 | 0.8203 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ILMI` | 0.4075 | 0.4036 | +0.0039 | [-0.0084, +0.0145] | 7/10 | 0.3750 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ASMI` | 0.8090 | 0.7993 | +0.0097 | [-0.0018, +0.0211] | 6/9 | 0.2031 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ASMI` | 0.8100 | 0.8120 | -0.0020 | [-0.0060, +0.0019] | 5/10 | 0.4922 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ASMI` | 0.7993 | 0.8103 | -0.0110 | [-0.0174, -0.0042] | 2/9 | 0.0195 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ASMI` | 0.8120 | 0.7933 | +0.0187 | [+0.0118, +0.0247] | 9/10 | 0.0039 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/AMI` | 0.1736 | 0.1902 | -0.0166 | [-0.0396, +0.0028] | 4/9 | 0.3594 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/AMI` | 0.1891 | 0.1944 | -0.0052 | [-0.0276, +0.0186] | 3/10 | 0.5566 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/AMI` | 0.1902 | 0.2129 | -0.0228 | [-0.0422, -0.0027] | 2/9 | 0.0742 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/AMI` | 0.1944 | 0.1762 | +0.0181 | [-0.0087, +0.0408] | 7/10 | 0.1934 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/macro` | 0.5120 | 0.5061 | +0.0060 | [-0.0018, +0.0140] | 5/9 | 0.3008 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/macro` | 0.5162 | 0.5181 | -0.0019 | [-0.0095, +0.0060] | 5/10 | 0.6953 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/macro` | 0.5061 | 0.5173 | -0.0112 | [-0.0173, -0.0054] | 1/9 | 0.0117 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/macro` | 0.5181 | 0.4986 | +0.0195 | [+0.0130, +0.0266] | 10/10 | 0.0020 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `S/composite` | 0.6994 | 0.7008 | -0.0013 | [-0.0059, +0.0029] | 3/9 | 0.4961 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `S/composite` | 0.6980 | 0.6944 | +0.0036 | [-0.0049, +0.0136] | 3/10 | 0.9219 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `S/composite` | 0.7008 | 0.7043 | -0.0035 | [-0.0087, +0.0018] | 3/9 | 0.3008 |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `S/composite` | 0.6944 | 0.6940 | +0.0004 | [-0.0107, +0.0107] | 6/10 | 0.7695 |

## Detail

### ARCH-12 - ARCH-14 (partial: contrast on - off): abl_arch12_partial  -  abl_arch14_partial_nogate

- seeds (n=9): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345]
- arm means: abl_arch14_partial_nogate 0.6246 | abl_arch12_partial 0.6587
- delta: mean +0.0342, median +0.0448, SD 0.0188
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0222, +0.0455]
- Cohen d_z: 1.82   variant wins 9/9 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0039
- verdict: **significant**

### ARCH-10 - ARCH-13 (decoupled: contrast on - off): abl_arch10_contrast  -  abl_arch13_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch13_nogate 0.6587 | abl_arch10_contrast 0.6609
- delta: mean +0.0022, median +0.0012, SD 0.0082
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0025, +0.0071]
- Cohen d_z: 0.27   variant wins 5/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=19.0, p=0.4316
- verdict: **not significant**

### ARCH-14 - ARCH-05 (projection + heads, no contrast): abl_arch14_partial_nogate  -  abl_arch05_shared_tf

- seeds (n=9): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345]
- arm means: abl_arch05_shared_tf 0.6308 | abl_arch14_partial_nogate 0.6246
- delta: mean -0.0063, median -0.0022, SD 0.0214
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0198, +0.0067]
- Cohen d_z: -0.29   variant wins 3/9 seeds
- Wilcoxon signed-rank (exact, two-sided): W=16.0, p=0.4961
- verdict: **not significant**

### ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast): abl_arch13_nogate  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6213 | abl_arch13_nogate 0.6587
- delta: mean +0.0374, median +0.0423, SD 0.0232
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0228, +0.0498]
- Cohen d_z: 1.61   variant wins 9/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=1.0, p=0.0039
- verdict: **significant**

