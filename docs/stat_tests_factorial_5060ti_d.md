# Contrast arms - exploratory paired tests (validation)

Run folders: `checkpoints/box_5060ti_d/run_*`

Post-hoc: added after the protocol was frozen. No Holm correction; report as descriptive evidence.
Positive delta = the arm WITH the added factor (contrast, gate or layout, as each pair's label says) is higher.
Bootstrap seed 20260810.

## Arms found

- `abl_arch05_shared_tf` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch09_decoupled` n=10: 0.6213 +/- 0.0169
- `abl_arch10_contrast` n=10: 0.6609 +/- 0.0078
- `abl_arch12_partial` n=10: 0.6577 +/- 0.0130
- `abl_arch13_nogate` n=10: 0.6602 +/- 0.0092
- `abl_arch14_partial_nogate` n=10: 0.6164 +/- 0.0145
- `abl_arch15_partial_ungated` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch16_decoupled_contrast` n=10: 0.6506 +/- 0.0106
- `abl_arch17_partial_cluster` n=10: 0.6676 +/- 0.0108
- `abl_arch18_partial_cluster_contrast` n=10: 0.6627 +/- 0.0141

## Summary

| pair | metric | with | without | delta | 95% CI | wins | p |
|---|---|---|---|---|---|---|---|
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/IMI` | 0.6577 | 0.6164 | +0.0413 | [+0.0303, +0.0531] | 10/10 | 0.0020 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/IMI` | 0.6609 | 0.6602 | +0.0007 | [-0.0018, +0.0029] | 6/10 | 0.4316 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/IMI` | 0.6602 | 0.6213 | +0.0389 | [+0.0256, +0.0505] | 9/10 | 0.0039 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `auprc/mi/IMI` | 0.6506 | 0.6213 | +0.0293 | [+0.0215, +0.0353] | 9/10 | 0.0039 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `auprc/mi/IMI` | 0.6609 | 0.6506 | +0.0103 | [+0.0010, +0.0202] | 7/10 | 0.1055 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `auprc/mi/IMI` | 0.6676 | 0.6164 | +0.0512 | [+0.0421, +0.0602] | 10/10 | 0.0020 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `auprc/mi/IMI` | 0.6627 | 0.6676 | -0.0049 | [-0.0164, +0.0061] | 5/10 | 0.4922 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `auprc/mi/IMI` | 0.6627 | 0.6577 | +0.0050 | [-0.0063, +0.0163] | 6/10 | 0.4922 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ILMI` | 0.4061 | 0.4102 | -0.0041 | [-0.0206, +0.0126] | 5/10 | 0.7695 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ILMI` | 0.4048 | 0.4028 | +0.0021 | [-0.0025, +0.0065] | 6/10 | 0.4316 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ILMI` | 0.4028 | 0.4036 | -0.0009 | [-0.0122, +0.0103] | 4/10 | 1.0000 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `auprc/mi/ILMI` | 0.4012 | 0.4036 | -0.0024 | [-0.0108, +0.0060] | 5/10 | 0.5566 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `auprc/mi/ILMI` | 0.4048 | 0.4012 | +0.0036 | [-0.0068, +0.0127] | 6/10 | 0.6250 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `auprc/mi/ILMI` | 0.3789 | 0.4102 | -0.0313 | [-0.0534, -0.0110] | 3/10 | 0.0273 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `auprc/mi/ILMI` | 0.3776 | 0.3789 | -0.0013 | [-0.0262, +0.0219] | 5/10 | 0.9219 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `auprc/mi/ILMI` | 0.3776 | 0.4061 | -0.0285 | [-0.0503, -0.0084] | 1/10 | 0.0195 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ASMI` | 0.8106 | 0.8069 | +0.0037 | [-0.0044, +0.0110] | 8/10 | 0.1055 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ASMI` | 0.8100 | 0.8102 | -0.0002 | [-0.0024, +0.0026] | 5/10 | 0.5566 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ASMI` | 0.8102 | 0.7933 | +0.0169 | [+0.0095, +0.0237] | 9/10 | 0.0039 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `auprc/mi/ASMI` | 0.7966 | 0.7933 | +0.0033 | [-0.0022, +0.0088] | 6/10 | 0.3750 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `auprc/mi/ASMI` | 0.8100 | 0.7966 | +0.0134 | [+0.0068, +0.0208] | 9/10 | 0.0059 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `auprc/mi/ASMI` | 0.8078 | 0.8069 | +0.0009 | [-0.0061, +0.0083] | 5/10 | 1.0000 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `auprc/mi/ASMI` | 0.8143 | 0.8078 | +0.0065 | [-0.0002, +0.0133] | 6/10 | 0.1934 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `auprc/mi/ASMI` | 0.8143 | 0.8106 | +0.0037 | [-0.0050, +0.0162] | 4/10 | 0.9219 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/AMI` | 0.1756 | 0.1745 | +0.0011 | [-0.0227, +0.0225] | 5/10 | 0.7695 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/AMI` | 0.1891 | 0.1842 | +0.0049 | [-0.0119, +0.0229] | 6/10 | 0.9219 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/AMI` | 0.1842 | 0.1762 | +0.0080 | [-0.0153, +0.0338] | 5/10 | 0.7695 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `auprc/mi/AMI` | 0.1609 | 0.1762 | -0.0154 | [-0.0291, -0.0017] | 3/10 | 0.1055 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `auprc/mi/AMI` | 0.1891 | 0.1609 | +0.0283 | [+0.0120, +0.0444] | 9/10 | 0.0137 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `auprc/mi/AMI` | 0.2011 | 0.1745 | +0.0266 | [+0.0095, +0.0408] | 9/10 | 0.0371 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `auprc/mi/AMI` | 0.1832 | 0.2011 | -0.0179 | [-0.0356, +0.0028] | 2/10 | 0.1309 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `auprc/mi/AMI` | 0.1832 | 0.1756 | +0.0076 | [-0.0209, +0.0347] | 7/10 | 0.6250 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/macro` | 0.5125 | 0.5020 | +0.0105 | [+0.0014, +0.0207] | 8/10 | 0.1055 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/macro` | 0.5162 | 0.5144 | +0.0019 | [-0.0026, +0.0063] | 6/10 | 0.5566 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/macro` | 0.5144 | 0.4986 | +0.0157 | [+0.0079, +0.0236] | 9/10 | 0.0039 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `auprc/mi/macro` | 0.5023 | 0.4986 | +0.0037 | [-0.0015, +0.0092] | 6/10 | 0.2324 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `auprc/mi/macro` | 0.5162 | 0.5023 | +0.0139 | [+0.0091, +0.0191] | 10/10 | 0.0020 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `auprc/mi/macro` | 0.5139 | 0.5020 | +0.0118 | [+0.0044, +0.0195] | 8/10 | 0.0371 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `auprc/mi/macro` | 0.5095 | 0.5139 | -0.0044 | [-0.0157, +0.0071] | 5/10 | 0.4316 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `auprc/mi/macro` | 0.5095 | 0.5125 | -0.0030 | [-0.0125, +0.0076] | 4/10 | 1.0000 |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `S/composite` | 0.7005 | 0.6983 | +0.0022 | [-0.0044, +0.0094] | 6/10 | 0.5566 |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `S/composite` | 0.6980 | 0.6983 | -0.0003 | [-0.0032, +0.0025] | 6/10 | 0.8457 |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `S/composite` | 0.6983 | 0.6940 | +0.0043 | [-0.0014, +0.0103] | 6/10 | 0.3223 |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-16 - ARCH-09 (decoupled: contrast, no layout) | `S/composite` | 0.6987 | 0.6940 | +0.0047 | [-0.0025, +0.0101] | 8/10 | 0.1055 |
| ARCH-10 - ARCH-16 (decoupled: layout on top of contrast) | `S/composite` | 0.6980 | 0.6987 | -0.0008 | [-0.0062, +0.0061] | 3/10 | 0.4922 |
| ARCH-17 - ARCH-14 (partial: layout, no contrast) | `S/composite` | 0.6985 | 0.6983 | +0.0002 | [-0.0064, +0.0069] | 4/10 | 0.7695 |
| ARCH-18 - ARCH-17 (partial: contrast on top of layout) | `S/composite` | 0.7018 | 0.6985 | +0.0033 | [-0.0023, +0.0089] | 6/10 | 0.3223 |
| ARCH-18 - ARCH-12 (partial: layout on top of contrast) | `S/composite` | 0.7018 | 0.7005 | +0.0013 | [-0.0042, +0.0076] | 5/10 | 1.0000 |

## Detail

### ARCH-12 - ARCH-14 (partial: contrast on - off): abl_arch12_partial  -  abl_arch14_partial_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch14_partial_nogate 0.6164 | abl_arch12_partial 0.6577
- delta: mean +0.0413, median +0.0360, SD 0.0195
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0303, +0.0531]
- Cohen d_z: 2.12   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020
- verdict: **significant**

### ARCH-10 - ARCH-13 (decoupled: contrast on - off): abl_arch10_contrast  -  abl_arch13_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch13_nogate 0.6602 | abl_arch10_contrast 0.6609
- delta: mean +0.0007, median +0.0009, SD 0.0040
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0018, +0.0029]
- Cohen d_z: 0.18   variant wins 6/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=19.0, p=0.4316
- verdict: **not significant**

### ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast): abl_arch13_nogate  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6213 | abl_arch13_nogate 0.6602
- delta: mean +0.0389, median +0.0397, SD 0.0217
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0256, +0.0505]
- Cohen d_z: 1.79   variant wins 9/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=1.0, p=0.0039
- verdict: **significant**

### ARCH-16 - ARCH-09 (decoupled: contrast, no layout): abl_arch16_decoupled_contrast  -  abl_arch09_decoupled

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch09_decoupled 0.6213 | abl_arch16_decoupled_contrast 0.6506
- delta: mean +0.0293, median +0.0304, SD 0.0119
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0215, +0.0353]
- Cohen d_z: 2.47   variant wins 9/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=1.0, p=0.0039
- verdict: **significant**

### ARCH-10 - ARCH-16 (decoupled: layout on top of contrast): abl_arch10_contrast  -  abl_arch16_decoupled_contrast

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch16_decoupled_contrast 0.6506 | abl_arch10_contrast 0.6609
- delta: mean +0.0103, median +0.0073, SD 0.0164
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0010, +0.0202]
- Cohen d_z: 0.63   variant wins 7/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=11.0, p=0.1055
- verdict: **not significant**

### ARCH-17 - ARCH-14 (partial: layout, no contrast): abl_arch17_partial_cluster  -  abl_arch14_partial_nogate

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch14_partial_nogate 0.6164 | abl_arch17_partial_cluster 0.6676
- delta: mean +0.0512, median +0.0518, SD 0.0154
- 95% CI (percentile bootstrap, 10000 resamples): [+0.0421, +0.0602]
- Cohen d_z: 3.32   variant wins 10/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=0.0, p=0.0020
- verdict: **significant**

### ARCH-18 - ARCH-17 (partial: contrast on top of layout): abl_arch18_partial_cluster_contrast  -  abl_arch17_partial_cluster

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch17_partial_cluster 0.6676 | abl_arch18_partial_cluster_contrast 0.6627
- delta: mean -0.0049, median +0.0022, SD 0.0191
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0164, +0.0061]
- Cohen d_z: -0.26   variant wins 5/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=20.0, p=0.4922
- verdict: **not significant**

### ARCH-18 - ARCH-12 (partial: layout on top of contrast): abl_arch18_partial_cluster_contrast  -  abl_arch12_partial

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch12_partial 0.6577 | abl_arch18_partial_cluster_contrast 0.6627
- delta: mean +0.0050, median +0.0090, SD 0.0192
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0063, +0.0163]
- Cohen d_z: 0.26   variant wins 6/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=20.0, p=0.4922
- verdict: **not significant**

