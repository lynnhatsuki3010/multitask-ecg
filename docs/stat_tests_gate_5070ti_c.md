# Contrast arms - exploratory paired tests (validation)

Run folders: `checkpoints/box_5070ti_c/run_*`

Post-hoc: added after the protocol was frozen. No Holm correction; report as descriptive evidence.
Positive delta = the arm WITH the explicit gated contrast is higher.
Bootstrap seed 20260810.

## Arms found

- `abl_arch05_shared_tf` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch09_decoupled` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch10_contrast` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch12_partial` n=10: 0.6593 +/- 0.0123
- `abl_arch13_nogate` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch14_partial_nogate` n=0: -  [missing seeds: [42, 123, 2024, 7, 13, 99, 3407, 12345, 2025, 31415]]
- `abl_arch15_partial_ungated` n=10: 0.6592 +/- 0.0142

## Summary

| pair | metric | with | without | delta | 95% CI | wins | p |
|---|---|---|---|---|---|---|---|
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/IMI` | 0.6593 | 0.6592 | +0.0002 | [-0.0065, +0.0067] | 5/10 | 1.0000 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/IMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/ILMI` | 0.3968 | 0.4087 | -0.0120 | [-0.0313, +0.0112] | 3/10 | 0.2324 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/ILMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/ASMI` | 0.8099 | 0.8071 | +0.0028 | [-0.0072, +0.0119] | 5/10 | 0.6953 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/ASMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/AMI` | 0.1772 | 0.1647 | +0.0126 | [-0.0131, +0.0371] | 7/10 | 0.2324 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/AMI` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `auprc/mi/macro` | 0.5108 | 0.5099 | +0.0009 | [-0.0093, +0.0109] | 5/10 | 0.9219 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `auprc/mi/macro` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-14 (partial: contrast on - off) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-10 - ARCH-13 (decoupled: contrast on - off) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-14 - ARCH-05 (projection + heads, no contrast) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-13 - ARCH-09 (contrast_v2 layout, no contrast) | `S/composite` | - | - | no shared seeds | | | |
| ARCH-12 - ARCH-15 (partial: gate on - off) | `S/composite` | 0.7047 | 0.6996 | +0.0051 | [-0.0000, +0.0103] | 7/10 | 0.1055 |
| ARCH-15 - ARCH-14 (partial: ungated contrast - none) | `S/composite` | - | - | no shared seeds | | | |

## Detail

### ARCH-12 - ARCH-15 (partial: gate on - off): abl_arch12_partial  -  abl_arch15_partial_ungated

- seeds (n=10): [7, 13, 42, 99, 123, 2024, 2025, 3407, 12345, 31415]
- arm means: abl_arch15_partial_ungated 0.6592 | abl_arch12_partial 0.6593
- delta: mean +0.0002, median -0.0001, SD 0.0114
- 95% CI (percentile bootstrap, 10000 resamples): [-0.0065, +0.0067]
- Cohen d_z: 0.01   variant wins 5/10 seeds
- Wilcoxon signed-rank (exact, two-sided): W=27.0, p=1.0000
- verdict: **not significant**

