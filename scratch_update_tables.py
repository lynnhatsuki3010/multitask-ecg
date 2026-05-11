import os

path = "e:/KLTN/KL Project/ECG_HRV/EXPERIMENT_LOG.md"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Stage 1
old_s1 = """| Experiment | Split Method | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|-------------|-----------|-----------|--------|----------------|--------|---------|
| SPLIT-E01 | `strat_fold` | 0.937 | 0.457 | 0.456 | 0.733 | `run_20260506_231153_hybrid-tf` | |
| SPLIT-E02 | `random_grouped` | 0.947 | **0.519** | **0.517** | **0.828** | `run_20260507_003151_hybrid-tf` | ✅ |
| SPLIT-E03 | `stratified_group_kfold (IMI)` | **0.959** | 0.506 | 0.558 | 0.768 | `run_20260507_013801_hybrid-tf` | |"""
new_s1 = """| Experiment | Split Method | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder | Winner? |
|------------|-------------|-----------|-----------|-------------|----------------|--------|---------|
| SPLIT-E01 | `strat_fold` | 0.937 | 0.457 | 0.564 | 0.733 | `run_20260506_231153_hybrid-tf` | |
| SPLIT-E02 | `random_grouped` | 0.947 | **0.519** | 0.640 | **0.828** | `run_20260507_003151_hybrid-tf` | ✅ |
| SPLIT-E03 | `stratified_group_kfold (IMI)` | **0.959** | 0.506 | **0.655** | 0.768 | `run_20260507_013801_hybrid-tf` | |"""
text = text.replace(old_s1, new_s1)

# Stage 2
old_s2 = """| Experiment | Normalization | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|---------------|-----------|-----------|--------|----------------|--------|---------|
| NORM-E01 | `zscore` | **0.948** | **0.525** | **0.511** | **0.830** | `run_20260507_204106_hybrid-tf` | ✅ |
| NORM-E02 | `robust` | 0.946 | 0.512 | **0.511** | 0.740 | `run_20260507_220132_hybrid-tf` | |
| NORM-E03 | `minmax` | 0.946 | 0.485 | 0.501 | 0.759 | `run_20260507_231720_hybrid-tf` | |"""
new_s2 = """| Experiment | Normalization | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder | Winner? |
|------------|---------------|-----------|-----------|-------------|----------------|--------|---------|
| NORM-E01 | `zscore` | **0.948** | **0.525** | 0.635 | **0.830** | `run_20260507_204106_hybrid-tf` | ✅ |
| NORM-E02 | `robust` | 0.946 | 0.512 | **0.639** | 0.740 | `run_20260507_220132_hybrid-tf` | |
| NORM-E03 | `minmax` | 0.946 | 0.485 | 0.618 | 0.759 | `run_20260507_231720_hybrid-tf` | |"""
text = text.replace(old_s2, new_s2)

# Stage 3
old_s3 = """| Experiment | Loss Configuration | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------------|-----------|-----------|--------|----------------|--------|---------|
| LOSS-E01 | `pos_weight` | 0.949 | 0.504 | 0.508 | 0.774 | `run_20260508_005129_hybrid-tf` | |
| LOSS-E02 | `Focal Loss` | 0.949 | 0.520 | 0.494 | 0.806 | `run_20260508_015300_hybrid-tf-focal` | |
| LOSS-E03 | `Focal + GradIso (0.3)`| 0.945 | 0.493 | 0.503 | **0.822** | `run_20260508_025238_hybrid-tf-focal` | ✅ |"""
new_s3 = """| Experiment | Loss Configuration | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------------|-----------|-----------|-------------|----------------|--------|---------|
| LOSS-E01 | `pos_weight` | 0.949 | 0.504 | **0.631** | 0.774 | `run_20260508_005129_hybrid-tf` | |
| LOSS-E02 | `Focal Loss` | 0.949 | 0.520 | 0.624 | 0.806 | `run_20260508_015300_hybrid-tf-focal` | |
| LOSS-E03 | `Focal + GradIso (0.3)`| 0.945 | 0.493 | 0.625 | **0.822** | `run_20260508_025238_hybrid-tf-focal` | ✅ |"""
text = text.replace(old_s3, new_s3)

# Stage 4
old_s4 = """| Experiment | Augmentation | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------|-----------|-----------|--------|----------------|--------|---------|
| AUG-E01 | `MixUp` | 0.944 | 0.497 | 0.494 | 0.807 | `run_20260508_071142_hybrid-tf-focal-aug-mxp` | |
| AUG-E02 | `Random Crop` | 0.940 | 0.484 | 0.471 | 0.626 | `run_20260508_081931_hybrid-tf-focal-aug` | |
| AUG-E03 | `Noise + Warp`| **0.949** | **0.516** | **0.527** | **0.825** | `run_20260508_092233_hybrid-tf-focal-aug` | ✅ |
| AUG-E04 | `Lead Dropout`| 0.950 | 0.495 | 0.513 | 0.812 | `run_20260508_102608_hybrid-tf-focal-aug` | |"""
new_s4 = """| Experiment | Augmentation | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder | Winner? |
|------------|--------------|-----------|-----------|-------------|----------------|--------|---------|
| AUG-E01 | `MixUp` | 0.944 | 0.497 | 0.632 | 0.807 | `run_20260508_071142_hybrid-tf-focal-aug-mxp` | |
| AUG-E02 | `Random Crop` | 0.940 | 0.484 | 0.615 | 0.626 | `run_20260508_081931_hybrid-tf-focal-aug` | |
| AUG-E03 | `Noise + Warp`| **0.949** | **0.517** | **0.643** | **0.826** | `run_20260508_092233_hybrid-tf-focal-aug` | ✅ |
| AUG-E04 | `Lead Dropout`| 0.950 | 0.496 | 0.631 | 0.813 | `run_20260508_102608_hybrid-tf-focal-aug` | |"""
text = text.replace(old_s4, new_s4)

# Stage 5
old_s5 = """| Experiment | Sampling | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder | Winner? |
|------------|----------|-----------|-----------|--------|----------------|--------|---------|
| SAMP-E01 | `Uniform` | **0.949** | **0.516** | **0.518** | **0.823** | `run_20260508_185741_hybrid-tf-focal-aug` | ✅ |
| SAMP-E02 | `Weighted` | 0.938 | 0.471 | 0.486 | 0.807 | `run_20260508_200759_hybrid-tf-focal-wrs-aug` | |"""
new_s5 = """| Experiment | Sampling | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder | Winner? |
|------------|----------|-----------|-----------|-------------|----------------|--------|---------|
| SAMP-E01 | `Uniform` | **0.949** | **0.516** | **0.639** | **0.824** | `run_20260508_185741_hybrid-tf-focal-aug` | ✅ |
| SAMP-E02 | `Weighted` | 0.938 | 0.471 | 0.601 | 0.808 | `run_20260508_200759_hybrid-tf-focal-wrs-aug` | |"""
text = text.replace(old_s5, new_s5)

# Stage 6
old_s6 = """| Run | Seed | IMI AUROC | IMI AUPRC | IMI F1 | Arrhy macro F1 | Folder |
|-----|------|-----------|-----------|--------|----------------|--------|
| FINAL-E01 | 42 | 0.949 | 0.492 | 0.520 | 0.837 | `run_20260508_214025_hybrid-tf-focal-aug` |
| FINAL-E02 | 123 | 0.946 | 0.480 | 0.481 | 0.822 | `run_20260509_004809_hybrid-tf-focal-aug` |
| FINAL-E03 | 2024 | 0.950 | 0.514 | 0.485 | 0.820 | `run_20260509_033053_hybrid-tf-focal-aug` |
| **Mean ± Std** | — | **0.9483 ± 0.0017** | **0.4955 ± 0.0142** | **0.4955 ± 0.0176** | **0.8263 ± 0.0079** | — |"""
new_s6 = """| Run | Seed | IMI AUROC | IMI AUPRC | MI macro F1 | Arrhy macro F1 | Folder |
|-----|------|-----------|-----------|-------------|----------------|--------|
| FINAL-E01 | 42 | 0.949 | 0.492 | 0.636 | 0.837 | `run_20260508_214025_hybrid-tf-focal-aug` |
| FINAL-E02 | 123 | 0.946 | 0.480 | 0.621 | 0.822 | `run_20260509_004809_hybrid-tf-focal-aug` |
| FINAL-E03 | 2024 | 0.950 | 0.514 | 0.623 | 0.820 | `run_20260509_033053_hybrid-tf-focal-aug` |
| **Mean ± Std** | — | **0.9483 ± 0.0017** | **0.4955 ± 0.0142** | **0.6267 ± 0.0066** | **0.8263 ± 0.0079** | — |"""
text = text.replace(old_s6, new_s6)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
