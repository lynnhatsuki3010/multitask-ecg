import json
import os

runs = [
    ("SPLIT-E01", "run_20260506_231153_hybrid-tf"),
    ("SPLIT-E02", "run_20260507_003151_hybrid-tf"),
    ("SPLIT-E03", "run_20260507_013801_hybrid-tf"),
    
    ("NORM-E01", "run_20260507_204106_hybrid-tf"),
    ("NORM-E02", "run_20260507_220132_hybrid-tf"),
    ("NORM-E03", "run_20260507_231720_hybrid-tf"),
    
    ("LOSS-E01", "run_20260508_005129_hybrid-tf"),
    ("LOSS-E02", "run_20260508_015300_hybrid-tf-focal"),
    ("LOSS-E03", "run_20260508_025238_hybrid-tf-focal"),
    
    ("AUG-E01", "run_20260508_071142_hybrid-tf-focal-aug-mxp"),
    ("AUG-E02", "run_20260508_081931_hybrid-tf-focal-aug"),
    ("AUG-E03", "run_20260508_092233_hybrid-tf-focal-aug"),
    ("AUG-E04", "run_20260508_102608_hybrid-tf-focal-aug"),
    
    ("SAMP-E01", "run_20260508_185741_hybrid-tf-focal-aug"),
    ("SAMP-E02", "run_20260508_200759_hybrid-tf-focal-wrs-aug")
]

print(f"{'Experiment':<10} | {'IMI AUROC':<10} | {'IMI AUPRC':<10} | {'MI F1 (mac)':<12} | {'Arrhy F1 (mac)':<14}")
print("-" * 70)

for name, folder in runs:
    path = os.path.join("checkpoints", folder, "test_metrics.json")
    if not os.path.exists(path):
        print(f"{name:<10} | MISSING {path}")
        continue
        
    try:
        with open(path, "r") as f:
            d = json.load(f)
            
            arrhy_f1 = d.get("f1/arrhy/macro", 0)
            mi_f1 = d.get("f1/mi/macro", 0)
            imi_auroc = d.get("auroc/mi/IMI", 0)
            imi_auprc = d.get("auprc/mi/IMI", 0)
            
            print(f"{name:<10} | {imi_auroc:.3f}      | {imi_auprc:.3f}      | {mi_f1:.3f}        | {arrhy_f1:.3f}")
    except Exception as e:
        print(f"{name:<10} | ERROR {e}")
