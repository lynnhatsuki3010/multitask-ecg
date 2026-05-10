import json

with open("e:/KLTN/KL Project/ECG_HRV/checkpoints/finetune_georgia/history.json", "r") as f:
    h = json.load(f)

best = max(h["val"], key=lambda x: x.get("auprc/mi/IMI", 0))

print("--- Best Epoch ---")
print(f"Arrhythmia F1 (Tuned): {best.get('tuned/f1/arrhy/macro', 0):.4f}")
print(f"MI F1 (Tuned): {best.get('tuned/f1/mi/macro', 0):.4f}")
print(f"IMI AUPRC: {best.get('auprc/mi/IMI', 0):.4f}")
print(f"IMI AUROC: {best.get('auroc/mi/IMI', 0):.4f}")
print(f"ASMI AUROC: {best.get('auroc/mi/ASMI', 0):.4f}")
