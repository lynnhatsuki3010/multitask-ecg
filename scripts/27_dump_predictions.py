"""
27_dump_predictions.py
----------------------
Save one run's logits so every CV analysis can run later without the weights.

Writes <run>/predictions.npz with
  val_idx, test_idx                    row indices into metadata.csv
  val_<task>, test_<task>              raw logits per task (arrhythmia, mi, conduction)
  test_mi_occ_V5V6, test_mi_occ_V3V4   MI logits with those leads set to 0
  temperature                          [T_arrhythmia, T_mi, T_conduction]
  thresholds, label_names              per-label thresholds in flat label order

Zeroing a lead is the operation the lead-dropout augmentation uses in training.
Run it after 03_calibrate.py, on the machine that holds best_model.pth.

Usage:
  python scripts/27_dump_predictions.py --dir checkpoints/run_...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import PTBXLDataset  # noqa: E402
from src.models.factory import build_model        # noqa: E402

TASKS = ("arrhythmia", "mi", "conduction")
# lead order I II III aVR aVL aVF V1..V6
OCCLUSIONS = {"V5V6": [10, 11], "V3V4": [8, 9]}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()
run = Path(args.dir)

cfg = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
cfg["hrv"] = {"enabled": False}
fs = cfg["dataset"]["sampling_rate"]
pr = cfg["preprocessing"]
proc = ROOT / f"{cfg['paths']['processed']}_{fs}hz"
splits = ROOT / f"{cfg['paths']['splits']}_{fs}hz"
df = pd.read_csv(proc / "metadata.csv")
lm = np.load(proc / "label_matrix.npy")

g = cfg["label_groups"]
names = {l["index"]: l["name"] for l in cfg["labels"]}
flat = [names[i] for t in TASKS for i in g[t]]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg, len(g["arrhythmia"]), len(g["mi"]), 0, len(g["conduction"])).to(dev).eval()
ck = torch.load(str(run / "best_model.pth"), map_location=dev, weights_only=False)
model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)), strict=True)
cal = json.loads((run / "calibration_results.json").read_text(encoding="utf-8"))


def signals(idx):
    ds = PTBXLDataset(metadata=df.iloc[idx], label_matrix=lm[idx], hrv_matrix=None,
                      base_path=str(ROOT / cfg["paths"]["raw_data"]), sampling_rate=fs,
                      target_length=cfg["dataset"]["signal_length"],
                      bandpass=(pr["bandpass_low"], pr["bandpass_high"]), notch=pr["notch_freq"],
                      normalize=pr["normalize"], augment=False)
    return torch.stack([ds[i]["signal"] for i in range(len(ds))])


@torch.no_grad()
def logits(X, zero=None):
    out = {t: [] for t in TASKS}
    for i in range(0, len(X), args.batch_size):
        xb = X[i:i + args.batch_size].to(dev)
        if zero:
            xb = xb.clone()
            xb[:, zero, :] = 0.0
        o = model(xb)
        for t in TASKS:
            out[t].append(o[t].float().cpu().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


save = {}
for split in ("val", "test"):
    idx = np.load(splits / f"{split}_indices.npy")
    X = signals(idx)
    save[f"{split}_idx"] = idx
    for t, z in logits(X).items():
        save[f"{split}_{t}"] = z
    if split == "test":
        for name, leads in OCCLUSIONS.items():
            save[f"test_mi_occ_{name}"] = logits(X, leads)["mi"]

save["temperature"] = np.array([cal["temperature"].get(t, 1.0) for t in TASKS], dtype=np.float64)
save["thresholds"] = np.array([cal["thresholds"].get(n, 0.5) for n in flat], dtype=np.float64)
save["label_names"] = np.array(flat)
np.savez_compressed(run / "predictions.npz", **save)
print(f"saved {run / 'predictions.npz'}  val={len(save['val_idx'])} test={len(save['test_idx'])}")
