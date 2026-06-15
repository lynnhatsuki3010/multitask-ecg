"""
04_ensemble_eval.py
E11 ensemble: Arrhythmia+Conduction from multitask (E08) + MI from MI-only model.

Usage:
    python scripts/04_ensemble_eval.py \
        --multitask-dir checkpoints/run_20260612_170141_multi_branch-tf-aug \
        --mi-dir checkpoints/run_YYYYMMDD_HHMMSS_mi_only-tf-aug
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import PTBXLDataset, collate_fn
from src.models.factory import build_model
from src.utils.metrics import (
    apply_thresholds,
    compute_classification_metrics,
    find_optimal_thresholds,
)


class TemperatureScaler(nn.Module):
    def __init__(self, initial_t: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_t)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-3)


def load_data_from_cfg(cfg: dict):
    fs = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits_dir = f"{cfg['paths']['splits']}_{fs}hz"
    df = pd.read_csv(os.path.join(processed, "metadata.csv"), index_col="ecg_id")
    label_matrix = np.load(os.path.join(processed, "label_matrix.npy"))
    hrv_path = os.path.join(processed, "hrv_matrix.npy")
    hrv_matrix = np.load(hrv_path) if os.path.exists(hrv_path) else None
    split_indices = {}
    for split in ["train", "val", "test"]:
        idx_path = os.path.join(splits_dir, f"{split}_indices.npy")
        split_indices[split] = np.load(idx_path)
    return df, label_matrix, hrv_matrix, split_indices


def build_loader(cfg, df, label_matrix, hrv_matrix, indices):
    raw_path = cfg["paths"]["raw_data"]
    fs = cfg["dataset"]["sampling_rate"]
    length = cfg["dataset"]["signal_length"]
    prep = cfg.get("preprocessing", {})
    ds = PTBXLDataset(
        metadata=df.iloc[indices],
        label_matrix=label_matrix[indices],
        hrv_matrix=hrv_matrix[indices] if hrv_matrix is not None else None,
        base_path=raw_path,
        sampling_rate=fs,
        target_length=length,
        bandpass=(prep.get("bandpass_low", 0.5), prep.get("bandpass_high", 40.0)),
        notch=prep.get("notch_freq", 50.0),
        normalize=prep.get("normalize", "zscore"),
        augment=False,
    )
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                    num_workers=0, collate_fn=collate_fn)


def optimize_temperature(logits, targets, task_name):
    scaler = TemperatureScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = scaler.temperature.item()
    print(f"  [{task_name}] T = {T:.4f}")
    return T


def load_model_from_dir(checkpoint_dir, device):
    with open(os.path.join(checkpoint_dir, "config_snapshot.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, "best_model.pt")

    label_cfgs = cfg["labels"]
    arrhy_idx = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_idx = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    cond_idx = [l["index"] for l in label_cfgs if l["task"] == "conduction"]
    arrhy_names = [l["name"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_names = [l["name"] for l in label_cfgs if l["task"] == "mi"]
    cond_names = [l["name"] for l in label_cfgs if l["task"] == "conduction"]

    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    model = build_model(cfg, len(arrhy_idx), len(mi_idx), len(hrv_features), len(cond_idx)).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model, arrhy_idx, mi_idx, cond_idx, arrhy_names, mi_names, cond_names


@torch.no_grad()
def extract_logits(model, loader, device, arrhy_idx, mi_idx, cond_idx):
    la, lm, lc, ta, tm, tc = [], [], [], [], [], []
    for batch in loader:
        signal = batch["signal"].to(device)
        labels = batch["labels"].to(device)
        preds = model(signal)
        if arrhy_idx and "arrhythmia" in preds:
            la.append(preds["arrhythmia"].cpu())
            ta.append(labels[:, arrhy_idx].cpu())
        if mi_idx and "mi" in preds:
            lm.append(preds["mi"].cpu())
            tm.append(labels[:, mi_idx].cpu())
        if cond_idx and "conduction" in preds:
            lc.append(preds["conduction"].cpu())
            tc.append(labels[:, cond_idx].cpu())
    return (
        torch.cat(la) if la else None, torch.cat(lm) if lm else None, torch.cat(lc) if lc else None,
        torch.cat(ta) if ta else None, torch.cat(tm) if tm else None, torch.cat(tc) if tc else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--multitask-dir", required=True)
    p.add_argument("--mi-dir", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt_cfg, mt_model, mt_ai, _, mt_ci, arrhy_names, _, cond_names = load_model_from_dir(args.multitask_dir, device)
    _, mi_model, _, mi_mi, _, _, mi_names, _ = load_model_from_dir(args.mi_dir, device)

    df, lm, hrv, splits = load_data_from_cfg(mt_cfg)
    val_loader = build_loader(mt_cfg, df, lm, hrv, splits["val"])
    test_loader = build_loader(mt_cfg, df, lm, hrv, splits["test"])

    print("[+] Extracting validation logits...")
    val_ma, _, val_mc, val_ta, _, val_tc = extract_logits(mt_model, val_loader, device, mt_ai, [], mt_ci)
    _, val_mm, _, _, val_tm, _ = extract_logits(mi_model, val_loader, device, [], mi_mi, [])

    print("[+] Temperature scaling...")
    T_a = optimize_temperature(val_ma, val_ta.float(), "Arrhythmia")
    T_m = optimize_temperature(val_mm, val_tm.float(), "MI")
    T_c = optimize_temperature(val_mc, val_tc.float(), "Conduction") if val_mc is not None else 1.0

    thr_cfg = mt_cfg.get("eval", {}).get("threshold_search", {})
    floor = float(thr_cfg.get("min_threshold", 0.15))
    rare = float(thr_cfg.get("max_rare_threshold", 0.4))
    min_pos = int(thr_cfg.get("min_pos_count", 20))
    beta = thr_cfg.get("class_beta", {})

    thresholds = {}
    thresholds.update(find_optimal_thresholds(
        val_ta.numpy(), torch.sigmoid(val_ma / T_a).numpy(), arrhy_names,
        min_pos_count=min_pos, max_rare_threshold=rare, min_threshold=floor, class_beta=beta))
    thresholds.update(find_optimal_thresholds(
        val_tm.numpy(), torch.sigmoid(val_mm / T_m).numpy(), mi_names,
        min_pos_count=min_pos, max_rare_threshold=rare, min_threshold=floor, class_beta=beta))
    if val_mc is not None:
        thresholds.update(find_optimal_thresholds(
            val_tc.numpy(), torch.sigmoid(val_mc / T_c).numpy(), cond_names,
            min_pos_count=min_pos, max_rare_threshold=rare, min_threshold=floor, class_beta=beta))

    print("[+] Test evaluation...")
    test_ma, _, test_mc, test_ta, _, test_tc = extract_logits(mt_model, test_loader, device, mt_ai, [], mt_ci)
    _, test_mm, _, _, test_tm, _ = extract_logits(mi_model, test_loader, device, [], mi_mi, [])

    pa = torch.sigmoid(test_ma / T_a).numpy()
    pm = torch.sigmoid(test_mm / T_m).numpy()
    pred_a = apply_thresholds(pa, arrhy_names, thresholds)
    pred_m = apply_thresholds(pm, mi_names, thresholds)
    metrics = {}
    metrics.update(compute_classification_metrics(test_ta.numpy(), pa, pred_a, arrhy_names, prefix="arrhy/"))
    metrics.update(compute_classification_metrics(test_tm.numpy(), pm, pred_m, mi_names, prefix="mi/"))
    if test_mc is not None:
        pc = torch.sigmoid(test_mc / T_c).numpy()
        pred_c = apply_thresholds(pc, cond_names, thresholds)
        metrics.update(compute_classification_metrics(test_tc.numpy(), pc, pred_c, cond_names, prefix="cond/"))

    print("\n=== E11 Ensemble (calibrated) ===")
    print(f"  Arrhythmia Macro F1: {metrics.get('f1/arrhy/macro', 0):.4f}")
    print(f"  MI Macro F1:         {metrics.get('f1/mi/macro', 0):.4f}")
    print(f"  IMI AUPRC:           {metrics.get('auprc/mi/IMI', 0):.4f}")
    print(f"  Conduction Macro F1: {metrics.get('f1/cond/macro', 0):.4f}")

    out = args.output or os.path.join(args.multitask_dir, "ensemble_results.json")
    with open(out, "w") as f:
        json.dump({
            "ensemble": "E11",
            "multitask_dir": args.multitask_dir,
            "mi_dir": args.mi_dir,
            "temperatures": {"arrhythmia": T_a, "mi": T_m, "conduction": T_c},
            "thresholds": thresholds,
            "test_metrics": {k: float(v) for k, v in metrics.items()},
        }, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
