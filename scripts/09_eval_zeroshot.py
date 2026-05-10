"""
09_eval_zeroshot.py
─────────────────────────────────────────────────────────────────
Run zero-shot evaluation of PTB-XL trained model on Georgia dataset.

Usage:
    python scripts/09_eval_zeroshot.py --checkpoint checkpoints/run_20260509_203405_hybrid-tf-focal-aug/best_model.pth
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model

class GeorgiaDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.X = np.load(features_path) # (N, 5000, 12)
        self.df = pd.read_csv(labels_path)
        
        # Define the exact label order matching the training model
        self.arrhy_names = ["NORM", "AFIB", "STACH", "PVC", "AFLT"]
        self.mi_names = ["IMI", "ASMI"]
        
        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32)
        self.Y_mi = self.df[self.mi_names].values.astype(np.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Shape: (5000, 12)
        sig = self.X[idx].copy()
        
        # Z-score normalization (per lead) - crucial for transfer learning!
        sig_mean = np.mean(sig, axis=0)
        sig_std = np.std(sig, axis=0) + 1e-8
        sig = (sig - sig_mean) / sig_std
        
        # Convert (5000, 12) -> (12, 5000) for PyTorch 1D convolutions
        sig = torch.from_numpy(sig.T).float()
        y_a = torch.from_numpy(self.Y_arrhy[idx])
        y_m = torch.from_numpy(self.Y_mi[idx])
        return {"signal": sig, "arrhythmia_labels": y_a, "mi_labels": y_m}

def evaluate(model, loader, device):
    model.eval()
    
    all_y_true_a = []
    all_y_pred_a = []
    
    all_y_true_m = []
    all_y_pred_m = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            sig = batch["signal"].to(device)
            y_a = batch["arrhythmia_labels"].numpy()
            y_m = batch["mi_labels"].numpy()
            
            out = model(sig)
            
            p_a = torch.sigmoid(out["arrhythmia"]).cpu().numpy()
            p_m = torch.sigmoid(out["mi"]).cpu().numpy()
            
            all_y_true_a.append(y_a)
            all_y_pred_a.append(p_a)
            all_y_true_m.append(y_m)
            all_y_pred_m.append(p_m)
            
    y_true_a = np.vstack(all_y_true_a)
    y_pred_a = np.vstack(all_y_pred_a)
    y_true_m = np.vstack(all_y_true_m)
    y_pred_m = np.vstack(all_y_pred_m)
    
    return y_true_a, y_pred_a, y_true_m, y_pred_m

def print_metrics(y_true, y_score, label_names, group_name):
    print(f"\n{'='*40}")
    print(f"{group_name} METRICS")
    print(f"{'='*40}")
    
    threshold = 0.5
    y_pred = (y_score >= threshold).astype(int)
    
    for i, name in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auprc = average_precision_score(y_true[:, i], y_score[:, i])
            f1 = f1_score(y_true[:, i], y_pred[:, i])
            
            print(f"{name:5s} | AUROC: {auc:.3f} | AUPRC: {auprc:.3f} | F1: {f1:.3f}")
        except ValueError:
            print(f"{name:5s} | AUROC: N/A | AUPRC: N/A | F1: N/A (Only 1 class present)")
            
    print(f"\n{group_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--config", default="configs/experiments/xval_georgia_zeroshot.yaml")
    args = parser.parse_args()
    # Instead of a hardcoded config, use the exact config snapshot saved with the checkpoint!
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config_snapshot.yaml")
    print(f"Loading config snapshot from {config_path}...")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load dataset (Override paths)
    processed_path = "data/processed_georgia"
    ds = GeorgiaDataset(
        os.path.join(processed_path, "features.npy"),
        os.path.join(processed_path, "labels.csv")
    )
    # Use batch size from config or default to 64
    batch_size = cfg.get("training", {}).get("batch_size", 64)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Build model
    cfg["hrv"] = {"enabled": False}
    model = build_model(cfg, num_arrhythmia_labels=5, num_mi_labels=2, num_hrv_targets=3).to(device)
    
    # 3. Load weights
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    # 4. Evaluate
    y_true_a, y_score_a, y_true_m, y_score_m = evaluate(model, loader, device)
    
    # 5. Report
    print_metrics(y_true_a, y_score_a, ds.arrhy_names, "ARRHYTHMIA")
    print_metrics(y_true_m, y_score_m, ds.mi_names, "MI (Proxy: Ischaemia)")

if __name__ == "__main__":
    main()
