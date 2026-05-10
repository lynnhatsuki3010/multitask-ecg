"""
11_finetune_georgia.py
─────────────────────────────────────────────────────────────────
Fine-tune the best PTB-XL model on the Georgia dataset.
Only trains the MI head to learn Ischaemia mapping, while freezing
the backbone to preserve its excellent general feature extraction.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model
from src.training.trainer import Trainer

class GeorgiaFinetuneDataset(Dataset):
    def __init__(self, features_path, labels_path, indices):
        self.X = np.load(features_path) # (N, 5000, 12)
        self.df = pd.read_csv(labels_path)
        self.indices = indices
        
        # Define the exact label order matching the training model
        self.arrhy_names = ["NORM", "AFIB", "STACH", "PVC", "AFLT"]
        self.mi_names = ["IMI", "ASMI"]
        
        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32)
        self.Y_mi = self.df[self.mi_names].values.astype(np.float32)
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, i):
        idx = self.indices[i]
        
        # Shape: (5000, 12)
        sig = self.X[idx].copy()
        
        # Z-score normalization (per lead)
        sig_mean = np.mean(sig, axis=0)
        sig_std = np.std(sig, axis=0) + 1e-8
        sig = (sig - sig_mean) / sig_std
        
        # Convert (5000, 12) -> (12, 5000) for PyTorch 1D convolutions
        sig = torch.from_numpy(sig.T).float()
        
        # For Trainer, we need "labels" as a concatenated vector [arrhy, mi]
        y_a = torch.from_numpy(self.Y_arrhy[idx])
        y_m = torch.from_numpy(self.Y_mi[idx])
        full_labels = torch.cat([y_a, y_m])
        
        return {
            "signal": sig, 
            "labels": full_labels,
            "hrv": torch.zeros(3),
            "hrv_valid": torch.tensor(False, dtype=torch.bool)
        }

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        l_arrhy = self.bce(preds["arrhythmia"], targets["arrhythmia"])
        l_mi = self.bce(preds["mi"], targets["mi"])
        loss = l_arrhy + l_mi
        return {
            "total": loss,
            "arrhythmia": l_arrhy,
            "mi": l_mi
        }

def main():
    checkpoint_path = "checkpoints/run_20260508_185741_hybrid-tf-focal-aug/best_model.pth"
    config_path = "configs/experiments/xval_georgia_finetune.yaml"
    
    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data Splits
    processed_dir = "data/processed_georgia"
    splits_dir = "data/splits_georgia"
    
    train_idx = np.load(os.path.join(splits_dir, "train_indices.npy"))
    val_idx = np.load(os.path.join(splits_dir, "val_indices.npy"))
    
    train_ds = GeorgiaFinetuneDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        train_idx
    )
    val_ds = GeorgiaFinetuneDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        val_idx
    )
    
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. Build Model
    cfg["hrv"] = {"enabled": False}
    model = build_model(cfg, num_arrhythmia_labels=5, num_mi_labels=2, num_hrv_targets=3)
    
    # 3. Load Pretrained Weights
    print(f"Loading pretrained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # 4. Freeze Backbone and Arrhythmia Head! (Only train MI Head)
    model.freeze_for_phase2()
    
    # 5. Initialize Trainer
    loss_fn = MultiTaskLoss()
    
    # Pass 0-4 for arrhythmia, 5-6 for mi to Trainer
    arrhy_indices = [0, 1, 2, 3, 4]
    mi_indices = [5, 6]
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        arrhythmia_label_indices=arrhy_indices,
        mi_label_indices=mi_indices,
        arrhythmia_label_names=["NORM", "AFIB", "STACH", "PVC", "AFLT"],
        mi_label_names=["IMI", "ASMI"],
    )
    
    # 6. Run Fine-Tuning
    print("\nStarting Phase 2 Fine-Tuning on Georgia Dataset...")
    trainer.train()

if __name__ == "__main__":
    main()
