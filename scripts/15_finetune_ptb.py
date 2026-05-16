"""
15_finetune_ptb.py
─────────────────────────────────────────────────────────────────
Fine-tune the best PTB-XL model on the PTB Diagnostic ECG Database.

Strategy (identical to Georgia fine-tuning):
  - Freeze Backbone (CNN + Transformer) and Arrhythmia Head completely.
  - Only train MI Head (IMI + ASMI MLP layers) for 15 epochs.
  - This proves that the backbone already contains the correct features
    for MI detection — the head just needs to learn the amplitude/gain
    offset caused by the different ECG machine (1000 Hz vs 500 Hz source).

Key advantage over Georgia:
  - PTB labels are DIRECT matches (IMI = Inferior MI, ASMI = Anterior MI)
    rather than using Ischaemia as a proxy.
  - This is the strongest possible validation of the model's generalizability.

Usage:
    python scripts/15_finetune_ptb.py
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


class PTBFinetuneDataset(Dataset):
    """
    Dataset for PTB fine-tuning. Arrhythmia labels are all 0 because PTB
    doesn't have fine-grained arrhythmia sub-labels. The Arrhythmia Head is
    frozen anyway, so this has zero effect on training.
    """
    def __init__(self, features_path, labels_path, indices):
        self.X       = np.load(features_path)   # (N, 5000, 12)
        self.df      = pd.read_csv(labels_path)
        self.indices = indices

        self.arrhy_names = ["NORM", "AFIB", "STACH", "PVC", "AFLT"]
        self.mi_names    = ["IMI", "ASMI"]

        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32)
        self.Y_mi    = self.df[self.mi_names].values.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy()   # (5000, 12)

        # Per-lead Z-score normalization
        sig_mean = np.mean(sig, axis=0)
        sig_std  = np.std(sig, axis=0) + 1e-8
        sig = (sig - sig_mean) / sig_std

        # (5000, 12) → (12, 5000)
        sig = torch.from_numpy(sig.T).float()

        y_a = torch.from_numpy(self.Y_arrhy[idx])
        y_m = torch.from_numpy(self.Y_mi[idx])
        full_labels = torch.cat([y_a, y_m])   # shape (7,)

        return {
            "signal":    sig,
            "labels":    full_labels,
            "hrv":       torch.zeros(3),
            "hrv_valid": torch.tensor(False, dtype=torch.bool),
        }


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        l_arrhy = self.bce(preds["arrhythmia"], targets["arrhythmia"])
        l_mi    = self.bce(preds["mi"],         targets["mi"])
        loss = l_arrhy + l_mi
        return {"total": loss, "arrhythmia": l_arrhy, "mi": l_mi}


def main():
    checkpoint_path = "checkpoints/run_20260508_185741_hybrid-tf-focal-aug/best_model.pth"
    config_path     = "configs/experiments/cross_validation/xval_ptb_finetune.yaml"

    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load Splits ──
    processed_dir = "data/processed_ptb"
    splits_dir    = "data/splits_ptb"

    train_idx = np.load(os.path.join(splits_dir, "train_indices.npy"))
    val_idx   = np.load(os.path.join(splits_dir, "val_indices.npy"))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_ds = PTBFinetuneDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        train_idx,
    )
    val_ds = PTBFinetuneDataset(
        os.path.join(processed_dir, "features.npy"),
        os.path.join(processed_dir, "labels.csv"),
        val_idx,
    )

    batch_size   = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 2. Build Model ──
    cfg["hrv"] = {"enabled": False}
    model = build_model(cfg, num_arrhythmia_labels=5, num_mi_labels=2, num_hrv_targets=3)

    # ── 3. Load Pretrained Weights ──
    print(f"Loading pretrained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    # ── 4. Freeze Backbone + Arrhythmia Head → only MI Head trains ──
    model.freeze_for_phase2()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)")

    # ── 5. Initialize Trainer ──
    loss_fn = MultiTaskLoss()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        arrhythmia_label_indices=[0, 1, 2, 3, 4],
        mi_label_indices=[5, 6],
        arrhythmia_label_names=["NORM", "AFIB", "STACH", "PVC", "AFLT"],
        mi_label_names=["IMI", "ASMI"],
    )

    # ── 6. Fine-tune ──
    print("\nStarting Phase 2 Fine-Tuning on PTB Dataset...")
    print("  → Backbone + Arrhythmia Head: FROZEN")
    print("  → MI Head (IMI + ASMI):       TRAINING")
    trainer.train()


if __name__ == "__main__":
    main()
