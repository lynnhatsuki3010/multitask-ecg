"""
02_train.py
─────────────────────────────────────────────────────────────────
Step 2: Train the multi-task ECG Transformer.

Requires 01_build_metadata.py to have been run first.

Usage:
    python scripts/02_train.py
    python scripts/02_train.py --config configs/config.yaml
    python scripts/02_train.py --no-hrv          # disable HRV task
    python scripts/02_train.py --epochs 10       # override epochs
    python scripts/02_train.py --batch-size 32
    python scripts/02_train.py --debug           # 200 samples only, 3 epochs
"""
import os
import sys
import argparse
import random
import json
import time
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import PTBXLDataset, collate_fn
from src.models.ecg_transformer import ECGTransformer
from src.models.multitask_head import MultiTaskECGModel
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Multi-Task ECG Transformer")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--no-hrv",     action="store_true", help="Disable HRV regression task")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--debug",      action="store_true", help="Quick debug run (200 samples, 3 epochs)")
    p.add_argument("--resume",     type=str,   default=None, help="Path to checkpoint .pth to resume from")
    return p.parse_args()


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Using CPU (no CUDA available)")
    return dev


def load_splits_and_data(cfg: dict, args) -> dict:
    """Load metadata, label matrix, HRV matrix, and split indices."""
    processed = cfg["paths"]["processed"]
    splits    = cfg["paths"]["splits"]

    meta_path = os.path.join(processed, "metadata.csv")
    lm_path   = os.path.join(processed, "label_matrix.npy")
    hrv_path  = os.path.join(processed, "hrv_matrix.npy")
    pw_path   = os.path.join(processed, "pos_weights.json")

    for p in [meta_path, lm_path]:
        if not os.path.exists(p):
            print(f"\n❌ Missing: {p}")
            print("   Please run:  python scripts/01_build_metadata.py  first.\n")
            sys.exit(1)

    df           = pd.read_csv(meta_path, index_col="ecg_id")
    label_matrix = np.load(lm_path)
    hrv_matrix   = np.load(hrv_path) if os.path.exists(hrv_path) else None
    pos_weights  = json.load(open(pw_path)) if os.path.exists(pw_path) else {}

    split_indices = {}
    for split in ["train", "val", "test"]:
        idx_path = os.path.join(splits, f"{split}_indices.npy")
        if os.path.exists(idx_path):
            split_indices[split] = np.load(idx_path)
        else:
            print(f"⚠ Missing split file: {idx_path}  — will derive from strat_fold")
            test_fold = cfg["dataset"]["test_fold"]
            val_fold  = cfg["dataset"]["val_fold"]
            fold_arr  = df["strat_fold"].values
            all_idx   = np.arange(len(df))
            split_indices["test"]  = all_idx[fold_arr == test_fold]
            split_indices["val"]   = all_idx[fold_arr == val_fold]
            split_indices["train"] = all_idx[(fold_arr != test_fold) & (fold_arr != val_fold)]

    return {
        "df":           df,
        "label_matrix": label_matrix,
        "hrv_matrix":   hrv_matrix,
        "pos_weights":  pos_weights,
        "splits":       split_indices,
    }


def build_datasets(cfg: dict, data: dict, args) -> dict:
    """Build PTBXLDataset objects for each split."""
    raw_path   = cfg["paths"]["raw_data"]
    fs         = cfg["dataset"]["sampling_rate"]
    length     = cfg["dataset"]["signal_length"]
    prep_cfg   = cfg.get("preprocessing", {})
    bandpass   = (prep_cfg.get("bandpass_low", 0.5), prep_cfg.get("bandpass_high", 40.0))
    notch      = prep_cfg.get("notch_freq", 50.0)
    normalize  = prep_cfg.get("normalize", "zscore")

    df           = data["df"]
    label_matrix = data["label_matrix"]
    hrv_matrix   = data["hrv_matrix"]
    splits       = data["splits"]

    datasets = {}
    for split, indices in splits.items():
        if args.debug and split == "train":
            indices = indices[:200]
        elif args.debug and split == "val":
            indices = indices[:50]

        sub_df  = df.iloc[indices]
        sub_lm  = label_matrix[indices]
        sub_hrv = hrv_matrix[indices] if hrv_matrix is not None else None

        datasets[split] = PTBXLDataset(
            metadata     = sub_df,
            label_matrix = sub_lm,
            hrv_matrix   = sub_hrv,
            base_path    = raw_path,
            sampling_rate= fs,
            target_length= length,
            bandpass     = bandpass,
            notch        = notch,
            normalize    = normalize,
            augment      = (split == "train"),
        )
        print(f"  {split:6s}: {len(datasets[split]):6d} samples")

    return datasets


def build_loaders(datasets: dict, cfg: dict, args) -> dict:
    """Build DataLoaders for each split."""
    train_cfg  = cfg.get("training", {})
    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    num_workers = 0 if args.debug else train_cfg.get("num_workers", 4)
    pin_memory  = torch.cuda.is_available() and train_cfg.get("pin_memory", True)

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            collate_fn  = collate_fn,
            drop_last   = (split == "train"),
        )
    return loaders


def build_pos_weight_tensors(pos_weights: dict, cfg: dict, device: torch.device):
    """Kept for future use when enabling pos_weight trick."""
    pass



# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Overrides from CLI
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.no_hrv:
        cfg["hrv"]["enabled"] = False
    if args.debug:
        cfg["training"]["epochs"] = 3
        print("⚡ DEBUG mode: 200 train / 50 val samples, 3 epochs")

    set_seed(cfg["training"].get("seed", 42))
    device = get_device()

    print(f"\n{'='*60}")
    print("  ECG Multi-Task Transformer — Training")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("► Loading preprocessed data ...")
    data = load_splits_and_data(cfg, args)

    # ── Build label index lists ───────────────────────────────────────────────
    label_cfgs     = cfg["labels"]
    label_names    = [l["name"] for l in label_cfgs]
    arrhy_indices  = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_indices     = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    arrhy_names    = [label_names[i] for i in arrhy_indices]
    mi_names       = [label_names[i] for i in mi_indices]
    hrv_features   = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    hrv_enabled    = cfg.get("hrv", {}).get("enabled", True)

    print(f"  Arrhythmia labels [{len(arrhy_names)}]: {arrhy_names}")
    print(f"  MI labels         [{len(mi_names)}]: {mi_names}")
    print(f"  HRV enabled: {hrv_enabled}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("\n► Building datasets ...")
    datasets = build_datasets(cfg, data, args)
    loaders  = build_loaders(datasets, cfg, args)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n► Building model ...")
    model_cfg = cfg["model"]
    backbone = ECGTransformer(
        num_leads          = cfg["dataset"]["num_leads"],
        signal_length      = cfg["dataset"]["signal_length"],
        patch_size         = model_cfg["patch_size"],
        d_model            = model_cfg["d_model"],
        nhead              = model_cfg["nhead"],
        num_encoder_layers = model_cfg["num_encoder_layers"],
        dim_feedforward    = model_cfg["dim_feedforward"],
        dropout            = model_cfg["dropout"],
    )
    model = MultiTaskECGModel(
        backbone               = backbone,
        d_model                = model_cfg["d_model"],
        num_arrhythmia_labels  = len(arrhy_indices),
        num_mi_labels          = len(mi_indices),
        hrv_enabled            = hrv_enabled,
        num_hrv_targets        = len(hrv_features),
        head_hidden_dim        = model_cfg["dim_feedforward"] // 4,
        dropout                = model_cfg["dropout"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    if args.resume:
        print(f"\n► Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # ── Loss (baseline: plain BCE, no pos_weight, no label smoothing) ────────
    lw = cfg["training"].get("loss_weights", {})

    loss_fn = MultiTaskLoss(
        arrhythmia_weight = lw.get("arrhythmia", 1.0),
        mi_weight         = lw.get("mi",         1.0),
        hrv_weight        = lw.get("hrv",        0.1),
        hrv_enabled       = hrv_enabled,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model                    = model,
        loss_fn                  = loss_fn,
        train_loader             = loaders["train"],
        val_loader               = loaders["val"],
        cfg                      = cfg,
        device                   = device,
        arrhythmia_label_indices = arrhy_indices,
        mi_label_indices         = mi_indices,
        arrhythmia_label_names   = arrhy_names,
        mi_label_names           = mi_names,
        hrv_feature_names        = hrv_features,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n► Loading best model for test evaluation ...")
    best_ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("  Loaded best_model.pth")
    else:
        print("  best_model.pth not found — using last model state")

    print("\n► Test set evaluation:")
    test_metrics = trainer.evaluate(loaders["test"])

    # Save test metrics
    test_metrics_path = os.path.join(cfg["paths"]["checkpoints"], "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Test metrics saved → {test_metrics_path}")


if __name__ == "__main__":
    main()
