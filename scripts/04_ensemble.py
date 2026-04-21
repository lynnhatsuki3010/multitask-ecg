"""
04_ensemble.py
─────────────────────────────────────────────────────────────────
Run an ensemble of multiple trained models to improve F1 scores.
It averages the predictions of multiple models and evaluates the
result on the test dataset.

Usage:
    # Option 1: Provide existing run directories
    python scripts/04_ensemble.py --dirs checkpoints/run_A checkpoints/run_B

    # Option 2: Train N models automatically with different seeds then ensemble
    python scripts/04_ensemble.py --train --n-runs 3
"""
import os
import sys
import argparse
import subprocess
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ecg_transformer import ECGTransformer
from src.models.multitask_head import MultiTaskECGModel
from src.utils.metrics import find_optimal_thresholds, compute_classification_metrics, apply_thresholds

import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "scripts/02_train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
plot_confusion_matrices = train_module.plot_confusion_matrices
load_splits_and_data = train_module.load_splits_and_data
build_datasets = train_module.build_datasets
build_loaders = train_module.build_loaders

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", help="List of checkpoint directories to ensemble")
    p.add_argument("--train", action="store_true", help="Train models automatically before ensembling")
    p.add_argument("--n-runs", type=int, default=3, help="Number of models to train if --train is used")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds to use for auto-training")
    p.add_argument("--config", default="configs/config.yaml", help="Config file for auto-training")
    return p.parse_args()

def run_training(seeds, config):
    """Automatically run 02_train.py for the given seeds and return the generated directories."""
    trained_dirs = []
    base_checkpoints = set(glob.glob("checkpoints/run_*"))
    
    for seed in seeds:
        print(f"\n{'='*50}\nTraining model with Seed {seed}\n{'='*50}")
        cmd = [
            sys.executable, "scripts/02_train.py",
            "--config", config,
            "--seed", str(seed),
            "--use-focal", "--weighted-sampler",
            "--aug-baseline-wander", "--aug-lead-dropout", "--aug-time-warp"
        ]
        # execute
        subprocess.check_call(cmd)
        
        # find the newly created dir
        current_checkpoints = set(glob.glob("checkpoints/run_*"))
        new_dirs = current_checkpoints - base_checkpoints
        if new_dirs:
            # get the latest
            latest_dir = max(new_dirs, key=os.path.getmtime)
            trained_dirs.append(latest_dir)
            base_checkpoints.add(latest_dir)
            print(f"-> Model saved in: {latest_dir}")
            
    return trained_dirs

def load_model_from_dir(run_dir, device):
    cfg_path = os.path.join(run_dir, "config_snapshot.yaml")
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        print(f"Skipping {run_dir}: config or best_model.pth missing")
        return None, None
        
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    arrhy_labels = [l for l in cfg["labels"] if l["task"] in ("arrhythmia", "normal")]
    mi_labels = [l for l in cfg["labels"] if l["task"] == "mi"]
    hrv_enabled = cfg.get("hrv", {}).get("enabled", True)
    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    fs = cfg["dataset"]["sampling_rate"]
    
    model_cfg = cfg["model"]
    backbone = ECGTransformer(
        num_leads=cfg["dataset"]["num_leads"],
        signal_length=cfg["dataset"]["signal_length"],
        patch_size=model_cfg.get("patch_size", 25),
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg.get("dim_feedforward", model_cfg["d_model"]*4),
        dropout=model_cfg["dropout"],
        fs=fs,
    )
    model = MultiTaskECGModel(
        backbone=backbone,
        d_model=model_cfg["d_model"],
        num_arrhythmia_labels=len(arrhy_labels),
        num_mi_labels=len(mi_labels),
        hrv_enabled=hrv_enabled,
        num_hrv_targets=len(hrv_features),
        head_hidden_dim=model_cfg.get("dim_feedforward", model_cfg["d_model"]*4) // 4,
        dropout=model_cfg["dropout"],
        mi_lead_dim=model_cfg.get("mi_lead_dim", 64),
        fs=fs,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.to(device)
    model.eval()
    return model, cfg

def get_predictions(model, loader, device):
    a_scores, mi_scores, a_trues, mi_trues = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            labels = batch["labels"]
            preds = model(signal)
            
            a_idx = [i for i, l in enumerate(loader.dataset.metadata["label_names"]) if l in [x["name"] for x in cfg["labels"] if x["task"] in ("arrhythmia", "normal")]]
            mi_idx = [i for i, l in enumerate(loader.dataset.metadata["label_names"]) if l in [x["name"] for x in cfg["labels"] if x["task"] == "mi"]]
            
            # The indices mapping logic is a bit complex, let's use the exact targets like Trainer does
            # We assume labels tensor from batch matches the dataset's label_matrix 7D vector
            
            # Actually, `batch["labels"]` contains the subset of selected columns mapped exactly to 0..6
            pass

def simple_predict(model, loader, device, cfg):
    arrhy_indices = [l["index"] for l in cfg["labels"] if l["task"] in ("arrhythmia", "normal")]
    mi_indices = [l["index"] for l in cfg["labels"] if l["task"] == "mi"]
    
    a_scores, mi_scores, y_trues = [], [], []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            labels = batch["labels"]
            preds = model(signal)
            
            a_scores.append(F.sigmoid(preds["arrhythmia"]).cpu().numpy())
            mi_scores.append(F.sigmoid(preds["mi"]).cpu().numpy())
            y_trues.append(labels.numpy())
            
    return np.concatenate(a_scores), np.concatenate(mi_scores), np.concatenate(y_trues)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dirs = args.dirs or []
    if args.train:
        seeds = args.seeds[:args.n_runs]
        new_dirs = run_training(seeds, args.config)
        dirs.extend(new_dirs)
        
    if not dirs:
        print("No run directories provided and --train not specified. Example:")
        print("  python scripts/04_ensemble.py --dims checkpoints/run_A checkpoints/run_B")
        sys.exit(1)
        
    print(f"\nEnsembling {len(dirs)} models:")
    models = []
    base_cfg = None
    for d in dirs:
        m, c = load_model_from_dir(d, device)
        if m:
            models.append(m)
            if base_cfg is None:
                base_cfg = c
            print(f"  Loaded {d}")
            
    if not models:
        print("No valid models loaded.")
        sys.exit(1)
        
    print("\nLoading data...")
    class MockArgs:
        debug = False
        batch_size = None
        num_workers = None
        pin_memory = None
    mock_args = MockArgs()
    data = load_splits_and_data(base_cfg, mock_args)
    datasets = build_datasets(base_cfg, data, mock_args)
    loaders = build_loaders(datasets, base_cfg, mock_args)
    
    arrhy_names = [l["name"] for l in base_cfg["labels"] if l["task"] in ("arrhythmia", "normal")]
    mi_names = [l["name"] for l in base_cfg["labels"] if l["task"] == "mi"]
    arrhy_indices = [l["index"] for l in base_cfg["labels"] if l["task"] in ("arrhythmia", "normal")]
    mi_indices = [l["index"] for l in base_cfg["labels"] if l["task"] == "mi"]
    
    # Validation Inference
    print("\nRunning inference on VAL set for threshold optimization...")
    all_val_a_scores, all_val_mi_scores = [], []
    for i, m in enumerate(models):
        a_sc, mi_sc, val_true = simple_predict(m, loaders["val"], device, base_cfg)
        all_val_a_scores.append(a_sc)
        all_val_mi_scores.append(mi_sc)
        
    avg_val_a = np.mean(all_val_a_scores, axis=0)
    avg_val_mi = np.mean(all_val_mi_scores, axis=0)
    
    val_true_a = val_true[:, arrhy_indices]
    val_true_mi = val_true[:, mi_indices]
    
    opt_a_thr = find_optimal_thresholds(val_true_a, avg_val_a, arrhy_names)
    opt_mi_thr = find_optimal_thresholds(val_true_mi, avg_val_mi, mi_names)
    thresholds = {**opt_a_thr, **opt_mi_thr}
    print("Optimal ensemble thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.3f}")
        
    # Test Inference
    print("\nRunning inference on TEST set...")
    all_test_a_scores, all_test_mi_scores = [], []
    for i, m in enumerate(models):
        a_sc, mi_sc, test_true = simple_predict(m, loaders["test"], device, base_cfg)
        all_test_a_scores.append(a_sc)
        all_test_mi_scores.append(mi_sc)
        
    avg_test_a = np.mean(all_test_a_scores, axis=0)
    avg_test_mi = np.mean(all_test_mi_scores, axis=0)
    
    test_true_a = test_true[:, arrhy_indices]
    test_true_mi = test_true[:, mi_indices]
    
    test_pred_a = apply_thresholds(avg_test_a, arrhy_names, thresholds)
    test_pred_mi = apply_thresholds(avg_test_mi, mi_names, thresholds)
    
    metrics = {}
    metrics.update(compute_classification_metrics(test_true_a, avg_test_a, test_pred_a, arrhy_names, prefix="arrhy/"))
    metrics.update(compute_classification_metrics(test_true_mi, avg_test_mi, test_pred_mi, mi_names, prefix="mi/"))
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"checkpoints/ensemble_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n=== FINAL ENSEMBLE METRICS ===")
    from src.training.trainer import print_metrics
    print_metrics(metrics)
    
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Plot CM
    # We need to mock a model or modify plot_confusion_matrices
    # Actually plot_confusion_matrices accepts `model`, but we can bypass it by overriding its content,
    # or just saving predictions. We'll simply use print_metrics which prints the detailed classification report.
    print(f"\nDone! Ensemble results saved in {out_dir}")
