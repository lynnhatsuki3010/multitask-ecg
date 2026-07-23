"""
03_calibrate.py
Post-hoc Temperature Scaling and Per-class Threshold Tuning.

Usage:
    python scripts/03_calibrate.py --dir checkpoints/run_XXXX

Steps:
    1. Load model & config from checkpoint dir.
    2. Extract logits from Validation set.
    3. Learn Temperature T (separately for Arrhythmia & MI) via L-BFGS.
    4. Tune per-class thresholds tau_c on temperature-scaled Val logits.
    5. Evaluate Test set with calibrated T and tau_c.
    6. Save calibration_results.json to the checkpoint dir.
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.factory import build_model
from src.data.preprocessing import PTBXLDataset, collate_fn
from src.utils.metrics import (
    compute_classification_metrics,
    find_optimal_thresholds,
    class_performance_report,
)


# ─── Temperature Scaler ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T such that:
        calibrated_logits = logits / T
    A good T makes the sigmoid probabilities well-calibrated.
    """
    def __init__(self, initial_t: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_t)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-3)


# ─── Data Loading (mirrors 02_train.py) ──────────────────────────────────────

def load_data_from_cfg(cfg: dict):
    """Load df, label_matrix, hrv_matrix, and split_indices exactly as 02_train.py does."""
    fs        = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits_dir = f"{cfg['paths']['splits']}_{fs}hz"

    meta_path = os.path.join(processed, "metadata.csv")
    lm_path   = os.path.join(processed, "label_matrix.npy")
    hrv_path  = os.path.join(processed, "hrv_matrix.npy")

    if not os.path.exists(meta_path) or not os.path.exists(lm_path):
        raise FileNotFoundError(f"Missing data files in {processed}. Run 01_build_metadata.py first.")

    df           = pd.read_csv(meta_path, index_col="ecg_id")
    label_matrix = np.load(lm_path)
    hrv_matrix   = np.load(hrv_path) if os.path.exists(hrv_path) else None

    split_indices = {}
    for split in ["train", "val", "test"]:
        idx_path = os.path.join(splits_dir, f"{split}_indices.npy")
        if os.path.exists(idx_path):
            split_indices[split] = np.load(idx_path)
        else:
            # Fallback: derive from strat_fold column
            test_fold = cfg["dataset"]["test_fold"]
            val_fold  = cfg["dataset"]["val_fold"]
            fold_arr  = df["strat_fold"].values
            all_idx   = np.arange(len(df))
            split_indices["test"]  = all_idx[fold_arr == test_fold]
            split_indices["val"]   = all_idx[fold_arr == val_fold]
            split_indices["train"] = all_idx[(fold_arr != test_fold) & (fold_arr != val_fold)]
            break

    return df, label_matrix, hrv_matrix, split_indices


def build_loader(cfg: dict, df, label_matrix, hrv_matrix, indices, augment=False):
    """Build a DataLoader for a given split (no augmentation for val/test)."""
    raw_path  = cfg["paths"]["raw_data"]
    fs        = cfg["dataset"]["sampling_rate"]
    length    = cfg["dataset"]["signal_length"]
    prep_cfg  = cfg.get("preprocessing", {})
    bandpass  = (prep_cfg.get("bandpass_low", 0.5), prep_cfg.get("bandpass_high", 40.0))
    notch     = prep_cfg.get("notch_freq", 50.0)
    normalize = prep_cfg.get("normalize", "zscore")

    sub_df  = df.iloc[indices]
    sub_lm  = label_matrix[indices]
    sub_hrv = hrv_matrix[indices] if hrv_matrix is not None else None

    ds = PTBXLDataset(
        metadata      = sub_df,
        label_matrix  = sub_lm,
        hrv_matrix    = sub_hrv,
        base_path     = raw_path,
        sampling_rate = fs,
        target_length = length,
        bandpass      = bandpass,
        notch         = notch,
        normalize     = normalize,
        augment       = augment,
    )

    return DataLoader(
        ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )


# ─── Logit Extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_logits(model, loader, arrhy_idx, mi_idx, cond_idx, device):
    """Run inference and collect raw logits + targets for all samples."""
    model.eval()
    all_logits_arrhy, all_logits_mi, all_logits_cond = [], [], []
    all_targets_arrhy, all_targets_mi, all_targets_cond = [], [], []

    for batch in tqdm(loader, desc="  extracting logits", unit="batch"):
        signal = batch["signal"].to(device)
        labels = batch["labels"].to(device)          # "labels" not "label"

        target_arrhy = labels[:, arrhy_idx]
        target_mi    = labels[:, mi_idx]
        target_cond  = labels[:, cond_idx]

        preds = model(signal)

        all_logits_arrhy.append(preds["arrhythmia"].cpu())
        all_logits_mi.append(preds["mi"].cpu())
        if "conduction" in preds:
            all_logits_cond.append(preds["conduction"].cpu())
        all_targets_arrhy.append(target_arrhy.cpu())
        all_targets_mi.append(target_mi.cpu())
        all_targets_cond.append(target_cond.cpu())

    return (
        torch.cat(all_logits_arrhy, dim=0),
        torch.cat(all_logits_mi, dim=0),
        torch.cat(all_logits_cond, dim=0) if all_logits_cond else None,
        torch.cat(all_targets_arrhy, dim=0),
        torch.cat(all_targets_mi, dim=0),
        torch.cat(all_targets_cond, dim=0),
    )


# ─── Temperature Optimization ────────────────────────────────────────────────

def optimize_temperature(logits: torch.Tensor, targets: torch.Tensor, task_name: str):
    """
    Optimize temperature T on validation logits using L-BFGS.
    Returns the scalar T value.
    """
    scaler    = TemperatureScaler(initial_t=1.5)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

    initial_nll = criterion(logits, targets).item()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    final_nll = criterion(scaler(logits), targets).item()
    T = scaler.temperature.item()
    print(f"  [{task_name}] T = {T:.4f}  (NLL: {initial_nll:.4f} -> {final_nll:.4f})")
    return T


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(checkpoint_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    # ── Load config
    config_path = os.path.join(checkpoint_dir, "config_snapshot.yaml")
    model_path  = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config_snapshot.yaml not found in {checkpoint_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth or best_model.pt not found in {checkpoint_dir}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Build label index lists (same as 02_train.py)
    label_cfgs   = cfg["labels"]
    label_names  = [l["name"] for l in label_cfgs]
    arrhy_indices = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_indices    = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    cond_indices  = [l["index"] for l in label_cfgs if l["task"] == "conduction"]
    arrhy_names   = [label_names[i] for i in arrhy_indices]
    mi_names      = [label_names[i] for i in mi_indices]
    cond_names    = [label_names[i] for i in cond_indices]

    print(f"Arrhythmia labels: {arrhy_names}")
    print(f"MI labels:         {mi_names}")
    print(f"Conduction labels: {cond_names}")

    # ── Load data
    print("\n[+] Loading preprocessed data...")
    df, label_matrix, hrv_matrix, splits = load_data_from_cfg(cfg)

    val_loader  = build_loader(cfg, df, label_matrix, hrv_matrix, splits["val"],  augment=False)
    test_loader = build_loader(cfg, df, label_matrix, hrv_matrix, splits["test"], augment=False)
    print(f"  Val:  {len(splits['val']):5d} samples")
    print(f"  Test: {len(splits['test']):5d} samples")

    # ── Load model
    print("\n[+] Loading model...")
    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    model = build_model(
        cfg=cfg,
        num_arrhythmia_labels=len(arrhy_indices),
        num_mi_labels=len(mi_indices),
        num_conduction_labels=len(cond_indices),
        num_hrv_targets=len(hrv_features),
    ).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded from epoch {epoch}")

    # ── Extract logits
    print("\n[+] Extracting Val logits...")
    val_la, val_lm, val_lc, val_ta, val_tm, val_tc = extract_logits(model, val_loader, arrhy_indices, mi_indices, cond_indices, device)

    print("\n[+] Extracting Test logits...")
    test_la, test_lm, test_lc, test_ta, test_tm, test_tc = extract_logits(model, test_loader, arrhy_indices, mi_indices, cond_indices, device)

    # ── Baseline Test metrics (no calibration)
    print("\n=== Baseline Test Metrics (T=1.0, tau=0.5) ===")
    base_probs_a = torch.sigmoid(test_la).numpy()
    base_probs_m = torch.sigmoid(test_lm).numpy()
    base_pred_a  = (base_probs_a >= 0.5).astype(float)
    base_pred_m  = (base_probs_m >= 0.5).astype(float)
    base_met_a = compute_classification_metrics(test_ta.numpy(), base_probs_a, base_pred_a, arrhy_names, prefix="arrhy/")
    base_met_m = compute_classification_metrics(test_tm.numpy(), base_probs_m, base_pred_m, mi_names,   prefix="mi/")
    print(f"  Arrhythmia Macro F1: {base_met_a.get('f1/arrhy/macro', 0):.4f}")
    print(f"  MI Macro F1:         {base_met_m.get('f1/mi/macro', 0):.4f}")
    print(f"  IMI AUPRC:           {base_met_m.get('auprc/mi/IMI', 0):.4f}")
    
    base_met_c = {}
    if test_lc is not None:
        base_probs_c = torch.sigmoid(test_lc).numpy()
        base_pred_c  = (base_probs_c >= 0.5).astype(float)
        base_met_c = compute_classification_metrics(test_tc.numpy(), base_probs_c, base_pred_c, cond_names, prefix="cond/")
        print(f"  Conduction Macro F1: {base_met_c.get('f1/cond/macro', 0):.4f}")

    # ── Temperature Scaling
    print("\n[+] Optimizing Temperature (T) on Validation Set...")
    T_a = optimize_temperature(val_la, val_ta.float(), "Arrhythmia")
    T_m = optimize_temperature(val_lm, val_tm.float(), "MI")
    T_c = 1.0
    if val_lc is not None:
        T_c = optimize_temperature(val_lc, val_tc.float(), "Conduction")

    # Scale logits
    val_la_s  = val_la  / T_a
    val_lm_s  = val_lm  / T_m
    test_la_s = test_la / T_a
    test_lm_s = test_lm / T_m
    val_lc_s  = val_lc / T_c if val_lc is not None else None
    test_lc_s = test_lc / T_c if test_lc is not None else None

    # ── Threshold Tuning on scaled Val logits
    print("\n[+] Tuning Per-class Thresholds on Scaled Val Set...")
    # Read threshold search params from config
    th_cfg = cfg.get("eval", {}).get("threshold_search", {})

    val_probs_a = torch.sigmoid(val_la_s).numpy()
    val_probs_m = torch.sigmoid(val_lm_s).numpy()

    arrhy_thresholds = find_optimal_thresholds(
        val_ta.numpy(), val_probs_a, arrhy_names,
        min_threshold = th_cfg.get("min_threshold", 0.15),
    )
    mi_thresholds = find_optimal_thresholds(
        val_tm.numpy(), val_probs_m, mi_names,
        min_threshold = th_cfg.get("min_threshold", 0.15),
    )
    cond_thresholds = {}
    if val_lc_s is not None:
        val_probs_c = torch.sigmoid(val_lc_s).numpy()
        cond_thresholds = find_optimal_thresholds(
            val_tc.numpy(), val_probs_c, cond_names,
            min_threshold = th_cfg.get("min_threshold", 0.15),
        )

    print("  Optimal thresholds:")
    for name, tau in {**arrhy_thresholds, **mi_thresholds, **cond_thresholds}.items():
        print(f"    {name}: {tau:.3f}")

    # ── Calibrated Test Metrics
    print("\n=== Calibrated Test Metrics ===")
    test_probs_a = torch.sigmoid(test_la_s).numpy()
    test_probs_m = torch.sigmoid(test_lm_s).numpy()

    test_pred_a = np.stack([
        (test_probs_a[:, i] >= arrhy_thresholds[name]).astype(float)
        for i, name in enumerate(arrhy_names)
    ], axis=1)
    test_pred_m = np.stack([
        (test_probs_m[:, i] >= mi_thresholds[name]).astype(float)
        for i, name in enumerate(mi_names)
    ], axis=1)

    cal_met_a = compute_classification_metrics(test_ta.numpy(), test_probs_a, test_pred_a, arrhy_names, prefix="arrhy/")
    cal_met_m = compute_classification_metrics(test_tm.numpy(), test_probs_m, test_pred_m, mi_names,   prefix="mi/")

    print(f"  Arrhythmia Macro F1: {cal_met_a.get('f1/arrhy/macro', 0):.4f}  (was {base_met_a.get('f1/arrhy/macro', 0):.4f})")
    print(f"  MI Macro F1:         {cal_met_m.get('f1/mi/macro', 0):.4f}  (was {base_met_m.get('f1/mi/macro', 0):.4f})")
    print(f"  IMI AUPRC:           {cal_met_m.get('auprc/mi/IMI', 0):.4f}  (was {base_met_m.get('auprc/mi/IMI', 0):.4f})")

    cal_met_c = {}
    if test_lc_s is not None:
        test_probs_c = torch.sigmoid(test_lc_s).numpy()
        test_pred_c = np.stack([
            (test_probs_c[:, i] >= cond_thresholds[name]).astype(float)
            for i, name in enumerate(cond_names)
        ], axis=1)
        cal_met_c = compute_classification_metrics(test_tc.numpy(), test_probs_c, test_pred_c, cond_names, prefix="cond/")
        print(f"  Conduction Macro F1: {cal_met_c.get('f1/cond/macro', 0):.4f}  (was {base_met_c.get('f1/cond/macro', 0):.4f})")

    # ── Save results
    all_metrics = {}
    for k, v in cal_met_a.items(): all_metrics[k] = v
    for k, v in cal_met_m.items(): all_metrics[k] = v
    for k, v in cal_met_c.items(): all_metrics[k] = v

    results = {
        "temperature": {"arrhythmia": T_a, "mi": T_m, "conduction": T_c},
        "thresholds": {**arrhy_thresholds, **mi_thresholds, **cond_thresholds},
        "baseline_metrics": {
            "f1/arrhy/macro": base_met_a.get("f1/arrhy/macro", 0),
            "f1/mi/macro":    base_met_m.get("f1/mi/macro", 0),
            "auprc/mi/IMI":   base_met_m.get("auprc/mi/IMI", 0),
            "f1/cond/macro":  base_met_c.get("f1/cond/macro", 0),
        },
        "calibrated_metrics": {k: v for k, v in all_metrics.items() if isinstance(v, float)},
    }

    save_path = os.path.join(checkpoint_dir, "calibration_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved calibration results -> {save_path}")

    # ── Phase 4: common vs rare class report + micro/weighted-F1 + bootstrap CI
    print("\n[+] Building class performance report (common vs rare, bootstrap CIs)...")
    class_report = {
        "arrhythmia": class_performance_report(test_ta.numpy(), test_probs_a, test_pred_a, arrhy_names),
        "mi":         class_performance_report(test_tm.numpy(), test_probs_m, test_pred_m, mi_names),
    }
    if test_lc_s is not None:
        class_report["conduction"] = class_performance_report(test_tc.numpy(), test_probs_c, test_pred_c, cond_names)

    for task, rep in class_report.items():
        print(f"  [{task}] micro-F1={rep['f1_micro']:.4f}  weighted-F1={rep['f1_weighted']:.4f}  "
              f"macroF1 common={rep['macro_f1_common']}  rare={rep['macro_f1_rare']}")
        for c in rep["classes"]:
            if c["rare"]:
                lo, hi = c["f1_ci95"]
                print(f"      rare {c['label']:6s} n={c['support']:3d}  F1={c['f1']:.3f} "
                      f"[{lo:.3f}, {hi:.3f}]")

    report_path = os.path.join(checkpoint_dir, "class_report.json")
    with open(report_path, "w") as f:
        json.dump(class_report, f, indent=2)
    print(f"[OK] Saved class report -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc Temperature Scaling for ECG Multi-Task Model")
    parser.add_argument("--dir", type=str, required=True,
                        help="Path to checkpoint directory (e.g. checkpoints/run_20260525_222856_hybrid-tf-aug)")
    args = parser.parse_args()
    main(args.dir)
