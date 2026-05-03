"""
Train the rebuilt phased multitask ECG model.

This script keeps the old training pipeline intact and introduces a new
phase-based workflow:
  - Phase 1: MI-focused bootstrap
  - Phase 2: full multitask refinement
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import PTBXLDataset, collate_fn
from src.data.policy import audit_dataset_policy
from src.models.factory import build_model
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train phased multitask ECG model")
    parser.add_argument("--config", default="configs/phased_multitask_transformer.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-hrv", action="store_true", help="Disable HRV branch for all phases")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Using CPU (no CUDA available)")
    return device


def load_splits_and_data(cfg: dict) -> dict:
    fs = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits = f"{cfg['paths']['splits']}_{fs}hz"

    meta_path = os.path.join(processed, "metadata.csv")
    lm_path = os.path.join(processed, "label_matrix.npy")
    hrv_path = os.path.join(processed, "hrv_matrix.npy")
    pw_path = os.path.join(processed, "pos_weights.json")

    for path in [meta_path, lm_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(meta_path, index_col="ecg_id")
    label_matrix = np.load(lm_path)
    hrv_matrix = np.load(hrv_path) if os.path.exists(hrv_path) else None
    pos_weights = json.load(open(pw_path, "r", encoding="utf-8")) if os.path.exists(pw_path) else {}

    split_indices = {}
    for split in ["train", "val", "test"]:
        idx_path = os.path.join(splits, f"{split}_indices.npy")
        if os.path.exists(idx_path):
            split_indices[split] = np.load(idx_path)
        else:
            test_fold = cfg["dataset"]["test_fold"]
            val_fold = cfg["dataset"]["val_fold"]
            fold_arr = df["strat_fold"].values
            all_idx = np.arange(len(df))
            split_indices["test"] = all_idx[fold_arr == test_fold]
            split_indices["val"] = all_idx[fold_arr == val_fold]
            split_indices["train"] = all_idx[(fold_arr != test_fold) & (fold_arr != val_fold)]
            break

    return {
        "df": df,
        "label_matrix": label_matrix,
        "hrv_matrix": hrv_matrix,
        "pos_weights": pos_weights,
        "splits": split_indices,
        "processed_path": processed,
    }


def normalize_hrv(
    hrv_matrix: np.ndarray,
    train_indices: np.ndarray,
    processed_path: str,
    feature_names: list,
) -> np.ndarray:
    train_hrv = hrv_matrix[train_indices]
    hrv_mean = np.nanmean(train_hrv, axis=0)
    hrv_std = np.nanstd(train_hrv, axis=0)
    hrv_std = np.where(hrv_std < 1e-6, 1.0, hrv_std)

    stats = {"mean": hrv_mean.tolist(), "std": hrv_std.tolist(), "features": feature_names}
    stats_path = os.path.join(processed_path, "hrv_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return (hrv_matrix - hrv_mean) / hrv_std


def build_datasets(cfg: dict, data: dict, debug: bool) -> dict:
    raw_path = cfg["paths"]["raw_data"]
    fs = cfg["dataset"]["sampling_rate"]
    length = cfg["dataset"]["signal_length"]
    prep_cfg = cfg.get("preprocessing", {})
    bandpass = (prep_cfg.get("bandpass_low", 0.5), prep_cfg.get("bandpass_high", 40.0))
    notch = prep_cfg.get("notch_freq", 50.0)
    normalize = prep_cfg.get("normalize", "zscore")

    datasets = {}
    for split, indices in data["splits"].items():
        if debug and split == "train":
            indices = indices[:200]
        elif debug and split == "val":
            indices = indices[:50]
        elif debug and split == "test":
            indices = indices[:50]

        sub_df = data["df"].iloc[indices]
        sub_lm = data["label_matrix"][indices]
        sub_hrv = data["hrv_matrix"][indices] if data["hrv_matrix"] is not None else None

        ds = PTBXLDataset(
            metadata=sub_df,
            label_matrix=sub_lm,
            hrv_matrix=sub_hrv,
            base_path=raw_path,
            sampling_rate=fs,
            target_length=length,
            bandpass=bandpass,
            notch=notch,
            normalize=normalize,
            augment=(split == "train"),
        )
        aug_cfg = cfg.get("augmentation", {})
        if split == "train":
            ds.aug_lead_dropout = aug_cfg.get("aug_lead_dropout", False)
            ds.aug_random_crop = aug_cfg.get("aug_random_crop", False)
            ds.aug_baseline_wander = aug_cfg.get("aug_baseline_wander", False)
            ds.aug_time_warp = aug_cfg.get("aug_time_warp", False)
        datasets[split] = ds
        print(f"  {split:6s}: {len(ds):6d} samples")

    return datasets


def build_weighted_sampler(dataset, max_ratio: float = 5.0):
    from torch.utils.data import WeightedRandomSampler

    label_matrix = dataset.label_matrix
    n_samples = len(label_matrix)
    class_counts = label_matrix.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts

    sample_weights = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        pos_mask = label_matrix[i] > 0
        sample_weights[i] = class_weights[pos_mask].max() if pos_mask.any() else class_weights.min()

    median_w = float(np.median(sample_weights[sample_weights > 0]))
    cap = median_w * max_ratio
    sample_weights = np.clip(sample_weights, 0, cap)

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=n_samples,
        replacement=True,
    )


def build_loaders(datasets: dict, cfg: dict, debug: bool) -> dict:
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = 0 if debug else train_cfg.get("num_workers", 4)
    pin_memory = torch.cuda.is_available() and train_cfg.get("pin_memory", True)
    use_wrs = train_cfg.get("weighted_sampler", False)

    loaders = {}
    for split, ds in datasets.items():
        if split == "train" and use_wrs:
            sampler = build_weighted_sampler(ds)
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                drop_last=(split == "train"),
            )
    return loaders


def build_pos_weight_tensors(
    pos_weights: dict,
    arrhy_names: list,
    mi_names: list,
    device: torch.device,
    max_weight: float,
    power: float,
):
    def _build(names):
        values = []
        for name in names:
            raw_weight = float(pos_weights.get(name, 1.0))
            values.append(min(max(raw_weight, 1.0) ** power, max_weight))
        return torch.tensor(values, dtype=torch.float32, device=device)

    return _build(arrhy_names), _build(mi_names)


def build_loss_fn(cfg: dict, data: dict, arrhy_names: list, mi_names: list, device: torch.device) -> MultiTaskLoss:
    train_cfg = cfg.get("training", {})
    lw = train_cfg.get("loss_weights", {})
    use_pos_weight = train_cfg.get("use_pos_weight", True)

    arrhy_pw, mi_pw = None, None
    if use_pos_weight:
        arrhy_pw, mi_pw = build_pos_weight_tensors(
            data["pos_weights"],
            arrhy_names,
            mi_names,
            device,
            max_weight=float(train_cfg.get("max_pos_weight", 50.0)),
            power=float(train_cfg.get("pos_weight_power", 1.0)),
        )

    return MultiTaskLoss(
        arrhythmia_weight=lw.get("arrhythmia", 1.0),
        mi_weight=lw.get("mi", 1.0),
        imi_weight=lw.get("imi", 1.0),
        asmi_weight=lw.get("asmi", 1.0),
        hrv_weight=lw.get("hrv", 0.1),
        hrv_enabled=cfg.get("hrv", {}).get("enabled", True),
        arrhythmia_pos_weight=arrhy_pw,
        mi_pos_weight=mi_pw,
        use_focal=train_cfg.get("use_focal", False),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        hrv_loss_type=train_cfg.get("hrv_loss", "smooth_l1"),
    )


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def prepare_phase_cfg(base_cfg: dict, phase_cfg: dict, phase_dir: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["paths"]["checkpoints"] = phase_dir
    deep_update(cfg["training"], phase_cfg.get("training", {}))
    return cfg


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def audit_and_save_dataset_policy(cfg: dict, data: dict, run_dir: str) -> None:
    report = audit_dataset_policy(
        cfg,
        data["processed_path"],
        metadata_columns=list(data["df"].columns),
    )
    report_path = os.path.join(run_dir, "dataset_policy_audit.json")
    save_json(report_path, report)

    print("\n► Dataset policy audit ...")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"  [warn] {warning}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"  [error] {error}")
        raise ValueError(
            "Dataset policy audit failed. Rebuild metadata or align config before training."
        )
    print(f"  Audit passed → {report_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.no_hrv:
        cfg["hrv"]["enabled"] = False
    if args.debug:
        cfg["training"]["batch_size"] = min(cfg["training"].get("batch_size", 64), 16)

    phases = cfg.get("phases", [])
    if not phases:
        raise ValueError("Config must define at least one phase under 'phases'.")

    set_seed(cfg["training"].get("seed", 42))
    device = get_device()
    fs = cfg["dataset"]["sampling_rate"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}_phased-mt"
    run_dir = os.path.join(cfg["paths"]["checkpoints"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg_snapshot = os.path.join(run_dir, "config_snapshot.yaml")
    with open(cfg_snapshot, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\n{'=' * 60}")
    print("  Phased ECG Multitask Training")
    print(f"{'=' * 60}")
    print(f"  Run directory: {run_dir}")

    data = load_splits_and_data(cfg)
    audit_and_save_dataset_policy(cfg, data, run_dir)

    label_cfgs = cfg["labels"]
    label_names = [label["name"] for label in label_cfgs]
    arrhy_indices = [label["index"] for label in label_cfgs if label["task"] in ("arrhythmia", "normal")]
    mi_indices = [label["index"] for label in label_cfgs if label["task"] == "mi"]
    arrhy_names = [label_names[i] for i in arrhy_indices]
    mi_names = [label_names[i] for i in mi_indices]
    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])

    print(f"  Arrhythmia labels [{len(arrhy_names)}]: {arrhy_names}")
    print(f"  MI labels         [{len(mi_names)}]: {mi_names}")
    print(f"  HRV enabled: {cfg.get('hrv', {}).get('enabled', True)}")

    if cfg.get("hrv", {}).get("enabled", True) and data["hrv_matrix"] is not None:
        processed_path = f"{cfg['paths']['processed']}_{fs}hz"
        data["hrv_matrix"] = normalize_hrv(
            data["hrv_matrix"],
            data["splits"]["train"],
            processed_path,
            hrv_features,
        )

    print("\n► Building datasets ...")
    datasets = build_datasets(cfg, data, args.debug)

    print("\n► Building model ...")
    model = build_model(
        cfg=cfg,
        num_arrhythmia_labels=len(arrhy_indices),
        num_mi_labels=len(mi_indices),
        num_hrv_targets=len(hrv_features),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: {cfg['model'].get('architecture')}")
    print(f"  Parameters: total={total_params:,} | trainable={trainable_params:,}")

    phase_summaries = []
    final_trainer = None
    final_thresholds = None
    final_phase_dir = None

    for phase_idx, phase in enumerate(phases, start=1):
        phase_name = phase["name"]
        phase_dir = os.path.join(run_dir, phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        phase_cfg = prepare_phase_cfg(cfg, phase, phase_dir)
        with open(os.path.join(phase_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(phase_cfg, f, default_flow_style=False, allow_unicode=True)

        if args.debug:
            phase_cfg["training"]["epochs"] = min(phase_cfg["training"].get("epochs", 3), 3)
            phase_cfg["training"]["early_stopping_patience"] = min(
                phase_cfg["training"].get("early_stopping_patience", 3),
                3,
            )

        if hasattr(model, "configure_phase"):
            model.configure_phase(phase.get("phase_control", {}))

        phase_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'-' * 60}")
        print(f"  Phase {phase_idx}: {phase_name}")
        print(f"  Trainable params this phase: {phase_trainable:,}")
        print(f"{'-' * 60}")

        loaders = build_loaders(datasets, phase_cfg, args.debug)
        loss_fn = build_loss_fn(phase_cfg, data, arrhy_names, mi_names, device)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            cfg=phase_cfg,
            device=device,
            arrhythmia_label_indices=arrhy_indices,
            mi_label_indices=mi_indices,
            arrhythmia_label_names=arrhy_names,
            mi_label_names=mi_names,
            hrv_feature_names=hrv_features,
        )
        trainer.train()

        best_ckpt = os.path.join(phase_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            checkpoint = load_checkpoint(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model"])

        thresholds = trainer.find_thresholds(loaders["val"])
        save_json(os.path.join(phase_dir, "thresholds.json"), thresholds)
        val_metrics = trainer.evaluate(loaders["val"], thresholds=thresholds)
        save_json(os.path.join(phase_dir, "val_metrics.json"), val_metrics)

        phase_summary = {
            "phase": phase_name,
            "trainable_params": phase_trainable,
            "best_monitor_value": trainer.best_monitor_value,
            "monitor_metric": phase_cfg["training"].get("monitor_metric", "macro_f1"),
        }
        save_json(os.path.join(phase_dir, "phase_summary.json"), phase_summary)
        phase_summaries.append(phase_summary)

        final_trainer = trainer
        final_thresholds = thresholds
        final_phase_dir = phase_dir

    if final_trainer is None or final_thresholds is None or final_phase_dir is None:
        raise RuntimeError("No phase completed successfully.")

    print("\n► Final test evaluation ...")
    test_metrics = final_trainer.evaluate(loaders["test"], thresholds=final_thresholds)
    save_json(os.path.join(final_phase_dir, "test_metrics.json"), test_metrics)
    save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    save_json(os.path.join(run_dir, "phase_summary.json"), {"phases": phase_summaries})
    save_json(os.path.join(run_dir, "thresholds.json"), final_thresholds)

    print(f"\n{'=' * 60}")
    print(f"  Finished phased training. Final phase: {phases[-1]['name']}")
    print(f"  Final test metrics saved to: {os.path.join(run_dir, 'test_metrics.json')}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
