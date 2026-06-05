"""
03_tune.py
─────────────────────────────────────────────────────────────────
Phase C: Hyperparameter tuning với Optuna.

Tối ưu các siêu tham số:
  - lr, weight_decay
  - d_model, nhead, num_encoder_layers, dim_feedforward
  - dropout
  - batch_size
  - use_focal, focal_gamma (optional)

Metric tối ưu: avg AUROC (arrhythmia + MI) trên val set.

Usage:
    python scripts/03_tune.py                          # 30 trials, 15 epochs mỗi trial
    python scripts/03_tune.py --trials 50 --epochs 20
    python scripts/03_tune.py --trials 20 --epochs 10 --debug
    python scripts/03_tune.py --show-best              # in kết quả trial tốt nhất

Install trước:
    pip install optuna
"""
import os
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Subset
from src.data.preprocessing import PTBXLDataset, collate_fn
from src.models.ecg_transformer import ECGTransformer
from src.models.multitask_head import MultiTaskECGModel
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer

warnings.filterwarnings("ignore")


# ─── Args ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--trials",     type=int,   default=30,  help="Number of Optuna trials")
    p.add_argument("--epochs",     type=int,   default=15,  help="Epochs per trial (short)")
    p.add_argument("--subset",     type=float, default=0.3, help="Fraction of training data to use (runs faster)")
    p.add_argument("--study-name", type=str,   default="ecg_hrv_tuning")
    p.add_argument("--storage",    type=str,   default=None, help="Optuna DB URL (optional)")
    p.add_argument("--show-best",  action="store_true",  help="Print best params from existing study and exit")
    p.add_argument("--debug",      action="store_true",  help="200 samples, 3 epochs, 3 trials")
    return p.parse_args()


# ─── Utilities ───────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    processed = cfg["paths"]["processed"]
    splits    = cfg["paths"]["splits"]
    df           = pd.read_csv(os.path.join(processed, "metadata.csv"), index_col="ecg_id")
    label_matrix = np.load(os.path.join(processed, "label_matrix.npy"))
    hrv_path     = os.path.join(processed, "hrv_matrix.npy")
    hrv_matrix   = np.load(hrv_path) if os.path.exists(hrv_path) else None
    pw_path      = os.path.join(processed, "pos_weights.json")
    pos_weights  = json.load(open(pw_path)) if os.path.exists(pw_path) else {}

    splits_idx = {}
    for split in ["train", "val"]:
        idx_path = os.path.join(splits, f"{split}_indices.npy")
        if os.path.exists(idx_path):
            splits_idx[split] = np.load(idx_path)
        else:
            test_fold = cfg["dataset"]["test_fold"]
            val_fold  = cfg["dataset"]["val_fold"]
            fold_arr  = df["strat_fold"].values
            all_idx   = np.arange(len(df))
            splits_idx["val"]   = all_idx[fold_arr == val_fold]
            splits_idx["train"] = all_idx[(fold_arr != test_fold) & (fold_arr != val_fold)]

    return df, label_matrix, hrv_matrix, pos_weights, splits_idx


def normalize_hrv(hrv_matrix, train_indices):
    train_hrv = hrv_matrix[train_indices]
    mean = np.nanmean(train_hrv, axis=0)
    std  = np.nanstd(train_hrv,  axis=0)
    std  = np.where(std < 1e-6, 1.0, std)
    return (hrv_matrix - mean) / std


def build_loaders(cfg, df, label_matrix, hrv_matrix, pos_weights, splits_idx,
                  batch_size, debug=False):
    raw_path  = cfg["paths"]["raw_data"]
    fs        = cfg["dataset"]["sampling_rate"]
    length    = cfg["dataset"]["signal_length"]
    prep      = cfg.get("preprocessing", {})
    bandpass  = (prep.get("bandpass_low", 0.5), prep.get("bandpass_high", 40.0))
    notch     = prep.get("notch_freq", 50.0)
    normalize = prep.get("normalize", "zscore")

    loaders = {}
    for split, indices in splits_idx.items():
        if debug:
            indices = indices[:200] if split == "train" else indices[:50]

        ds = PTBXLDataset(
            metadata      = df.iloc[indices],
            label_matrix  = label_matrix[indices],
            hrv_matrix    = hrv_matrix[indices] if hrv_matrix is not None else None,
            base_path     = raw_path,
            sampling_rate = fs,
            target_length = length,
            bandpass      = bandpass,
            notch         = notch,
            normalize     = normalize,
            augment       = (split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = 0,
            pin_memory  = torch.cuda.is_available(),
            collate_fn  = collate_fn,
            drop_last   = (split == "train"),
        )
    return loaders


# ─── Objective ───────────────────────────────────────────────────────────────

def objective(trial, cfg, df, label_matrix, hrv_matrix, pos_weights,
              splits_idx, device, n_epochs, subset, debug):
    # ── Sample hyperparameters ──────────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout      = trial.suggest_float("dropout",      0.0, 0.3, step=0.05)

    # Model capacity — nhead must divide d_model
    d_model      = trial.suggest_categorical("d_model", [64, 128, 192, 256])
    nhead_choices = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead        = trial.suggest_categorical("nhead", nhead_choices)
    num_layers   = trial.suggest_int("num_encoder_layers", 2, 6)
    ff_mult      = trial.suggest_categorical("ff_mult", [2, 4])  # dim_feedforward = d_model * ff_mult
    dim_ff       = d_model * ff_mult

    use_focal    = trial.suggest_categorical("use_focal", [False, True])
    focal_gamma  = trial.suggest_float("focal_gamma", 1.0, 3.0) if use_focal else 2.0

    # ── Labels ─────────────────────────────────────────────────────────────
    label_cfgs   = cfg["labels"]
    label_names  = [l["name"] for l in label_cfgs]
    arrhy_idx    = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi_idx       = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    arrhy_names  = [label_names[i] for i in arrhy_idx]
    mi_names     = [label_names[i] for i in mi_idx]
    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    hrv_enabled  = cfg.get("hrv", {}).get("enabled", True)

    # ── Dataloaders (with Subsetting) ───────────────────────────────────────
    # Slice the train indices to speed up each epoch
    tuning_splits = dict(splits_idx)
    if subset < 1.0 and not debug:
        train_len = len(tuning_splits["train"])
        tuning_splits["train"] = tuning_splits["train"][:int(train_len * subset)]

    loaders = build_loaders(
        cfg, df, label_matrix, hrv_matrix, pos_weights,
        tuning_splits, batch_size, debug=debug,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    backbone = ECGTransformer(
        num_leads          = cfg["dataset"]["num_leads"],
        signal_length      = cfg["dataset"]["signal_length"],
        patch_size         = cfg["model"]["patch_size"],
        d_model            = d_model,
        nhead              = nhead,
        num_encoder_layers = num_layers,
        dim_feedforward    = dim_ff,
        dropout            = dropout,
        fs                 = cfg["dataset"]["sampling_rate"],
    )
    model = MultiTaskECGModel(
        backbone               = backbone,
        d_model                = d_model,
        num_arrhythmia_labels  = len(arrhy_idx),
        num_mi_labels          = len(mi_idx),
        hrv_enabled            = hrv_enabled,
        num_hrv_targets        = len(hrv_features),
        head_hidden_dim        = max(32, dim_ff // 4),
        dropout                = dropout,
        fs                     = cfg["dataset"]["sampling_rate"],
    )

    # ── Loss ────────────────────────────────────────────────────────────────
    def _build_pw(names):
        w = [min(pos_weights.get(n, 1.0), 50.0) for n in names]
        return torch.tensor(w, dtype=torch.float32, device=device)

    arrhy_pw = _build_pw(arrhy_names) if not use_focal else None
    mi_pw    = _build_pw(mi_names)    if not use_focal else None

    lw = cfg["training"].get("loss_weights", {})
    loss_fn = MultiTaskLoss(
        arrhythmia_weight     = lw.get("arrhythmia", 1.0),
        mi_weight             = lw.get("mi",         1.0),
        imi_weight            = lw.get("imi",        1.0),
        asmi_weight           = lw.get("asmi",       1.0),
        hrv_weight            = lw.get("hrv",        0.1),
        mi_contrastive_weight = cfg["training"].get("mi_contrastive_weight", 0.0),
        hrv_enabled           = hrv_enabled,
        arrhythmia_pos_weight = arrhy_pw,
        mi_pos_weight         = mi_pw,
        use_focal             = use_focal,
        focal_gamma           = focal_gamma,
        focal_alpha           = 0.25,
    )

    # ── Trainer config override ─────────────────────────────────────────────
    trial_cfg = {
        k: v for k, v in cfg.items()
    }
    trial_cfg["training"] = dict(cfg["training"])
    trial_cfg["training"].update({
        "lr":                       lr,
        "weight_decay":             weight_decay,
        "batch_size":               batch_size,
        "epochs":                   n_epochs,
        "scheduler":                "cosine",
        "warmup_epochs":            0,
        "early_stopping_patience":  n_epochs + 1,  # disable early stopping in short trials
        "save_every":               9999,          # no checkpoint saves during tuning
    })

    trainer = Trainer(
        model                    = model,
        loss_fn                  = loss_fn,
        train_loader             = loaders["train"],
        val_loader               = loaders["val"],
        cfg                      = trial_cfg,
        device                   = device,
        arrhythmia_label_indices = arrhy_idx,
        mi_label_indices         = mi_idx,
        arrhythmia_label_names   = arrhy_names,
        mi_label_names           = mi_names,
        hrv_feature_names        = hrv_features,
        optuna_trial             = trial,
    )

    # ── Train ───────────────────────────────────────────────────────────────
    # Suppress checkpoint saves during tuning
    trainer.checkpoint_dir = None

    trainer.train()

    # ── Score = avg AUROC across both task groups ───────────────────────────
    val_metrics = trainer.history["val"][-1] if trainer.history["val"] else {}
    arrhy_auroc = val_metrics.get("auroc/arrhy/macro", 0.0)
    mi_auroc    = val_metrics.get("auroc/mi/macro",    0.0)
    score       = (arrhy_auroc + mi_auroc) / 2.0

    print(f"  Trial {trial.number:3d} | score={score:.4f} | "
          f"arrhy={arrhy_auroc:.4f} mi={mi_auroc:.4f} | "
          f"lr={lr:.2e} d_model={d_model} nhead={nhead} layers={num_layers} "
          f"bs={batch_size} dropout={dropout:.2f} focal={use_focal}")

    return score


# ─── Patch Trainer to skip checkpointing ─────────────────────────────────────

_orig_save = None

def _patch_trainer_no_save(trainer_instance):
    """Monkey-patch save_checkpoint to no-op during tuning."""
    import src.training.trainer as trainer_mod
    global _orig_save
    if _orig_save is None:
        _orig_save = trainer_mod.save_checkpoint
    trainer_mod.save_checkpoint = lambda *a, **kw: None


def _unpatch_trainer():
    import src.training.trainer as trainer_mod
    global _orig_save
    if _orig_save is not None:
        trainer_mod.save_checkpoint = _orig_save


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("❌ Optuna not installed. Run:  pip install optuna")
        sys.exit(1)

    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if args.debug:
        args.trials = 3
        args.epochs = 3
        print("⚡ DEBUG mode: 3 trials, 3 epochs each")

    # ── Load & normalize data ────────────────────────────────────────────────
    print("► Loading data ...")
    df, label_matrix, hrv_matrix, pos_weights, splits_idx = load_data(cfg)
    if hrv_matrix is not None:
        hrv_matrix = normalize_hrv(hrv_matrix, splits_idx["train"])

    # ── Create / load study ──────────────────────────────────────────────────
    study = optuna.create_study(
        study_name   = args.study_name,
        direction    = "maximize",
        storage      = args.storage,
        load_if_exists = True,
        sampler      = optuna.samplers.TPESampler(seed=42),
        pruner       = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    if args.show_best:
        if len(study.trials) == 0:
            print("No trials found in study.")
        else:
            best = study.best_trial
            print(f"\nBest trial: #{best.number}  score={best.value:.4f}")
            print("\nBest hyperparameters:")
            for k, v in best.params.items():
                print(f"  {k}: {v}")
        return

    # ── Patch away checkpoint saves ──────────────────────────────────────────
    _patch_trainer_no_save(None)

    print(f"\n► Starting Optuna tuning: {args.trials} trials × {args.epochs} epochs each")
    print(f"  Study: {args.study_name}")
    print(f"  Metric: avg AUROC (arrhy + MI)\n")

    study.optimize(
        lambda trial: objective(
            trial, cfg, df, label_matrix, hrv_matrix, pos_weights,
            splits_idx, device, args.epochs, args.subset, args.debug,
        ),
        n_trials   = args.trials,
        show_progress_bar = False,
    )

    _unpatch_trainer()

    # ── Results ──────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial: #{best.number}  avg AUROC = {best.value:.4f}")
    print(f"{'='*60}")
    print("\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save best params to JSON
    results = {
        "best_value":  best.value,
        "best_trial":  best.number,
        "best_params": best.params,
        "n_trials":    len(study.trials),
    }
    out_path = os.path.join(cfg["paths"]["checkpoints"], "tune_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")

    print("\n► To train with best params, run:")
    p = best.params
    dim_ff = p.get("d_model", 128) * p.get("ff_mult", 2)
    print(f"  python scripts/02_train.py --epochs 50 "
          f"--lr {p.get('lr', 1e-4):.2e} "
          + (f"--use-focal --focal-gamma {p.get('focal_gamma', 2.0):.1f} " if p.get("use_focal") else "")
    )
    print(f"  (also update config.yaml: d_model={p.get('d_model')}, "
          f"nhead={p.get('nhead')}, layers={p.get('num_encoder_layers')}, "
          f"dim_feedforward={dim_ff}, dropout={p.get('dropout'):.2f})")


if __name__ == "__main__":
    main()
