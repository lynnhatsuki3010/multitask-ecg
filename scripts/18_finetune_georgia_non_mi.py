"""
18_finetune_georgia_non_mi.py
─────────────────────────────────────────────────────────────────
Fine-tune the non-MI task heads of a DIR5 decoupled model on Georgia.

Georgia/PhysioNet-2020 in this project can contain:
  - Arrhythmia labels: NORM, AFIB, STACH, PVC, AFLT
  - MI labels: IMI, ASMI
  - Conduction labels after rebuilding with scripts/08_build_georgia.py:
    LBBB, RBBB, IRBBB, 1AVB

This script fine-tunes Arrhythmia and, when the rebuilt labels.csv has
conduction columns, Conduction. It automatically skips any task whose
label columns are missing.

Protocol:
  - Train on data/splits_georgia/train_indices.npy
  - Select best by mean validation AUPRC over available non-MI tasks
  - Report zero-shot -> fine-tuned metrics on held-out Georgia test split
  - Freeze the whole model except task-specific heads:
      * Arrhythmia: arrhythmia_pool + arrhythmia_head
      * Conduction: conduction_head (only if labels exist)

Usage:
    python scripts/18_finetune_georgia_non_mi.py \
        --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth \
        --epochs 30 --lr 1e-4
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import bandpass_filter, normalize_signal, notch_filter
from src.models.factory import build_model


PROCESSED_DIR = "data/processed_georgia"
SPLITS_DIR = "data/splits_georgia"


class GeorgiaNonMIDataset(Dataset):
    """Raw Georgia features -> training preprocessing -> non-MI labels."""

    def __init__(self, processed_dir, arrhy_pool, cond_pool, preproc, sampling_rate, indices):
        self.X = np.load(os.path.join(processed_dir, "features.npy"))
        self.df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))
        self.indices = indices

        self.arrhy_names = [name for name in arrhy_pool if name in self.df.columns]
        self.cond_names = [name for name in cond_pool if name in self.df.columns]
        self.Y_arrhy = self.df[self.arrhy_names].values.astype(np.float32) if self.arrhy_names else None
        self.Y_cond = self.df[self.cond_names].values.astype(np.float32) if self.cond_names else None

        self.bandpass = (preproc.get("bandpass_low", 0.5), preproc.get("bandpass_high", 40.0))
        self.notch = preproc.get("notch_freq", 50.0)
        self.normalize_method = preproc.get("normalize", "zscore")
        self.fs = float(sampling_rate)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy().astype(np.float32)  # (5000, 12), raw mV

        if self.bandpass is not None:
            sig = bandpass_filter(sig, self.fs, low=self.bandpass[0], high=self.bandpass[1])
        if self.notch is not None:
            sig = notch_filter(sig, self.fs, freq=self.notch)

        sig = np.ascontiguousarray(sig.T)  # (12, 5000)
        sig = normalize_signal(sig, method=self.normalize_method)

        item = {"signal": torch.from_numpy(sig.astype(np.float32))}
        if self.Y_arrhy is not None:
            item["arrhythmia_labels"] = torch.from_numpy(self.Y_arrhy[idx])
        if self.Y_cond is not None:
            item["conduction_labels"] = torch.from_numpy(self.Y_cond[idx])
        return item


def group_label_names(cfg, group):
    idx_to_name = {label["index"]: label["name"] for label in cfg["labels"]}
    return [idx_to_name[i] for i in cfg.get("label_groups", {}).get(group, [])]


def metric_block(y_true, y_score, label_names):
    y_pred = (y_score >= 0.5).astype(int)
    out = {}
    for i, name in enumerate(label_names):
        support = int(y_true[:, i].sum())
        if support == 0:
            out[name] = {
                "auroc": float("nan"),
                "auprc": float("nan"),
                "f1": float("nan"),
                "support": 0,
            }
            continue
        out[name] = {
            "auroc": float(roc_auc_score(y_true[:, i], y_score[:, i])),
            "auprc": float(average_precision_score(y_true[:, i], y_score[:, i])),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "support": support,
        }
    return out


@torch.no_grad()
def evaluate(model, loader, device, tasks, arrhy_cols, cond_cols, arrhy_names, cond_names):
    model.eval()
    buckets = {
        "arrhythmia_true": [],
        "arrhythmia_score": [],
        "conduction_true": [],
        "conduction_score": [],
    }

    for batch in loader:
        sig = batch["signal"].to(device)
        preds = model(sig)

        if "arrhythmia" in tasks:
            logits = preds["arrhythmia"][:, arrhy_cols]
            buckets["arrhythmia_score"].append(torch.sigmoid(logits).cpu().numpy())
            buckets["arrhythmia_true"].append(batch["arrhythmia_labels"].numpy())
        if "conduction" in tasks:
            logits = preds["conduction"][:, cond_cols]
            buckets["conduction_score"].append(torch.sigmoid(logits).cpu().numpy())
            buckets["conduction_true"].append(batch["conduction_labels"].numpy())

    results = {}
    if "arrhythmia" in tasks:
        results["arrhythmia"] = metric_block(
            np.vstack(buckets["arrhythmia_true"]),
            np.vstack(buckets["arrhythmia_score"]),
            arrhy_names,
        )
    if "conduction" in tasks:
        results["conduction"] = metric_block(
            np.vstack(buckets["conduction_true"]),
            np.vstack(buckets["conduction_score"]),
            cond_names,
        )
    return results


def fmt_task(metrics):
    parts = []
    for name, row in metrics.items():
        parts.append(
            f"{name}: AUROC={row['auroc']:.3f} AUPRC={row['auprc']:.3f} "
            f"F1={row['f1']:.3f} (n={row['support']})"
        )
    return "  |  ".join(parts)


def mean_auprc(results):
    values = []
    for task_metrics in results.values():
        for row in task_metrics.values():
            if not np.isnan(row["auprc"]):
                values.append(row["auprc"])
    return float(np.mean(values)) if values else float("nan")


def set_trainable_modules(model, tasks):
    for param in model.parameters():
        param.requires_grad = False

    modules = []
    if "arrhythmia" in tasks:
        modules.extend([model.arrhythmia_pool, model.arrhythmia_head])
    if "conduction" in tasks:
        modules.append(model.conduction_head)

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
    return modules


def set_head_train_mode(model, tasks):
    model.eval()
    if "arrhythmia" in tasks:
        model.arrhythmia_pool.train()
        model.arrhythmia_head.train()
    if "conduction" in tasks:
        model.conduction_head.train()


def print_delta_table(zero_shot, finetuned):
    print("\n" + "=" * 80)
    print("  GEORGIA NON-MI FINE-TUNE RESULT (held-out test)")
    print("=" * 80)
    print(f"{'Task':<11} | {'Label':<6} | {'AUROC (zs->ft)':>18} | {'AUPRC (zs->ft)':>18} | {'F1 (zs->ft)':>14}")
    print("-" * 86)
    for task, task_metrics in finetuned.items():
        for label, ft in task_metrics.items():
            zs = zero_shot[task][label]
            print(
                f"{task:<11} | {label:<6} | {zs['auroc']:.3f} -> {ft['auroc']:.3f}      | "
                f"{zs['auprc']:.3f} -> {ft['auprc']:.3f}      | {zs['f1']:.3f} -> {ft['f1']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="checkpoints/finetune_georgia_non_mi_decoupled")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.checkpoint), "config_snapshot.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arrhy_pool = group_label_names(cfg, "arrhythmia")
    mi_pool = group_label_names(cfg, "mi")
    cond_pool = group_label_names(cfg, "conduction")

    train_idx = np.load(os.path.join(SPLITS_DIR, "train_indices.npy"))
    val_idx = np.load(os.path.join(SPLITS_DIR, "val_indices.npy"))
    test_idx = np.load(os.path.join(SPLITS_DIR, "test_indices.npy"))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    preproc = cfg.get("preprocessing", {})
    sr = cfg.get("dataset", {}).get("sampling_rate", 500)

    def make_ds(indices):
        return GeorgiaNonMIDataset(PROCESSED_DIR, arrhy_pool, cond_pool, preproc, sr, indices)

    train_ds, val_ds, test_ds = make_ds(train_idx), make_ds(val_idx), make_ds(test_idx)
    tasks = []
    if train_ds.arrhy_names:
        tasks.append("arrhythmia")
    if train_ds.cond_names:
        tasks.append("conduction")
    if not tasks:
        raise RuntimeError("No non-MI labels found in Georgia labels.csv.")

    print(f"Georgia arrhythmia labels: {train_ds.arrhy_names}")
    print(f"Georgia conduction labels: {train_ds.cond_names or 'NONE - skipped'}")

    arrhy_cols = [arrhy_pool.index(name) for name in train_ds.arrhy_names]
    cond_cols = [cond_pool.index(name) for name in train_ds.cond_names]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg["hrv"] = {"enabled": False}
    model = build_model(
        cfg,
        num_arrhythmia_labels=len(arrhy_pool),
        num_mi_labels=len(mi_pool),
        num_hrv_targets=3,
        num_conduction_labels=len(cond_pool),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    modules = set_trainable_modules(model, tasks)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    criteria = {}
    if "arrhythmia" in tasks:
        y = train_ds.Y_arrhy
        pos = y.sum(axis=0)
        neg = len(y) - pos
        pw = np.clip(neg / np.maximum(pos, 1), 1.0, 30.0).astype(np.float32)
        print(f"arrhythmia pos_weight {train_ds.arrhy_names} = {pw.tolist()}")
        criteria["arrhythmia"] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    if "conduction" in tasks:
        y = train_ds.Y_cond
        pos = y.sum(axis=0)
        neg = len(y) - pos
        pw = np.clip(neg / np.maximum(pos, 1), 1.0, 30.0).astype(np.float32)
        print(f"conduction pos_weight {train_ds.cond_names} = {pw.tolist()}")
        criteria["conduction"] = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    optimizer = torch.optim.AdamW(
        [p for module in modules for p in module.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    zero_shot = evaluate(model, test_loader, device, tasks, arrhy_cols, cond_cols, train_ds.arrhy_names, train_ds.cond_names)
    for task, metrics in zero_shot.items():
        print(f"\n[Zero-shot TEST] {task}: {fmt_task(metrics)}")

    best_val = -1.0
    best_state = {name: copy.deepcopy(module.state_dict()) for name, module in model.named_modules()
                  if (name in {"arrhythmia_pool", "arrhythmia_head", "conduction_head"})}

    for epoch in range(1, args.epochs + 1):
        set_head_train_mode(model, tasks)
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            sig = batch["signal"].to(device)
            preds = model(sig)
            losses = []
            if "arrhythmia" in tasks:
                y = batch["arrhythmia_labels"].to(device)
                losses.append(criteria["arrhythmia"](preds["arrhythmia"][:, arrhy_cols], y))
            if "conduction" in tasks:
                y = batch["conduction_labels"].to(device)
                losses.append(criteria["conduction"](preds["conduction"][:, cond_cols], y))
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * sig.size(0)
            n_seen += sig.size(0)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, tasks, arrhy_cols, cond_cols, train_ds.arrhy_names, train_ds.cond_names)
        val_score = mean_auprc(val_metrics)
        print(f"Ep {epoch:02d} | loss={running / max(n_seen, 1):.4f} | val mean AUPRC={val_score:.4f}")
        for task, metrics in val_metrics.items():
            print(f"  {task}: {fmt_task(metrics)}")
        if val_score > best_val:
            best_val = val_score
            best_state = {name: copy.deepcopy(module.state_dict()) for name, module in model.named_modules()
                          if (name in {"arrhythmia_pool", "arrhythmia_head", "conduction_head"})}

    for name, module in model.named_modules():
        if name in best_state:
            module.load_state_dict(best_state[name])
    finetuned = evaluate(model, test_loader, device, tasks, arrhy_cols, cond_cols, train_ds.arrhy_names, train_ds.cond_names)
    print_delta_table(zero_shot, finetuned)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict()}, os.path.join(args.out, "best_model.pth"))
    with open(os.path.join(args.out, "finetune_result.json"), "w") as f:
        json.dump(
            {
                "dataset": "georgia",
                "tasks": tasks,
                "best_val_mean_auprc": best_val,
                "zero_shot_test": zero_shot,
                "finetuned_test": finetuned,
                "source_checkpoint": args.checkpoint,
                "note": "Conduction is trained only when Georgia labels.csv contains LBBB/RBBB/IRBBB/1AVB.",
            },
            f,
            indent=2,
        )
    print(f"\n[OK] Saved -> {args.out}")


if __name__ == "__main__":
    main()
