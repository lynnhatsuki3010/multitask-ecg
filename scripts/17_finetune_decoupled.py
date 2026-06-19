"""
17_finetune_decoupled.py
─────────────────────────────────────────────────────────────────
Fine-tune the MI head of a DECOUPLED PTB-XL model on an external
dataset (PTB / Georgia) and report the "adaptation ceiling".

Why a dedicated script (vs the legacy 15_finetune_ptb.py):
  - The DIR5 decoupled MI head takes the RAW signal directly
    (`model.mi_head(x)`), independent of the shared backbone, and
    emits 4 labels [IMI, ASMI, ILMI, AMI]. The external datasets only
    have IMI/ASMI, so we supervise just those 2 output columns and
    leave ILMI/AMI untouched.
  - Everything except the MI head is frozen → this isolates how much
    a light, head-only adaptation recovers MI on the target domain.
  - Same preprocessing chain as training (bandpass + notch + robust)
    is applied to the raw external features.

Protocol (no leakage):
  - Train on splits_<ds>/train, select best by mean(val AUPRC),
    report final metrics on the HELD-OUT splits_<ds>/test.
  - Prints zero-shot (epoch 0) vs fine-tuned test metrics side by side.

Usage:
    python scripts/17_finetune_decoupled.py \
        --checkpoint checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug/best_model.pth \
        --dataset ptb --epochs 30 --lr 1e-3
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model
from src.data.preprocessing import bandpass_filter, notch_filter, normalize_signal


DATASET_PATHS = {
    "ptb":     {"processed": "data/processed_ptb",     "splits": "data/splits_ptb"},
    "georgia": {"processed": "data/processed_georgia", "splits": "data/splits_georgia"},
}


class MIFinetuneDataset(Dataset):
    """Raw external features -> training preprocessing -> signal + MI labels."""

    def __init__(self, processed_dir, mi_label_pool, preproc, sampling_rate, indices):
        self.X  = np.load(os.path.join(processed_dir, "features.npy"))   # (N, 5000, 12) raw mV
        self.df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))
        self.mi_names = [c for c in mi_label_pool if c in self.df.columns]
        self.indices  = indices
        self.Y_mi     = self.df[self.mi_names].values.astype(np.float32)

        self.bandpass = (preproc.get("bandpass_low", 0.5), preproc.get("bandpass_high", 40.0))
        self.notch    = preproc.get("notch_freq", 50.0)
        self.normalize_method = preproc.get("normalize", "zscore")
        self.fs = float(sampling_rate)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        sig = self.X[idx].copy().astype(np.float32)        # (5000, 12)
        if self.bandpass is not None:
            sig = bandpass_filter(sig, self.fs, low=self.bandpass[0], high=self.bandpass[1])
        if self.notch is not None:
            sig = notch_filter(sig, self.fs, freq=self.notch)
        sig = np.ascontiguousarray(sig.T)                  # (12, 5000)
        sig = normalize_signal(sig, method=self.normalize_method)
        return {
            "signal":    torch.from_numpy(sig.astype(np.float32)),
            "mi_labels": torch.from_numpy(self.Y_mi[idx]),
        }


def group_label_names(cfg, group):
    idx_to_name = {l["index"]: l["name"] for l in cfg["labels"]}
    return [idx_to_name[i] for i in cfg.get("label_groups", {}).get(group, [])]


@torch.no_grad()
def eval_mi(model, loader, device, mi_col_idx, label_names):
    model.mi_head.eval()
    scores, trues = [], []
    for batch in loader:
        sig = batch["signal"].to(device)
        logits = model.mi_head(sig)[:, mi_col_idx]
        scores.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(batch["mi_labels"].numpy())
    y_score = np.vstack(scores)
    y_true  = np.vstack(trues)
    y_pred  = (y_score >= 0.5).astype(int)
    out = {}
    for i, name in enumerate(label_names):
        sup = int(y_true[:, i].sum())
        if sup == 0:
            out[name] = {"auroc": float("nan"), "auprc": float("nan"),
                         "f1": float("nan"), "support": 0}
            continue
        out[name] = {
            "auroc":   float(roc_auc_score(y_true[:, i], y_score[:, i])),
            "auprc":   float(average_precision_score(y_true[:, i], y_score[:, i])),
            "f1":      float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "support": sup,
        }
    return out


def fmt(metrics, names):
    parts = []
    for n in names:
        m = metrics[n]
        parts.append(f"{n}: AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    return "  |  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="ptb", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None, help="Output dir (default checkpoints/finetune_<ds>_decoupled)")
    args = parser.parse_args()

    ckpt_dir    = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config_snapshot.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arrhy_names = group_label_names(cfg, "arrhythmia")
    mi_names    = group_label_names(cfg, "mi")
    cond_names  = group_label_names(cfg, "conduction")

    paths = DATASET_PATHS[args.dataset]
    processed_dir, splits_dir = paths["processed"], paths["splits"]
    preproc = cfg.get("preprocessing", {})
    sr = cfg.get("dataset", {}).get("sampling_rate", 500)

    train_idx = np.load(os.path.join(splits_dir, "train_indices.npy"))
    val_idx   = np.load(os.path.join(splits_dir, "val_indices.npy"))
    test_idx  = np.load(os.path.join(splits_dir, "test_indices.npy"))
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    def make_ds(idx):
        return MIFinetuneDataset(processed_dir, mi_names, preproc, sr, idx)

    train_ds, val_ds, test_ds = make_ds(train_idx), make_ds(val_idx), make_ds(test_idx)
    ext_mi = train_ds.mi_names                       # external labels present (e.g. IMI, ASMI)
    mi_col_idx = [mi_names.index(n) for n in ext_mi]  # map to model output columns
    print(f"External MI labels: {ext_mi}  -> model cols {mi_col_idx}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Build model + load pretrained weights ──
    cfg["hrv"] = {"enabled": False}
    model = build_model(
        cfg,
        num_arrhythmia_labels=len(arrhy_names),
        num_mi_labels=len(mi_names),
        num_hrv_targets=3,
        num_conduction_labels=len(cond_names),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    # ── Freeze everything except the MI head ──
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mi_head.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable (MI head): {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── pos_weight from train labels (IMI/ASMI) ──
    Y = train_ds.Y_mi
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    pw = np.clip(neg / np.maximum(pos, 1), 1.0, 30.0).astype(np.float32)
    pos_weight = torch.tensor(pw, device=device)
    print(f"pos_weight {ext_mi} = {pw.tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.mi_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ── Zero-shot (pre-finetune) test baseline ──
    zs_test = eval_mi(model, test_loader, device, mi_col_idx, ext_mi)
    print("\n[Zero-shot TEST] " + fmt(zs_test, ext_mi))

    best_val = -1.0
    best_state = copy.deepcopy(model.mi_head.state_dict())
    for ep in range(1, args.epochs + 1):
        model.mi_head.train()
        running = 0.0
        for batch in train_loader:
            sig = batch["signal"].to(device)
            y   = batch["mi_labels"].to(device)
            logits = model.mi_head(sig)[:, mi_col_idx]
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * sig.size(0)
        sched.step()

        val_m = eval_mi(model, val_loader, device, mi_col_idx, ext_mi)
        val_auprc = np.nanmean([val_m[n]["auprc"] for n in ext_mi])
        print(f"Ep {ep:02d} | loss={running/len(train_ds):.4f} | "
              f"val mAUPRC={val_auprc:.4f} | " + fmt(val_m, ext_mi))
        if val_auprc > best_val:
            best_val = val_auprc
            best_state = copy.deepcopy(model.mi_head.state_dict())

    # ── Final test with best MI head ──
    model.mi_head.load_state_dict(best_state)
    ft_test = eval_mi(model, test_loader, device, mi_col_idx, ext_mi)

    print("\n" + "=" * 64)
    print(f"  FINE-TUNE RESULT — {args.dataset.upper()} (held-out test)")
    print("=" * 64)
    print(f"{'Label':<6} | {'AUROC (zs->ft)':>20} | {'AUPRC (zs->ft)':>20} | {'F1 (zs->ft)':>16}")
    print("-" * 72)
    for n in ext_mi:
        z, t = zs_test[n], ft_test[n]
        print(f"{n:<6} | {z['auroc']:.3f} -> {t['auroc']:.3f}        | "
              f"{z['auprc']:.3f} -> {t['auprc']:.3f}        | "
              f"{z['f1']:.3f} -> {t['f1']:.3f}")

    out_dir = args.out or f"checkpoints/finetune_{args.dataset}_decoupled"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "best_model.pth"))
    with open(os.path.join(out_dir, "finetune_result.json"), "w") as f:
        json.dump({"dataset": args.dataset, "best_val_mAUPRC": best_val,
                   "zero_shot_test": zs_test, "finetuned_test": ft_test,
                   "source_checkpoint": args.checkpoint}, f, indent=2)
    print(f"\n[OK] Saved -> {out_dir}")


if __name__ == "__main__":
    main()
