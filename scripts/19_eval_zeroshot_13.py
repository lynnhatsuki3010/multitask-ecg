"""
19_eval_zeroshot_13.py
----------------------
Zero-shot evaluation of a 13-label model on an external dataset.

Replaces 09_eval_zeroshot.py, which hard-codes 5 arrhythmia + 2 MI labels and no
conduction head - shapes that no longer match the model and make load_state_dict
fail. Here the label counts come from the checkpoint's own config_snapshot.yaml, so
the script follows the model rather than the other way round.

External sets carry a subset of the 13 labels (Georgia has 11, PTB has 7); only the
columns actually present in labels.csv are scored, and the rest are reported as
absent rather than silently counted as negatives.

Thresholds are taken from the source domain (calibration_results.json when present,
otherwise 0.5). Nothing is tuned on the target - that is the point of the
evaluation, and the paper has to be able to say so.

Usage:
  python scripts/19_eval_zeroshot_13.py --checkpoint checkpoints/<run>/best_model.pth \
      --data data/processed_georgia --name Georgia
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ExternalECG(Dataset):
    """(N, 5000, 12) float array + per-lead z-score, matching the training pipeline."""

    def __init__(self, features_path: str):
        self.X = np.load(features_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sig = np.asarray(self.X[idx], dtype=np.float32)
        sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)
        return torch.from_numpy(sig.T)          # (12, 5000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="processed dir with features.npy + labels.csv")
    ap.add_argument("--name", default="external")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint).parent
    cfg = yaml.safe_load((ckpt_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    cfg["hrv"] = {"enabled": False}

    groups = cfg["label_groups"]
    names_by_index = {l["index"]: l["name"] for l in cfg["labels"]}
    order = {task: [names_by_index[i] for i in groups[task]]
             for task in ("arrhythmia", "mi", "conduction")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(groups["arrhythmia"]), len(groups["mi"]), 0,
                        len(groups["conduction"])).to(device).eval()
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)))

    df = pd.read_csv(Path(args.data) / "labels.csv")
    ds = ExternalECG(str(Path(args.data) / "features.npy"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    scores = {t: [] for t in order}
    with torch.no_grad():
        for sig in loader:
            out = model(sig.to(device))
            for t in order:
                scores[t].append(torch.sigmoid(out[t]).cpu().numpy())
    scores = {t: np.concatenate(v) for t, v in scores.items()}

    cal_path = ckpt_dir / "calibration_results.json"
    thresholds = {}
    if cal_path.exists():
        thresholds = json.loads(cal_path.read_text(encoding="utf-8")).get("thresholds", {})

    lines = [f"# Zero-shot — {args.name}", "",
             f"Checkpoint: `{args.checkpoint}`", f"Số bản ghi: {len(ds)}", "",
             "Ngưỡng lấy từ miền nguồn (PTB-XL), **không** tinh chỉnh trên miền đích.", ""]
    absent = []
    for task, cols in order.items():
        rows = []
        for j, name in enumerate(cols):
            if name not in df.columns:
                absent.append(name)
                continue
            y = df[name].values.astype(int)
            if y.sum() == 0:
                absent.append(f"{name} (0 mẫu dương)")
                continue
            s = scores[task][:, j]
            tau = thresholds.get(name, 0.5)
            pred_tau = (s >= tau).astype(int)
            pred_half = (s >= 0.5).astype(int)
            rows.append((name, int(y.sum()), roc_auc_score(y, s), average_precision_score(y, s),
                         f1_score(y, pred_tau, zero_division=0),
                         precision_score(y, pred_tau, zero_division=0),
                         recall_score(y, pred_tau, zero_division=0),
                         f1_score(y, pred_half, zero_division=0), tau))
        if not rows:
            continue
        lines += [f"## {task}", "",
                  "| Nhãn | Support | AUROC | AUPRC | F1@τ | Precision | Recall | F1@0,5 | τ |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |"
                         .format(*r).replace(".", ","))
        m = np.mean([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows], axis=0)
        lines.append("| **Macro** | – | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | – |"
                     .format(*m).replace(".", ","))
        lines.append("")

    if absent:
        lines += [f"**Nhãn không có trong tập này:** {', '.join(absent)}.", ""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
