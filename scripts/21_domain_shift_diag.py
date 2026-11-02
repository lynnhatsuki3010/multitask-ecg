"""
21_domain_shift_diag.py
-----------------------
Why does MI localisation collapse to 0.462 on Georgia while arrhythmia and
conduction transfer fine? This script separates the candidate causes instead of
arguing about them, and it does so WITHOUT retraining: the checkpoint is frozen
and only the input transform and the calibration change.

Two pipeline defects motivated it, both confirmed in the code that produced the
published zero-shot numbers (19_eval_zeroshot_13.py):

  1. NO FILTERING. Training applies bandpass 0.5-40 Hz + notch before the
     per-lead z-score (preprocessing.load_signal). The zero-shot path applies
     the z-score alone, and neither 08_build_georgia.py nor 12_build_ptb.py
     filters at build time. The model therefore sees baseline wander it was
     never trained on. That drift sits directly on the ST segment and T wave -
     the evidence MI localisation depends on - while QRS timing and width, which
     carry arrhythmia and conduction, are largely immune. The observed pattern
     (arrhythmia 0.899, conduction 0.943, MI 0.462) is what this defect predicts.

  2. NO TEMPERATURE. Source thresholds in calibration_results.json were tuned on
     TEMPERATURE-SCALED probabilities (T_mi ~ 0.33, i.e. much sharper), but the
     zero-shot path thresholds a raw sigmoid against them. A threshold like 0.90
     then almost never fires, which inflates the apparent "threshold does not
     transfer" effect. The same bug was found and fixed in 20_error_analysis.py.

Georgia is recorded in Atlanta, so its mains interference is 60 Hz, not the
50 Hz of PTB-XL. A 50 Hz notch does not remove it. E2 tests that.

IN-DISTRIBUTION vs EXPLORATORY
  The model was trained on per-lead z-scored inputs, so every lead it ever saw
  had unit variance - inter-lead amplitude ratio is already discarded at training
  time. Variants that keep the same normalisation family (E0, E1, E2, E5) are
  fair tests of the trained model. Variants that change it (E3a, E3b, E4) feed
  the model a distribution it never saw; they are reported, but flagged, and a
  gain there is a reason to retrain, not a result on its own.

Usage:
  python scripts/21_domain_shift_diag.py --checkpoint checkpoints/<run>/best_model.pth
  python scripts/21_domain_shift_diag.py --checkpoint ... --variants E0 E1 --datasets georgia
  python scripts/21_domain_shift_diag.py --checkpoint ... --no-source   # skip the control
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import bandpass_filter, notch_filter  # noqa: E402
from src.models.factory import build_model                        # noqa: E402
from src.utils.metrics import find_optimal_thresholds             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FS = 500.0
TASKS = ("arrhythmia", "mi", "conduction")

# name -> (bandpass, notch_hz, norm, in_distribution, description)
VARIANTS: Dict[str, tuple] = {
    "E0":  (None,        None, "perlead_z",  True,  "z-score theo đạo trình (pipeline hiện tại)"),
    "E1":  ((0.5, 40.0), 50.0, "perlead_z",  True,  "+ bandpass 0,5–40 + notch 50 (khớp huấn luyện)"),
    "E2":  ((0.5, 40.0), 60.0, "perlead_z",  True,  "+ bandpass 0,5–40 + notch 60 (điện lưới Mỹ)"),
    "E3a": ((0.5, 40.0), 50.0, "global_z",   False, "z-score toàn bản ghi (giữ tỷ lệ biên độ liên đạo trình)"),
    "E3b": ((0.5, 40.0), 50.0, "perlead_r",  False, "robust median/IQR theo đạo trình"),
    "E4":  ((0.5, 40.0), 50.0, "affine_src", False, "affine theo đạo trình, khớp thống kê miền nguồn"),
    "E5":  ((0.5, 40.0), 50.0, "morphology", False, "slope-only (vi phân, xóa thông tin mức trong từng đạo trình)"),
}


# --------------------------------------------------------------------------
# input transforms - every variant takes and returns (T, 12)
# --------------------------------------------------------------------------

def _perlead_z(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd < 1e-8, 1.0, sd)


def _global_z(x: np.ndarray) -> np.ndarray:
    """One mean/std for the whole record, so inter-lead amplitude ratio survives."""
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def _perlead_robust(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=0, keepdims=True)
    iqr = q75 - q25
    return (x - med) / np.where(iqr < 1e-8, 1.0, iqr)


def _affine_to_source(x: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Map each lead's (median, IQR) onto the source lead's (median, IQR).

    src is (2, 12): row 0 median, row 1 IQR, measured on filtered PTB-XL train.
    Unlike a per-lead z-score this keeps the source's inter-lead amplitude
    structure, which is exactly what a per-lead z-score throws away.
    """
    med = np.median(x, axis=0, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25], axis=0, keepdims=True)
    iqr = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
    return (x - med) / iqr * src[1:2, :] + src[0:1, :]


def _morphology(x: np.ndarray) -> np.ndarray:
    """Slope-only: per-lead z-score, first difference, per-lead z-score again.

    Note on what "morphology-only" can still mean here. The obvious construction -
    z-score per lead then rescale each lead to unit L2 norm - is an identity
    operation, because a z-scored lead already has ||z||_2 = sqrt(T) exactly. That
    is itself the answer to the question: the per-lead z-score the model trains on
    ALREADY discards how large one lead is relative to another, so the trained
    model never had access to inter-lead amplitude in the first place. Testing
    "remove inter-lead amplitude" on top of it removes nothing.

    What is left to strip is the within-lead level, so this variant differentiates:
    ST elevation is a level phenomenon and a derivative destroys it, while QRS
    slope survives. If MI collapses here but arrhythmia and conduction hold, the
    MI evidence is level-based, which is the claim worth testing.

    The inter-lead amplitude question is answered instead by E3a (global z-score,
    which keeps the ratio) against E1 (per-lead, which does not).
    """
    d = np.diff(_perlead_z(x), axis=0, prepend=0.0).astype(np.float32)
    return _perlead_z(d)


def apply_variant(x: np.ndarray, variant: str, src_stats: Optional[np.ndarray]) -> np.ndarray:
    band, notch, norm, _, _ = VARIANTS[variant]
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if band is not None:
        x = np.ascontiguousarray(bandpass_filter(x, FS, low=band[0], high=band[1]))
    if notch is not None:
        x = np.ascontiguousarray(notch_filter(x, FS, freq=notch))
    if norm == "perlead_z":
        x = _perlead_z(x)
    elif norm == "global_z":
        x = _global_z(x)
    elif norm == "perlead_r":
        x = _perlead_robust(x)
    elif norm == "morphology":
        x = _morphology(x)
    elif norm == "affine_src":
        if src_stats is None:
            raise RuntimeError("E4 needs --source stats; run without --no-source once.")
        x = _affine_to_source(x, src_stats)
    return np.ascontiguousarray(x, dtype=np.float32)


class VariantDataset(Dataset):
    def __init__(self, features: np.ndarray, variant: str, src_stats: Optional[np.ndarray]):
        self.X, self.variant, self.src = features, variant, src_stats

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = apply_variant(np.asarray(self.X[i]), self.variant, self.src)
        return torch.from_numpy(x.T)          # (12, T)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_target(data_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    X = np.load(data_dir / "features.npy", mmap_mode="r")
    df = pd.read_csv(data_dir / "labels.csv")
    if len(X) != len(df):
        raise RuntimeError(f"{data_dir}: {len(X)} signals vs {len(df)} label rows")
    return X, df


def build_source_cache(cfg: dict, cache: Path, fold: str = "test") -> Tuple[np.ndarray, pd.DataFrame]:
    """PTB-XL fold as RAW (unfiltered) signals, so it goes through the exact same
    variant code as the target sets. Without this the control would silently use
    a different code path than the thing it is controlling for."""
    import wfdb
    fs = cfg["dataset"]["sampling_rate"]
    proc = Path(f"{cfg['paths']['processed']}_{fs}hz")
    splits = Path(f"{cfg['paths']['splits']}_{fs}hz")
    idx = np.load(splits / f"{fold}_indices.npy")
    meta = pd.read_csv(proc / "metadata.csv").iloc[idx].reset_index(drop=True)
    lm = np.load(proc / "label_matrix.npy")[idx]

    names = {l["index"]: l["name"] for l in cfg["labels"]}
    flat = [names[i] for t in TASKS for i in cfg["label_groups"][t]]
    labels = pd.DataFrame(lm, columns=flat)

    sig_path = cache / f"source_{fold}_raw.npy"
    if not sig_path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        col = "filename_hr" if "filename_hr" in meta.columns else "filename"
        target_len = cfg["dataset"]["signal_length"]
        out = np.zeros((len(meta), target_len, 12), dtype=np.float32)
        print(f"  caching {len(meta)} raw source records -> {sig_path}")
        for i, fn in enumerate(meta[col].values):
            rec, _ = wfdb.rdsamp(os.path.join(cfg["paths"]["raw_data"], fn))
            rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            n = min(len(rec), target_len)
            out[i, :n, :] = rec[:n, :12]
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(meta)}")
        np.save(sig_path, out)
    return np.load(sig_path, mmap_mode="r"), labels


def source_lead_stats(cfg: dict, cache: Path, n_sample: int = 300) -> np.ndarray:
    """Per-lead (median, IQR) on FILTERED PTB-XL train, for E4."""
    f = cache / "source_lead_stats.npy"
    if f.exists():
        return np.load(f)
    import wfdb
    fs = cfg["dataset"]["sampling_rate"]
    proc = Path(f"{cfg['paths']['processed']}_{fs}hz")
    splits = Path(f"{cfg['paths']['splits']}_{fs}hz")
    idx = np.load(splits / "train_indices.npy")
    meta = pd.read_csv(proc / "metadata.csv").iloc[idx].reset_index(drop=True)
    col = "filename_hr" if "filename_hr" in meta.columns else "filename"
    rng = np.random.default_rng(20260912)
    pick = rng.choice(len(meta), size=min(n_sample, len(meta)), replace=False)

    pooled = []
    print(f"  measuring source per-lead stats on {len(pick)} filtered train records")
    for i in pick:
        rec, _ = wfdb.rdsamp(os.path.join(cfg["paths"]["raw_data"], meta[col].values[i]))
        x = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)[:, :12]
        x = np.ascontiguousarray(bandpass_filter(x, FS, 0.5, 40.0))
        x = np.ascontiguousarray(notch_filter(x, FS, 50.0))
        pooled.append(x)
    P = np.concatenate(pooled, axis=0)
    q75, q25 = np.percentile(P, [75, 25], axis=0)
    stats = np.stack([np.median(P, axis=0), np.maximum(q75 - q25, 1e-8)]).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(f, stats)
    return stats


# --------------------------------------------------------------------------
# inference + calibration
# --------------------------------------------------------------------------

def collect_logits(model, X, variant, src_stats, device, batch_size, order) -> Dict[str, np.ndarray]:
    """Logits, not probabilities: temperature is applied afterwards, so one
    forward pass serves every calibration mode."""
    loader = DataLoader(VariantDataset(X, variant, src_stats),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    acc = {t: [] for t in order}
    with torch.no_grad():
        for sig in loader:
            out = model(sig.to(device))
            for t in order:
                acc[t].append(out[t].float().cpu().numpy())
    return {t: np.concatenate(v) for t, v in acc.items()}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising BCE, matching 03_calibrate.py's objective."""
    lg = torch.from_numpy(logits.astype(np.float32))
    tg = torch.from_numpy(y.astype(np.float32))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(lg / torch.exp(log_t).clamp(min=1e-3), tg)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(min=1e-3).item())


def metrics_for(y: np.ndarray, p: np.ndarray, tau: float) -> Optional[tuple]:
    if y.sum() == 0 or y.sum() == len(y):
        return None
    pred = (p >= tau).astype(int)
    return (roc_auc_score(y, p), average_precision_score(y, p),
            f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--datasets", nargs="*", default=["georgia", "ptb"])
    ap.add_argument("--no-source", action="store_true", help="skip the PTB-XL control")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache", default=None, help="dir for the raw source cache")
    ap.add_argument("--out", default="docs/domain_shift_diag.md")
    args = ap.parse_args()

    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; pick from {list(VARIANTS)}")

    ckpt = Path(args.checkpoint)
    cfg = yaml.safe_load((ckpt.parent / "config_snapshot.yaml").read_text(encoding="utf-8"))
    cfg["hrv"] = {"enabled": False}
    cache = Path(args.cache) if args.cache else ckpt.parent / "_diag_cache"

    groups = cfg["label_groups"]
    names = {l["index"]: l["name"] for l in cfg["labels"]}
    order = {t: [names[i] for i in groups[t]] for t in TASKS}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(groups["arrhythmia"]), len(groups["mi"]), 0,
                        len(groups["conduction"])).to(device).eval()
    ck = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)))

    cal = json.loads((ckpt.parent / "calibration_results.json").read_text(encoding="utf-8"))
    src_T = cal.get("temperature", {})
    src_tau = cal.get("thresholds", {})
    beta = (cfg.get("eval", {}).get("threshold_search", {}) or {}).get("class_beta", {})

    need_src_stats = "E4" in args.variants
    src_stats = source_lead_stats(cfg, cache) if need_src_stats else None

    sets: List[tuple] = []
    if not args.no_source:
        Xs, Ys = build_source_cache(cfg, cache)
        sets.append(("PTB-XL test (đối chứng nguồn)", Xs, Ys, None))
    for name in args.datasets:
        d = Path(f"data/processed_{name}")
        sp = Path(f"data/splits_{name}")
        if not (d / "features.npy").exists():
            print(f"[bỏ qua] không thấy {d}/features.npy")
            continue
        X, df = load_target(d)
        split = None
        if (sp / "train_indices.npy").exists() and (sp / "test_indices.npy").exists():
            split = (np.load(sp / "train_indices.npy"), np.load(sp / "test_indices.npy"))
        sets.append((name, X, df, split))

    L = ["# Chẩn đoán domain shift", "",
         f"Checkpoint: `{args.checkpoint}`  ·  thiết bị: {device.type}", "",
         "Mô hình đóng băng. Chỉ đổi phép biến đổi đầu vào và cách hiệu chỉnh; "
         "không huấn luyện lại tham số nào.", "",
         "## Các biến thể", "",
         "| Mã | Biến thể | Trong phân phối huấn luyện |", "|---|---|---|"]
    for v in args.variants:
        _, _, _, ind, desc = VARIANTS[v]
        L.append(f"| {v} | {desc} | {'có' if ind else '**không** (chỉ để tham khảo)'} |")
    L += ["", "*Chú thích.* Mô hình được huấn luyện trên đầu vào z-score theo đạo trình, "
          "nên mọi đạo trình nó từng thấy đều có phương sai đơn vị. Các biến thể đổi họ "
          "chuẩn hóa (E3a, E3b, E4) đưa mô hình vào phân phối chưa từng thấy; mức cải "
          "thiện ở đó là lý do để huấn luyện lại, không phải một kết quả tự thân.", ""]

    for ds_name, X, labels, split in sets:
        print(f"\n=== {ds_name} ({len(X)} bản ghi) ===")
        n_tgt = len(split[1]) if split is not None else 0
        L += [f"## {ds_name}", "", f"Số bản ghi: {len(X)}", ""]
        if n_tgt:
            L += [f"Cột F1@τ_đích đo trên split test của miền đích ({n_tgt} bản ghi); "
                  f"nhiệt độ và ngưỡng của nó khớp trên split train. Mọi cột còn lại "
                  f"không dùng nhãn miền đích nên đo trên toàn bộ {len(X)} bản ghi, "
                  f"so sánh trực tiếp được với số trong bài.", ""]

        rows = []
        detail: Dict[str, Dict[str, tuple]] = {}
        for v in args.variants:
            print(f"  [{v}] {VARIANTS[v][4]}")
            logits = collect_logits(model, X, v, src_stats, device, args.batch_size, order)

            tgt_T, tgt_tau = {}, {}
            if split is not None:
                tr, _ = split
                for t in TASKS:
                    cols = [c for c in order[t] if c in labels.columns]
                    if not cols:
                        continue
                    y_tr = labels.loc[tr, cols].values.astype(np.float32)
                    if y_tr.sum() == 0:
                        continue
                    tgt_T[t] = fit_temperature(logits[t][tr][:, [order[t].index(c) for c in cols]], y_tr)
                    p_tr = 1.0 / (1.0 + np.exp(-logits[t][tr][:, [order[t].index(c) for c in cols]] / tgt_T[t]))
                    tgt_tau.update(find_optimal_thresholds(
                        y_tr.astype(int), p_tr, cols, class_beta=beta))

            # Source-calibrated columns use NO target labels, so they are measured on
            # the whole target set - that is what the paper reports and what these
            # numbers have to be comparable with. Only the target-calibrated column
            # is restricted to the held-out target test split, because its threshold
            # was fitted on the target train split.
            ev_all = np.arange(len(X))
            ev_tgt = split[1] if split is not None else ev_all
            per_task = {}
            for t in TASKS:
                acc = {"auroc": [], "auprc": [], "f1_bug": [], "f1_src": [], "f1_tgt": []}
                per_label = {}
                for j, c in enumerate(order[t]):
                    if c not in labels.columns:
                        continue
                    y_all = labels[c].values.astype(int)
                    lg_all = logits[t][:, j]
                    p_raw = 1.0 / (1.0 + np.exp(-lg_all))                  # bug hiện tại
                    p_src = 1.0 / (1.0 + np.exp(-lg_all / float(src_T.get(t, 1.0))))
                    m = metrics_for(y_all[ev_all], p_src[ev_all], float(src_tau.get(c, 0.5)))
                    if m is None:
                        continue
                    acc["auroc"].append(m[0]); acc["auprc"].append(m[1]); acc["f1_src"].append(m[2])
                    mb = metrics_for(y_all[ev_all], p_raw[ev_all], float(src_tau.get(c, 0.5)))
                    acc["f1_bug"].append(mb[2] if mb else np.nan)
                    if c in tgt_tau:
                        p_t = 1.0 / (1.0 + np.exp(-lg_all / tgt_T.get(t, 1.0)))
                        mt = metrics_for(y_all[ev_tgt], p_t[ev_tgt], float(tgt_tau[c]))
                        acc["f1_tgt"].append(mt[2] if mt else np.nan)
                    per_label[c] = (int(y_all[ev_all].sum()), m)
                if acc["auroc"]:
                    per_task[t] = {k: (float(np.nanmean(x)) if x else float("nan"))
                                   for k, x in acc.items()}
                    detail.setdefault(v, {})[t] = per_label
            rows.append((v, per_task))

        L += ["| Biến thể | MI AUROC | MI AUPRC | MI F1@τ_nguồn | MI F1 (không scale, pipeline cũ) | "
              "MI F1@τ_đích | Nhịp AUROC | Dẫn truyền AUROC |",
              "|---|---|---|---|---|---|---|---|"]
        for v, pt in rows:
            g = lambda t, k: pt.get(t, {}).get(k, float("nan"))
            L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                v, g("mi", "auroc"), g("mi", "auprc"), g("mi", "f1_src"), g("mi", "f1_bug"),
                g("mi", "f1_tgt"), g("arrhythmia", "auroc"), g("conduction", "auroc")
            ).replace("nan", "–").replace(".", ","))
        L += ["", "*Chú thích.* AUROC không phụ thuộc ngưỡng nên cột AUROC tách bạch "
              "\"mô hình xếp hạng sai\" với \"ngưỡng đặt sai\". Cột F1 không scale tái lập "
              "đúng pipeline cũ (ngưỡng nguồn áp lên sigmoid chưa chia nhiệt độ); chênh lệch "
              "giữa nó và cột F1@τ_nguồn là phần do lỗi thiếu Temperature Scaling gây ra, "
              "không phải do đổi miền.", ""]

        best = max((r for r in rows if not np.isnan(r[1].get("mi", {}).get("auprc", float("nan")))),
                   key=lambda r: r[1]["mi"]["auprc"], default=None)
        if best is not None and best[0] in detail:
            L += [f"### Chi tiết nhãn — biến thể tốt nhất theo MI AUPRC: {best[0]}", "",
                  "| Nhãn | Support | AUROC | AUPRC | F1 | Precision | Recall |",
                  "|---|---|---|---|---|---|---|"]
            for t in TASKS:
                for c, (sup, m) in detail[best[0]].get(t, {}).items():
                    L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |"
                             .format(c, sup, *m).replace(".", ","))
            L.append("")

    report = "\n".join(L)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\nĐã lưu {args.out}")


if __name__ == "__main__":
    main()
