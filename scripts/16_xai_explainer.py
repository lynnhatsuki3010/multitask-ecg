"""
16_xai_explainer.py  —  XAI-EXP1
==================================
Generates per-lead ECG heatmaps using **Gradient x Input** (GxI) saliency.

WHY Gradient x Input?
---------------------
The Hybrid-Transformer backbone fuses all 12 leads into shared channels after
the depthwise→pointwise projection step.  Attention-weight visualisation
(the previous approach) only has access to **temporal** weights over the
fused feature space — so all 12 leads necessarily receive an identical
overlay.

GxI computes  saliency[lead, t] = |dL/d_input[lead, t]  ×  input[lead, t]|
directly on the raw (12, T) input tensor.  Each lead receives its own
independent gradient signal, revealing which time-steps AND which leads
actually drive each diagnostic decision.

Output directory structure
---------------------------
artifacts/xai_v2/
    <LABEL>/                      <- e.g. AFIB, IMI, NORM+PVC ...
        <ecg_id>_gxi.png          <- 12-lead heatmap (GxI)

Usage
------
    python scripts/16_xai_explainer.py \\
        --checkpoint checkpoints/run_20260526_235533_hybrid-tf-aug \\
        --split      test \\
        --max-per-class 5 \\
        --out        artifacts/xai_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import PTBXLDataset, collate_fn
from src.models.factory import build_model

# ── Constants ─────────────────────────────────────────────────────────────────

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
HEAT_CMAP  = plt.cm.YlOrRd   # Yellow → Orange → Red
HEAT_ALPHA = 0.65

# Lead groups for the MI head (used to annotate clinical relevance)
_MI_LEAD_GROUPS = {
    "inferior":   [1, 2, 5],    # II, III, aVF
    "reciprocal": [0, 4],       # I, aVL
    "anterior":   [6, 7, 8, 9], # V1-V4
}

# Lead groups for arrhythmia labels
_ARRHY_LEAD_GROUPS = {
    "rhythm":  [1, 6],          # II, V1  — P wave & flutter wave rõ nhất
    "lateral": [0, 4, 10, 11],  # I, aVL, V5, V6 — AFIB thường rõ ở đây
    "septal":  [7, 8],          # V2, V3 — PVC morphology
}

_MI_LABELS = {"IMI", "ASMI"}


# ── Gradient × Input saliency ─────────────────────────────────────────────────

def gradient_x_input(
    model: torch.nn.Module,
    signal: torch.Tensor,        # (1, 12, T)  already on device, requires_grad set outside
    task: str,                   # "arrhythmia" | "mi"
    class_idx: int,              # index within the task output
    device: torch.device,
) -> np.ndarray:
    """
    Returns absolute GxI saliency map of shape (12, T).

    Steps:
        1. Forward pass — keep graph (no torch.no_grad).
        2. Select scalar output = logit[task][class_idx].
        3. Backward to input.
        4. saliency = |grad * input|, then per-lead min-max normalised.
    """
    model.eval()
    x = signal.clone().detach().requires_grad_(True).to(device)

    out = model(x)                            # forward
    logit = out[task][0, class_idx]           # scalar
    model.zero_grad()
    logit.backward()                          # backward

    grad = x.grad.detach()                    # (1, 12, T)
    saliency = (grad * x.detach()).abs()      # element-wise product
    saliency = saliency[0].cpu().numpy()      # (12, T)

    # Smooth nhẹ thôi để giữ chi tiết QRS
    for i in range(saliency.shape[0]):
        saliency[i] = gaussian_filter1d(saliency[i], sigma=5)

    # Global normalize — giữ tương quan GIỮA các lead
    global_max = saliency.max()
    if global_max > 1e-8:
        saliency = saliency / global_max
        saliency = np.clip(saliency, 0, 1) ** 0.6

    return saliency


def integrated_gradients(
    model: torch.nn.Module,
    signal: torch.Tensor,        # (1, 12, T)
    task: str,
    class_idx: int,
    device: torch.device,
    n_steps: int = 20,
) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al. 2017).
    Baseline = zero signal.  Returns (12, T) saliency.
    Smoother than plain GxI; recommended for publication figures.
    """
    model.eval()
    x = signal.clone().detach().to(device)
    baseline = torch.zeros_like(x)

    integrated_grads = torch.zeros_like(x)
    for step in range(n_steps):
        alpha = step / n_steps
        interp = (baseline + alpha * (x - baseline)).requires_grad_(True)
        out = model(interp)
        logit = out[task][0, class_idx]
        model.zero_grad()
        logit.backward()
        integrated_grads += interp.grad.detach()

    # Riemann approximation × (x - baseline)
    ig = (integrated_grads / n_steps) * (x - baseline)
    saliency = ig.abs()[0].cpu().numpy()      # (12, T)

    # Smooth nhẹ thôi để giữ chi tiết QRS
    for i in range(saliency.shape[0]):
        saliency[i] = gaussian_filter1d(saliency[i], sigma=5)

    # Global normalize — giữ tương quan GIỮA các lead
    global_max = saliency.max()
    if global_max > 1e-8:
        saliency = saliency / global_max
        saliency = np.clip(saliency, 0, 1) ** 0.6

    return saliency


# ── Plotting ──────────────────────────────────────────────────────────────────

def _lead_group_label(lead_idx: int, label: str = "") -> str:
    """Return a short clinical group tag for the lead axis label.

    MI labels (IMI, ASMI) use anatomical MI lead groups (inferior/reciprocal/anterior).
    Arrhythmia labels use rhythm-oriented groups (rhythm/lateral/septal).
    """
    groups = _MI_LEAD_GROUPS if label in _MI_LABELS else _ARRHY_LEAD_GROUPS
    for group, indices in groups.items():
        if lead_idx in indices:
            return f" [{group[:3].upper()}]"
    return ""


def plot_gxi_heatmap(
    saliency: np.ndarray,        # (12, T)  per-lead GxI map
    signal: np.ndarray,          # (12, T)  raw normalised ECG
    subtitle: str,
    fs: int = 500,
    label: str = "",             # label đang được visualize — chọn đúng lead group tag
) -> plt.Figure:
    """
    Creates a 12-subplot figure where each subplot is one lead.
    The heatmap background varies PER LEAD, showing where each
    specific lead contributed to the prediction.
    """
    n_leads, sig_len = signal.shape
    t = np.arange(sig_len) / fs

    fig, axes = plt.subplots(
        n_leads, 1,
        figsize=(24, n_leads * 1.5),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    cmap = HEAT_CMAP
    norm = mcolors.Normalize(0, 1)

    # Global max saliency per lead for title annotation
    lead_importance = saliency.max(axis=1)   # (12,)

    for i, (ax, lead_name) in enumerate(zip(axes, LEAD_NAMES)):
        lead_sig  = signal[i]
        lead_sal  = saliency[i]                     # unique per lead!

        y_min, y_max = lead_sig.min(), lead_sig.max()
        y_pad = (y_max - y_min) * 0.25 + 1e-4

        ax.set_facecolor("white")

        # ── Heatmap overlay (per-lead unique) ─────────────────────────────────
        rgba = cmap(norm(lead_sal))                 # (T, 4)
        rgba[:, 3] = lead_sal * HEAT_ALPHA          # alpha modulated by importance
        rgba_img = rgba.reshape(1, -1, 4)

        ax.imshow(
            rgba_img,
            aspect="auto",
            extent=[t[0], t[-1], y_min - y_pad, y_max + y_pad],
            zorder=1,
            interpolation="bilinear",
        )

        # ── ECG signal line ────────────────────────────────────────────────────
        line_color = "#1a1a2e"
        ax.plot(t, lead_sig, color=line_color, linewidth=0.65, zorder=2, alpha=0.92)

        # ── Lead label with clinical group ─────────────────────────────────────
        group_tag  = _lead_group_label(i, label)
        imp_pct    = int(lead_importance[i] * 100)
        ax.set_ylabel(
            f"{lead_name}{group_tag}\n{imp_pct}%",
            rotation=0,
            labelpad=52,
            fontsize=7.5,
            color="black",
            va="center",
        )
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(colors="black", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    axes[-1].set_xlabel("Time (s)", color="black", fontsize=9)

    fig.suptitle(subtitle, color="black", fontsize=12, y=1.005, fontweight="bold")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.012, pad=0.005)
    cbar.set_label("Saliency", color="black", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    return fig


# ── Data loading ──────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_dir: str):
    """Load config + model from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config_snapshot.yaml"
    model_path  = checkpoint_dir / "best_model.pth"
    if not model_path.exists():
        model_path = checkpoint_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in {checkpoint_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"best_model.pth not found in {checkpoint_dir}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, model_path


def build_dataset(cfg: dict, split: str) -> PTBXLDataset:
    fs        = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits_dir = f"{cfg['paths']['splits']}_{fs}hz"

    meta_path = os.path.join(processed, "metadata.csv")
    lm_path   = os.path.join(processed, "label_matrix.npy")
    hrv_path  = os.path.join(processed, "hrv_matrix.npy")

    df           = pd.read_csv(meta_path, index_col="ecg_id")
    label_matrix = np.load(lm_path)
    hrv_matrix   = np.load(hrv_path) if os.path.exists(hrv_path) else None

    idx_path = os.path.join(splits_dir, f"{split}_indices.npy")
    if os.path.exists(idx_path):
        indices = np.load(idx_path)
    else:
        test_fold = cfg["dataset"]["test_fold"]
        val_fold  = cfg["dataset"]["val_fold"]
        if split == "test":
            indices = df.index[df["strat_fold"] == test_fold].tolist()
        elif split == "val":
            indices = df.index[df["strat_fold"] == val_fold].tolist()
        else:
            indices = df.index[~df["strat_fold"].isin([test_fold, val_fold])].tolist()

    sub_df  = df.loc[indices]
    sub_lm  = label_matrix[df.index.isin(indices)]
    sub_hrv = hrv_matrix[df.index.isin(indices)] if hrv_matrix is not None else None

    norm_method = cfg.get("preprocessing", {}).get("normalize", "robust")
    bandpass    = (
        cfg.get("preprocessing", {}).get("bandpass_low",  0.5),
        cfg.get("preprocessing", {}).get("bandpass_high", 40.0),
    )
    notch = cfg.get("preprocessing", {}).get("notch_freq", 50.0)

    return PTBXLDataset(
        metadata      = sub_df,
        label_matrix  = sub_lm,
        hrv_matrix    = sub_hrv,
        base_path     = cfg["paths"]["raw_data"],
        sampling_rate = fs,
        target_length = cfg["dataset"]["signal_length"],
        bandpass      = bandpass,
        notch         = notch,
        normalize     = norm_method,
        augment       = False,
    )


# ── Label utilities ───────────────────────────────────────────────────────────

def get_label_config(cfg: dict):
    """Return ordered label names for arrhythmia and MI tasks."""
    labels = cfg.get("labels", [])
    arrhy_names = [l["name"] for l in labels if l.get("task") in ("arrhythmia", "normal")]
    mi_names    = [l["name"] for l in labels if l.get("task") == "mi"]
    return arrhy_names, mi_names


def predicted_label_str(
    arrhy_probs: np.ndarray,
    mi_probs: np.ndarray,
    arrhy_names: List[str],
    mi_names: List[str],
    tau_a: float = 0.5,
    tau_m: float = 0.5,
) -> str:
    """Return '+'-joined string of all predicted class names."""
    predicted = []
    for i, p in enumerate(arrhy_probs):
        if p >= tau_a:
            predicted.append(arrhy_names[i])
    for i, p in enumerate(mi_probs):
        if p >= tau_m:
            predicted.append(mi_names[i])
    return "+".join(sorted(predicted)) if predicted else "UNKNOWN"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    cfg, model_path = load_checkpoint(args.checkpoint)
    arrhy_names, mi_names = get_label_config(cfg)
    print(f"Arrhythmia labels : {arrhy_names}")
    print(f"MI labels         : {mi_names}")

    hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])
    model = build_model(
        cfg                  = cfg,
        num_arrhythmia_labels = len(arrhy_names),
        num_mi_labels         = len(mi_names),
        num_hrv_targets       = len(hrv_features),
    ).to(device)

    state = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in state:                    # key chuẩn từ Trainer (02_train.py)
        state = state["model"]
    elif "model_state_dict" in state:       # fallback cho checkpoint format cũ
        state = state["model_state_dict"]
    # else: bare state dict
    model.load_state_dict(state, strict=True)   # strict=True: crash sớm nếu key lệch
    model.eval()
    print(f"[+] Model loaded from {model_path}")
    # Sanity check: probs trên zero input phải lệch xa 0.5
    with torch.no_grad():
        _dummy = torch.zeros(1, 12, cfg["dataset"]["signal_length"]).to(device)
        _out   = model(_dummy)
        _p_arr = torch.sigmoid(_out["arrhythmia"][0]).cpu().numpy().round(3)
        _p_mi  = torch.sigmoid(_out["mi"][0]).cpu().numpy().round(3)
    print(f"[+] Sanity check — zero-input probs  arrhy={_p_arr}  mi={_p_mi}")
    print(f"    (neu tat ca ~0.5 thi weights chua load dung)")

    # ── Load calibration thresholds + temperature if available ──────────────────
    cal_path = Path(args.checkpoint) / "calibration_results.json"
    thresholds: Dict[str, float] = {}
    temp_arrhy = 1.0
    temp_mi    = 1.0
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        thresholds  = cal.get("thresholds", {})
        temp_arrhy  = cal.get("temperature", {}).get("arrhythmia", 1.0)
        temp_mi     = cal.get("temperature", {}).get("mi", 1.0)
        print(f"[+] Temperature : arrhythmia={temp_arrhy:.4f}  mi={temp_mi:.4f}")
        print(f"[+] Thresholds  : {thresholds}")
    else:
        print("[!] No calibration_results.json found — using T=1.0, tau=0.5 for all classes")

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = build_dataset(cfg, args.split)
    fs      = cfg["dataset"]["sampling_rate"]
    print(f"[+] Dataset split '{args.split}': {len(dataset)} samples")

    # ── Output directory ───────────────────────────────────────────────────────
    out_root = Path(args.out)
    if args.clear and out_root.exists():
        import shutil
        shutil.rmtree(out_root)
        print(f"[+] Cleared existing output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Per-label counters ─────────────────────────────────────────────────────
    non_norm_labels = [lbl for lbl in (arrhy_names + mi_names) if lbl != "NORM"]
    tp_counts:    Dict[str, int] = {lbl: 0 for lbl in non_norm_labels}
    fp_counts:    Dict[str, int] = {lbl: 0 for lbl in non_norm_labels}
    fn_counts:    Dict[str, int] = {lbl: 0 for lbl in non_norm_labels}
    combo_counts: Dict[str, int] = {}
    saved_total = 0

    method = args.method   # "gxi" or "ig"

    for sample_idx in range(len(dataset)):
        sample  = dataset[sample_idx]
        signal  = sample["signal"].unsqueeze(0).to(device)   # (1, 12, T)
        ecg_id  = sample["ecg_id"]

        # ── Forward + apply Temperature Scaling (same as calibrate.py) ─────────
        with torch.no_grad():
            out = model(signal)

        arrhy_probs = torch.sigmoid(out["arrhythmia"][0] / temp_arrhy).cpu().numpy()
        mi_probs    = torch.sigmoid(out["mi"][0]         / temp_mi).cpu().numpy()
        all_probs   = np.concatenate([arrhy_probs, mi_probs])
        combined_names = arrhy_names + mi_names

        # Build thresholded label string
        def _get_tau(name: str, default: float) -> float:
            return thresholds.get(name, default)

        # Predictions
        predicted = [lbl for j, lbl in enumerate(combined_names)
                     if all_probs[j] >= _get_tau(lbl, 0.5)]
        pathological = [lbl for lbl in predicted if lbl != "NORM"]

        # Ground truth
        labels_gt = sample["labels"].cpu().numpy()
        gt_labels = {lbl for j, lbl in enumerate(combined_names) if labels_gt[j] > 0.5}
        gt_path = set(gt_labels - {"NORM"})

        pred_set  = set(pathological)
        tp_labels = sorted(pred_set & gt_path)        # predict đúng
        fp_labels = sorted(pred_set - gt_path)        # predict thừa
        fn_labels = sorted(gt_path  - pred_set)       # bỏ sót

        # Track FN counts (không lưu ảnh, chỉ thống kê)
        for lbl in fn_labels:
            if lbl in fn_counts:
                fn_counts[lbl] += 1

        if not tp_labels and not fp_labels and not args.explain_all:
            continue

        # Filename tags
        pred_tag = "+".join(sorted(pred_set)) if pred_set else "NONE"
        gt_tag   = "+".join(sorted(gt_path))  if gt_path else "NORM"
        prob_tag = "_".join(
            f"{lbl}{all_probs[combined_names.index(lbl)]:.3f}"
            for lbl in sorted(pred_set)
        )
        patient_id = dataset.metadata.iloc[sample_idx].get("patient_id", 0)
        file_stem = f"pat{patient_id}_ecg{ecg_id}_pred[{pred_tag}]_gt[{gt_tag}]_{prob_tag}"

        # Check quotas
        tp_done = all(tp_counts.get(lbl, 0) >= args.max_per_class for lbl in tp_labels) if tp_labels else True
        fp_done = all(fp_counts.get(lbl, 0) >= args.max_per_class for lbl in fp_labels) if fp_labels else True

        tp_combo_key = "+".join(sorted(tp_labels)) if len(tp_labels) > 1 else None
        tp_combo_done = combo_counts.get(f"TP_COMBO_{tp_combo_key}", 0) >= args.max_per_class if tp_combo_key else True

        fp_combo_key = "+".join(sorted(pathological)) if fp_labels and len(pathological) > 1 else None
        fp_combo_done = combo_counts.get(f"FP_COMBO_{fp_combo_key}", 0) >= args.max_per_class if fp_combo_key else True

        if tp_done and fp_done and tp_combo_done and fp_combo_done:
            continue
            
        # Helper to compute saliency
        def get_sal(lbl):
            j = combined_names.index(lbl)
            task = "arrhythmia" if j < len(arrhy_names) else "mi"
            idx = j if j < len(arrhy_names) else j - len(arrhy_names)
            if method == "ig":
                return integrated_gradients(model, signal, task, idx, device, n_steps=args.ig_steps)
            else:
                return gradient_x_input(model, signal, task, idx, device)
                
        sal_maps = {}
        for lbl in pathological:
            sal_maps[lbl] = get_sal(lbl)
            
        raw_signal = signal[0].cpu().numpy()
        any_saved = False
        
        # ── 7a. Lưu TP — ảnh đơn nhãn ────────────────────────────────────
        for lbl in tp_labels:
            if tp_counts.get(lbl, 0) >= args.max_per_class: continue

            subtitle = f"[TP] GT: {gt_tag} | Pred: {pred_tag} | Focus: {lbl}"
            fig = plot_gxi_heatmap(sal_maps[lbl], raw_signal, subtitle, fs, label=lbl)

            folder = out_root / "TP" / lbl
            folder.mkdir(parents=True, exist_ok=True)
            fname = f"{file_stem}_map.png"
            fig.savefig(folder / fname, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            tp_counts[lbl] = tp_counts.get(lbl, 0) + 1
            any_saved = True
            print(f"[XAI]  TP {lbl:<8} ({tp_counts[lbl]}/{args.max_per_class})  {fname}")

        # ── 7b. Lưu TP combo (≥2 TP labels) ─────────────────────────────
        if tp_combo_key and not tp_combo_done:
            key = f"TP_COMBO_{tp_combo_key}"
            subtitle = f"[TP-COMBO] GT: {gt_tag} | Pred: {pred_tag} | Focus: {tp_combo_key}"

            combo_sal = np.zeros_like(raw_signal)
            for lbl in tp_labels: combo_sal += sal_maps[lbl]
            combo_sal /= len(tp_labels)

            fig = plot_gxi_heatmap(combo_sal, raw_signal, subtitle, fs, label=tp_labels[0] if len(tp_labels) == 1 else "")
            folder = out_root / "TP" / tp_combo_key
            folder.mkdir(parents=True, exist_ok=True)
            fname = f"{file_stem}_map.png"
            fig.savefig(folder / fname, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            combo_counts[key] = combo_counts.get(key, 0) + 1
            any_saved = True
            print(f"[XAI]  TP_COMBO {tp_combo_key:<15} ({combo_counts[key]}/{args.max_per_class})  {fname}")

        # ── 7c. Lưu FP — ảnh đơn nhãn ────────────────────────────────────
        for lbl in fp_labels:
            if fp_counts.get(lbl, 0) >= args.max_per_class: continue

            subtitle = f"[FP] GT: {gt_tag} | Pred: {pred_tag} | Focus: {lbl}"
            fig = plot_gxi_heatmap(sal_maps[lbl], raw_signal, subtitle, fs, label=lbl)

            folder = out_root / "FP" / lbl
            folder.mkdir(parents=True, exist_ok=True)
            fname = f"{file_stem}_map.png"
            fig.savefig(folder / fname, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            fp_counts[lbl] = fp_counts.get(lbl, 0) + 1
            any_saved = True
            print(f"[XAI]  FP {lbl:<8} ({fp_counts[lbl]}/{args.max_per_class})  {fname}")

        # ── 7d. Lưu FP combo (predict nhiều nhãn, một số sai) ────────────
        if fp_combo_key and not fp_combo_done:
            key = f"FP_COMBO_{fp_combo_key}"
            tp_str = "+".join(tp_labels) if tp_labels else "NONE"
            fp_str = "+".join(fp_labels)
            subtitle = f"[FP-COMBO] GT: {gt_tag} | TP: {tp_str} | FP: {fp_str} | Focus: {fp_combo_key}"

            combo_sal = np.zeros_like(raw_signal)
            for lbl in pathological: combo_sal += sal_maps[lbl]
            combo_sal /= len(pathological)

            fig = plot_gxi_heatmap(combo_sal, raw_signal, subtitle, fs)
            folder = out_root / "FP" / fp_combo_key
            folder.mkdir(parents=True, exist_ok=True)
            fname = f"{file_stem}_map.png"
            fig.savefig(folder / fname, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

            combo_counts[key] = combo_counts.get(key, 0) + 1
            any_saved = True
            print(f"[XAI]  FP_COMBO {fp_combo_key:<15} ({combo_counts[key]}/{args.max_per_class})  {fname}")

        if any_saved:
            saved_total += 1
            
        if saved_total >= args.max_total:
            print(f"[XAI] Reached max_total={args.max_total}. Stopping.")
            break

    print(f"\n[DONE] Saved {saved_total} heatmaps to: {out_root}")

    # ── Summary table (matches old version format) ─────────────────────────
    print(f"\n{'Label':<12} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 34)
    for lbl in sorted(non_norm_labels):
        print(f"  {lbl:<10} {tp_counts.get(lbl, 0):>6} {fp_counts.get(lbl, 0):>6} {fn_counts.get(lbl, 0):>6}")

    if combo_counts:
        print(f"\n{'Combo':<35} {'Count':>5}")
        print("-" * 43)
        for combo, n in sorted(combo_counts.items()):
            print(f"  {combo:<33}  {n:>5}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG XAI Explainer — Gradient x Input (XAI-EXP1)")
    parser.add_argument("--checkpoint",    required=True,  help="Checkpoint directory (contains best_model.pth + config_snapshot.yaml)")
    parser.add_argument("--split",         default="test", choices=["train", "val", "test"], help="Dataset split to explain")
    parser.add_argument("--max-per-class", type=int, default=5,  help="Maximum samples to save per predicted label combo")
    parser.add_argument("--max-total",     type=int, default=100, help="Maximum total heatmaps to generate")
    parser.add_argument("--min-classes",   type=int, default=3,  help="Minimum number of unique label groups before early exit")
    parser.add_argument("--out",           default="artifacts/xai_v2", help="Output directory root")
    parser.add_argument("--clear",         action="store_true", help="Clear the output directory before running")
    parser.add_argument("--method",        default="gxi", choices=["gxi", "ig"], help="Saliency method: 'gxi' (Gradient x Input, fast) or 'ig' (Integrated Gradients, smoother)")
    parser.add_argument("--ig-steps",      type=int, default=20, help="Number of interpolation steps for Integrated Gradients (only used with --method ig)")
    parser.add_argument("--explain-all",   action="store_true",  help="Also explain samples where no class is predicted (show top arrhythmia)")
    args = parser.parse_args()
    main(args)