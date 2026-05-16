"""
16_xai_explainer.py
====================
Generates attention-based ECG heatmaps for each predicted diagnostic label.

Output directory structure
---------------------------
artifacts/xai/
    IMI/                    <- samples where ONLY IMI is predicted
    ASMI/                   <- samples where ONLY ASMI is predicted
    IMI+ASMI/               <- samples where IMI AND ASMI co-occur
    AFIB/
    ...
    AFIB+PVC/               <- any multi-label combo, sorted and joined with '+'

    {label_folder}/
        {patient_id}_{ecg_id}_map.png   <- Full 12-lead heatmap

Usage
------
    python scripts/16_xai_explainer.py \\
        --checkpoint checkpoints/run_20260515_232907_hybrid-tf-aug \\
        --config    configs/experiments/dynth_e02_perclass.yaml \\
        --split     test \\
        --max-per-class 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import ECGDataset
from src.models.factory import build_model
from src.training.trainer import load_config_and_labels

# ── lead names (standard 12-lead order) ─────────────────────────────────────
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── colour map for heatmap overlay ───────────────────────────────────────────
HEAT_CMAP = plt.cm.YlOrRd          # Yellow → Orange → Red
HEAT_ALPHA = 0.45                   # overlay transparency


# ────────────────────────────────────────────────────────────────────────────
# Attention extraction helpers
# ────────────────────────────────────────────────────────────────────────────

def _get_weights(pool_module, seq_len: int) -> np.ndarray:
    """Upsamples saved attention weights from token-space to signal-space."""
    if not hasattr(pool_module, "last_weights"):
        return None
    w = pool_module.last_weights  # (B, T_token, 1) or (B, T_token)
    if w.dim() == 3:
        w = w.squeeze(-1)         # (B, T_token)
    w = w[0].float().cpu().numpy()    # (T_token,)
    # Interpolate back to original signal length
    w_tensor = torch.tensor(w).unsqueeze(0).unsqueeze(0)   # (1,1,T_token)
    w_up = F.interpolate(w_tensor, size=seq_len, mode="linear", align_corners=False)
    return w_up.squeeze().numpy()     # (seq_len,)


def extract_attention_maps(model, signal_len: int) -> dict[str, np.ndarray | None]:
    """
    Collects attention maps from all pooling layers in the model.

    Returns a dict of label_name → 1-D attention array (length = signal_len).
    Strategy: use the task-specific TaskTokenPool for each label when available,
    else fall back to the global AttentionPool from the backbone.
    """
    backbone = model.backbone
    mi_head  = model.mi_head

    # --- Global fallback from backbone (used for arrhythmia labels) ---
    global_map = _get_weights(backbone.pool, signal_len) if hasattr(backbone, "pool") else None

    maps: dict[str, np.ndarray | None] = {}

    # Arrhythmia labels: use arrhythmia_pool if it exists, else global
    arrhy_map = None
    if hasattr(model, "arrhythmia_pool"):
        arrhy_map = _get_weights(model.arrhythmia_pool, signal_len)
    arrhy_map = arrhy_map if arrhy_map is not None else global_map

    for name in ["NORM", "AFIB", "STACH", "PVC", "AFLT"]:
        maps[name] = arrhy_map

    # IMI — uses imi_token_pool
    imi_map = None
    if hasattr(mi_head, "imi_token_pool"):
        imi_map = _get_weights(mi_head.imi_token_pool, signal_len)
    maps["IMI"] = imi_map if imi_map is not None else global_map

    # ASMI — uses asmi_token_pool
    asmi_map = None
    if hasattr(mi_head, "asmi_token_pool"):
        asmi_map = _get_weights(mi_head.asmi_token_pool, signal_len)
    maps["ASMI"] = asmi_map if asmi_map is not None else global_map

    return maps


# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────

def _plot_single_label_focus(attn: np.ndarray, signal: np.ndarray,
                              label: str, fs: int = 500) -> plt.Figure:
    """Full 12-lead ECG with one attention map overlaid."""
    n_leads, sig_len = signal.shape
    t = np.arange(sig_len) / fs          # time axis in seconds

    # Normalise attention to [0, 1] for colour mapping
    a_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    fig, axes = plt.subplots(n_leads, 1, figsize=(22, n_leads * 1.6),
                              sharex=True, constrained_layout=True)
    if n_leads == 1:
        axes = [axes]

    cmap  = HEAT_CMAP
    norm  = mcolors.Normalize(0, 1)

    for i, (ax, lead_name) in enumerate(zip(axes, LEAD_NAMES)):
        lead_sig = signal[i]
        y_min, y_max = lead_sig.min(), lead_sig.max()
        y_pad = (y_max - y_min) * 0.2 + 1e-4

        # Draw heatmap as filled background bands
        for j in range(len(t) - 1):
            ax.axvspan(t[j], t[j + 1],
                       color=cmap(norm(a_norm[j])),
                       alpha=HEAT_ALPHA, linewidth=0)

        ax.plot(t, lead_sig, color="#1a1a2e", linewidth=0.7, zorder=2)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel(lead_name, rotation=0, labelpad=28,
                      fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_facecolor("#f8f9fa")
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"ECG Attention Heatmap — Predicted: {label}",
                 fontsize=13, fontweight="bold", y=1.01)

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=8)

    return fig


def _plot_multi_label(attn_maps: dict[str, np.ndarray],
                      signal: np.ndarray,
                      active_labels: list[str], fs: int = 500) -> plt.Figure:
    """
    For co-occurring labels: show each label's attention map as a
    separate coloured overlay on the same 12-lead ECG.

    Each label gets a distinct hue band; the legend explains which colour
    corresponds to which diagnosis.
    """
    n_leads, sig_len = signal.shape
    t = np.arange(sig_len) / fs

    # Distinct colour palettes per label (max 5 visible at once)
    palette = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
    label_colors = {lbl: palette[i % len(palette)]
                    for i, lbl in enumerate(active_labels)}

    fig, axes = plt.subplots(n_leads, 1, figsize=(22, n_leads * 1.6),
                              sharex=True, constrained_layout=True)
    if n_leads == 1:
        axes = [axes]

    for i, (ax, lead_name) in enumerate(zip(axes, LEAD_NAMES)):
        lead_sig = signal[i]
        y_min, y_max = lead_sig.min(), lead_sig.max()
        y_pad = (y_max - y_min) * 0.2 + 1e-4

        # Stack attention overlays — each label's map drawn independently
        for lbl in active_labels:
            attn = attn_maps[lbl]
            a_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            color  = label_colors[lbl]
            for j in range(len(t) - 1):
                ax.axvspan(t[j], t[j + 1],
                           color=color,
                           alpha=a_norm[j] * 0.28,
                           linewidth=0)

        ax.plot(t, lead_sig, color="#1a1a2e", linewidth=0.7, zorder=2)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel(lead_name, rotation=0, labelpad=28,
                      fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_facecolor("#f8f9fa")
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    combo_str = "+".join(sorted(active_labels))
    fig.suptitle(f"ECG Attention Heatmap — Predicted: {combo_str}  (multi-label)",
                 fontsize=13, fontweight="bold", y=1.01)

    # Legend patch per label
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=label_colors[lbl], alpha=0.7, label=lbl)
               for lbl in active_labels]
    fig.legend(handles=patches, loc="upper right",
               fontsize=9, framealpha=0.85, ncol=len(active_labels))

    return fig


# ────────────────────────────────────────────────────────────────────────────
# Core routine
# ────────────────────────────────────────────────────────────────────────────

def generate_xai(
    checkpoint_dir: str,
    config_path: str,
    split: str = "test",
    max_per_class: int = 5,
    out_root: str = "artifacts/xai",
    fs: int = 500,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir  = Path(checkpoint_dir)
    out_root  = Path(out_root)

    # ── 1. Load config ───────────────────────────────────────────────────────
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    # ── 2. Build model & load weights ───────────────────────────────────────
    with open(ckpt_dir / "config_snapshot.yaml", encoding="utf-8") as f:
        ckpt_cfg = yaml.safe_load(f)

    arrhythmia_labels = [l["name"] for l in ckpt_cfg["labels"]
                         if l["task"] == "arrhythmia"]
    mi_labels         = [l["name"] for l in ckpt_cfg["labels"]
                         if l["task"] == "mi"]
    all_labels        = arrhythmia_labels + mi_labels

    model = build_model(ckpt_cfg, len(arrhythmia_labels), len(mi_labels))
    best_ckpt = ckpt_dir / "best_model.pt"
    state = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    print(f"[XAI] Model loaded from: {best_ckpt}")

    # ── 3. Load tuned thresholds (from per-class tuning if available) ────────
    threshold_path = ckpt_dir / "best_thresholds.json"
    if threshold_path.exists():
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        print(f"[XAI] Using per-class thresholds: {thresholds}")
    else:
        thresholds = {lbl: 0.5 for lbl in all_labels}
        print("[XAI] Using fixed threshold 0.5 for all classes.")

    # ── 4. Build dataset ─────────────────────────────────────────────────────
    processed_dir = Path(cfg["paths"]["processed"])
    splits_dir    = Path(cfg["paths"]["splits"])
    dataset = ECGDataset(
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        split=split,
        config=cfg,
    )
    print(f"[XAI] Loaded {split} set: {len(dataset)} samples")

    # ── 5. Per-class counter to limit output ─────────────────────────────────
    combo_count: dict[str, int] = {}

    # ── 6. Iterate & generate ────────────────────────────────────────────────
    for idx in range(len(dataset)):
        sample  = dataset[idx]
        signal  = sample["signal"]                  # (12, 5000)
        sig_len = signal.shape[-1]

        # Stack into batch dim
        signal_t = signal.unsqueeze(0).to(device)   # (1, 12, 5000)

        with torch.no_grad():
            preds = model(signal_t)

        # Collect sigmoid probabilities and apply thresholds
        arrhy_prob = torch.sigmoid(preds["arrhythmia"])[0].cpu().numpy()  # (n_arrhy,)
        mi_prob    = torch.sigmoid(preds["mi"])[0].cpu().numpy()           # (n_mi,)
        all_probs  = np.concatenate([arrhy_prob, mi_prob])

        predicted = []
        for j, lbl in enumerate(all_labels):
            t = thresholds.get(lbl, 0.5)
            if all_probs[j] >= t:
                predicted.append(lbl)

        # Skip NORM-only predictions (usually uninteresting for XAI)
        if not predicted or predicted == ["NORM"]:
            continue

        # Exclude NORM from the combo label (keep only pathological)
        pathological = [l for l in predicted if l != "NORM"]
        if not pathological:
            continue

        combo_key = "+".join(sorted(pathological))

        # Enforce per-combo limit
        if combo_count.get(combo_key, 0) >= max_per_class:
            continue

        # Extract attention maps
        attn_maps = extract_attention_maps(model, sig_len)

        # Patient/ECG identifier
        meta_id   = sample.get("ecg_id", idx)
        patient   = sample.get("patient_id", "unk")
        file_stem = f"pat{patient}_ecg{meta_id}"

        # ── 7. Save image(s) ─────────────────────────────────────────────────

        # Determine folders: combo folder + individual-label folders (for easy
        # browsing when the same sample appears in multiple single-label searches)
        target_folders = [combo_key]
        if len(pathological) > 1:
            # Also copy into each individual label's folder
            target_folders += sorted(pathological)

        signal_np = signal.numpy()   # (12, 5000)

        if len(pathological) == 1:
            lbl   = pathological[0]
            attn  = attn_maps.get(lbl)
            if attn is None:
                continue
            fig = _plot_single_label_focus(attn, signal_np, lbl, fs=fs)
        else:
            # Multi-label: draw all relevant attention maps together
            fig = _plot_multi_label(
                {lbl: attn_maps[lbl] for lbl in pathological if attn_maps.get(lbl) is not None},
                signal_np,
                [lbl for lbl in pathological if attn_maps.get(lbl) is not None],
                fs=fs,
            )

        # Save to primary combo folder
        primary_folder = out_root / combo_key
        primary_folder.mkdir(parents=True, exist_ok=True)
        primary_path   = primary_folder / f"{file_stem}_map.png"
        fig.savefig(primary_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        # For multi-label: hard-copy into each individual label folder so
        # researchers can quickly browse all IMI images regardless of co-labels
        if len(pathological) > 1:
            for lbl in pathological:
                single_folder = out_root / lbl
                single_folder.mkdir(parents=True, exist_ok=True)
                dest = single_folder / f"{file_stem}_[{combo_key}]_map.png"
                shutil.copy2(primary_path, dest)

        combo_count[combo_key] = combo_count.get(combo_key, 0) + 1

        total_saved = sum(combo_count.values())
        print(f"[XAI] [{total_saved:4d}] {combo_key:30s}  -> {primary_path.name}")

        # Stop if every combo is saturated
        if all(v >= max_per_class for v in combo_count.values()):
            break

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n[XAI] Generation complete.")
    print(f"{'Label / Combo':<30}  {'Saved':>5}")
    print("-" * 40)
    for combo, n in sorted(combo_count.items()):
        print(f"  {combo:<28}  {n:>5}")
    print(f"\nOutput root: {out_root.resolve()}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate XAI attention heatmaps for ECG diagnoses.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint directory (e.g. checkpoints/run_xxx).")
    p.add_argument("--config", required=True,
                   help="Config YAML used to build the dataset (e.g. configs/experiments/dynth_e02_perclass.yaml).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Which dataset split to run XAI on. Default: test.")
    p.add_argument("--max-per-class", type=int, default=5,
                   help="Maximum heatmaps saved per label/combo. Default: 5.")
    p.add_argument("--out", default="artifacts/xai",
                   help="Output root directory. Default: artifacts/xai.")
    p.add_argument("--fs", type=int, default=500,
                   help="ECG sampling rate in Hz. Default: 500.")
    p.add_argument("--device", default=None,
                   help="Torch device (cuda / cpu). Auto-detected if not set.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_xai(
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        split=args.split,
        max_per_class=args.max_per_class,
        out_root=args.out,
        fs=args.fs,
        device=args.device,
    )
