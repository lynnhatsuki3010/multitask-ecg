"""
scripts/analyze_mi_stability.py
────────────────────────────────────────────────────────────────────
So sánh độ ổn định của training curve MI metrics (IMI AUPRC)
giữa Baseline (BCE only) và Direction 1 (BCE + Contrastive Loss).

Đọc file history.json từ 2 checkpoint directory và vẽ:
  1. IMI AUPRC theo epoch (train + val)
  2. ASMI AUPRC theo epoch (train + val)
  3. MI Total Loss theo epoch
  4. Gradient Norm của MI head theo epoch

Usage:
    python scripts/analyze_mi_stability.py \\
        --baseline-dir checkpoints/run_20260526_235533_hybrid-tf-aug \\
        --dir1-dir     checkpoints/run_20260605_114713_hybrid-tf-aug \\
        --output       results/stability_analysis
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Load history ──────────────────────────────────────────────────────────────

def load_history(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"history.json not found in: {ckpt_dir}")
    with open(path) as f:
        return json.load(f)


def extract_metric(history_list, key):
    """Pull a metric across all epochs, return list (NaN if missing)."""
    return [epoch.get(key, float("nan")) for epoch in history_list]


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "baseline_train": "#7f7f7f",
    "baseline_val":   "#c7c7c7",
    "dir1_train":     "#d62728",
    "dir1_val":       "#ff9896",
}


def plot_metric_comparison(
    epochs_bl, bl_train, bl_val,
    epochs_d1, d1_train, d1_val,
    metric_name, ylabel, title, save_path,
    ymin=None, ymax=None,
):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs_bl, bl_train, color=COLORS["baseline_train"],
            lw=1.8, label="Baseline (train)", linestyle="--")
    if bl_val is not None:
        ax.plot(epochs_bl, bl_val, color=COLORS["baseline_val"],
                lw=1.8, label="Baseline (val)", linestyle="-")

    ax.plot(epochs_d1, d1_train, color=COLORS["dir1_train"],
            lw=1.8, label="Dir1 Contrastive (train)", linestyle="--")
    if d1_val is not None:
        ax.plot(epochs_d1, d1_val, color=COLORS["dir1_val"],
                lw=1.8, label="Dir1 Contrastive (val)", linestyle="-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Chart] Saved → {save_path}")


def compute_stability(values):
    """Compute std-dev of epoch-to-epoch deltas as a stability metric."""
    vals = [v for v in values if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    deltas = np.abs(np.diff(vals))
    return float(np.std(deltas))


def print_stability_report(name, train_vals, val_vals, metric_key):
    train_stable = compute_stability(train_vals)
    val_stable   = compute_stability(val_vals) if val_vals else float("nan")
    peak_train   = max((v for v in train_vals if not np.isnan(v)), default=float("nan"))
    peak_val     = max((v for v in val_vals   if not np.isnan(v)), default=float("nan")) if val_vals else float("nan")
    print(f"  {name:<35} | train_std={train_stable:.4f}  val_std={val_stable:.4f}  "
          f"peak_train={peak_train:.4f}  peak_val={peak_val:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MI Stability Analysis")
    parser.add_argument("--baseline-dir", required=True,
                        help="Checkpoint dir of baseline (BCE only) model")
    parser.add_argument("--dir1-dir",     required=True,
                        help="Checkpoint dir of Direction 1 (BCE + Contrastive) model")
    parser.add_argument("--output",       default="results/stability_analysis",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n[+] Loading history from Baseline: {args.baseline_dir}")
    bl_hist = load_history(args.baseline_dir)

    print(f"[+] Loading history from Direction 1: {args.dir1_dir}")
    d1_hist = load_history(args.dir1_dir)

    bl_train = bl_hist.get("train", [])
    bl_val   = bl_hist.get("val",   [])
    d1_train = d1_hist.get("train", [])
    d1_val   = d1_hist.get("val",   [])

    epochs_bl = list(range(1, len(bl_train) + 1))
    epochs_d1 = list(range(1, len(d1_train) + 1))

    print(f"  Baseline epochs: {len(epochs_bl)}")
    print(f"  Direction1 epochs: {len(epochs_d1)}")

    # ── Metrics to plot ──
    metrics = [
        {
            "train_key": "auprc/mi/IMI",
            "val_key":   "auprc/mi/IMI",
            "ylabel":    "AUPRC",
            "title":     "IMI AUPRC — Baseline vs Direction 1 (Contrastive)",
            "filename":  "imi_auprc_stability.png",
            "ymin":      0.0,
        },
        {
            "train_key": "auprc/mi/ASMI",
            "val_key":   "auprc/mi/ASMI",
            "ylabel":    "AUPRC",
            "title":     "ASMI AUPRC — Baseline vs Direction 1 (Contrastive)",
            "filename":  "asmi_auprc_stability.png",
            "ymin":      0.0,
        },
        {
            "train_key": "auroc/mi/IMI",
            "val_key":   "auroc/mi/IMI",
            "ylabel":    "AUROC",
            "title":     "IMI AUROC — Baseline vs Direction 1 (Contrastive)",
            "filename":  "imi_auroc_stability.png",
            "ymin":      0.5,
        },
        {
            "train_key": "loss/mi",
            "val_key":   "loss/mi",
            "ylabel":    "MI Loss",
            "title":     "MI Branch Loss — Baseline vs Direction 1 (Contrastive)",
            "filename":  "mi_loss_stability.png",
            "ymin":      0.0,
        },
        {
            "train_key": "grad_norm/mi",
            "val_key":   None,
            "ylabel":    "Gradient Norm",
            "title":     "MI Head Gradient Norm — Baseline vs Direction 1",
            "filename":  "mi_grad_norm.png",
            "ymin":      0.0,
        },
    ]

    print(f"\n{'─'*90}")
    print(f"  {'Metric':<35} | train_std / val_std / peak_train / peak_val")
    print(f"{'─'*90}")

    for m in metrics:
        key_t = m["train_key"]
        key_v = m["val_key"]

        bl_t = extract_metric(bl_train, key_t)
        bl_v = extract_metric(bl_val,   key_v) if bl_val and key_v else None
        d1_t = extract_metric(d1_train, key_t)
        d1_v = extract_metric(d1_val,   key_v) if d1_val and key_v else None

        print_stability_report(f"[BL] {key_t}", bl_t, bl_v, key_t)
        print_stability_report(f"[D1] {key_t}", d1_t, d1_v, key_t)
        print()

        plot_metric_comparison(
            epochs_bl, bl_t, bl_v,
            epochs_d1, d1_t, d1_v,
            metric_name=key_t,
            ylabel=m["ylabel"],
            title=m["title"],
            save_path=os.path.join(args.output, m["filename"]),
            ymin=m.get("ymin"),
            ymax=m.get("ymax"),
        )

    # ── Epoch-to-epoch variance summary ──
    print(f"\n{'='*55}")
    print("  STABILITY SUMMARY (Lower std = More Stable)")
    print(f"{'='*55}")
    print(f"  {'Model':<30} | {'IMI AUPRC val std':>18} | {'ASMI AUPRC val std':>18}")
    print(f"  {'─'*68}")

    imi_bl  = compute_stability(extract_metric(bl_val,  "auprc/mi/IMI")) if bl_val else float("nan")
    imi_d1  = compute_stability(extract_metric(d1_val,  "auprc/mi/IMI")) if d1_val else float("nan")
    asmi_bl = compute_stability(extract_metric(bl_val,  "auprc/mi/ASMI")) if bl_val else float("nan")
    asmi_d1 = compute_stability(extract_metric(d1_val,  "auprc/mi/ASMI")) if d1_val else float("nan")

    print(f"  {'Baseline (BCE only)':<30} | {imi_bl:>18.4f} | {asmi_bl:>18.4f}")
    print(f"  {'Direction 1 (+ Contrastive)':<30} | {imi_d1:>18.4f} | {asmi_d1:>18.4f}")

    if imi_d1 < imi_bl:
        print("\n  ✅ Contrastive Loss giúp IMI training ổn định hơn (std giảm)")
    elif imi_d1 > imi_bl:
        print("\n  ⚠️  Contrastive Loss làm IMI training kém ổn định hơn (std tăng)")
    else:
        print("\n  ➖ Không có sự thay đổi rõ ràng về độ ổn định")

    print(f"\n[OK] Tất cả biểu đồ đã lưu tại: {args.output}/")


if __name__ == "__main__":
    main()
