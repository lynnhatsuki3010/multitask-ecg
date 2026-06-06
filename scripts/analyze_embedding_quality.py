"""
scripts/analyze_embedding_quality.py
────────────────────────────────────────────────────────────────────
Kiểm tra chất lượng MI embedding (Direction 1 - Contrastive Loss):

  1. Trích xuất vector h (128-d, L2-normalized) từ tập TEST.
  2. Vẽ t-SNE 2D scatter plot, màu theo nhãn IMI/ASMI.
  3. Tính intra-class vs inter-class cosine distance.
  4. So sánh với Baseline (không có Contrastive Loss) nếu cung cấp.

Usage:
    python scripts/analyze_embedding_quality.py \\
        --dir checkpoints/run_20260605_114713_hybrid-tf-aug \\
        --config configs/experiments/final_e01_seed42.yaml \\
        --output results/embedding_analysis

Options:
    --baseline-dir    Thư mục checkpoint Baseline để so sánh (tuỳ chọn)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import build_model
from src.data.preprocessing import PTBXLDataset, collate_fn


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_test_dataset(cfg: dict) -> PTBXLDataset:
    """Build the test split using the same PTBXLDataset as training."""
    import pandas as pd

    fs        = cfg["dataset"]["sampling_rate"]
    processed = f"{cfg['paths']['processed']}_{fs}hz"
    splits    = f"{cfg['paths']['splits']}_{fs}hz"
    raw_path  = cfg["paths"]["raw_data"]
    length    = cfg["dataset"]["signal_length"]
    prep      = cfg.get("preprocessing", {})
    bandpass  = (prep.get("bandpass_low", 0.5), prep.get("bandpass_high", 40.0))
    notch     = prep.get("notch_freq", 50.0)
    normalize = prep.get("normalize", "robust")

    meta         = pd.read_csv(os.path.join(processed, "metadata.csv"), index_col="ecg_id")
    label_matrix = np.load(os.path.join(processed, "label_matrix.npy"))
    hrv_matrix   = np.load(os.path.join(processed, "hrv_matrix.npy")) \
                   if os.path.exists(os.path.join(processed, "hrv_matrix.npy")) else None

    test_indices = np.load(os.path.join(splits, "test_indices.npy"))
    sub_meta     = meta.iloc[test_indices]
    sub_lm       = label_matrix[test_indices]
    sub_hrv      = hrv_matrix[test_indices] if hrv_matrix is not None else None

    # MI label indices from config
    mi_group     = cfg.get("label_groups", {}).get("mi", [])
    label_cfgs   = cfg.get("labels", [])
    all_names    = [l["name"] for l in label_cfgs]
    mi_names     = [all_names[i] for i in mi_group]

    ds = PTBXLDataset(
        metadata      = sub_meta,
        label_matrix  = sub_lm,
        hrv_matrix    = sub_hrv,
        base_path     = raw_path,
        sampling_rate = fs,
        target_length = length,
        bandpass      = bandpass,
        notch         = notch,
        normalize     = normalize,
        augment       = False,
    )
    # Attach MI indices so extract_embeddings can slice them
    ds.mi_indices = mi_group
    ds.mi_names   = mi_names
    return ds


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, cfg: dict, device: torch.device):
    """Build model from cfg and load weights from best_model.pth."""
    n_arrhy = len(cfg.get("label_groups", {}).get("arrhythmia", []))
    n_mi    = len(cfg.get("label_groups", {}).get("mi", []))

    model = build_model(cfg, num_arrhythmia_labels=n_arrhy,
                        num_mi_labels=n_mi, num_hrv_targets=3)

    ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    projection_missing = [
        k for k in missing if k.startswith("mi_head.mi_projection.")
    ]
    legacy_without_projection = bool(projection_missing) and missing == projection_missing

    if legacy_without_projection and not unexpected:
        model.load_state_dict(state, strict=False)
        model._embedding_analysis_use_logits = True
        print("    [legacy] mi_projection weights not found; using MI logits as embedding proxy.")
    else:
        model.load_state_dict(state)
        model._embedding_analysis_use_logits = False

    model.to(device).eval()
    return model


# ── Extract embeddings ────────────────────────────────────────────────────────

def extract_embeddings(model, loader, device, mi_indices):
    """
    Run forward pass and collect mi_proj (h) and mi labels.
    Returns:
        embeddings: (N, 128) numpy array
        labels:     (N, 2)   numpy array  [IMI, ASMI]
    """
    all_h, all_y = [], []
    has_proj = True

    with torch.no_grad():
        for batch in loader:
            sig = batch["signal"].to(device)
            out = model(sig)

            if "mi_proj" in out and not getattr(model, "_embedding_analysis_use_logits", False):
                h = out["mi_proj"].cpu().numpy()
            else:
                # Baseline model: use raw MI logits as embedding proxy
                h = out["mi"].cpu().numpy()
                has_proj = False

            all_h.append(h)
            # Slice MI labels from full label tensor
            all_y.append(batch["labels"][:, mi_indices].numpy())

    embeddings = np.vstack(all_h)
    labels     = np.vstack(all_y)
    return embeddings, labels, has_proj


# ── t-SNE Visualisation ───────────────────────────────────────────────────────

LABEL_COLORS = {
    "Neither IMI/ASMI": "#7f7f7f",
    "IMI only":         "#d62728",   # red
    "ASMI only":        "#1f77b4",   # blue
    "Both IMI+ASMI":    "#ff7f0e",   # orange
}

def label_to_group(y_mi):
    """Map (N, 2) binary array to group strings."""
    imi, asmi = y_mi[:, 0], y_mi[:, 1]
    groups = []
    for i, a in zip(imi, asmi):
        if i == 1 and a == 1:
            groups.append("Both IMI+ASMI")
        elif i == 1:
            groups.append("IMI only")
        elif a == 1:
            groups.append("ASMI only")
        else:
            groups.append("Neither IMI/ASMI")
    return np.array(groups)


def plot_tsne(embeddings, labels, title, save_path, perplexity=40, seed=42):
    print(f"  [t-SNE] Running on {len(embeddings)} samples ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                max_iter=1000, init="pca")
    coords = tsne.fit_transform(embeddings)

    groups = label_to_group(labels)
    unique_groups = list(LABEL_COLORS.keys())

    fig, ax = plt.subplots(figsize=(9, 7))
    for grp in unique_groups:
        mask = groups == grp
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=LABEL_COLORS[grp], label=grp,
                   s=12, alpha=0.65, edgecolors="none")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [t-SNE] Saved → {save_path}")


# ── Cosine Distance Analysis ──────────────────────────────────────────────────

def cosine_distance_matrix(A):
    """Compute pairwise cosine distance matrix for rows of A (L2-normed)."""
    # A is already L2-normalized (mi_proj), so dot product == cosine similarity
    sim  = A @ A.T
    dist = 1.0 - sim
    return dist


def compute_distance_stats(embeddings, labels):
    """
    Compute intra-class and inter-class mean cosine distances.
    Returns a dict of stats.
    """
    # Normalize just in case
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / (norms + 1e-8)

    groups = label_to_group(labels)
    unique = [g for g in LABEL_COLORS.keys() if (groups == g).sum() > 1]

    stats = {}

    # Overall intra-class: within same group
    intra_dists, inter_dists = [], []

    dist_mat = cosine_distance_matrix(emb_n)

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            d = dist_mat[i, j]
            if groups[i] == groups[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    stats["intra_mean"] = float(np.mean(intra_dists)) if intra_dists else 0.0
    stats["inter_mean"] = float(np.mean(inter_dists)) if inter_dists else 0.0
    stats["separation_ratio"] = stats["inter_mean"] / (stats["intra_mean"] + 1e-8)

    # Per-group counts
    unique_groups_list = list(LABEL_COLORS.keys())
    for g in unique_groups_list:
        stats[f"n_{g.replace('/', '_').replace(' ', '_')}"] = int((groups == g).sum())

    return stats


def print_distance_report(stats, model_name):
    print(f"\n{'─'*55}")
    print(f"  Cosine Distance Analysis: {model_name}")
    print(f"{'─'*55}")
    for k, v in stats.items():
        if k.startswith("n_"):
            lbl = k[2:].replace("_", " ")
            print(f"  {lbl:<30}: {v:>6d} samples")
    print()
    print(f"  {'Intra-class mean cosine dist':<30}: {stats['intra_mean']:.4f}")
    print(f"  {'Inter-class mean cosine dist':<30}: {stats['inter_mean']:.4f}")
    print(f"  {'Separation ratio (inter/intra)':<30}: {stats['separation_ratio']:.4f}")
    print(f"  → {'GOOD cluster separation' if stats['separation_ratio'] > 1.2 else 'WEAK or NO cluster separation'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze MI Embedding Quality")
    parser.add_argument("--dir",          required=True,
                        help="Checkpoint dir of Direction 1 model (with mi_proj)")
    parser.add_argument("--config",       required=True,
                        help="Path to YAML config used in training")
    parser.add_argument("--baseline-dir", default=None,
                        help="Checkpoint dir of baseline model (optional, for comparison)")
    parser.add_argument("--output",       default="results/embedding_analysis",
                        help="Output directory for figures and stats")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--perplexity",   type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Build dataset ──
    print("\n[+] Loading test dataset ...")
    dataset = build_test_dataset(cfg)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0,
                         collate_fn=collate_fn)
    mi_indices = dataset.mi_indices
    print(f"    Test samples: {len(dataset)}")
    print(f"    MI labels: {dataset.mi_names}")

    # ── Direction 1 model ──
    print(f"\n[+] Loading Direction 1 model from: {args.dir}")
    model_d1 = load_model(args.dir, cfg, device)
    emb_d1, y_mi, has_proj = extract_embeddings(model_d1, loader, device, mi_indices)
    print(f"    Embedding shape: {emb_d1.shape}")
    print(f"    Using mi_proj: {has_proj}")

    title_d1 = "MI Embedding Space — Direction 1 (BCE + Contrastive)"
    if not has_proj:
        title_d1 = "MI Logit Space — Baseline (no mi_proj found)"

    plot_tsne(emb_d1, y_mi,
              title    = title_d1,
              save_path= os.path.join(args.output, "tsne_direction1.png"),
              perplexity=args.perplexity)

    stats_d1 = compute_distance_stats(emb_d1, y_mi)
    print_distance_report(stats_d1, "Direction 1 (Contrastive)")

    all_stats = {"direction1": stats_d1}

    # ── Baseline model (optional) ──
    if args.baseline_dir:
        print(f"\n[+] Loading Baseline model from: {args.baseline_dir}")
        model_bl = load_model(args.baseline_dir, cfg, device)
        emb_bl, y_mi_bl, has_proj_bl = extract_embeddings(model_bl, loader, device, mi_indices)

        plot_tsne(emb_bl, y_mi_bl,
                  title    = "MI Logit Space — Baseline (BCE only)",
                  save_path= os.path.join(args.output, "tsne_baseline.png"),
                  perplexity=args.perplexity)

        stats_bl = compute_distance_stats(emb_bl, y_mi_bl)
        print_distance_report(stats_bl, "Baseline (BCE only)")
        all_stats["baseline"] = stats_bl

        # ── Comparison bar chart ──
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2)
        bars_intra = [stats_bl["intra_mean"], stats_d1["intra_mean"]]
        bars_inter = [stats_bl["inter_mean"], stats_d1["inter_mean"]]
        w = 0.3
        ax.bar(x - w/2, bars_intra, w, label="Intra-class dist", color="#1f77b4", alpha=0.8)
        ax.bar(x + w/2, bars_inter, w, label="Inter-class dist", color="#d62728", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline\n(BCE only)", "Direction 1\n(+ Contrastive)"])
        ax.set_ylabel("Mean Cosine Distance")
        ax.set_title("Intra vs Inter-class Cosine Distance Comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        cmp_path = os.path.join(args.output, "distance_comparison.png")
        plt.savefig(cmp_path, dpi=150)
        plt.close()
        print(f"\n  [Chart] Comparison saved → {cmp_path}")

    # ── Save stats JSON ──
    stats_path = os.path.join(args.output, "embedding_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n[OK] Stats saved → {stats_path}")
    print(f"[OK] All figures saved to: {args.output}/")


if __name__ == "__main__":
    main()
