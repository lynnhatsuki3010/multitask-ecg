"""
analyze_task_embedding_quality.py
────────────────────────────────────────────────────────────────────
Task-agnostic extension of the DIR1 MI embedding analysis.

For a trained checkpoint, this script:
  1. Rebuilds the test dataset from the checkpoint/config.
  2. Extracts task logits as the embedding proxy for one task:
     arrhythmia, mi, or conduction.
  3. Draws a t-SNE scatter colored by ground-truth task group.
  4. Computes intra/inter-group cosine-distance statistics.
  5. Optionally compares against a baseline checkpoint.

The original DIR1 script used `mi_proj` when available and fell back to MI
logits for BCE-only models. The current decoupled architecture does not expose a
shared projection for all tasks, so logits are used consistently across tasks.

Usage:
    python scripts/analyze_task_embedding_quality.py \
        --dir checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug \
        --task conduction \
        --output results/task_embedding_b10_05
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import PTBXLDataset, collate_fn
from src.models.factory import build_model


TASK_TO_OUTPUT = {
    "arrhythmia": "arrhythmia",
    "mi": "mi",
    "conduction": "conduction",
}

TASK_TO_GROUP = {
    "arrhythmia": "arrhythmia",
    "mi": "mi",
    "conduction": "conduction",
}


def load_cfg(ckpt_dir: str, config_path: str = None) -> dict:
    path = config_path or os.path.join(ckpt_dir, "config_snapshot.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_dir(base: str, fs: int, required_file: str) -> str:
    candidates = [base, f"{base}_{fs}hz"]
    for path in candidates:
        if os.path.exists(os.path.join(path, required_file)):
            return path
    raise FileNotFoundError(f"Could not find {required_file} under: {candidates}")


def group_label_names(cfg: dict, group: str) -> Tuple[List[str], List[int]]:
    idx_to_name = {label["index"]: label["name"] for label in cfg.get("labels", [])}
    indices = list(cfg.get("label_groups", {}).get(group, []))
    names = [idx_to_name[i] for i in indices]
    return names, indices


def build_test_dataset(cfg: dict) -> Tuple[PTBXLDataset, Dict[str, Tuple[List[str], List[int]]]]:
    fs = int(cfg["dataset"]["sampling_rate"])
    processed = resolve_data_dir(cfg["paths"]["processed"], fs, "metadata.csv")
    splits = resolve_data_dir(cfg["paths"]["splits"], fs, "test_indices.npy")
    raw_path = cfg["paths"]["raw_data"]
    length = int(cfg["dataset"]["signal_length"])
    prep = cfg.get("preprocessing", {})
    bandpass = (prep.get("bandpass_low", 0.5), prep.get("bandpass_high", 40.0))
    notch = prep.get("notch_freq", 50.0)
    normalize = prep.get("normalize", "robust")

    meta = pd.read_csv(os.path.join(processed, "metadata.csv"), index_col="ecg_id")
    label_matrix = np.load(os.path.join(processed, "label_matrix.npy"))
    hrv_path = os.path.join(processed, "hrv_matrix.npy")
    hrv_matrix = np.load(hrv_path) if os.path.exists(hrv_path) else None
    test_indices = np.load(os.path.join(splits, "test_indices.npy"))

    ds = PTBXLDataset(
        metadata=meta.iloc[test_indices],
        label_matrix=label_matrix[test_indices],
        hrv_matrix=hrv_matrix[test_indices] if hrv_matrix is not None else None,
        base_path=raw_path,
        sampling_rate=fs,
        target_length=length,
        bandpass=bandpass,
        notch=notch,
        normalize=normalize,
        augment=False,
    )
    groups = {
        group: group_label_names(cfg, group)
        for group in ["arrhythmia", "mi", "conduction"]
    }
    return ds, groups


def load_model(ckpt_dir: str, cfg: dict, device: torch.device):
    arrhy_names, _ = group_label_names(cfg, "arrhythmia")
    mi_names, _ = group_label_names(cfg, "mi")
    cond_names, _ = group_label_names(cfg, "conduction")
    cfg = dict(cfg)
    cfg["hrv"] = {"enabled": False}
    model = build_model(
        cfg,
        num_arrhythmia_labels=len(arrhy_names),
        num_mi_labels=len(mi_names),
        num_hrv_targets=3,
        num_conduction_labels=len(cond_names),
    )
    ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    return model.to(device).eval()


def label_to_group(y_task: np.ndarray, label_names: List[str]) -> np.ndarray:
    """Map a multi-hot task label row to a compact color group."""
    groups = []
    for row in y_task.astype(int):
        active = [name for name, val in zip(label_names, row) if val == 1]
        if not active:
            groups.append("None")
        elif len(active) == 1:
            groups.append(active[0])
        else:
            groups.append("Multi-label")
    return np.array(groups)


def extract_task_logits(model, loader, task: str, task_indices: List[int], device: torch.device):
    out_key = TASK_TO_OUTPUT[task]
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            preds = model(signal)
            logits.append(preds[out_key].cpu().numpy())
            labels.append(batch["labels"][:, task_indices].numpy())
    return np.vstack(logits), np.vstack(labels)


def maybe_subsample(X: np.ndarray, Y: np.ndarray, max_samples: int, seed: int):
    if max_samples <= 0 or len(X) <= max_samples:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], Y[idx]


def plot_tsne(embeddings, labels, label_names, title, save_path, perplexity=40, seed=42):
    perplexity = min(perplexity, max(5, (len(embeddings) - 1) // 3))
    print(f"  [t-SNE] Running on {len(embeddings)} samples (perplexity={perplexity}) ...")
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    ).fit_transform(embeddings)

    groups = label_to_group(labels, label_names)
    unique_groups = sorted(set(groups), key=lambda g: (g == "None", g == "Multi-label", g))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, group in enumerate(unique_groups):
        mask = groups == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.68,
            edgecolors="none",
            color=cmap(i % 10),
            label=f"{group} (n={mask.sum()})",
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [t-SNE] Saved -> {save_path}")


def cosine_distance_matrix(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / (norms + 1e-8)
    return 1.0 - (A @ A.T)


def compute_distance_stats(embeddings, labels, label_names):
    groups = label_to_group(labels, label_names)
    dist = cosine_distance_matrix(embeddings)
    intra, inter = [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] == groups[j]:
                intra.append(dist[i, j])
            else:
                inter.append(dist[i, j])
    counts = {group: int((groups == group).sum()) for group in sorted(set(groups))}
    return {
        "group_counts": counts,
        "intra_mean": float(np.mean(intra)) if intra else float("nan"),
        "inter_mean": float(np.mean(inter)) if inter else float("nan"),
        "separation_ratio": float(np.mean(inter) / (np.mean(intra) + 1e-8)) if intra and inter else float("nan"),
    }


def run_one(ckpt_dir, cfg, task, output, tag, args, device):
    dataset, groups = build_test_dataset(cfg)
    label_names, label_indices = groups[TASK_TO_GROUP[task]]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = load_model(ckpt_dir, cfg, device)
    logits, labels = extract_task_logits(model, loader, task, label_indices, device)
    logits, labels = maybe_subsample(logits, labels, args.max_samples, args.seed)

    task_out = os.path.join(output, task)
    os.makedirs(task_out, exist_ok=True)
    plot_tsne(
        logits,
        labels,
        label_names,
        f"{task.title()} logit space — {tag}",
        os.path.join(task_out, f"tsne_{tag}.png"),
        perplexity=args.perplexity,
        seed=args.seed,
    )
    stats = compute_distance_stats(logits, labels, label_names)
    with open(os.path.join(task_out, f"embedding_stats_{tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  [Stats] {task}/{tag}: separation_ratio={stats['separation_ratio']:.4f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Task embedding/logit quality analysis (t-SNE + distance stats)")
    parser.add_argument("--dir", required=True, help="Checkpoint directory to analyze")
    parser.add_argument("--task", required=True, choices=["arrhythmia", "mi", "conduction"])
    parser.add_argument("--config", default=None, help="Optional config path; defaults to checkpoint config_snapshot.yaml")
    parser.add_argument("--baseline-dir", default=None, help="Optional checkpoint dir for comparison")
    parser.add_argument("--baseline-config", default=None, help="Optional baseline config path")
    parser.add_argument("--output", default="results/task_embedding_analysis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=2500, help="Subsample for t-SNE; <=0 disables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_cfg(args.dir, args.config)
    all_stats = {
        "candidate": run_one(args.dir, cfg, args.task, args.output, "candidate", args, device),
    }
    if args.baseline_dir:
        base_cfg = load_cfg(args.baseline_dir, args.baseline_config)
        all_stats["baseline"] = run_one(args.baseline_dir, base_cfg, args.task, args.output, "baseline", args, device)

    summary_path = os.path.join(args.output, args.task, "embedding_stats_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[OK] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
