"""
analyze_task_stability.py
────────────────────────────────────────────────────────────────────
Task-agnostic extension of the DIR1 MI stability analysis.

Reads history.json from two checkpoint directories and plots train/val curves
plus an epoch-to-epoch stability score (std of absolute deltas). Supports
arrhythmia, MI, and conduction metrics in the current decoupled architecture.

Usage:
    python scripts/analyze_task_stability.py \
        --baseline-dir checkpoints/run_20260618_233545_dir5_p3a_imi50-decoupled_multitask-aug \
        --candidate-dir checkpoints/run_20260619_193427_branche10_05_cond_depth-decoupled_multitask-aug \
        --task conduction \
        --output results/task_stability_b10_05
"""

import argparse
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml


TASK_PREFIX = {
    "arrhythmia": "arrhy",
    "mi": "mi",
    "conduction": "cond",
}

TASK_LOSS = {
    "arrhythmia": "loss/arrhythmia",
    "mi": "loss/mi",
    "conduction": "loss/conduction",
}


def load_history(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"history.json not found in: {ckpt_dir}")
    with open(path, "r") as f:
        return json.load(f)


def load_cfg(ckpt_dir: str, config_path: str = None) -> dict:
    path = config_path or os.path.join(ckpt_dir, "config_snapshot.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def group_label_names(cfg: dict, task: str) -> List[str]:
    idx_to_name = {label["index"]: label["name"] for label in cfg.get("labels", [])}
    group = "arrhythmia" if task == "arrhythmia" else task
    return [idx_to_name[i] for i in cfg.get("label_groups", {}).get(group, [])]


def extract_metric(history_list, key):
    return [epoch.get(key, float("nan")) for epoch in history_list]


def finite(values):
    return [v for v in values if not np.isnan(v)]


def compute_stability(values):
    vals = finite(values)
    if len(vals) < 2:
        return float("nan")
    return float(np.std(np.abs(np.diff(vals))))


def metric_summary(values):
    vals = finite(values)
    if not vals:
        return {"stability": float("nan"), "peak": float("nan"), "final": float("nan")}
    return {
        "stability": compute_stability(values),
        "peak": float(np.max(vals)),
        "final": float(vals[-1]),
    }


def plot_metric_comparison(
    epochs_bl,
    bl_train,
    bl_val,
    epochs_cand,
    cand_train,
    cand_val,
    ylabel,
    title,
    save_path,
    ymin=None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_bl, bl_train, color="#7f7f7f", lw=1.8, linestyle="--", label="Baseline train")
    if bl_val is not None:
        ax.plot(epochs_bl, bl_val, color="#c7c7c7", lw=1.8, linestyle="-", label="Baseline val")
    ax.plot(epochs_cand, cand_train, color="#d62728", lw=1.8, linestyle="--", label="Candidate train")
    if cand_val is not None:
        ax.plot(epochs_cand, cand_val, color="#ff9896", lw=1.8, linestyle="-", label="Candidate val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Chart] Saved -> {save_path}")


def build_metric_specs(task: str, labels: List[str]):
    prefix = TASK_PREFIX[task]
    specs = [
        {
            "key": f"auprc/{prefix}/macro",
            "ylabel": "Macro AUPRC",
            "title": f"{task.title()} Macro AUPRC Stability",
            "filename": f"{task}_macro_auprc_stability.png",
            "ymin": 0.0,
        },
        {
            "key": f"f1/{prefix}/macro",
            "ylabel": "Macro F1",
            "title": f"{task.title()} Macro F1 Stability",
            "filename": f"{task}_macro_f1_stability.png",
            "ymin": 0.0,
        },
        {
            "key": TASK_LOSS[task],
            "ylabel": "Loss",
            "title": f"{task.title()} Loss Stability",
            "filename": f"{task}_loss_stability.png",
            "ymin": 0.0,
        },
    ]
    for label in labels:
        specs.append(
            {
                "key": f"auprc/{prefix}/{label}",
                "ylabel": "AUPRC",
                "title": f"{label} AUPRC Stability",
                "filename": f"{task}_{label}_auprc_stability.png",
                "ymin": 0.0,
            }
        )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Task stability analysis across training histories")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--task", required=True, choices=["arrhythmia", "mi", "conduction"])
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--candidate-config", default=None)
    parser.add_argument("--output", default="results/task_stability_analysis")
    args = parser.parse_args()

    task_out = os.path.join(args.output, args.task)
    os.makedirs(task_out, exist_ok=True)

    bl_hist = load_history(args.baseline_dir)
    cand_hist = load_history(args.candidate_dir)
    cand_cfg = load_cfg(args.candidate_dir, args.candidate_config)
    labels = group_label_names(cand_cfg, args.task)

    bl_train, bl_val = bl_hist.get("train", []), bl_hist.get("val", [])
    cand_train, cand_val = cand_hist.get("train", []), cand_hist.get("val", [])
    epochs_bl = list(range(1, len(bl_train) + 1))
    epochs_cand = list(range(1, len(cand_train) + 1))

    specs = build_metric_specs(args.task, labels)
    summary = {}

    print(f"\nTask: {args.task}")
    print(f"Baseline epochs: {len(epochs_bl)} | Candidate epochs: {len(epochs_cand)}")
    print("Stability = std(abs(epoch-to-epoch delta)); lower is smoother.\n")

    for spec in specs:
        key = spec["key"]
        bl_t = extract_metric(bl_train, key)
        bl_v = extract_metric(bl_val, key) if bl_val else None
        cand_t = extract_metric(cand_train, key)
        cand_v = extract_metric(cand_val, key) if cand_val else None

        summary[key] = {
            "baseline_train": metric_summary(bl_t),
            "baseline_val": metric_summary(bl_v or []),
            "candidate_train": metric_summary(cand_t),
            "candidate_val": metric_summary(cand_v or []),
        }

        print(
            f"{key:<28} | "
            f"BL val std={summary[key]['baseline_val']['stability']:.4f}, "
            f"CAND val std={summary[key]['candidate_val']['stability']:.4f}, "
            f"BL peak={summary[key]['baseline_val']['peak']:.4f}, "
            f"CAND peak={summary[key]['candidate_val']['peak']:.4f}"
        )

        plot_metric_comparison(
            epochs_bl,
            bl_t,
            bl_v,
            epochs_cand,
            cand_t,
            cand_v,
            ylabel=spec["ylabel"],
            title=spec["title"],
            save_path=os.path.join(task_out, spec["filename"]),
            ymin=spec.get("ymin"),
        )

    summary_path = os.path.join(task_out, "stability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
