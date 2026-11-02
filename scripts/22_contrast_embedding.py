"""
22_contrast_embedding.py
------------------------
What the gated contrast actually does to the per-label region representations.

The t-SNE currently in the paper embeds the four MI *logits*, so it can only show
where the decision lands - not whether the proposed mechanism did anything. This
script embeds the 128-d region-context vectors instead, and measures the module's
effect directly by capturing the same vectors twice: before the contrast (c) and
after it (c-tilde).

Three outputs, in increasing order of what they prove:

  1. t-SNE of the refined context vectors, one point per record. The direct
     replacement for the paper's Figure 3, on representations rather than logits.

  2. Sibling gap, before vs after. For one record the model builds both c_IMI and
     c_ILMI; the contrast claims to push them apart. Measuring ||c_a - c_b|| before
     and after the module quantifies exactly that claim, per record, with no
     appeal to downstream metrics.

  3. Gate activation. Section 3.5 argues the module starts as an identity map
     (b_g = -2, so g ~ 0.12) and "only opens where the gradient rewards it". That
     is an assertion about training dynamics the paper never checks. The learned
     gate is readable straight off the checkpoint: if it stayed near its
     initialisation the mechanism was dormant and the IMI gain came from
     somewhere else.

Nothing is monkey-patched. Forward hooks capture the four pooling outputs, and the
contrast modules are then re-applied in the same order the model uses, which
reproduces c-tilde exactly because both modules are deterministic in eval mode.

Usage:
  python scripts/22_contrast_embedding.py --checkpoint checkpoints/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

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

from src.data.preprocessing import PTBXLDataset   # noqa: E402
from src.models.factory import build_model        # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MI_LABELS = ["IMI", "ASMI", "ILMI", "AMI"]
PAIRS = [("IMI", "ILMI", "inf_contrast"), ("ASMI", "AMI", "ant_contrast")]


def label_group(y: np.ndarray, names) -> np.ndarray:
    """One group per record: its single MI label, 'Multi-label', or 'None'."""
    out = []
    for row in y:
        hits = [n for n, v in zip(names, row) if v == 1]
        out.append(hits[0] if len(hits) == 1 else ("Multi-label" if hits else "None"))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="run directory")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--perplexity", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/fig3_contrast")
    args = ap.parse_args()

    run = Path(args.checkpoint)
    cfg = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
    cfg["hrv"] = {"enabled": False}

    mode = cfg["model"].get("mi_head_mode")
    routing = cfg["model"].get("routing_mode")
    if mode != "contrast_v2" or routing != "decoupled":
        raise SystemExit(f"Cần mi_head_mode=contrast_v2 và routing_mode=decoupled; "
                         f"checkpoint này là {mode!r} / {routing!r}.")

    fs = cfg["dataset"]["sampling_rate"]
    proc = Path(f"{cfg['paths']['processed']}_{fs}hz")
    splits = Path(f"{cfg['paths']['splits']}_{fs}hz")
    df = pd.read_csv(proc / "metadata.csv")
    lm = np.load(proc / "label_matrix.npy")
    idx = np.load(splits / f"{args.split}_indices.npy")

    pr = cfg.get("preprocessing", {})
    ds = PTBXLDataset(
        metadata=df.iloc[idx], label_matrix=lm[idx], hrv_matrix=None,
        base_path=cfg["paths"]["raw_data"], sampling_rate=fs,
        target_length=cfg["dataset"]["signal_length"],
        bandpass=(pr.get("bandpass_low", 0.5), pr.get("bandpass_high", 40.0)),
        notch=pr.get("notch_freq", 50.0), normalize=pr.get("normalize", "zscore"),
        augment=False,
    )

    groups = cfg["label_groups"]
    names = {l["index"]: l["name"] for l in cfg["labels"]}
    mi_names = [names[i] for i in groups["mi"]]
    if mi_names != MI_LABELS:
        raise SystemExit(f"Thứ tự nhãn MI bất ngờ: {mi_names}")
    flat = [names[i] for t in ("arrhythmia", "mi", "conduction") for i in groups[t]]
    mi_cols = [flat.index(n) for n in MI_LABELS]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(groups["arrhythmia"]), len(groups["mi"]), 0,
                        len(groups["conduction"])).to(device).eval()
    ck = torch.load(str(run / "best_model.pth"), map_location=device)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)))
    head = model.mi_head

    # Hook the four label poolings: their outputs are exactly c_IMI..c_AMI, the
    # vectors the contrast consumes.
    caught: Dict[str, torch.Tensor] = {}
    handles = []
    for lab in MI_LABELS:
        pool = getattr(head, f"{lab.lower()}_pool")
        handles.append(pool.register_forward_hook(
            lambda m, i, o, lab=lab: caught.__setitem__(lab, o.detach())))

    pre = {l: [] for l in MI_LABELS}
    post = {l: [] for l in MI_LABELS}
    gates = {p[2]: [] for p in PAIRS}

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Suy luận trên {len(ds)} bản ghi ({args.split}) ...")
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            caught.clear()
            model(batch["signal"].to(device))
            for a, b, mod_name in PAIRS:
                mod = getattr(head, mod_name)
                ca, cb = caught[a], caught[b]
                # Same call order as _contrast_v2_refs.
                post[a].append(mod(ca, cb).cpu().numpy())
                post[b].append(mod(cb, ca).cpu().numpy())
                pre[a].append(ca.cpu().numpy())
                pre[b].append(cb.cpu().numpy())
                # Gate for the a-direction, recomputed from the module's own weights.
                g = torch.sigmoid(mod.gate(torch.cat([ca, ca - cb], dim=-1)))
                gates[mod_name].append(g.cpu().numpy())
            if (bi + 1) % 20 == 0:
                print(f"    {(bi + 1) * args.batch_size}/{len(ds)}")
    for h in handles:
        h.remove()

    pre = {k: np.concatenate(v) for k, v in pre.items()}
    post = {k: np.concatenate(v) for k, v in post.items()}
    gates = {k: np.concatenate(v) for k, v in gates.items()}
    Y = lm[idx][:, mi_cols]
    grp = label_group(Y, MI_LABELS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. t-SNE
    X = np.concatenate([post[l] for l in MI_LABELS], axis=1)   # (N, 4*d)
    print(f"[+] t-SNE trên {X.shape[1]} chiều, {len(X)} điểm ...")
    perp = min(args.perplexity, max(5, (len(X) - 1) // 3))
    coords = TSNE(n_components=2, perplexity=perp, random_state=args.seed,
                  max_iter=1000, init="pca", learning_rate="auto").fit_transform(X)

    order = sorted(set(grp), key=lambda g: (g == "None", g == "Multi-label", g))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, g in enumerate(order):
        m = grp == g
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.68, edgecolors="none",
                   color=cmap(i % 10), label=f"{g} (n={m.sum()})")
    ax.set_title("MI region-context space after gated contrast", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig_path = out / "tsne_context.png"
    plt.savefig(fig_path, dpi=200); plt.close()
    print(f"    lưu -> {fig_path}")

    # ------------------------------------------- 2. sibling gap, before vs after
    stats: Dict[str, object] = {"n_records": int(len(X)), "split": args.split,
                                "group_counts": {g: int((grp == g).sum()) for g in order}}
    print("\nKHOẢNG CÁCH GIỮA HAI NHÃN ANH EM (chuẩn L2, trung bình mỗi bản ghi)")
    print(f"{'cặp':14s} {'trước':>8s} {'sau':>8s} {'tỷ số':>8s}")
    for a, b, mod_name in PAIRS:
        d0 = np.linalg.norm(pre[a] - pre[b], axis=1)
        d1 = np.linalg.norm(post[a] - post[b], axis=1)
        stats[f"gap_{a}_{b}"] = {
            "before_mean": float(d0.mean()), "after_mean": float(d1.mean()),
            "ratio": float(d1.mean() / (d0.mean() + 1e-8)),
            "pct_records_increased": float((d1 > d0).mean() * 100),
        }
        print(f"{a+'/'+b:14s} {d0.mean():8.3f} {d1.mean():8.3f} {d1.mean()/d0.mean():8.3f}"
              f"   (tăng ở {(d1 > d0).mean()*100:.1f}% bản ghi)")

    # ------------------------------------------------------- 3. gate activation
    print("\nCỔNG ĐÃ HỌC  (khởi tạo b_g = −2  ->  g ≈ 0,119)")
    print(f"{'mô-đun':14s} {'g trung bình':>13s} {'g trung vị':>11s} {'% chiều > 0,5':>14s}")
    for a, b, mod_name in PAIRS:
        g = gates[mod_name]
        per_dim = g.mean(axis=0)
        stats[f"gate_{mod_name}"] = {
            "mean": float(g.mean()), "median": float(np.median(g)),
            "p05": float(np.percentile(g, 5)), "p95": float(np.percentile(g, 95)),
            "pct_dims_above_half": float((per_dim > 0.5).mean() * 100),
            "init_value": float(1 / (1 + np.exp(2))),
        }
        print(f"{mod_name:14s} {g.mean():13.3f} {np.median(g):11.3f} "
              f"{(per_dim > 0.5).mean()*100:13.1f}%")

    (out / "contrast_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLưu thống kê -> {out / 'contrast_stats.json'}")


if __name__ == "__main__":
    main()
