"""
mc6_ami_threshold.py
--------------------
Quantify the false-negative / false-positive trade-off for the rare MI labels.

Review-2 MC6/MC9: AMI recall is 0.238 +/- 0.126, so roughly three in four true
anterior infarctions are missed. In an emergency-cardiology setting a missed
anterior MI and a false alarm are not equally costly, and the paper reports a
single operating point chosen to maximise F1 - a criterion that weights the two
errors equally and therefore encodes a clinical assumption nobody stated.

For each label this reports three operating points a reader can reason about:
the one the paper actually deploys, the one that reaches a target recall, and
the F-beta-optimal one for a stated cost ratio. Numbers, not adjectives, for the
discussion MC9 asks for.

EVERY threshold here is chosen on VALIDATION and then measured on TEST. Sweeping
the threshold on test and reporting the best point would be choosing with the
answer in hand: it inflates recall (for AMI, 0.33 instead of the 0.18 the
deployed threshold actually delivers) and is the first thing a reviewer would
catch. The deployed row is read straight from calibration_results.json, so it
reproduces the number in the paper's own table.

Both folds are re-run through the model because calibration_results.json stores
thresholds and summary metrics, not per-sample scores.

Usage:
  python scripts/mc6_ami_threshold.py --glob "checkpoints/run_*abl_arch10_contrast*"
  python scripts/mc6_ami_threshold.py --glob "..." --labels AMI ILMI --beta 3 --target-recall 0.8
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory import build_model               # noqa: E402

# Reuse 03_calibrate's own data path rather than re-deriving it: the bandpass,
# notch, normalisation, collate and split handling all have to match the run
# being analysed, and a second copy of that logic would drift silently. The
# module name starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location(
    "calibrate_mod", Path(__file__).resolve().parent / "03_calibrate.py")
_cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cal)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_run(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    cal = json.loads((run_dir / "calibration_results.json").read_text(encoding="utf-8"))
    return cfg, cal


def label_groups(cfg: dict):
    """Index groups exactly as 03_calibrate derives them (task names included)."""
    label_cfgs = cfg["labels"]
    names = [l["name"] for l in label_cfgs]
    arrhy = [l["index"] for l in label_cfgs if l["task"] in ("arrhythmia", "normal")]
    mi    = [l["index"] for l in label_cfgs if l["task"] == "mi"]
    cond  = [l["index"] for l in label_cfgs if l["task"] == "conduction"]
    return names, arrhy, mi, cond


def sweep(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """Precision / recall / F1 at every threshold, for one label."""
    tp = ((probs[None, :] >= thresholds[:, None]) & (targets[None, :] == 1)).sum(axis=1)
    fp = ((probs[None, :] >= thresholds[:, None]) & (targets[None, :] == 0)).sum(axis=1)
    fn = ((probs[None, :] <  thresholds[:, None]) & (targets[None, :] == 1)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall    = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.where(denom > 0, (1 + b2) * precision * recall / np.maximum(denom, 1e-12), 0.0)


def dataset_fingerprint(cfg: dict) -> tuple:
    """Everything that decides what the test fold contains and how it is filtered.

    All ten seeds of one config share this, which is what makes caching sound;
    if a future glob mixes configs that differ here, the cache is rebuilt rather
    than silently reused across incompatible preprocessing.
    """
    ds = cfg["dataset"]
    paths = cfg["paths"]
    prep = cfg.get("preprocessing", {})
    return (
        paths["processed"], paths["splits"], paths["raw_data"],
        ds["sampling_rate"], ds["signal_length"], ds.get("label_threshold"),
        ds.get("test_fold"), ds.get("val_fold"), ds.get("split_seed"),
        prep.get("bandpass_low"), prep.get("bandpass_high"),
        prep.get("notch_freq"), prep.get("normalize"),
        cfg["training"]["batch_size"],
    )


def load_folds(cfg: dict):
    """Materialise the val and test folds once: {split: (signals, labels)}.

    Reading them per run is what made this slow - every sample is re-read with
    wfdb and re-filtered, and 03_calibrate's loader is fixed at num_workers=0.
    The ten seeds of one config share both folds, so they are read once.
    """
    df, label_matrix, hrv_matrix, splits = _cal.load_data_from_cfg(cfg)
    out = {}
    for name in ("val", "test"):
        loader = _cal.build_loader(cfg, df, label_matrix, hrv_matrix,
                                   splits[name], augment=False)
        signals, labels = [], []
        for batch in tqdm(loader, desc=f"  caching {name} fold", unit="batch"):
            signals.append(batch["signal"])
            labels.append(batch["labels"])
        out[name] = (torch.cat(signals), torch.cat(labels))
    return out


@torch.no_grad()
def mi_logits(model, signals: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """MI-head logits for the cached fold."""
    model.eval()
    out = []
    for i in tqdm(range(0, len(signals), batch_size), desc="  scoring", unit="batch"):
        chunk = signals[i:i + batch_size].to(device)
        out.append(model(chunk)["mi"].cpu())
    return torch.cat(out)


def measure_at(probs: np.ndarray, targets: np.ndarray, threshold: float) -> Dict[str, float]:
    """Precision / recall / false positives at one fixed threshold."""
    pred = probs >= threshold
    tp = int((pred & (targets == 1)).sum())
    fp = int((pred & (targets == 0)).sum())
    fn = int((~pred & (targets == 1)).sum())
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "fp": float(fp),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="checkpoints/run_*abl_arch10_contrast*",
                    help="run folders to sweep (default: the reported model)")
    ap.add_argument("--labels", nargs="*", default=["AMI", "IMI", "ASMI", "ILMI"])
    ap.add_argument("--beta", type=float, default=3.0,
                    help="F-beta weight: beta=3 treats one missed infarction as "
                         "costly as nine false alarms. State the value you use.")
    ap.add_argument("--target-recall", type=float, default=0.80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dirs = sorted(Path(p) for p in glob.glob(args.glob)
                      if (Path(p) / "calibration_results.json").exists())
    if not run_dirs:
        sys.exit(f"no calibrated runs matched {args.glob!r}")

    grid = np.linspace(0.01, 0.99, 197)
    per_label: Dict[str, Dict[str, List[float]]] = {
        name: {"dep_thr": [], "dep_recall": [], "dep_precision": [],
               "fb_thr": [], "fb_recall": [], "fb_precision": [],
               "tr_thr": [], "tr_recall": [], "tr_precision": [], "tr_fp": [],
               "tr_reachable": [], "n_pos": []}
        for name in args.labels
    }

    cache_key = None
    folds = None

    for run_dir in run_dirs:
        cfg, cal = load_run(run_dir)
        names, arrhy_idx, mi_idx, cond_idx = label_groups(cfg)
        mi_names = [names[i] for i in mi_idx]
        hrv_features = cfg.get("hrv", {}).get("features", ["rmssd", "sdnn", "mean_hr"])

        fp_key = dataset_fingerprint(cfg)
        if fp_key != cache_key:
            folds = load_folds(cfg)
            cache_key = fp_key

        model = build_model(
            cfg=cfg,
            num_arrhythmia_labels=len(arrhy_idx),
            num_mi_labels=len(mi_idx),
            num_conduction_labels=len(cond_idx),
            num_hrv_targets=len(hrv_features),
        ).to(args.device)
        ckpt = torch.load(run_dir / "best_model.pth", map_location=args.device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])

        temp = float(cal["temperature"]["mi"])
        scored = {}
        for split in ("val", "test"):
            sig, lab = folds[split]
            logits = mi_logits(model, sig, args.batch_size, args.device)
            scored[split] = (torch.sigmoid(logits / temp).numpy(), lab[:, mi_idx].numpy())

        for name in args.labels:
            if name not in mi_names:
                continue
            col = mi_names.index(name)
            vp, vt = scored["val"][0][:, col], scored["val"][1][:, col]
            tp_, tt = scored["test"][0][:, col], scored["test"][1][:, col]
            rec = per_label[name]
            rec["n_pos"].append(int(tt.sum()))

            # Deployed point: threshold from calibration, i.e. the paper's number.
            thr_dep = float(cal["thresholds"][name])
            m = measure_at(tp_, tt, thr_dep)
            rec["dep_thr"].append(thr_dep)
            rec["dep_recall"].append(m["recall"])
            rec["dep_precision"].append(m["precision"])

            # Every other point: chosen on val, measured on test.
            sv = sweep(vp, vt, grid)

            i_fb = int(np.argmax(fbeta(sv["precision"], sv["recall"], args.beta)))
            m = measure_at(tp_, tt, float(grid[i_fb]))
            rec["fb_thr"].append(float(grid[i_fb]))
            rec["fb_recall"].append(m["recall"])
            rec["fb_precision"].append(m["precision"])

            hit = np.where(sv["recall"] >= args.target_recall)[0]
            rec["tr_reachable"].append(1.0 if len(hit) else 0.0)
            if len(hit):
                thr_tr = float(grid[int(hit[-1])])
                m = measure_at(tp_, tt, thr_tr)
                rec["tr_thr"].append(thr_tr)
                rec["tr_recall"].append(m["recall"])
                rec["tr_precision"].append(m["precision"])
                rec["tr_fp"].append(m["fp"])

    def cell(values: List[float], fmt: str = "{:.3f}") -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return fmt.format(values[0])
        return f"{fmt.format(stats.mean(values))} +/- {fmt.format(stats.stdev(values))}"

    lines = [
        "# MI operating points (thresholds chosen on validation, measured on test)",
        "",
        f"Runs: {len(run_dirs)}  |  F-beta with beta={args.beta}  |  "
        f"target recall {args.target_recall:.2f} (on validation)",
        "",
        "The deployed row is the operating point the paper reports: the threshold",
        "tuned during calibration and applied unchanged to test. The other rows",
        "move that threshold under a different cost assumption, chosen on",
        "validation and then measured on test - never swept on test itself.",
        "",
    ]
    for name in args.labels:
        rec = per_label[name]
        if not rec["dep_thr"]:
            continue
        lines += [
            f"## {name}  (test positives: {cell(rec['n_pos'], '{:.0f}')})",
            "",
            "| operating point | threshold | recall (test) | precision (test) |",
            "|---|---|---|---|",
            f"| deployed, F1-tuned (reported) | {cell(rec['dep_thr'])} | "
            f"{cell(rec['dep_recall'])} | {cell(rec['dep_precision'])} |",
            f"| F-beta={args.beta:g}, tuned on val | {cell(rec['fb_thr'])} | "
            f"{cell(rec['fb_recall'])} | {cell(rec['fb_precision'])} |",
        ]
        reach = sum(rec["tr_reachable"])
        if rec["tr_thr"]:
            lines.append(
                f"| val recall >= {args.target_recall:.2f} | {cell(rec['tr_thr'])} | "
                f"{cell(rec['tr_recall'])} | {cell(rec['tr_precision'])} |")
            lines += [
                "",
                f"Chasing {args.target_recall:.0%} recall on validation costs "
                f"{cell(rec['tr_fp'], '{:.0f}')} false positives on test, and the "
                f"target is reachable on validation in {reach:.0f}/{len(run_dirs)} "
                f"seeds. Test recall at that same threshold is "
                f"{cell(rec['tr_recall'])}; the shortfall is how badly the "
                f"threshold transfers between folds, which for a label this rare "
                f"is itself a finding.",
            ]
        else:
            lines += ["", f"**No threshold reaches {args.target_recall:.0%} recall on "
                          f"validation in any seed.**"]
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
