"""
make_paper_tables.py
--------------------
Build the paper's result tables from a finished run's calibration output.

Reads calibration_results.json, which holds both `baseline_metrics` (thresholds at
their trained values) and `calibrated_metrics` (temperature scaling + per-class
threshold search). The paper reports the calibrated numbers, so every table here
comes from the same block - mixing the two is how a "+0.068 from calibration" and a
"0.545 in test_metrics.json" end up describing the same run.

With several seeds it also prints mean+/-SD, since single-seed cells on rare labels
carry more spread than the differences the paper draws conclusions from.

Usage:
  python scripts/make_paper_tables.py --glob "checkpoints/run_*abl_arch10b*"
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TASKS = [
    ("Rối loạn nhịp", "arrhy", ["NORM", "AFIB", "STACH", "PVC", "AFLT"]),
    ("Nhồi máu cơ tim", "mi", ["IMI", "ASMI", "ILMI", "AMI"]),
    ("Rối loạn dẫn truyền", "cond", ["LBBB", "RBBB", "IRBBB", "1AVB"]),
]
COLS = ["auroc", "auprc", "f1", "prec", "rec"]


def load_runs(pattern: str) -> List[Dict]:
    runs = []
    for d in sorted(glob.glob(pattern)):
        p = Path(d) / "calibration_results.json"
        if not p.exists():
            print(f"[skip] no calibration_results.json in {d}")
            continue
        cal = json.loads(p.read_text(encoding="utf-8"))
        seed = None
        snap = Path(d) / "config_snapshot.yaml"
        if snap.exists():
            seed = (yaml.safe_load(snap.read_text(encoding="utf-8")).get("training") or {}).get("seed")
        runs.append({"dir": d, "seed": seed, "cal": cal})
    return runs


def cell(runs: List[Dict], key: str, block: str = "calibrated_metrics") -> str:
    vals = [r["cal"][block][key] for r in runs if key in r["cal"].get(block, {})]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3f}".replace(".", ",")
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return f"{m:.3f} ± {s:.3f}".replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. "checkpoints/run_*abl_arch10b*"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load_runs(args.glob)
    if not runs:
        print("No runs matched.")
        return
    seeds = [r["seed"] for r in runs]

    L = [f"# Bảng kết quả — {len(runs)} run, seed {seeds}", ""]
    L += [f"Nguồn: `calibrated_metrics` trong `calibration_results.json`.", ""]
    for r in runs:
        L.append(f"- seed {r['seed']}: `{r['dir']}`")
    L += ["", "---", "", "## Tổng hợp theo nhóm tác vụ (test, sau hiệu chỉnh)", "",
          "| Nhóm tác vụ | AUROC | AUPRC | Macro F1 | Precision | Recall |",
          "|---|---|---|---|---|---|"]
    for label, tag, _ in TASKS:
        cells = [cell(runs, f"{c}/{tag}/macro") for c in COLS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += ["", "## Tác động của hiệu chỉnh hậu kỳ", "",
          "| Nhóm/Nhãn | Chỉ số | Trước | Sau |", "|---|---|---|---|"]
    for label, tag, _ in TASKS:
        k = f"f1/{tag}/macro"
        L.append(f"| {label} | Macro F1 | {cell(runs, k, 'baseline_metrics')} | {cell(runs, k)} |")
    L.append(f"| IMI | AUPRC | {cell(runs, 'auprc/mi/IMI', 'baseline_metrics')} | "
             f"{cell(runs, 'auprc/mi/IMI')} |")

    for label, tag, names in TASKS:
        L += ["", f"## Chi tiết — {label}", "",
              "| Nhãn | AUROC | AUPRC | F1 | Precision | Recall |", "|---|---|---|---|---|---|"]
        for n in names:
            cells = [cell(runs, f"{c}/{tag}/{n}") for c in COLS]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        cells = [cell(runs, f"{c}/{tag}/macro") for c in COLS]
        L.append("| **Macro** | " + " | ".join(cells) + " |")

    temps = [r["cal"].get("temperature", {}) for r in runs]
    if temps and temps[0]:
        L += ["", "## Nhiệt độ hiệu chỉnh", "", "| Nhánh | T |", "|---|---|"]
        for t in temps[0]:
            vals = [x[t] for x in temps if t in x]
            v = f"{statistics.mean(vals):.3f}" + (f" ± {statistics.stdev(vals):.3f}" if len(vals) > 1 else "")
            L.append(f"| {t} | {v.replace('.', ',')} |")
        L.append("")
        L.append("T > 1 làm phân phối mềm đi (mô hình quá tự tin trước hiệu chỉnh); "
                 "T < 1 làm nó sắc hơn (thiếu tự tin). Đọc dấu trước khi diễn giải.")

    thr = runs[0]["cal"].get("thresholds", {})
    if thr:
        L += ["", "## Ngưỡng quyết định theo lớp (seed đầu tiên)", "", "| Nhãn | τ |", "|---|---|"]
        for k, v in thr.items():
            L.append(f"| {k} | {v:.2f} |".replace(".", ","))

    report = "\n".join(L)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
