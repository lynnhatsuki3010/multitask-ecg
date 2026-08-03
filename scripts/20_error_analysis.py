"""
20_error_analysis.py
--------------------
Where the model's mistakes go, on the test fold, for a finished run.

The paper claims the two anatomically overlapping MI pairs - IMI/ILMI and
ASMI/AMI - are where the errors concentrate, and that the gated contrast module
exists to separate them. That claim needs the actual confusion structure, not a
per-label F1 table: a false positive on IMI means something different depending on
whether the record was ILMI or NORM.

Predictions use the per-class thresholds from calibration_results.json, so the
numbers here match the F1 values reported in the results tables.

Usage:
  python scripts/20_error_analysis.py --run checkpoints/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessing import PTBXLDataset
from src.models.factory import build_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SIBLINGS = [("IMI", "ILMI"), ("ASMI", "AMI")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
    cfg["hrv"] = {"enabled": False}

    fs = cfg["dataset"]["sampling_rate"]
    proc = f"{cfg['paths']['processed']}_{fs}hz"
    splits = f"{cfg['paths']['splits']}_{fs}hz"

    df = pd.read_csv(Path(proc) / "metadata.csv")
    lm = np.load(Path(proc) / "label_matrix.npy")
    idx = np.load(Path(splits) / "test_indices.npy")

    prep = cfg.get("preprocessing", {})
    ds = PTBXLDataset(
        metadata=df.iloc[idx], label_matrix=lm[idx], hrv_matrix=None,
        base_path=cfg["paths"]["raw_data"], sampling_rate=fs,
        target_length=cfg["dataset"]["signal_length"],
        bandpass=(prep.get("bandpass_low", 0.5), prep.get("bandpass_high", 40.0)),
        notch=prep.get("notch_freq", 50.0), normalize=prep.get("normalize", "zscore"),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    groups = cfg["label_groups"]
    names = {l["index"]: l["name"] for l in cfg["labels"]}
    order = {t: [names[i] for i in groups[t]] for t in ("arrhythmia", "mi", "conduction")}
    flat = [n for t in ("arrhythmia", "mi", "conduction") for n in order[t]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(groups["arrhythmia"]), len(groups["mi"]), 0,
                        len(groups["conduction"])).to(device).eval()
    ck = torch.load(str(run / "best_model.pth"), map_location=device)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)))

    probs = {t: [] for t in order}
    with torch.no_grad():
        for batch in loader:
            out = model(batch["signal"].to(device))
            for t in order:
                probs[t].append(torch.sigmoid(out[t]).cpu().numpy())
    P = np.concatenate([np.concatenate(probs[t]) for t in
                        ("arrhythmia", "mi", "conduction")], axis=1)
    Y = lm[idx]

    thr = json.loads((run / "calibration_results.json").read_text(encoding="utf-8"))["thresholds"]
    tau = np.array([thr.get(n, 0.5) for n in flat])
    pred = (P >= tau).astype(int)

    col = {n: i for i, n in enumerate(flat)}
    L = [f"# Phân tích lỗi — `{run.name}`", "",
         f"Tập test: {len(idx)} bản ghi. Ngưỡng lấy từ `calibration_results.json`.", ""]

    L += ["## Cặp nhãn anh em — cấu trúc nhầm lẫn", ""]
    for a, b in SIBLINGS:
        ia, ib = col[a], col[b]
        for x, y in ((a, b), (b, a)):
            ix, iy = col[x], col[y]
            fn = (Y[:, ix] == 1) & (pred[:, ix] == 0)          # bỏ sót x
            fp = (Y[:, ix] == 0) & (pred[:, ix] == 1)          # báo nhầm thành x
            fn_is_y = int((fn & (Y[:, iy] == 1)).sum())
            fn_pred_y = int((fn & (pred[:, iy] == 1)).sum())
            fp_is_y = int((fp & (Y[:, iy] == 1)).sum())
            fp_clean = int((fp & (Y[:, ia] == 0) & (Y[:, ib] == 0)).sum())
            L += [f"### {x} (support {int(Y[:, ix].sum())})", "",
                  f"- Bỏ sót: **{int(fn.sum())}** ca. Trong đó **{fn_is_y}** ca thực tế cũng mang "
                  f"nhãn {y}, và **{fn_pred_y}** ca được mô hình gán sang {y}.",
                  f"- Báo nhầm: **{int(fp.sum())}** ca. Trong đó **{fp_is_y}** ca thực tế là {y} "
                  f"(nhầm trong cặp), **{fp_clean}** ca không mang nhãn nào của cặp.", ""]

    L += ["## Nhãn nào bị nhầm thành nhãn nào (nhóm MI)", "",
          "Hàng = nhãn thật, cột = nhãn được dự đoán. Ô (i,j) = số ca mang nhãn i và được gán j.", "",
          "| thật \\ dự đoán | " + " | ".join(order["mi"]) + " | không gán MI nào |",
          "|---" * (len(order["mi"]) + 2) + "|"]
    for a in order["mi"]:
        ia = col[a]
        mask = Y[:, ia] == 1
        cells = [str(int((mask & (pred[:, col[b]] == 1)).sum())) for b in order["mi"]]
        none = int((mask & (pred[:, [col[b] for b in order["mi"]]].sum(axis=1) == 0)).sum())
        L.append(f"| **{a}** ({int(mask.sum())}) | " + " | ".join(cells) + f" | {none} |")
    L.append("")

    L += ["## Tỷ lệ dương tính giả đến từ đâu (mọi nhãn)", "",
          "| Nhãn | FP | FP là nhãn anh em | FP là NORM |", "|---|---|---|---|"]
    norm_i = col["NORM"]
    sib_of = {a: b for a, b in SIBLINGS} | {b: a for a, b in SIBLINGS}
    for n in flat:
        i = col[n]
        fp = (Y[:, i] == 0) & (pred[:, i] == 1)
        s = sib_of.get(n)
        sib = int((fp & (Y[:, col[s]] == 1)).sum()) if s else 0
        nrm = int((fp & (Y[:, norm_i] == 1)).sum())
        L.append(f"| {n} | {int(fp.sum())} | {sib if s else '–'} | {nrm} |")

    report = "\n".join(L)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
