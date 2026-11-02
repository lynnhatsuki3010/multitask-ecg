"""
23_sibling_confusion.py
-----------------------
Table 7 for every arm and seed, on the 320 single-MI-label test records.

Thresholded (Table 7 rule: temperature-scaled sigmoid >= per-class threshold from
calibration_results.json) and threshold-free (AUROC of logit_a - logit_b for
separating a-only from b-only records within each sibling pair).
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.data.preprocessing import PTBXLDataset
from src.models.factory import build_model
from sklearn.metrics import roc_auc_score

dev = torch.device("cuda")
ARMS = {"ARCH-05": "abl_arch05_shared_tf", "ARCH-09": "abl_arch09_decoupled",
        "ARCH-10": "abl_arch10_contrast", "ARCH-12": "abl_arch12_partial"}
MI = ["IMI", "ASMI", "ILMI", "AMI"]
cache = {}


def signals(cfg):
    fs = cfg["dataset"]["sampling_rate"]
    pr = cfg["preprocessing"]
    key = (fs, cfg["dataset"]["signal_length"], pr["bandpass_low"], pr["bandpass_high"],
           pr["notch_freq"], pr["normalize"])
    if key not in cache:
        proc, sp = ROOT / f"{cfg['paths']['processed']}_{fs}hz", ROOT / f"{cfg['paths']['splits']}_{fs}hz"
        df = pd.read_csv(proc / "metadata.csv"); lm = np.load(proc / "label_matrix.npy")
        idx = np.load(sp / "test_indices.npy")
        ds = PTBXLDataset(metadata=df.iloc[idx], label_matrix=lm[idx], hrv_matrix=None,
                          base_path=str(ROOT / cfg["paths"]["raw_data"]), sampling_rate=fs,
                          target_length=cfg["dataset"]["signal_length"],
                          bandpass=(pr["bandpass_low"], pr["bandpass_high"]), notch=pr["notch_freq"],
                          normalize=pr["normalize"], augment=False)
        X = torch.stack([ds[i]["signal"] for i in range(len(ds))])
        cache[key] = (X, lm[idx])
    return cache[key]


rows = []
for arm, pat in ARMS.items():
    for rp in sorted(glob.glob(str(ROOT / f"checkpoints/run_*{pat}-*"))):
        r = Path(rp)
        cfg = yaml.safe_load((r / "config_snapshot.yaml").read_text(encoding="utf-8"))
        cfg["hrv"] = {"enabled": False}
        X, Y = signals(cfg)
        g = cfg["label_groups"]; names = {l["index"]: l["name"] for l in cfg["labels"]}
        flat = [names[i] for t in ("arrhythmia", "mi", "conduction") for i in g[t]]
        mi_names = [names[i] for i in g["mi"]]
        m = build_model(cfg, len(g["arrhythmia"]), len(g["mi"]), 0, len(g["conduction"])).to(dev).eval()
        ck = torch.load(str(r / "best_model.pth"), map_location=dev, weights_only=False)
        m.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)), strict=True)
        cal = json.loads((r / "calibration_results.json").read_text(encoding="utf-8"))
        Tmi = float(cal["temperature"]["mi"])
        with torch.no_grad():
            Z = np.concatenate([m(X[i:i + 64].to(dev))["mi"].cpu().numpy() for i in range(0, len(X), 64)])
        P = 1 / (1 + np.exp(-Z / Tmi))
        tau = np.array([cal["thresholds"].get(n, 0.5) for n in mi_names])
        pred = P >= tau
        ycol = {n: flat.index(n) for n in MI}
        zc = {n: mi_names.index(n) for n in MI}
        Ymi = np.stack([Y[:, ycol[n]] for n in MI], 1)
        single = Ymi.sum(1) == 1
        rec = {"arm": arm, "seed": cfg["training"]["seed"]}
        for i, a in enumerate(MI):
            mask = single & (Ymi[:, i] == 1)
            rec[f"n_{a}"] = int(mask.sum())
            for b in MI:
                rec[f"{a}>{b}"] = int((mask & pred[:, zc[b]]).sum())
            rec[f"{a}>none"] = int((mask & (pred[:, [zc[b] for b in MI]].sum(1) == 0)).sum())
        for a, b in (("IMI", "ILMI"), ("ASMI", "AMI")):
            ma = single & (Y[:, ycol[a]] == 1); mb = single & (Y[:, ycol[b]] == 1)
            s = Z[:, zc[a]] - Z[:, zc[b]]
            yy = np.r_[np.ones(ma.sum()), np.zeros(mb.sum())]
            rec[f"auc_{a}_vs_{b}"] = roc_auc_score(yy, np.r_[s[ma], s[mb]])
            # each head on its own: does the a-head score a-only records above b-only ones?
            rec[f"head_{a}_on_{a}_vs_{b}"] = roc_auc_score(yy, np.r_[Z[ma, zc[a]], Z[mb, zc[a]]])
            rec[f"head_{b}_on_{b}_vs_{a}"] = roc_auc_score(1 - yy, np.r_[Z[ma, zc[b]], Z[mb, zc[b]]])
        rows.append(rec)
        print(arm, rec["seed"], "ILMI>IMI", rec["ILMI>IMI"], "IMI>ILMI", rec["IMI>ILMI"],
              "ILMI>ILMI", rec["ILMI>ILMI"], "auc", round(rec["auc_IMI_vs_ILMI"], 3),
              round(rec["auc_ASMI_vs_AMI"], 3), flush=True)

# usage: python scripts/23_sibling_confusion.py results/sibling_confusion_test.csv
out = Path(sys.argv[1]); pd.DataFrame(rows).to_csv(out, index=False); print("saved", out)
