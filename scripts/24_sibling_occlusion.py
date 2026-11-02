"""
24_sibling_occlusion.py
-----------------------
Two mechanism checks for the sibling-pair result, on the test fold, 4 arms x 10 seeds.

1. Lead occlusion. If the IMI head of ARCH-10/12 rejects ILMI by reading the lateral
   leads ("inferior AND NOT lateral"), zeroing V5-V6 (or I, aVL, V5, V6) on ILMI-only
   records should raise its IMI score and pull its IMI-vs-ILMI AUROC back toward the
   baseline. Zeroing a matched set of anterior leads is the control. Zeroing is the
   same operation as the lead-dropout augmentation the models were trained with.

2. Where the ASMI gain comes from. AUROC of the ASMI head (and IMI head) against
   each negative population: the sibling, NORM, LBBB, other non-MI records.

Probabilities are temperature-scaled with T_mi from calibration_results.json.

Usage:
  python scripts/24_sibling_occlusion.py results/sibling_occlusion_test.csv
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.data.preprocessing import PTBXLDataset
from src.models.factory import build_model
from sklearn.metrics import roc_auc_score, average_precision_score

dev = torch.device("cuda")
ARMS = {"ARCH-05": "abl_arch05_shared_tf", "ARCH-09": "abl_arch09_decoupled",
        "ARCH-10": "abl_arch10_contrast", "ARCH-12": "abl_arch12_partial"}
MI = ["IMI", "ASMI", "ILMI", "AMI"]
# lead order I II III aVR aVL aVF V1..V6
MASKS = {"none": [], "V5V6": [10, 11], "V3V4": [8, 9], "lat4": [0, 4, 10, 11],
         "ant4": [6, 7, 8, 9], "V1V2": [6, 7]}
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
        cache[key] = (torch.stack([ds[i]["signal"] for i in range(len(ds))]), lm[idx])
    return cache[key]


def auc(pos_scores, neg_scores):
    return roc_auc_score(np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))],
                         np.r_[pos_scores, neg_scores])


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
        T = float(cal["temperature"]["mi"])
        tau = {n: cal["thresholds"].get(n, 0.5) for n in mi_names}
        seed = cfg["training"]["seed"]
        y = {n: Y[:, flat.index(n)] == 1 for n in flat}
        ymi = np.stack([y[n] for n in MI], 1)
        single = {n: (ymi.sum(1) == 1) & y[n] for n in MI}
        no_mi = ymi.sum(1) == 0
        add = lambda metric, v: rows.append({"arm": arm, "seed": seed, "metric": metric, "value": float(v)})

        P = {}
        for mk, leads in MASKS.items():
            Xm = X.clone()
            if leads:
                Xm[:, leads, :] = 0.0
            with torch.no_grad():
                Z = np.concatenate([m(Xm[i:i + 64].to(dev))["mi"].cpu().numpy() for i in range(0, len(Xm), 64)])
            P[mk] = {n: 1 / (1 + np.exp(-Z[:, mi_names.index(n)] / T)) for n in mi_names}

        # 1. occlusion
        for mk in MASKS:
            for head, pos, neg in (("IMI", "IMI", "ILMI"), ("ASMI", "ASMI", "AMI")):
                s = P[mk][head]
                add(f"occ/{mk}/{head}_head_auc_{pos}_vs_{neg}", auc(s[single[pos]], s[single[neg]]))
                add(f"occ/{mk}/{head}_prob_on_{neg}", s[single[neg]].mean())
                add(f"occ/{mk}/{head}_prob_on_{pos}", s[single[pos]].mean())
                add(f"occ/{mk}/{neg}_to_{head}", (s[single[neg]] >= tau[head]).mean())
                add(f"occ/{mk}/{pos}_recall", (s[single[pos]] >= tau[head]).mean())

        # 2. negatives, unmasked
        s0 = P["none"]
        negs = {"NORM": no_mi & y["NORM"], "LBBB": no_mi & y["LBBB"],
                "otherNonMI": no_mi & ~y["NORM"], "allNonMI": no_mi}
        for head, sib in (("ASMI", "AMI"), ("IMI", "ILMI")):
            s = s0[head]
            add(f"neg/{head}_vs_{sib}", auc(s[single[head]], s[single[sib]]))
            for nn, mask in negs.items():
                add(f"neg/{head}_vs_{nn}", auc(s[single[head]], s[mask]))
            other_mi = ymi.any(1) & ~y[head] & ~y[sib]
            add(f"neg/{head}_vs_otherMI", auc(s[single[head]], s[other_mi]))
        for n in MI:
            add(f"test_auprc/{n}", average_precision_score(y[n], s0[n]))
        print(arm, seed, "done", flush=True)

out = Path(sys.argv[1]); pd.DataFrame(rows).to_csv(out, index=False); print("saved", out)
