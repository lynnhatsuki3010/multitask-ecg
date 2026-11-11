"""
25_sibling_posthoc_rules.py
---------------------------
Can a parameter-free post-hoc rule on an existing model's logits do what the
clustered head does?

Inside the IMI/ILMI pair the logit difference z_IMI - z_ILMI already separates
the two labels well in every arm (AUROC ~0.86-0.88 on single-label records),
while the IMI head on its own does not (0.54 in the baselines). The obvious
reviewer question is whether a rule that uses the sibling logit closes the gap
without any architecture change. IMI and ILMI never co-occur in PTB-XL, so the
pair is mutually exclusive by annotation convention and such rules are natural.

Scored on the FULL test fold (IMI AUPRC against every negative, not just ILMI):
  raw      : z_IMI                                   (what the model outputs)
  pairsoft : log sigma(z_IMI) + log softmax(z_IMI, z_ILMI)[IMI]
             i.e. P(IMI) = P(inferior evidence) * P(IMI | IMI or ILMI)
  lin      : z_IMI - lam * z_ILMI, lam picked on validation IMI AUPRC (grid 0..2)
and the same three for ILMI (roles swapped).

Usage:
  python scripts/25_sibling_posthoc_rules.py results/sibling_posthoc_rules.csv
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.data.preprocessing import PTBXLDataset
from src.models.factory import build_model
from sklearn.metrics import average_precision_score

ARMS = {"ARCH-05": "abl_arch05_shared_tf", "ARCH-09": "abl_arch09_decoupled",
        "ARCH-10": "abl_arch10_contrast", "ARCH-12": "abl_arch12_partial"}
LAMS = np.round(np.arange(0, 2.01, 0.1), 2)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = {}


def split_data(cfg, split):
    fs = cfg["dataset"]["sampling_rate"]; pr = cfg["preprocessing"]
    key = (split, fs)
    if key not in cache:
        proc, sp = ROOT / f"{cfg['paths']['processed']}_{fs}hz", ROOT / f"{cfg['paths']['splits']}_{fs}hz"
        df = pd.read_csv(proc / "metadata.csv"); lm = np.load(proc / "label_matrix.npy")
        idx = np.load(sp / f"{split}_indices.npy")
        ds = PTBXLDataset(metadata=df.iloc[idx], label_matrix=lm[idx], hrv_matrix=None,
                          base_path=str(ROOT / cfg["paths"]["raw_data"]), sampling_rate=fs,
                          target_length=cfg["dataset"]["signal_length"],
                          bandpass=(pr["bandpass_low"], pr["bandpass_high"]), notch=pr["notch_freq"],
                          normalize=pr["normalize"], augment=False)
        cache[key] = (torch.stack([ds[i]["signal"] for i in range(len(ds))]), lm[idx])
    return cache[key]


def logsig(z):
    return -np.logaddexp(0, -z)


def rules(z_a, z_b, lam):
    pair = z_a - np.logaddexp(z_a, z_b)            # log softmax over the pair
    return {"raw": z_a, "pairsoft": logsig(z_a) + pair, "lin": z_a - lam * z_b}


ap = argparse.ArgumentParser()
ap.add_argument("out")
args = ap.parse_args()
rows = []
for arm, pat in ARMS.items():
    for rp in sorted(glob.glob(str(ROOT / f"checkpoints/run_*{pat}-*"))):
        r = Path(rp)
        cfg = yaml.safe_load((r / "config_snapshot.yaml").read_text(encoding="utf-8")); cfg["hrv"] = {"enabled": False}
        g = cfg["label_groups"]; names = {l["index"]: l["name"] for l in cfg["labels"]}
        flat = [names[i] for t in ("arrhythmia", "mi", "conduction") for i in g[t]]
        mi = [names[i] for i in g["mi"]]
        m = build_model(cfg, len(g["arrhythmia"]), len(g["mi"]), 0, len(g["conduction"])).to(dev).eval()
        ck = torch.load(str(r / "best_model.pth"), map_location=dev, weights_only=False)
        m.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)), strict=True)
        Z, Y = {}, {}
        for split in ("val", "test"):
            X, lab = split_data(cfg, split)
            with torch.no_grad():
                Z[split] = np.concatenate([m(X[i:i + 64].to(dev))["mi"].cpu().numpy() for i in range(0, len(X), 64)])
            Y[split] = lab
        rec = {"arm": arm, "seed": cfg["training"]["seed"]}
        for a, b in (("IMI", "ILMI"), ("ILMI", "IMI")):
            ia, ib, ya = mi.index(a), mi.index(b), flat.index(a)
            val_ap = [average_precision_score(Y["val"][:, ya], Z["val"][:, ia] - lam * Z["val"][:, ib]) for lam in LAMS]
            lam = float(LAMS[int(np.argmax(val_ap))])
            rec[f"{a}_lam"] = lam
            for name, s in rules(Z["test"][:, ia], Z["test"][:, ib], lam).items():
                rec[f"{a}_{name}"] = average_precision_score(Y["test"][:, ya], s)
        rows.append(rec)
        print(arm, rec["seed"], {k: round(v, 4) for k, v in rec.items() if k.startswith("IMI_")}, flush=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print("saved", args.out)
