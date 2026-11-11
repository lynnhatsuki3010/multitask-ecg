"""
26_make_cv_splits.py
--------------------
Write the ten train/val/test index sets for 10-fold cross-validation on PTB-XL.

Fold k is the test set and fold k % 10 + 1 the validation set; the other eight
folds train. PTB-XL's strat_fold is already assigned per patient, so no patient
crosses sets. The script checks that, and that every record is a test record
exactly once, before writing anything.

Output: <paths.splits>_cvf<k>_<fs>hz/{train,val,test}_indices.npy for k = 1..10,
indices into the rows of <paths.processed>_<fs>hz/metadata.csv, the same
convention as the single-split files.

Usage:
  python scripts/26_make_cv_splits.py --config configs/cv/cv_layout.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
fs = cfg["dataset"]["sampling_rate"]
meta = f"{cfg['paths']['processed']}_{fs}hz/metadata.csv"
df = pd.read_csv(meta)
for col in ("strat_fold", "patient_id"):
    if col not in df.columns:
        sys.exit(f"{meta} has no '{col}' column")

fold = df["strat_fold"].to_numpy()
pid = df["patient_id"].to_numpy()
idx = np.arange(len(df))
assert set(np.unique(fold)) == set(range(1, 11)), "strat_fold must take values 1..10"

seen_test = np.zeros(len(df), dtype=int)
summary = {}
for k in range(1, 11):
    v = k % 10 + 1
    split = {"test": idx[fold == k], "val": idx[fold == v], "train": idx[(fold != k) & (fold != v)]}
    for a in split:
        for b in split:
            if a < b:
                shared = set(pid[split[a]]) & set(pid[split[b]])
                assert not shared, f"fold {k}: {len(shared)} patients in both {a} and {b}"
    seen_test[split["test"]] += 1
    out = f"{cfg['paths']['splits']}_cvf{k}_{fs}hz"
    os.makedirs(out, exist_ok=True)
    for name, ix in split.items():
        np.save(os.path.join(out, f"{name}_indices.npy"), ix)
    summary[k] = {"val_fold": v, **{n: int(len(ix)) for n, ix in split.items()}, "dir": out}
    print(f"fold {k:2d}: test={len(split['test']):5d} val={len(split['val']):5d} "
          f"train={len(split['train']):5d} -> {out}")

assert (seen_test == 1).all(), "every record must be a test record exactly once"
with open(f"{cfg['paths']['splits']}_cv_summary_{fs}hz.json", "w") as f:
    json.dump(summary, f, indent=2)
print("all 10 folds written; every record is out-of-fold exactly once; no patient crosses sets")
