"""
13_split_ptb.py
─────────────────────────────────────────────────────────────────
Creates stratified 70/15/15 splits for the PTB dataset.
Uses MultilabelStratifiedShuffleSplit, grouped by patient_id to
avoid data leakage (same patient can have multiple recordings).
"""

import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def main():
    processed_dir = "data/processed_ptb"
    out_dir       = "data/splits_ptb"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading labels from {processed_dir}/labels.csv...")
    df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))

    label_cols = ["NORM", "AFIB", "STACH", "PVC", "AFLT", "IMI", "ASMI"]
    Y = df[label_cols].values

    # --- Patient-level grouping ---
    # PTB has multiple recordings per patient. We group by patient_id
    # so that all recordings from one patient go to the same split.
    patients = df["patient_id"].unique()
    patient2idx = {p: [] for p in patients}
    for i, row in df.iterrows():
        patient2idx[row["patient_id"]].append(i)

    # For stratification, represent each patient by the OR of their labels
    patient_list = list(patients)
    patient_labels = np.array([
        df.loc[patient2idx[p], label_cols].values.max(axis=0)
        for p in patient_list
    ])

    X_pat = np.arange(len(patient_list))

    # 1. Split patients into (train+val) vs test  [85% / 15%]
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_pat_idx, te_pat_idx = next(msss_test.split(X_pat, patient_labels))

    # 2. Split (train+val) into train vs val  [82.4% / 17.6%  → 70% / 15% overall]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.17647, random_state=42)
    tr_pat_idx_rel, va_pat_idx_rel = next(
        msss_val.split(tv_pat_idx, patient_labels[tv_pat_idx])
    )
    tr_pat_idx = tv_pat_idx[tr_pat_idx_rel]
    va_pat_idx = tv_pat_idx[va_pat_idx_rel]

    # Expand patient-level indices → recording-level indices
    def expand(pat_idx_array):
        record_idx = []
        for pi in pat_idx_array:
            record_idx.extend(patient2idx[patient_list[pi]])
        return np.array(sorted(record_idx))

    train_idx = expand(tr_pat_idx)
    val_idx   = expand(va_pat_idx)
    test_idx  = expand(te_pat_idx)

    n_total = len(df)
    print(f"Total records : {n_total}")
    print(f"  Train : {len(train_idx)} ({len(train_idx)/n_total*100:.1f}%)")
    print(f"  Val   : {len(val_idx)}   ({len(val_idx)/n_total*100:.1f}%)")
    print(f"  Test  : {len(test_idx)}  ({len(test_idx)/n_total*100:.1f}%)")

    # Save flat indices
    np.save(os.path.join(out_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(out_dir, "val_indices.npy"),   val_idx)
    np.save(os.path.join(out_dir, "test_indices.npy"),  test_idx)

    # Save fold_0 format (compatible with finetune script)
    fold_dir = os.path.join(out_dir, "folds", "fold_0")
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, "train.npy"), train_idx)
    np.save(os.path.join(fold_dir, "val.npy"),   val_idx)
    np.save(os.path.join(fold_dir, "test.npy"),  test_idx)
    print(f"Saved splits to {fold_dir}")

    # Label distribution
    print("\nLabel Distribution:")
    print(f"{'Label':<10} | {'Train':>6} | {'Val':>6} | {'Test':>6}")
    print("-" * 40)
    for i, col in enumerate(label_cols):
        tr = int(Y[train_idx, i].sum())
        va = int(Y[val_idx, i].sum())
        te = int(Y[test_idx, i].sum())
        print(f"{col:<10} | {tr:>6} | {va:>6} | {te:>6}")

if __name__ == "__main__":
    main()
