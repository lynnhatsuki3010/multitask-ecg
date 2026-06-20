"""
10_split_georgia.py
─────────────────────────────────────────────────────────────────
Creates random 70/15/15 splits for the Georgia dataset.
Saves to data/splits_georgia/train_indices.npy, etc.
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ModuleNotFoundError:
    MultilabelStratifiedShuffleSplit = None


def split_indices(X, Y, test_size, seed):
    if MultilabelStratifiedShuffleSplit is not None:
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        return next(splitter.split(X, Y))

    print(
        "[WARN] Missing package 'iterative-stratification' (module 'iterstrat'). "
        "Falling back to deterministic random split. For better multilabel balance, run: "
        "python -m pip install iterative-stratification"
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx

def main():
    processed_dir = "data/processed_georgia"
    out_dir = "data/splits_georgia"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading labels from {processed_dir}/labels.csv...")
    df = pd.read_csv(os.path.join(processed_dir, "labels.csv"))
    
    # Stratify by all available task labels. After rebuilding Georgia with
    # scripts/08_build_georgia.py this includes conduction labels too.
    preferred_cols = [
        "NORM", "AFIB", "STACH", "PVC", "AFLT",
        "IMI", "ASMI",
        "LBBB", "RBBB", "IRBBB", "1AVB",
    ]
    label_cols = [col for col in preferred_cols if col in df.columns]
    Y = df[label_cols].values
    X = np.arange(len(df))
    
    # 1. Split off test (15%)
    train_val_idx, test_idx = split_indices(X, Y, test_size=0.15, seed=42)
    
    # 2. Split remaining (85%) into train (70%) and val (15%)
    # 15 / 85 = 0.17647
    train_idx_rel, val_idx_rel = split_indices(
        train_val_idx,
        Y[train_val_idx],
        test_size=0.17647,
        seed=42,
    )
    
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    
    print(f"Total samples: {len(df)}")
    print(f"  Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")
    
    # Save splits
    np.save(os.path.join(out_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(out_dir, "val_indices.npy"), val_idx)
    np.save(os.path.join(out_dir, "test_indices.npy"), test_idx)
    
    # Optional: Save folds format if needed by trainer
    # trainer.py expects splits/fold_0/train.npy etc. if using K-Fold.
    # But for a single random split, we can just save it like random_grouped:
    # Actually, `src.data.loader` supports 'random_grouped' which reads folds.
    # Or 'stratified_kfold'.
    # If we look at how PTB-XL is split, it creates `data/splits_split_e02/folds/...`
    # Let's just create 10 folds, where fold 0 is our split, and the rest are just empty or same, to satisfy the loader.
    # Wait, the loader just reads `fold_{i}/train.npy`, `val.npy`, `test.npy`.
    # Let's write fold_0:
    fold_dir = os.path.join(out_dir, "folds", "fold_0")
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, "train.npy"), train_idx)
    np.save(os.path.join(fold_dir, "val.npy"), val_idx)
    np.save(os.path.join(fold_dir, "test.npy"), test_idx)
    print(f"Saved to {fold_dir}")
    
    # Let's also print label distribution
    print("\nLabel Distribution:")
    print(f"{'Label':<10} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-" * 40)
    for i, col in enumerate(label_cols):
        tr_count = int(np.sum(Y[train_idx, i]))
        v_count = int(np.sum(Y[val_idx, i]))
        te_count = int(np.sum(Y[test_idx, i]))
        print(f"{col:<10} | {tr_count:<7} | {v_count:<7} | {te_count:<7}")

if __name__ == "__main__":
    main()
