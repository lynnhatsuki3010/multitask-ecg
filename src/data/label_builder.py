"""
label_builder.py
Build multi-label matrix from PTB-XL scp_codes column.
"""
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# ─── Default 7-label configuration ──────────────────────────────────────────
DEFAULT_LABELS = ["NORM", "AFIB", "STACH", "PVC", "AFLT", "IMI", "ASMI"]

LABEL_TO_TASK = {
    "NORM":  "normal",
    "AFIB":  "arrhythmia",
    "STACH": "arrhythmia",
    "PVC":   "arrhythmia",
    "SBRAD": "arrhythmia",  # kept for backward compat
    "AFLT":  "arrhythmia",
    "IMI":   "mi",
    "ASMI":  "mi",
}

# PTB-XL rhythm/form statements use confidence=0.0 to mean "Present".
# Diagnostic statements (NORM, IMI, ASMI, etc.) have numeric confidence 0-100.
RHYTHM_STATEMENTS = {
    "AFIB", "STACH", "PVC", "SBRAD", "AFLT", "SR", "SARRH",
    "SVTAC", "PSVT", "TRIGU", "BIGU", "PACE", "SVARR",
    "APTS", "VPTS",
}


def parse_scp_codes(scp_str: str) -> Dict[str, float]:
    """Parse the scp_codes string column into a Python dict."""
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def build_label_matrix(
    df: pd.DataFrame,
    selected_labels: List[str] = DEFAULT_LABELS,
    threshold: float = 50.0,
    label_threshold_overrides: Dict[str, float] = None,
    normal_label: str = "NORM",
    normal_mode: str = "exclusive",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a binary multi-label matrix from the scp_codes column.

    Args:
        df: PTB-XL metadata DataFrame (must have 'scp_codes' column).
        selected_labels: List of SCP label strings to include.
        threshold: Minimum confidence to assign a label (0–100).
        normal_label: Name of the normal rhythm label.
        normal_mode:
            - "exclusive": if any abnormal selected label is present, force normal_label=0
            - "independent": keep the raw PTB-XL assignments unchanged

    Returns:
        label_matrix: np.ndarray of shape (N, len(selected_labels)), dtype float32.
        selected_labels: The label list in order (same as column order).
    """
    if "scp_codes" not in df.columns:
        raise ValueError("DataFrame must contain a 'scp_codes' column.")

    # Parse scp_codes if not already done
    if df["scp_codes"].dtype == object and isinstance(df["scp_codes"].iloc[0], str):
        scp = df["scp_codes"].apply(parse_scp_codes)
    else:
        scp = df["scp_codes"]

    n = len(df)
    k = len(selected_labels)
    label_matrix = np.zeros((n, k), dtype=np.float32)

    label_to_idx = {lbl: i for i, lbl in enumerate(selected_labels)}
    label_threshold_overrides = label_threshold_overrides or {}

    for row_idx, codes in enumerate(scp):
        for code, confidence in codes.items():
            if code in label_to_idx:
                if code in RHYTHM_STATEMENTS:
                    # Rhythm/form statements: confidence=0.0 means "Present".
                    # Any non-negative confidence means the label is active.
                    label_matrix[row_idx, label_to_idx[code]] = 1.0
                else:
                    # Diagnostic statements: apply confidence threshold.
                    min_confidence = float(label_threshold_overrides.get(code, threshold))
                    if confidence >= min_confidence:
                        label_matrix[row_idx, label_to_idx[code]] = 1.0

    if normal_mode == "exclusive" and normal_label in label_to_idx:
        normal_idx = label_to_idx[normal_label]
        abnormal_indices = [i for i, name in enumerate(selected_labels) if name != normal_label]
        if abnormal_indices:
            abnormal_mask = label_matrix[:, abnormal_indices].sum(axis=1) > 0
            label_matrix[abnormal_mask, normal_idx] = 0.0

    return label_matrix, selected_labels


def get_label_statistics(
    label_matrix: np.ndarray,
    selected_labels: List[str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    """
    Print and return per-label statistics.

    Returns:
        DataFrame with columns: label, count, prevalence(%).
    """
    n = label_matrix.shape[0]
    counts = label_matrix.sum(axis=0).astype(int)
    prevalences = counts / n * 100

    stats = pd.DataFrame({
        "label":       selected_labels,
        "task":        [LABEL_TO_TASK.get(l, "unknown") for l in selected_labels],
        "count":       counts,
        "prevalence%": prevalences.round(2),
    })
    return stats


def compute_pos_weights(
    label_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute positive class weights for BCEWithLogitsLoss.
    pos_weight[i] = (N - n_pos[i]) / n_pos[i]
    """
    n = label_matrix.shape[0]
    n_pos = label_matrix.sum(axis=0)
    n_pos = np.clip(n_pos, 1, None)  # avoid division by zero
    pos_weights = (n - n_pos) / n_pos
    return pos_weights.astype(np.float32)


if __name__ == "__main__":
    # Quick test
    import os
    path = "data/raw/PTB-XL"
    df = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    mat, labels = build_label_matrix(df, DEFAULT_LABELS, threshold=50.0)
    stats = get_label_statistics(mat, labels)
    print(stats.to_string(index=False))
    print(f"\nTotal samples: {len(df)}")
    print(f"Samples with at least 1 label: {(mat.sum(axis=1) > 0).sum()}")
    print(f"Multi-label samples: {(mat.sum(axis=1) > 1).sum()}")
