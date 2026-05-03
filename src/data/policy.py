"""
Dataset policy manifest and audit helpers.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


POLICY_FILENAME = "dataset_policy.json"


def build_dataset_policy(cfg: dict, processed_path: str) -> dict:
    dataset_cfg = cfg.get("dataset", {})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_path": processed_path,
        "sampling_rate": dataset_cfg.get("sampling_rate"),
        "signal_length": dataset_cfg.get("signal_length"),
        "label_threshold": dataset_cfg.get("label_threshold"),
        "label_threshold_overrides": dataset_cfg.get("label_threshold_overrides", {}),
        "normal_label": dataset_cfg.get("normal_label", "NORM"),
        "normal_mode": dataset_cfg.get("normal_mode", "exclusive"),
        "split_method": dataset_cfg.get("split_method", "strat_fold"),
        "split_seed": dataset_cfg.get("split_seed", 42),
        "split_group_key": dataset_cfg.get("split_group_key", "patient_id"),
        "split_train_ratio": dataset_cfg.get("split_train_ratio", 0.7),
        "split_val_ratio": dataset_cfg.get("split_val_ratio", 0.15),
        "split_test_ratio": dataset_cfg.get("split_test_ratio", 0.15),
        "split_num_folds": dataset_cfg.get("split_num_folds", 5),
        "split_test_fold_index": dataset_cfg.get("split_test_fold_index", 0),
        "split_val_fold_index": dataset_cfg.get("split_val_fold_index"),
        "split_stratify_label": dataset_cfg.get("split_stratify_label"),
        "test_fold": dataset_cfg.get("test_fold", 10),
        "val_fold": dataset_cfg.get("val_fold", 9),
        "selected_labels": [label["name"] for label in cfg.get("labels", [])],
    }


def save_dataset_policy(processed_path: str, policy: dict) -> str:
    os.makedirs(processed_path, exist_ok=True)
    policy_path = os.path.join(processed_path, POLICY_FILENAME)
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    return policy_path


def load_dataset_policy(processed_path: str) -> Optional[dict]:
    policy_path = os.path.join(processed_path, POLICY_FILENAME)
    if not os.path.exists(policy_path):
        return None
    with open(policy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_dataset_policy(
    cfg: dict,
    processed_path: str,
    metadata_columns: Optional[List[str]] = None,
) -> dict:
    expected = build_dataset_policy(cfg, processed_path)
    stored = load_dataset_policy(processed_path)
    warnings: List[str] = []
    errors: List[str] = []

    if metadata_columns is not None:
        expected_label_cols = [f"label_{name}" for name in expected["selected_labels"]]
        missing = [col for col in expected_label_cols if col not in metadata_columns]
        if missing:
            errors.append(
                "Metadata is missing configured label columns: " + ", ".join(missing)
            )

    if stored is None:
        warnings.append(
            "No dataset_policy.json found. Audit is heuristic-only until metadata is rebuilt."
        )
        basename = os.path.basename(os.path.normpath(processed_path)).lower()
        match = re.search(r"imi(\d+)", basename)
        if match:
            implied_threshold = int(match.group(1))
            overrides = expected.get("label_threshold_overrides", {}) or {}
            current_override = overrides.get("IMI")
            matches_name = (
                expected.get("label_threshold") == implied_threshold
                or current_override == implied_threshold
            )
            if not matches_name:
                warnings.append(
                    "Processed path suggests IMI threshold "
                    f"{implied_threshold}, but config uses label_threshold="
                    f"{expected.get('label_threshold')} and IMI override={current_override}."
                )
    else:
        for key in [
            "sampling_rate",
            "signal_length",
            "label_threshold",
            "normal_label",
            "normal_mode",
            "split_method",
            "split_seed",
            "split_group_key",
            "split_train_ratio",
            "split_val_ratio",
            "split_test_ratio",
            "split_num_folds",
            "split_test_fold_index",
            "split_val_fold_index",
            "split_stratify_label",
            "test_fold",
            "val_fold",
            "selected_labels",
            "label_threshold_overrides",
        ]:
            if stored.get(key) != expected.get(key):
                errors.append(
                    f"Dataset policy mismatch for '{key}': "
                    f"stored={stored.get(key)!r}, expected={expected.get(key)!r}"
                )

    return {
        "ok": len(errors) == 0,
        "processed_path": processed_path,
        "policy_path": os.path.join(processed_path, POLICY_FILENAME),
        "expected_policy": expected,
        "stored_policy": stored,
        "warnings": warnings,
        "errors": errors,
    }
