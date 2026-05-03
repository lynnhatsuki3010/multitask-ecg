"""
Model factory for clean architecture selection from config.
"""
from __future__ import annotations

from typing import List

from src.models.backbones import CNNBackbone, HybridTransformerBackbone
from src.models.ecg_multitask import ECGMultiTaskModel
from src.models.phased_multitask_transformer import PhasedMultitaskTransformer


def _parse_stage_dims(model_cfg: dict, d_model: int) -> List[int]:
    stage_dims = model_cfg.get("cnn_stage_dims")
    if stage_dims:
        return list(stage_dims)
    return [max(96, d_model // 2), max(128, int(d_model * 0.75)), d_model]


def build_model(
    cfg: dict,
    num_arrhythmia_labels: int,
    num_mi_labels: int,
    num_hrv_targets: int,
):
    model_cfg = cfg.get("model", {})
    architecture = model_cfg.get("architecture", "hybrid_transformer")
    d_model = int(model_cfg.get("d_model", 256))
    stem_dim = int(model_cfg.get("stem_dim", max(64, d_model // 3)))
    dropout = float(model_cfg.get("dropout", 0.15))
    downsample_factor = int(model_cfg.get("downsample_factor", model_cfg.get("patch_size", 20)))

    if architecture == "cnn":
        backbone = CNNBackbone(
            num_leads=cfg["dataset"]["num_leads"],
            d_model=d_model,
            stem_dim=stem_dim,
            downsample_factor=downsample_factor,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            dropout=dropout,
        )
    elif architecture == "hybrid_transformer":
        backbone = HybridTransformerBackbone(
            num_leads=cfg["dataset"]["num_leads"],
            d_model=d_model,
            stem_dim=stem_dim,
            downsample_factor=downsample_factor,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", d_model * 2)),
            dropout=dropout,
        )
    elif architecture == "phased_multitask_transformer":
        hrv_enabled = bool(cfg.get("hrv", {}).get("enabled", True))
        return PhasedMultitaskTransformer(
            num_leads=cfg["dataset"]["num_leads"],
            num_arrhythmia_labels=num_arrhythmia_labels,
            num_mi_labels=num_mi_labels,
            hrv_enabled=hrv_enabled,
            num_hrv_targets=num_hrv_targets,
            d_model=d_model,
            stem_dim=stem_dim,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            downsample_factor=downsample_factor,
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", d_model * 2)),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", d_model // 2)),
            mi_branch_dim=int(model_cfg.get("mi_branch_dim", 128)),
            dropout=dropout,
            hrv_detach=bool(model_cfg.get("hrv_detach", False)),
        )
    else:
        raise ValueError(
            f"Unsupported model.architecture='{architecture}'. "
            "Use 'cnn', 'hybrid_transformer', or 'phased_multitask_transformer'."
        )

    head_hidden_dim = int(model_cfg.get("head_hidden_dim", d_model // 2))
    mi_branch_dim = int(model_cfg.get("mi_branch_dim", 128))
    hrv_enabled = bool(cfg.get("hrv", {}).get("enabled", True))

    return ECGMultiTaskModel(
        backbone=backbone,
        num_arrhythmia_labels=num_arrhythmia_labels,
        num_mi_labels=num_mi_labels,
        hrv_enabled=hrv_enabled,
        num_hrv_targets=num_hrv_targets,
        head_hidden_dim=head_hidden_dim,
        mi_branch_dim=mi_branch_dim,
        dropout=dropout,
        hrv_detach=bool(model_cfg.get("hrv_detach", False)),
    )
