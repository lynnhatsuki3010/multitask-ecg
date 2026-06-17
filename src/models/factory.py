"""
Model factory for clean architecture selection from config.
"""
from __future__ import annotations

from typing import List

from src.models.backbones import (
    CNNBackbone,
    DecoupledMultiTaskBackbone,
    HybridTransformerBackbone,
    MultiBranchTransformerBackbone,
)
from src.models.ecg_multitask import ECGMIModel, ECGMultiTaskModel


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
    num_conduction_labels: int = 0,
):
    model_cfg = cfg.get("model", {})
    architecture = model_cfg.get("architecture", "hybrid_transformer")
    d_model = int(model_cfg.get("d_model", 256))
    stem_dim = int(model_cfg.get("stem_dim", max(64, d_model // 3)))
    dropout = float(model_cfg.get("dropout", 0.15))
    use_se = bool(model_cfg.get("use_se", False))
    use_multi_scale = bool(model_cfg.get("use_multi_scale", False))
    use_cross_attention = bool(model_cfg.get("use_cross_attention", False))
    downsample_factor = int(model_cfg.get("downsample_factor", model_cfg.get("patch_size", 20)))

    if architecture == "cnn":
        backbone = CNNBackbone(
            num_leads=cfg["dataset"]["num_leads"],
            d_model=d_model,
            stem_dim=stem_dim,
            downsample_factor=downsample_factor,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            dropout=dropout,
            use_se=use_se,
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
            use_se=use_se,
            use_multi_scale=use_multi_scale,
        )
    elif architecture == "multi_branch_transformer":
        backbone = MultiBranchTransformerBackbone(
            num_leads=cfg["dataset"]["num_leads"],
            d_model=d_model,
            stem_dim=stem_dim,
            downsample_factor=downsample_factor,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            nhead=int(model_cfg.get("nhead", 8)),
            num_shared_layers=int(model_cfg.get("num_shared_layers", 2)),
            num_expert_layers=int(model_cfg.get("num_expert_layers", 2)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", d_model * 2)),
            dropout=dropout,
            use_se=use_se,
            use_multi_scale=use_multi_scale,
            arrhy_layers=model_cfg.get("arrhy_layers"),
            arrhy_nhead=model_cfg.get("arrhy_nhead"),
            arrhy_ffn=model_cfg.get("arrhy_ffn"),
            mi_layers=model_cfg.get("mi_layers"),
            mi_nhead=model_cfg.get("mi_nhead"),
            mi_ffn=model_cfg.get("mi_ffn"),
            cond_layers=model_cfg.get("cond_layers"),
            cond_nhead=model_cfg.get("cond_nhead"),
            cond_ffn=model_cfg.get("cond_ffn"),
            cd_lead_out_dim=int(model_cfg.get("cd_lead_out_dim", 128)),
        )
    elif architecture == "decoupled_multitask":
        backbone = DecoupledMultiTaskBackbone(
            num_leads=cfg["dataset"]["num_leads"],
            d_model=d_model,
            stem_dim=stem_dim,
            downsample_factor=downsample_factor,
            stage_dims=_parse_stage_dims(model_cfg, d_model),
            nhead=int(model_cfg.get("nhead", 8)),
            num_expert_layers=int(model_cfg.get("num_expert_layers", 2)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", d_model * 2)),
            dropout=dropout,
            use_se=use_se,
            use_multi_scale=use_multi_scale,
            arrhy_layers=model_cfg.get("arrhy_layers"),
            arrhy_nhead=model_cfg.get("arrhy_nhead"),
            arrhy_ffn=model_cfg.get("arrhy_ffn"),
            cond_layers=model_cfg.get("cond_layers"),
            cond_nhead=model_cfg.get("cond_nhead"),
            cond_ffn=model_cfg.get("cond_ffn"),
            cd_lead_out_dim=int(model_cfg.get("cd_lead_out_dim", 128)),
        )
    elif architecture == "mi_only":
        mi_branch_dim = int(model_cfg.get("mi_branch_dim", 128))
        return ECGMIModel(
            num_mi_labels=num_mi_labels,
            mi_branch_dim=mi_branch_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unsupported model.architecture='{architecture}'. "
            "Use 'cnn', 'hybrid_transformer', 'multi_branch_transformer', "
            "'decoupled_multitask', or 'mi_only'."
        )

    head_hidden_dim = int(model_cfg.get("head_hidden_dim", d_model // 2))
    mi_branch_dim = int(model_cfg.get("mi_branch_dim", 128))
    hrv_enabled = bool(cfg.get("hrv", {}).get("enabled", True))
    mi_gradient_scale = float(model_cfg.get("mi_gradient_scale", 1.0))

    routing_mode = str(model_cfg.get("routing_mode", "soft"))
    if architecture == "decoupled_multitask":
        routing_mode = "decoupled"

    return ECGMultiTaskModel(
        backbone=backbone,
        num_arrhythmia_labels=num_arrhythmia_labels,
        num_mi_labels=num_mi_labels,
        num_conduction_labels=num_conduction_labels,
        hrv_enabled=hrv_enabled,
        num_hrv_targets=num_hrv_targets,
        head_hidden_dim=head_hidden_dim,
        mi_branch_dim=mi_branch_dim,
        dropout=dropout,
        hrv_detach=bool(model_cfg.get("hrv_detach", False)),
        mi_gradient_scale=mi_gradient_scale,
        use_cross_attention=use_cross_attention,
        routing_mode=routing_mode,
        mi_head_mode=str(model_cfg.get("mi_head_mode", "shared")),
    )

