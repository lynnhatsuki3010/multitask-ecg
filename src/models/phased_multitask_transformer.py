"""
Fresh multitask ECG architecture built for staged MI-first training.

Design goals:
  - CNN front-end captures local ECG morphology.
  - Transformer remains the core fusion/reasoning module.
  - MI receives anatomy-aware region tokens from clinically relevant lead groups.
  - The model can be trained in phases by freezing selected components.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.backbones import (
    AttentionPooling,
    ConvNormAct,
    ResidualConvBlock,
    SinusoidalPositionalEncoding,
)


class MLPHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalMorphologyEncoder(nn.Module):
    """Shared CNN encoder that converts 12-lead waveforms into temporal tokens."""

    def __init__(
        self,
        num_leads: int,
        stem_dim: int,
        stage_dims: list[int],
        downsample_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(stage_dims) < 3:
            raise ValueError("stage_dims must contain at least three channel sizes.")

        stride_a = 4 if downsample_factor >= 16 else 2
        stride_b = 2
        accumulated = stride_a * stride_b * 2
        stride_c = max(1, downsample_factor // accumulated)

        self.stem = nn.Sequential(
            ConvNormAct(num_leads, stem_dim, kernel_size=15, stride=stride_a, dropout=dropout),
            ResidualConvBlock(stem_dim, stage_dims[0], kernel_size=11, stride=stride_b, dropout=dropout),
            ResidualConvBlock(stage_dims[0], stage_dims[1], kernel_size=9, stride=2, dropout=dropout),
            ResidualConvBlock(stage_dims[1], stage_dims[2], kernel_size=7, stride=stride_c, dropout=dropout),
        )

        extra_blocks = []
        in_dim = stage_dims[2]
        for out_dim in stage_dims[3:]:
            extra_blocks.append(ResidualConvBlock(in_dim, out_dim, kernel_size=7, dropout=dropout))
            in_dim = out_dim
        self.refine = nn.Sequential(*extra_blocks) if extra_blocks else nn.Identity()
        self.out_dim = stage_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.refine(x)
        return x


class RegionalTokenEncoder(nn.Module):
    """Encodes clinically meaningful lead groups into compact tokens."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_channels, 32, kernel_size=11, stride=2, dropout=dropout),
            ResidualConvBlock(32, 64, kernel_size=7, stride=2, dropout=dropout),
            ResidualConvBlock(64, out_dim, kernel_size=5, stride=2, dropout=dropout),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        return self.proj(torch.cat([avg, mx], dim=-1))


class PhasedMultitaskTransformer(nn.Module):
    """New multitask model with MI-first staged training in mind."""

    INFERIOR_LEADS = [1, 2, 5]    # II, III, aVF
    RECIPROCAL_LEADS = [0, 4]     # I, aVL
    ANTERIOR_LEADS = [6, 7, 8, 9] # V1-V4

    QUERY_ORDER = ("rhythm", "imi", "asmi", "hrv")
    REGION_ORDER = ("inferior", "reciprocal", "anterior")

    def __init__(
        self,
        num_leads: int,
        num_arrhythmia_labels: int,
        num_mi_labels: int,
        hrv_enabled: bool,
        num_hrv_targets: int,
        d_model: int = 256,
        stem_dim: int = 96,
        stage_dims: list[int] | None = None,
        downsample_factor: int = 20,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        head_hidden_dim: int = 128,
        mi_branch_dim: int = 128,
        dropout: float = 0.1,
        hrv_detach: bool = False,
    ) -> None:
        super().__init__()
        if num_mi_labels != 2:
            raise ValueError("PhasedMultitaskTransformer currently expects exactly 2 MI labels: IMI and ASMI.")

        stage_dims = stage_dims or [128, 192, 256]
        self.hrv_enabled = hrv_enabled
        self.hrv_detach = hrv_detach

        self.temporal_encoder = TemporalMorphologyEncoder(
            num_leads=num_leads,
            stem_dim=stem_dim,
            stage_dims=stage_dims,
            downsample_factor=downsample_factor,
            dropout=dropout,
        )
        self.temporal_proj = nn.Conv1d(self.temporal_encoder.out_dim, d_model, kernel_size=1, bias=False)
        self.temporal_norm = nn.LayerNorm(d_model)

        self.region_encoders = nn.ModuleDict({
            "inferior": RegionalTokenEncoder(len(self.INFERIOR_LEADS), d_model, dropout),
            "reciprocal": RegionalTokenEncoder(len(self.RECIPROCAL_LEADS), d_model, dropout),
            "anterior": RegionalTokenEncoder(len(self.ANTERIOR_LEADS), d_model, dropout),
        })
        self.region_to_type_idx = {
            "inferior": 0,
            "reciprocal": 1,
            "anterior": 2,
        }

        self.query_tokens = nn.Parameter(torch.randn(len(self.QUERY_ORDER), d_model))
        self.query_type_embeddings = nn.Parameter(torch.randn(len(self.QUERY_ORDER), d_model))
        self.region_type_embeddings = nn.Parameter(torch.randn(len(self.REGION_ORDER), d_model))
        self.temporal_type_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.query_type_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.region_type_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_type_embedding, mean=0.0, std=0.02)

        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=1024, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.sequence_pool = AttentionPooling(d_model)

        self.arrhythmia_head = MLPHead(d_model * 2, num_arrhythmia_labels, head_hidden_dim, dropout)
        self.imi_head = MLPHead(d_model * 4, 1, mi_branch_dim, dropout)
        self.asmi_head = MLPHead(d_model * 3, 1, mi_branch_dim, dropout)

        if hrv_enabled:
            self.hrv_head = MLPHead(d_model * 2, num_hrv_targets, head_hidden_dim, dropout)

    def _build_region_tokens(self, x: torch.Tensor) -> torch.Tensor:
        region_inputs = {
            "inferior": x[:, self.INFERIOR_LEADS, :],
            "reciprocal": x[:, self.RECIPROCAL_LEADS, :],
            "anterior": x[:, self.ANTERIOR_LEADS, :],
        }
        tokens = []
        for name in self.REGION_ORDER:
            token = self.region_encoders[name](region_inputs[name])
            type_embed = self.region_type_embeddings[self.region_to_type_idx[name]]
            tokens.append(token + type_embed.unsqueeze(0))
        return torch.stack(tokens, dim=1)

    def configure_phase(self, phase_control: Dict[str, bool] | None = None) -> None:
        """Freeze/unfreeze module groups according to a phase config."""
        phase_control = phase_control or {}
        freeze_map = {
            "freeze_temporal_encoder": self.temporal_encoder,
            "freeze_region_encoders": self.region_encoders,
            "freeze_transformer": nn.ModuleList([
                self.temporal_proj,
                self.temporal_norm,
                self.transformer,
            ]),
            "freeze_arrhythmia_head": self.arrhythmia_head,
            "freeze_mi_head": nn.ModuleList([self.imi_head, self.asmi_head]),
        }
        if self.hrv_enabled:
            freeze_map["freeze_hrv_head"] = self.hrv_head

        for freeze_key, module in freeze_map.items():
            freeze = bool(phase_control.get(freeze_key, False))
            for param in module.parameters():
                param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal_features = self.temporal_encoder(x)
        temporal_tokens = self.temporal_proj(temporal_features).transpose(1, 2)
        temporal_tokens = self.temporal_norm(temporal_tokens)
        temporal_tokens = self.positional_encoding(temporal_tokens + self.temporal_type_embedding)

        batch_size = x.size(0)
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query_tokens = query_tokens + self.query_type_embeddings.unsqueeze(0)
        region_tokens = self._build_region_tokens(x)

        tokens = torch.cat([query_tokens, region_tokens, temporal_tokens], dim=1)
        encoded = self.transformer(tokens)

        num_queries = len(self.QUERY_ORDER)
        num_regions = len(self.REGION_ORDER)
        query_out = encoded[:, :num_queries]
        region_out = encoded[:, num_queries:num_queries + num_regions]
        temporal_out = encoded[:, num_queries + num_regions:]
        pooled_temporal = self.sequence_pool(temporal_out)

        rhythm_query = query_out[:, 0]
        imi_query = query_out[:, 1]
        asmi_query = query_out[:, 2]
        inferior_token = region_out[:, 0]
        reciprocal_token = region_out[:, 1]
        anterior_token = region_out[:, 2]

        outputs = {
            "arrhythmia": self.arrhythmia_head(torch.cat([rhythm_query, pooled_temporal], dim=-1)),
            "mi": torch.cat(
                [
                    self.imi_head(torch.cat([imi_query, inferior_token, reciprocal_token, pooled_temporal], dim=-1)),
                    self.asmi_head(torch.cat([asmi_query, anterior_token, pooled_temporal], dim=-1)),
                ],
                dim=-1,
            ),
        }

        if self.hrv_enabled:
            hrv_query = query_out[:, 3]
            hrv_features = torch.cat([hrv_query, pooled_temporal], dim=-1)
            if self.hrv_detach:
                hrv_features = hrv_features.detach()
            outputs["hrv"] = self.hrv_head(hrv_features)

        return outputs
