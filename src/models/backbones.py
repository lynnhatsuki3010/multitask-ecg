"""
Backbone modules for 12-lead ECG modeling.

Each backbone returns:
  - global_features: compact representation for downstream heads
  - sequence_features: temporal token map for optional task-specific pooling
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    """Small convenience block used throughout the backbones."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Residual 1D block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size, stride=stride, dropout=dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if (in_channels != out_channels or stride != 1)
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class AttentionPooling(nn.Module):
    """Learned temporal pooling that is less lossy than plain mean pooling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class SharedConvTokenizer(nn.Module):
    """Front-end that preserves local morphology before any global modeling."""

    def __init__(
        self,
        num_leads: int,
        stem_dim: int,
        token_dim: int,
        downsample_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        stride_a = 5 if downsample_factor % 5 == 0 else 4 if downsample_factor % 4 == 0 else 2
        stride_b = max(1, downsample_factor // stride_a)

        self.depthwise = ConvNormAct(
            num_leads,
            num_leads * 8,
            kernel_size=15,
            stride=stride_a,
            groups=num_leads,
            dropout=dropout,
        )
        self.pointwise = ConvNormAct(
            num_leads * 8,
            stem_dim,
            kernel_size=1,
            dropout=dropout,
        )
        self.refine = nn.Sequential(
            ResidualConvBlock(stem_dim, stem_dim, kernel_size=9, dropout=dropout),
            ResidualConvBlock(stem_dim, token_dim, kernel_size=7, stride=stride_b, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.refine(x)
        return x


class MorphologyConvFrontEnd(nn.Module):
    """Richer CNN front-end before the transformer.

    The goal is to let CNN layers absorb local ECG morphology
    (QRS shape, ST shift, reciprocal changes) before tokens are sent
    to the transformer for longer-range dependency modeling.
    """

    def __init__(
        self,
        num_leads: int,
        stem_dim: int,
        stage_dims: List[int],
        downsample_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not stage_dims:
            raise ValueError("stage_dims must contain at least one channel size.")

        self.tokenizer = SharedConvTokenizer(
            num_leads=num_leads,
            stem_dim=stem_dim,
            token_dim=stage_dims[0],
            downsample_factor=downsample_factor,
            dropout=dropout,
        )

        kernels = [11, 9, 7, 7]
        strides = [1] * len(stage_dims)
        blocks = []
        in_dim = stage_dims[0]
        for idx, out_dim in enumerate(stage_dims):
            kernel = kernels[min(idx, len(kernels) - 1)]
            blocks.append(ResidualConvBlock(in_dim, out_dim, kernel_size=kernel, stride=strides[idx], dropout=dropout))
            if idx >= 1:
                # A second block per stage stabilizes morphology extraction without changing sequence length.
                blocks.append(ResidualConvBlock(out_dim, out_dim, kernel_size=kernel, dropout=dropout))
            in_dim = out_dim
        self.stages = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.stages(x)
        return x


class CNNBackbone(nn.Module):
    """Pure CNN backbone that is still compatible with the multitask heads."""

    def __init__(
        self,
        num_leads: int = 12,
        d_model: int = 256,
        stem_dim: int = 96,
        downsample_factor: int = 20,
        stage_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        stage_dims = stage_dims or [128, 192, d_model]
        self.output_dim = d_model

        self.tokenizer = SharedConvTokenizer(
            num_leads=num_leads,
            stem_dim=stem_dim,
            token_dim=stage_dims[0],
            downsample_factor=downsample_factor,
            dropout=dropout,
        )

        stages = []
        in_dim = stage_dims[0]
        for out_dim in stage_dims:
            stages.append(ResidualConvBlock(in_dim, out_dim, kernel_size=7, dropout=dropout))
            in_dim = out_dim
        self.stages = nn.Sequential(*stages)
        self.sequence_proj = nn.Conv1d(stage_dims[-1], d_model, kernel_size=1, bias=False)
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.tokenizer(x)
        x = self.stages(x)
        seq = self.sequence_proj(x).transpose(1, 2)
        global_features = self.norm(self.pool(seq))
        return {
            "global_features": global_features,
            "sequence_features": seq,
        }


class HybridTransformerBackbone(nn.Module):
    """CNN tokenizer followed by a lightweight Transformer encoder."""

    def __init__(
        self,
        num_leads: int = 12,
        d_model: int = 256,
        stem_dim: int = 96,
        downsample_factor: int = 20,
        stage_dims: List[int] | None = None,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = d_model
        stage_dims = stage_dims or [128, 192, d_model]
        self.front_end = MorphologyConvFrontEnd(
            num_leads=num_leads,
            stem_dim=stem_dim,
            stage_dims=stage_dims,
            downsample_factor=downsample_factor,
            dropout=dropout,
        )
        self.sequence_proj = nn.Conv1d(stage_dims[-1], d_model, kernel_size=1, bias=False)
        self.pre_transformer_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=1024, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.pool = AttentionPooling(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        conv_features = self.front_end(x)
        seq = self.sequence_proj(conv_features).transpose(1, 2)
        seq = self.pre_transformer_norm(seq)
        batch_size = seq.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = self.positional_encoding(torch.cat([cls, seq], dim=1))
        encoded = self.encoder(tokens)
        cls_features = encoded[:, 0]
        sequence_features = encoded[:, 1:]
        pooled = self.pool(sequence_features)
        global_features = self.fuse(torch.cat([cls_features, pooled], dim=-1))
        return {
            "global_features": global_features,
            "sequence_features": sequence_features,
        }
