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


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D signals.
    Automatically learns channel-wise attention to enhance informative features
    and suppress noisy or irrelevant ones.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(1, channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualConvBlock(nn.Module):
    """Residual 1D block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.0,
        use_se: bool = False,
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
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = self.se(out)
        return self.activation(out + self.shortcut(x))


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
        use_se: bool = False,
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
            blocks.append(ResidualConvBlock(in_dim, out_dim, kernel_size=kernel, stride=strides[idx], dropout=dropout, use_se=use_se))
            if idx >= 1:
                # A second block per stage stabilizes morphology extraction without changing sequence length.
                blocks.append(ResidualConvBlock(out_dim, out_dim, kernel_size=kernel, dropout=dropout, use_se=use_se))
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
        use_se: bool = False,
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
            stages.append(ResidualConvBlock(in_dim, out_dim, kernel_size=7, dropout=dropout, use_se=use_se))
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


class MultiScaleTemporalBranch(nn.Module):
    """Multi-scale temporal processing to capture both high-frequency (QRS) and low-frequency (QT, T-wave) features."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        # Short scale: kernel 3, 5
        self.short_scale = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 2),
            nn.GELU(),
            nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 2),
            nn.GELU()
        )
        # Long scale: kernel 15
        self.long_scale = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels // 2),
            nn.GELU()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_scale(x)
        long = self.long_scale(x)
        return self.proj(torch.cat([short, long], dim=1))


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
        use_se: bool = False,
        use_multi_scale: bool = False,
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
            use_se=use_se,
        )
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.multi_scale = MultiScaleTemporalBranch(stage_dims[-1], d_model, dropout=dropout)
            self.sequence_proj = nn.Identity()
        else:
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
        if self.use_multi_scale:
            seq = self.multi_scale(conv_features)
            seq = self.sequence_proj(seq).transpose(1, 2)
        else:
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


class _ExpertBranch(nn.Module):
    """One expert Transformer branch with its own CLS token, PE and encoder.

    Receives the CNN feature sequence (shared) and produces its own
    independent global vector + sequence features.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=1024, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)
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
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.pool = AttentionPooling(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, shared_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            shared_seq: (B, T, d_model) — CNN token sequence from shared front-end.
        Returns:
            global_features:   (B, d_model)
            sequence_features: (B, T, d_model)
        """
        seq = self.pre_norm(shared_seq)
        B = seq.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = self.positional_encoding(torch.cat([cls, seq], dim=1))
        encoded = self.encoder(tokens)
        cls_feat = encoded[:, 0]
        seq_feat = encoded[:, 1:]
        pooled = self.pool(seq_feat)
        global_feat = self.fuse(torch.cat([cls_feat, pooled], dim=-1))
        return {"global_features": global_feat, "sequence_features": seq_feat}



class ConductionLeadGroupEncoder(nn.Module):
    """Dedicated lead-group encoder for Conduction Disturbance detection.

    Bundle-branch blocks manifest as QRS widening / morphology changes at:
      - Right precordial leads V1-V3  → RBBB (RSR' pattern, 'rabbit ears')
      - Left precordial + lateral leads V5, V6, I, aVL  → LBBB (wide R, absent q)

    This encoder extracts independent representations from each group and
    fuses them so the Conduction head has anatomy-aware raw-signal features,
    analogous to what MILeadGroupEncoder provides for MI.
    """

    # PTB-XL 12-lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    # Indices:               0   1   2    3    4    5   6   7   8   9  10  11
    RIGHT_LEADS = [6, 7, 8]        # V1, V2, V3  — RBBB focus
    LEFT_LEADS  = [10, 11, 0, 4]  # V5, V6, I, aVL  — LBBB focus

    def __init__(self, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Shared LeadGroupEncoder architecture (same as MI branch)
        self.right_encoder = _make_lead_encoder(
            in_channels=len(self.RIGHT_LEADS), out_dim=out_dim, dropout=dropout
        )
        self.left_encoder = _make_lead_encoder(
            in_channels=len(self.LEFT_LEADS), out_dim=out_dim, dropout=dropout
        )
        # Fuse right + left into a single conduction representation
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: (B, 12, T) raw ECG signal
        Returns:
            cond_lead_features: (B, out_dim)
        """
        right_feat = self.right_encoder(signal[:, self.RIGHT_LEADS, :])
        left_feat  = self.left_encoder(signal[:, self.LEFT_LEADS, :])
        return self.fuse(torch.cat([right_feat, left_feat], dim=-1))


def _make_lead_encoder(in_channels: int, out_dim: int, dropout: float) -> nn.Module:
    """Factory for a small 3-conv lead encoder (same design as LeadGroupEncoder)."""
    return nn.Sequential(
        _LeadEncoderNet(in_channels, out_dim, dropout),
    )


class _LeadEncoderNet(nn.Module):
    """Internal: CNN + dual-pool projection used by ConductionLeadGroupEncoder."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim), nn.GELU(),
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
        mx  = self.max_pool(h).squeeze(-1)
        return self.proj(torch.cat([avg, mx], dim=-1))


class MultiBranchTransformerBackbone(nn.Module):
    """Shared CNN front-end → 3 independent Transformer Expert Branches.

    Architecture:
        ECG (B, 12, T)
              ↓
        Shared CNN Front-End  [MorphologyConvFrontEnd]
              ↓
        Optional shared Transformer layers (num_shared_layers)
              ↓
        ┌──────────────────────────────────────────────────┐
        │  Arrhythmia Expert  (arrhy_layers, arrhy_nhead)  │
        │  MI Expert          (mi_layers,    mi_nhead)     │
        │  Conduction Expert  (cond_layers,  cond_nhead)   │
        └──────────────────────────────────────────────────┘
        Each expert builds its own attention map from scratch.

    Additionally exposes ConductionLeadGroupEncoder outputs so the
    ConductionHead can fuse anatomy-specific raw-signal features
    (V1-V3 for RBBB, V5+V6+I+aVL for LBBB).

    output_dim = d_model  (same contract as HybridTransformerBackbone)
    """

    def __init__(
        self,
        num_leads: int = 12,
        d_model: int = 256,
        stem_dim: int = 96,
        downsample_factor: int = 20,
        stage_dims: List[int] | None = None,
        # Shared transformer layers
        nhead: int = 8,
        num_shared_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # Per-expert configs — if None, falls back to shared defaults
        arrhy_layers: int | None = None,
        arrhy_nhead: int | None = None,
        arrhy_ffn: int | None = None,
        mi_layers: int | None = None,
        mi_nhead: int | None = None,
        mi_ffn: int | None = None,
        cond_layers: int | None = None,
        cond_nhead: int | None = None,
        cond_ffn: int | None = None,
        # Misc
        num_expert_layers: int = 2,   # fallback for all experts if per-expert not set
        use_se: bool = False,
        use_multi_scale: bool = False,
        cd_lead_out_dim: int = 128,   # output dim of ConductionLeadGroupEncoder
    ) -> None:
        super().__init__()
        self.output_dim = d_model
        stage_dims = stage_dims or [128, 192, d_model]

        # ── Shared CNN front-end ────────────────────────────────────────────
        self.front_end = MorphologyConvFrontEnd(
            num_leads=num_leads,
            stem_dim=stem_dim,
            stage_dims=stage_dims,
            downsample_factor=downsample_factor,
            dropout=dropout,
            use_se=use_se,
        )
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.multi_scale = MultiScaleTemporalBranch(stage_dims[-1], d_model, dropout=dropout)
            self.sequence_proj = nn.Identity()
        else:
            self.sequence_proj = nn.Conv1d(stage_dims[-1], d_model, kernel_size=1, bias=False)

        # ── Optional shared Transformer layers (before branching) ───────────
        self.num_shared_layers = num_shared_layers
        if num_shared_layers > 0:
            self.pre_norm_shared = nn.LayerNorm(d_model)
            shared_enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.shared_encoder = nn.TransformerEncoder(
                shared_enc_layer, num_layers=num_shared_layers, norm=nn.LayerNorm(d_model),
            )

        # ── Expert branches (heterogeneous configs) ─────────────────────────
        def _make_expert(layers, e_nhead, e_ffn):
            return _ExpertBranch(
                d_model=d_model,
                nhead=e_nhead or nhead,
                num_layers=layers or num_expert_layers,
                dim_feedforward=e_ffn or dim_feedforward,
                dropout=dropout,
            )

        self.arrhythmia_expert = _make_expert(arrhy_layers, arrhy_nhead, arrhy_ffn)
        self.mi_expert         = _make_expert(mi_layers,    mi_nhead,    mi_ffn)
        self.conduction_expert = _make_expert(cond_layers,  cond_nhead,  cond_ffn)

        # ── Conduction Lead Group Encoder ────────────────────────────────────
        self.cd_lead_encoder = ConductionLeadGroupEncoder(
            out_dim=cd_lead_out_dim, dropout=dropout
        )
        self.cd_lead_out_dim = cd_lead_out_dim

    def _cnn_to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN front-end and project to (B, T, d_model)."""
        conv = self.front_end(x)
        if self.use_multi_scale:
            seq = self.multi_scale(conv)
            return self.sequence_proj(seq).transpose(1, 2)
        return self.sequence_proj(conv).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_seq = self._cnn_to_seq(x)               # (B, T, d_model)

        if self.num_shared_layers > 0:
            shared_seq = self.shared_encoder(
                self.pre_norm_shared(shared_seq)
            )

        arrhy_out = self.arrhythmia_expert(shared_seq)
        mi_out    = self.mi_expert(shared_seq)
        cond_out  = self.conduction_expert(shared_seq)

        # Conduction lead-group encoder (raw signal features)
        cd_lead_feats = self.cd_lead_encoder(x)        # (B, cd_lead_out_dim)

        return {
            # Arrhythmia branch outputs
            "arrhy_global":   arrhy_out["global_features"],
            "arrhy_seq":      arrhy_out["sequence_features"],
            # MI branch outputs
            "mi_global":      mi_out["global_features"],
            "mi_seq":         mi_out["sequence_features"],
            # Conduction branch outputs
            "cond_global":    cond_out["global_features"],
            "cond_seq":       cond_out["sequence_features"],
            "cond_lead_feats": cd_lead_feats,
            # Backward-compat aliases (used by old code paths)
            "global_features":   arrhy_out["global_features"],
            "sequence_features": arrhy_out["sequence_features"],
        }


