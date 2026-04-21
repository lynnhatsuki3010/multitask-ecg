"""
ecg_transformer.py
Patch-based Transformer encoder for 12-lead ECG classification.

Architecture:
  Input: (B, 12, T)
  → PatchEmbedding: split each lead into patches → (B, num_patches, d_model)
  → [CLS] token prepended → (B, 1+num_patches, d_model)
  → Positional encoding
  → TransformerEncoder (N layers)
  → CLS output → feature vector (B, d_model)
"""
import math
import torch
import torch.nn as nn
from typing import Optional


# ─── Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─── Patch Embedding ─────────────────────────────────────────────────────────

class LinearPatchEmbedding(nn.Module):
    """
    Split 12-lead ECG into non-overlapping patches and project to d_model.

    Input:  (B, num_leads, T)
    Output: (B, num_patches, d_model)

    Each "patch" = all 12 leads × patch_size time steps
    → flattened to (12 * patch_size) → linear projection to d_model.
    """

    def __init__(self, num_leads: int = 12, patch_size: int = 25, d_model: int = 128):
        super().__init__()
        self.num_leads = num_leads
        self.patch_size = patch_size
        self.d_model = d_model
        self.proj = nn.Linear(num_leads * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_leads, T)"""
        B, C, T = x.shape
        assert T % self.patch_size == 0, (
            f"Signal length {T} must be divisible by patch_size {self.patch_size}"
        )
        num_patches = T // self.patch_size

        # (B, C, T) → (B, num_patches, C * patch_size)
        x = x.reshape(B, C, num_patches, self.patch_size)   # (B, C, NP, PS)
        x = x.permute(0, 2, 1, 3)                           # (B, NP, C, PS)
        x = x.reshape(B, num_patches, C * self.patch_size)  # (B, NP, C*PS)

        x = self.proj(x)   # (B, NP, d_model)
        x = self.norm(x)
        return x

class MultiScaleConvEmbedding(nn.Module):
    """
    Multi-scale parallel CNN stem for ECG patch embedding.

    3 branches with different receptive fields target distinct ECG morphologies:
      - Branch 1 (kernel=7,  70ms):  Fast spikes  → QRS complex, arrhythmia
      - Branch 2 (kernel=25, 250ms): Medium waves → P-wave, T-wave morphology
      - Branch 3 (kernel=51, 510ms): Slow trends  → ST elevation/depression (MI)

    Supports arbitrary patch_size: total downsampling stride = patch_size.
    Implemented as two-stage: stride_a × stride_b = patch_size.
      | patch_size | stride_a | stride_b | patches (1000 samples) |
      |------------|----------|----------|------------------------|
      |     25     |    5     |    5     |          40            |
      |     20     |    4     |    5     |          50            |
      |     16     |    4     |    4     |        62 (≈60)        |
      |     10     |    2     |    5     |         100            |
    """

    def __init__(self, num_leads: int = 12, patch_size: int = 25, d_model: int = 128, fs: int = 100):
        super().__init__()
        scale = max(1, fs // 100)
        # Factorize patch_size into two strides (stride_a × stride_b ≈ patch_size)
        # Keep stride_b = 5 when possible for consistent fusion kernel
        # Scale the branch block downsampling if a high patch_size ensures a large uniform stride
        if patch_size % (5 * scale) == 0:
            stride_a, stride_b = patch_size // 5, 5
        elif patch_size % (4 * scale) == 0:
            stride_a, stride_b = patch_size // 4, 4
        elif patch_size % (2 * scale) == 0:
            stride_a, stride_b = patch_size // 2, 2
        else:
            stride_a, stride_b = patch_size, 1

        branch_ch  = d_model // 3
        branch_ch3 = d_model - 2 * branch_ch   # absorb rounding

        # ── Branch 1: QRS / Arrhythmia (kernel 70ms) ──────
        b1_k, b1_p = 7 * scale, 3 * scale
        self.branch1 = nn.Sequential(
            nn.Conv1d(num_leads, branch_ch, kernel_size=b1_k,  stride=stride_a, padding=b1_p,  bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )

        # ── Branch 2: P/T-wave morphology (kernel 250ms) ─
        b2_k, b2_p = 25 * scale, 12 * scale
        self.branch2 = nn.Sequential(
            nn.Conv1d(num_leads, branch_ch, kernel_size=b2_k, stride=stride_a, padding=b2_p, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.GELU(),
        )

        # ── Branch 3: ST-segment / MI trends (kernel 510ms)
        b3_k, b3_p = 51 * scale, 25 * scale
        self.branch3 = nn.Sequential(
            nn.Conv1d(num_leads, branch_ch3, kernel_size=b3_k, stride=stride_a, padding=b3_p, bias=False),
            nn.BatchNorm1d(branch_ch3),
            nn.GELU(),
        )

        # ── Fusion: merge 3 branches + further downsample by stride_b ──────────
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, stride=stride_b, padding=3, bias=False),  # fusion kernel stays the same, operating on patch semantics
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )


        self._stride_a = stride_a
        self._stride_b = stride_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, T)  →  (B, T//patch_size, d_model)"""
        b1 = self.branch1(x)             # (B, d/3,      T//stride_a)
        b2 = self.branch2(x)             # (B, d/3,      T//stride_a)
        b3 = self.branch3(x)             # (B, d-2*(d/3),T//stride_a)

        fused = torch.cat([b1, b2, b3], dim=1)   # (B, d_model, T//stride_a)
        out   = self.fuse(fused)                  # (B, d_model, T//patch_size)
        return out.transpose(1, 2)                # (B, num_patches, d_model)





# ─── ECG Transformer ─────────────────────────────────────────────────────────

class ECGTransformer(nn.Module):
    """
    Transformer encoder for 12-lead ECG.

    Returns a feature vector of shape (B, d_model) representing the
    [CLS] token output, suitable for downstream task heads.
    """

    def __init__(
        self,
        num_leads: int = 12,
        signal_length: int = 1000,
        patch_size: int = 25,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        fs: int = 100,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        num_patches = signal_length // patch_size

        # Patch embedding (Multi-Scale CNN Stem: QRS / P-T wave / ST-segment)
        self.patch_embed = MultiScaleConvEmbedding(num_leads, patch_size, d_model, fs=fs)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding (seq includes CLS + patches)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_patches + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, T)
        Returns:
            features: (B, d_model)  — CLS token embedding
        """
        B = x.size(0)

        # Patch + embed
        patches = self.patch_embed(x)   # (B, num_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        tokens = torch.cat([cls, patches], dim=1)    # (B, 1+NP, d_model)

        # Positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer
        out = self.transformer(tokens)   # (B, 1+NP, d_model)

        # Return CLS token
        features = out[:, 0, :]          # (B, d_model)
        return features


if __name__ == "__main__":
    model = ECGTransformer(
        num_leads=12, signal_length=1000, patch_size=25,
        d_model=128, nhead=4, num_encoder_layers=4,
    )
    x = torch.randn(4, 12, 1000)
    out = model(x)
    print(f"Input: {x.shape}  →  Features: {out.shape}")  # (4, 128)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
