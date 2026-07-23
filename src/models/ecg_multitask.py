"""
Clean multitask ECG model built on top of interchangeable backbones.

routing_mode (model config):
  - "soft"        : E06-style expert fusion + LeadGroupEncoder (default)
  - "hard_graph"  : E07-style PerLeadEncoder + GraphTransformer on raw signal
  - "subset_lead" : Approach 2 — LeadGroupEncoder per anatomy group, no 12-lead graph mix
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.models.backbones import (
    AttentionPooling,
    DecoupledMultiTaskBackbone,
    MultiBranchTransformerBackbone,
)

# PTB-XL 12-lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
# Indices:               0   1   2    3    4    5   6   7   8   9  10  11

INFERIOR_LEADS   = [1, 2, 5]          # II, III, aVF
RECIPROCAL_LEADS = [0, 4]             # I, aVL
ANTERIOR_LEADS   = [6, 7, 8, 9]       # V1–V4
LATERAL_LEADS    = [10, 11]           # V5, V6
ANTERIOR6_LEADS  = [6, 7, 8, 9, 10, 11]  # V1–V6

# Clinical lead set per MI label — used as a soft anatomical prior for the
# dynamic anatomical attention head (mi_head_mode="dyn_anat").
MI_LABELS = ["imi", "asmi", "ilmi", "ami"]
MI_LABEL_LEADS: Dict[str, List[int]] = {
    "imi":  INFERIOR_LEADS + RECIPROCAL_LEADS,                       # II,III,aVF + I,aVL
    "asmi": ANTERIOR_LEADS,                                          # V1–V4
    "ilmi": INFERIOR_LEADS + RECIPROCAL_LEADS + LATERAL_LEADS,       # + V5,V6
    "ami":  ANTERIOR6_LEADS,                                         # V1–V6
}


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


class TaskTokenPooling(nn.Module):
    """Task-specific attention pooling over transformer/CNN token sequences."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.score_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.score_proj(tokens))
        scores = torch.matmul(projected, self.query)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(tokens * weights, dim=1)
        return self.dropout(pooled)


class LeadGroupEncoder(nn.Module):
    """Small dedicated encoder for clinically relevant MI lead groups."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_sequence: bool = False):
        h = self.net(x)
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        pooled = self.proj(torch.cat([avg, mx], dim=-1))
        if return_sequence:
            return pooled, h
        return pooled


class PerLeadEncoder(nn.Module):
    """Encodes each lead independently using a 1D CNN."""

    def __init__(self, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
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


class _GraphTransformerHead(nn.Module):
    """Shared graph-transformer stack for hard-routing heads."""

    NUM_LEADS = 12

    def __init__(self, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.per_lead_encoder = PerLeadEncoder(out_dim=branch_dim, dropout=dropout)
        self.lead_embeddings = nn.Parameter(torch.randn(1, self.NUM_LEADS, branch_dim))
        nn.init.normal_(self.lead_embeddings, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=4,
            dim_feedforward=branch_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.graph_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def encode_graph(self, signal: torch.Tensor) -> torch.Tensor:
        B, L, T = signal.size()
        x_reshaped = signal.view(B * L, 1, T)
        node_features = self.per_lead_encoder(x_reshaped)
        nodes = node_features.view(B, L, -1) + self.lead_embeddings
        return self.graph_transformer(nodes)


class MIHead(nn.Module):
    """Soft-routing MI head: mi_expert features + anatomy LeadGroupEncoders (4 labels)."""

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float, use_cross_attention: bool = False) -> None:
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.imi_token_pool  = TaskTokenPooling(shared_dim, dropout)
        self.asmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ilmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ami_token_pool  = TaskTokenPooling(shared_dim, dropout)

        self.inferior_encoder   = LeadGroupEncoder(len(INFERIOR_LEADS),   branch_dim,      dropout)
        self.reciprocal_encoder = LeadGroupEncoder(len(RECIPROCAL_LEADS), branch_dim // 2, dropout)
        self.anterior_encoder   = LeadGroupEncoder(len(ANTERIOR_LEADS),   branch_dim,      dropout)
        self.lateral_encoder    = LeadGroupEncoder(len(LATERAL_LEADS),    branch_dim,      dropout)
        self.anterior6_encoder  = LeadGroupEncoder(len(ANTERIOR6_LEADS),  branch_dim,      dropout)

        if use_cross_attention:
            self.cross_attn_inferior   = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_reciprocal = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_anterior   = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_lateral    = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.cross_attn_anterior6  = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=4, batch_first=True, dropout=dropout)
            self.proj_k_inf  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_rec  = nn.Conv1d(branch_dim // 2, shared_dim, kernel_size=1)
            self.proj_k_ant  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_lat  = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            self.proj_k_ant6 = nn.Conv1d(branch_dim,      shared_dim, kernel_size=1)
            imi_in  = shared_dim * 4
            asmi_in = shared_dim * 3
            ilmi_in = shared_dim * 5
            ami_in  = shared_dim * 3
        else:
            imi_in  = shared_dim * 2 + branch_dim + branch_dim // 2
            asmi_in = shared_dim * 2 + branch_dim
            ilmi_in = shared_dim * 2 + branch_dim + branch_dim // 2 + branch_dim
            ami_in  = shared_dim * 2 + branch_dim

        self.imi_head  = MLPHead(imi_in,  1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(asmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(ilmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(ami_in,  1, hidden_dim=branch_dim, dropout=dropout)

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        imi_tokens  = self.imi_token_pool(sequence_features)
        asmi_tokens = self.asmi_token_pool(sequence_features)
        ilmi_tokens = self.ilmi_token_pool(sequence_features)
        ami_tokens  = self.ami_token_pool(sequence_features)

        if self.use_cross_attention:
            _, inf_seq  = self.inferior_encoder(signal[:, INFERIOR_LEADS, :],   return_sequence=True)
            _, rec_seq  = self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :], return_sequence=True)
            _, ant_seq  = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :],   return_sequence=True)
            _, lat_seq  = self.lateral_encoder(signal[:, LATERAL_LEADS, :],     return_sequence=True)
            _, ant6_seq = self.anterior6_encoder(signal[:, ANTERIOR6_LEADS, :], return_sequence=True)
            inf_seq  = self.proj_k_inf(inf_seq).transpose(1, 2)
            rec_seq  = self.proj_k_rec(rec_seq).transpose(1, 2)
            ant_seq  = self.proj_k_ant(ant_seq).transpose(1, 2)
            lat_seq  = self.proj_k_lat(lat_seq).transpose(1, 2)
            ant6_seq = self.proj_k_ant6(ant6_seq).transpose(1, 2)
            q = shared_features.unsqueeze(1)
            inferior,   _ = self.cross_attn_inferior(q, inf_seq, inf_seq)
            reciprocal, _ = self.cross_attn_reciprocal(q, rec_seq, rec_seq)
            anterior,   _ = self.cross_attn_anterior(q, ant_seq, ant_seq)
            lateral,    _ = self.cross_attn_lateral(q, lat_seq, lat_seq)
            anterior6,  _ = self.cross_attn_anterior6(q, ant6_seq, ant6_seq)
            inferior   = inferior.squeeze(1)
            reciprocal = reciprocal.squeeze(1)
            anterior   = anterior.squeeze(1)
            lateral    = lateral.squeeze(1)
            anterior6  = anterior6.squeeze(1)
        else:
            inferior   = self.inferior_encoder(signal[:, INFERIOR_LEADS, :])
            reciprocal = self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :])
            anterior   = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :])
            lateral    = self.lateral_encoder(signal[:, LATERAL_LEADS, :])
            anterior6  = self.anterior6_encoder(signal[:, ANTERIOR6_LEADS, :])

        imi  = self.imi_head(torch.cat([shared_features, imi_tokens,  inferior, reciprocal],          dim=-1))
        asmi = self.asmi_head(torch.cat([shared_features, asmi_tokens, anterior],                     dim=-1))
        ilmi = self.ilmi_head(torch.cat([shared_features, ilmi_tokens, inferior, reciprocal, lateral], dim=-1))
        ami  = self.ami_head(torch.cat([shared_features,  ami_tokens,  anterior6],                    dim=-1))
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)


class SubsetLeadMIHead(nn.Module):
    """Approach 2: LeadGroupEncoder per anatomy group + small group-only transformer."""

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.imi_token_pool  = TaskTokenPooling(shared_dim, dropout)
        self.asmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ilmi_token_pool = TaskTokenPooling(shared_dim, dropout)
        self.ami_token_pool  = TaskTokenPooling(shared_dim, dropout)

        self.inferior_encoder   = LeadGroupEncoder(len(INFERIOR_LEADS),   branch_dim,      dropout)
        self.reciprocal_encoder = LeadGroupEncoder(len(RECIPROCAL_LEADS), branch_dim // 2, dropout)
        self.anterior_encoder   = LeadGroupEncoder(len(ANTERIOR_LEADS),   branch_dim,      dropout)
        self.lateral_encoder    = LeadGroupEncoder(len(LATERAL_LEADS),    branch_dim,      dropout)
        self.anterior6_encoder  = LeadGroupEncoder(len(ANTERIOR6_LEADS),  branch_dim,      dropout)
        self.reciprocal_proj = nn.Linear(branch_dim // 2, branch_dim)

        def _group_tf() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=branch_dim, nhead=4, dim_feedforward=branch_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=1)

        self.imi_group_tf  = _group_tf()
        self.asmi_group_tf = _group_tf()
        self.ilmi_group_tf = _group_tf()
        self.ami_group_tf  = _group_tf()

        self.imi_pool  = TaskTokenPooling(branch_dim, dropout)
        self.asmi_pool = TaskTokenPooling(branch_dim, dropout)
        self.ilmi_pool = TaskTokenPooling(branch_dim, dropout)
        self.ami_pool  = TaskTokenPooling(branch_dim, dropout)

        imi_in  = shared_dim * 2 + branch_dim
        asmi_in = shared_dim * 2 + branch_dim
        ilmi_in = shared_dim * 2 + branch_dim
        ami_in  = shared_dim * 2 + branch_dim

        self.imi_head  = MLPHead(imi_in,  1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(asmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(ilmi_in, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(ami_in,  1, hidden_dim=branch_dim, dropout=dropout)

    def _group_nodes(
        self,
        inferior: torch.Tensor,
        reciprocal: torch.Tensor,
        lateral: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rec = self.reciprocal_proj(reciprocal)
        parts = [inferior.unsqueeze(1), rec.unsqueeze(1)]
        if lateral is not None:
            parts.append(lateral.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        imi_tokens  = self.imi_token_pool(sequence_features)
        asmi_tokens = self.asmi_token_pool(sequence_features)
        ilmi_tokens = self.ilmi_token_pool(sequence_features)
        ami_tokens  = self.ami_token_pool(sequence_features)

        inferior   = self.inferior_encoder(signal[:, INFERIOR_LEADS, :])
        reciprocal = self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :])
        anterior   = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :])
        lateral    = self.lateral_encoder(signal[:, LATERAL_LEADS, :])
        anterior6  = self.anterior6_encoder(signal[:, ANTERIOR6_LEADS, :])

        imi_nodes  = self._group_nodes(inferior, reciprocal)
        ilmi_nodes = self._group_nodes(inferior, reciprocal, lateral)
        asmi_nodes = anterior.unsqueeze(1)
        ami_nodes  = anterior6.unsqueeze(1)

        imi_ctx  = self.imi_pool(self.imi_group_tf(imi_nodes))
        ilmi_ctx = self.ilmi_pool(self.ilmi_group_tf(ilmi_nodes))
        asmi_ctx = self.asmi_pool(self.asmi_group_tf(asmi_nodes))
        ami_ctx  = self.ami_pool(self.ami_group_tf(ami_nodes))

        imi  = self.imi_head(torch.cat([shared_features, imi_tokens,  imi_ctx],  dim=-1))
        asmi = self.asmi_head(torch.cat([shared_features, asmi_tokens, asmi_ctx], dim=-1))
        ilmi = self.ilmi_head(torch.cat([shared_features, ilmi_tokens, ilmi_ctx], dim=-1))
        ami  = self.ami_head(torch.cat([shared_features,  ami_tokens,  ami_ctx],  dim=-1))
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)


class SiblingContrast(nn.Module):
    """Gated contrast refinement between two sibling labels.

    out = x + gate * proj(x - sibling), where gate in [0,1] is learned per-feature
    from [x, x - sibling]. proj is zero-initialised so the module starts as an
    identity map and only ramps up contrast where it helps training — this avoids
    the hard symmetric subtraction impoverishing the "extended" sibling.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.constant_(self.gate.bias, -2.0)  # start with small gate (~0.12)

    def forward(self, x: torch.Tensor, sibling: torch.Tensor) -> torch.Tensor:
        delta = x - sibling
        g = torch.sigmoid(self.gate(torch.cat([x, delta], dim=-1)))
        return x + g * self.proj(delta)


class SharedPerLeadEncoder(nn.Module):
    """Weight-shared 1D CNN applied independently to each of the 12 leads.

    Produces one token per lead: (B, 12, out_dim). Sharing weights across leads
    keeps the parameter count low and lets the per-lead tokens live in a common
    space so a downstream attention can compare them.
    """

    def __init__(self, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        B, L, T = signal.size()
        h = self.net(signal.reshape(B * L, 1, T))
        avg = self.avg_pool(h).squeeze(-1)
        mx = self.max_pool(h).squeeze(-1)
        tok = self.proj(torch.cat([avg, mx], dim=-1))
        return tok.view(B, L, -1)


class DynamicAnatomicalAttention(nn.Module):
    """Per-patient dynamic attention over the 12 leads with a soft anatomical prior.

    Motivation. Fixed anatomical grouping (contrast_v2) always reads the same lead
    set for a label. Clinically, the diagnostic leads shift per patient: an inferior
    MI shows ST elevation in II/III/aVF, so attention there should rise, whereas an
    atypical/lateral extension should let V5/V6 or reciprocal leads speak up.

    Design. Each of the 12 leads is encoded independently (SharedPerLeadEncoder) and
    contextualised by a 1-layer transformer so leads can "see" each other (reciprocal
    changes). Each MI label owns a learnable query and an additive per-lead bias
    initialised from its clinical anatomy. Attention weights depend on the *content*
    of each lead token — hence adapt per patient — while the anatomical bias keeps the
    focus clinically grounded and lets off-group leads be up-weighted only when the
    signal supports it. The bias is learnable, so the prior is a starting point, not
    a hard mask.
    """

    NUM_LEADS = 12

    def __init__(self, dim: int, dropout: float, nhead: int = 4, prior_strength: float = 2.0) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim ({dim}) must be divisible by nhead ({nhead})")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.lead_encoder = SharedPerLeadEncoder(dim, dropout)
        self.lead_pos = nn.Parameter(torch.randn(1, self.NUM_LEADS, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.lead_context_tf = nn.TransformerEncoder(layer, num_layers=1)

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.query = nn.ParameterDict(
            {lbl: nn.Parameter(torch.randn(dim) * 0.02) for lbl in MI_LABELS}
        )
        self.out_proj = nn.ModuleDict({lbl: nn.Linear(dim, dim) for lbl in MI_LABELS})
        self.attn_dropout = nn.Dropout(dropout)

        lead_bias = {}
        for lbl in MI_LABELS:
            b = torch.zeros(self.NUM_LEADS)
            for l in MI_LABEL_LEADS[lbl]:
                b[l] = prior_strength
            lead_bias[lbl] = nn.Parameter(b)
        self.lead_bias = nn.ParameterDict(lead_bias)

    def _attend(self, lbl: str, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # k, v: (B, L, nhead, head_dim)
        B = k.size(0)
        q = self.query[lbl].view(self.nhead, self.head_dim)                     # (H, hd)
        scores = torch.einsum("blhd,hd->bhl", k, q) / (self.head_dim ** 0.5)    # (B, H, L)
        scores = scores + self.lead_bias[lbl].view(1, 1, -1)                    # anatomical prior
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))             # (B, H, L)
        ctx = torch.einsum("bhl,blhd->bhd", weights, v).reshape(B, self.dim)    # (B, dim)
        return self.out_proj[lbl](ctx)

    def forward(self, signal: torch.Tensor):
        tokens = self.lead_encoder(signal) + self.lead_pos                      # (B, 12, dim)
        tokens = self.lead_context_tf(tokens)
        B, L, _ = tokens.size()
        k = self.k_proj(tokens).view(B, L, self.nhead, self.head_dim)
        v = self.v_proj(tokens).view(B, L, self.nhead, self.head_dim)
        return tuple(self._attend(lbl, k, v) for lbl in MI_LABELS)


class DynamicLeadAttention(nn.Module):
    """Per-patient, per-lead attention producing a GATED ADDITIVE feature per MI label.

    Unlike DynamicAnatomicalAttention (which replaces the anatomy encoders), this
    module is meant to be *added on top of* the contrast_v2 joint region encoders:
    the joint cross-lead morphology is preserved, and this branch only injects a
    dynamic, per-patient "which leads matter now" signal. It is gated and
    zero-initialised so the head starts identical to contrast_v2 and the dynamic
    term ramps up only where it helps — avoiding the regression seen when the
    joint encoders were thrown away.

    It also exposes interpretable softmax attention weights over the 12 leads for
    each label (return_weights=True), e.g. for an inferior MI the IMI weights
    should concentrate on II/III/aVF.
    """

    NUM_LEADS = 12

    def __init__(self, dim: int, dropout: float, nhead: int = 4, prior_strength: float = 2.0) -> None:
        super().__init__()
        if dim % nhead != 0:
            raise ValueError(f"dim ({dim}) must be divisible by nhead ({nhead})")
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.lead_encoder = SharedPerLeadEncoder(dim, dropout)
        self.lead_pos = nn.Parameter(torch.randn(1, self.NUM_LEADS, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.lead_context_tf = nn.TransformerEncoder(layer, num_layers=1)

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.query = nn.ParameterDict(
            {lbl: nn.Parameter(torch.randn(dim) * 0.02) for lbl in MI_LABELS}
        )
        self.out_proj = nn.ModuleDict({lbl: nn.Linear(dim, dim) for lbl in MI_LABELS})
        # Gated + zero-init so the whole branch starts as a no-op (== contrast_v2).
        self.gate = nn.ParameterDict(
            {lbl: nn.Parameter(torch.full((dim,), -2.0)) for lbl in MI_LABELS}
        )
        for lbl in MI_LABELS:
            nn.init.zeros_(self.out_proj[lbl].weight)
            nn.init.zeros_(self.out_proj[lbl].bias)
        self.attn_dropout = nn.Dropout(dropout)

        lead_bias = {}
        for lbl in MI_LABELS:
            b = torch.zeros(self.NUM_LEADS)
            for l in MI_LABEL_LEADS[lbl]:
                b[l] = prior_strength
            lead_bias[lbl] = nn.Parameter(b)
        self.lead_bias = nn.ParameterDict(lead_bias)

    def forward(self, signal: torch.Tensor, return_weights: bool = False):
        tokens = self.lead_encoder(signal) + self.lead_pos                      # (B, 12, dim)
        tokens = self.lead_context_tf(tokens)
        B, L, _ = tokens.size()
        k = self.k_proj(tokens).view(B, L, self.nhead, self.head_dim)
        v = self.v_proj(tokens).view(B, L, self.nhead, self.head_dim)

        adds: Dict[str, torch.Tensor] = {}
        weights: Dict[str, torch.Tensor] = {}
        for lbl in MI_LABELS:
            q = self.query[lbl].view(self.nhead, self.head_dim)
            scores = torch.einsum("blhd,hd->bhl", k, q) / (self.head_dim ** 0.5)   # (B,H,L)
            scores = scores + self.lead_bias[lbl].view(1, 1, -1)
            w = torch.softmax(scores, dim=-1)                                      # (B,H,L)
            ctx = torch.einsum("bhl,blhd->bhd", self.attn_dropout(w), v).reshape(B, self.dim)
            g = torch.sigmoid(self.gate[lbl])
            adds[lbl] = g * self.out_proj[lbl](ctx)                               # gated additive term
            if return_weights:
                weights[lbl] = w.mean(dim=1)                                      # (B,L) avg over heads
        if return_weights:
            return adds, weights
        return adds


class DecoupledSubsetMIHead(nn.Module):
    """E10/E11: subset LeadGroupEncoder + group TF only — no shared CNN expert fusion.

    head_mode:
      - "shared"     : E10 default. Per-label group transformers; IMI/ASMI never
                       see the lateral leads, so the "narrow" siblings cannot learn
                       to exclude their "extended" siblings (IMI vs ILMI, ASMI vs
                       AMI) -> recall-heavy, precision-poor on overlapping anatomy.
      - "contrast"   : Phase 2. Encode each anatomical region once, run a shared
                       region transformer, and let every label attend over the FULL
                       region tokens (so IMI/ASMI also see the lateral leads). A
                       symmetric sibling contrast refinement pushes the paired logits
                       apart. Helped IMI but degraded the extended siblings
                       (ILMI/AMI lost their dedicated joint encoding).
      - "contrast_v2": Phase 2.1. Like "contrast" but (a) restores a dedicated joint
                       encoder for each extended sibling (anterior6 V1-V6 for AMI,
                       inferolateral for ILMI) as an extra region token, and (b) uses
                       a gated contrast (identity at init) instead of hard subtraction.
    """

    INFEROLATERAL_LEADS = INFERIOR_LEADS + RECIPROCAL_LEADS + LATERAL_LEADS

    def __init__(self, branch_dim: int, dropout: float, head_mode: str = "shared") -> None:
        super().__init__()
        self.head_mode = head_mode

        def _group_tf() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=branch_dim, nhead=4, dim_feedforward=branch_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=1)

        self.imi_pool  = TaskTokenPooling(branch_dim, dropout)
        self.asmi_pool = TaskTokenPooling(branch_dim, dropout)
        self.ilmi_pool = TaskTokenPooling(branch_dim, dropout)
        self.ami_pool  = TaskTokenPooling(branch_dim, dropout)

        self.imi_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)

        if head_mode == "contrast":
            # One encoder per anatomical region (encoded once, shared by siblings).
            self.inferior_encoder   = LeadGroupEncoder(len(INFERIOR_LEADS),   branch_dim,      dropout)
            self.reciprocal_encoder = LeadGroupEncoder(len(RECIPROCAL_LEADS), branch_dim // 2, dropout)
            self.lateral_encoder    = LeadGroupEncoder(len(LATERAL_LEADS),    branch_dim,      dropout)
            self.anterior_encoder   = LeadGroupEncoder(len(ANTERIOR_LEADS),   branch_dim,      dropout)
            self.reciprocal_proj    = nn.Linear(branch_dim // 2, branch_dim)
            # Shared region transformers over the full token set of each region.
            self.inferior_region_tf = _group_tf()   # tokens: [inferior, reciprocal, lateral]
            self.anterior_region_tf = _group_tf()    # tokens: [anterior(V1-V4), lateral(V5,V6)]
            # Sibling contrast refinement (soft mutual exclusion within a region).
            self.inf_contrast = nn.Linear(branch_dim, branch_dim)
            self.ant_contrast = nn.Linear(branch_dim, branch_dim)
        elif head_mode in ("contrast_v2", "dyn_anat_v2"):
            self.inferior_encoder     = LeadGroupEncoder(len(INFERIOR_LEADS),         branch_dim,      dropout)
            self.reciprocal_encoder   = LeadGroupEncoder(len(RECIPROCAL_LEADS),       branch_dim // 2, dropout)
            self.lateral_encoder      = LeadGroupEncoder(len(LATERAL_LEADS),          branch_dim,      dropout)
            self.anterior_encoder     = LeadGroupEncoder(len(ANTERIOR_LEADS),         branch_dim,      dropout)
            self.reciprocal_proj      = nn.Linear(branch_dim // 2, branch_dim)
            # Dedicated joint encoders for the extended siblings (restored from E10).
            self.anterior6_encoder    = LeadGroupEncoder(len(ANTERIOR6_LEADS),        branch_dim,      dropout)
            self.inferolateral_encoder = LeadGroupEncoder(len(self.INFEROLATERAL_LEADS), branch_dim,   dropout)
            self.inferior_region_tf = _group_tf()   # tokens: [inferior, reciprocal, lateral, inferolateral]
            self.anterior_region_tf = _group_tf()    # tokens: [anterior, lateral, anterior6]
            # Gated contrast refinement (identity at init).
            self.inf_contrast = SiblingContrast(branch_dim)
            self.ant_contrast = SiblingContrast(branch_dim)
            if head_mode == "dyn_anat_v2":
                # Phase 5: add a gated per-patient per-lead attention on TOP of the
                # contrast_v2 joint encoders (starts as a no-op, ramps up if helpful).
                self.lead_attn = DynamicLeadAttention(branch_dim, dropout)
        elif head_mode == "dyn_anat":
            # Phase 4. Per-patient dynamic anatomical attention over the 12 leads,
            # with a soft (learnable) anatomical prior per label, followed by the
            # same gated sibling contrast as contrast_v2.
            self.dyn_attn = DynamicAnatomicalAttention(branch_dim, dropout)
            self.inf_contrast = SiblingContrast(branch_dim)
            self.ant_contrast = SiblingContrast(branch_dim)
        else:
            self.inferior_encoder   = LeadGroupEncoder(len(INFERIOR_LEADS),   branch_dim,      dropout)
            self.reciprocal_encoder = LeadGroupEncoder(len(RECIPROCAL_LEADS), branch_dim // 2, dropout)
            self.anterior_encoder   = LeadGroupEncoder(len(ANTERIOR_LEADS),   branch_dim,      dropout)
            self.lateral_encoder    = LeadGroupEncoder(len(LATERAL_LEADS),    branch_dim,      dropout)
            self.anterior6_encoder  = LeadGroupEncoder(len(ANTERIOR6_LEADS),  branch_dim,      dropout)
            self.reciprocal_proj = nn.Linear(branch_dim // 2, branch_dim)
            self.imi_group_tf  = _group_tf()
            self.asmi_group_tf = _group_tf()
            self.ilmi_group_tf = _group_tf()
            self.ami_group_tf  = _group_tf()

    def _group_nodes(
        self,
        inferior: torch.Tensor,
        reciprocal: torch.Tensor,
        lateral: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rec = self.reciprocal_proj(reciprocal)
        parts = [inferior.unsqueeze(1), rec.unsqueeze(1)]
        if lateral is not None:
            parts.append(lateral.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if self.head_mode == "contrast":
            return self._forward_contrast(signal)
        if self.head_mode == "contrast_v2":
            return self._forward_contrast_v2(signal)
        if self.head_mode == "dyn_anat":
            return self._forward_dyn_anat(signal)
        if self.head_mode == "dyn_anat_v2":
            return self._forward_dyn_anat_v2(signal)
        return self._forward_shared(signal)

    def _forward_dyn_anat(self, signal: torch.Tensor) -> torch.Tensor:
        imi_ctx, asmi_ctx, ilmi_ctx, ami_ctx = self.dyn_attn(signal)

        # Gated contrast (identity at init) — same sibling pairs as contrast_v2.
        imi_ref  = self.inf_contrast(imi_ctx,  ilmi_ctx)
        ilmi_ref = self.inf_contrast(ilmi_ctx, imi_ctx)
        asmi_ref = self.ant_contrast(asmi_ctx, ami_ctx)
        ami_ref  = self.ant_contrast(ami_ctx,  asmi_ctx)

        imi  = self.imi_head(imi_ref)
        asmi = self.asmi_head(asmi_ref)
        ilmi = self.ilmi_head(ilmi_ref)
        ami  = self.ami_head(ami_ref)
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)

    def _forward_shared(self, signal: torch.Tensor) -> torch.Tensor:
        inferior   = self.inferior_encoder(signal[:, INFERIOR_LEADS, :])
        reciprocal = self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :])
        anterior   = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :])
        lateral    = self.lateral_encoder(signal[:, LATERAL_LEADS, :])
        anterior6  = self.anterior6_encoder(signal[:, ANTERIOR6_LEADS, :])

        imi_ctx  = self.imi_pool(self.imi_group_tf(self._group_nodes(inferior, reciprocal)))
        ilmi_ctx = self.ilmi_pool(self.ilmi_group_tf(self._group_nodes(inferior, reciprocal, lateral)))
        asmi_ctx = self.asmi_pool(self.asmi_group_tf(anterior.unsqueeze(1)))
        ami_ctx  = self.ami_pool(self.ami_group_tf(anterior6.unsqueeze(1)))

        imi  = self.imi_head(imi_ctx)
        asmi = self.asmi_head(asmi_ctx)
        ilmi = self.ilmi_head(ilmi_ctx)
        ami  = self.ami_head(ami_ctx)
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)

    def _forward_contrast(self, signal: torch.Tensor) -> torch.Tensor:
        inferior   = self.inferior_encoder(signal[:, INFERIOR_LEADS, :])
        reciprocal = self.reciprocal_proj(self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :]))
        lateral    = self.lateral_encoder(signal[:, LATERAL_LEADS, :])
        anterior   = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :])

        # Inferior region tokens: [inferior, reciprocal, lateral]
        inf_tokens = torch.stack([inferior, reciprocal, lateral], dim=1)
        inf_tokens = self.inferior_region_tf(inf_tokens)
        # Anterior region tokens: [anterior(V1-V4), lateral(V5,V6)] -> AMI = anterior + lateral
        ant_tokens = torch.stack([anterior, lateral], dim=1)
        ant_tokens = self.anterior_region_tf(ant_tokens)

        # Each label attends over the full region tokens with its own query, so the
        # "narrow" siblings (IMI, ASMI) also see the lateral leads.
        imi_ctx  = self.imi_pool(inf_tokens)
        ilmi_ctx = self.ilmi_pool(inf_tokens)
        asmi_ctx = self.asmi_pool(ant_tokens)
        ami_ctx  = self.ami_pool(ant_tokens)

        # Sibling contrast refinement: emphasize the difference within each region.
        imi_ref  = imi_ctx  + self.inf_contrast(imi_ctx  - ilmi_ctx)
        ilmi_ref = ilmi_ctx + self.inf_contrast(ilmi_ctx - imi_ctx)
        asmi_ref = asmi_ctx + self.ant_contrast(asmi_ctx - ami_ctx)
        ami_ref  = ami_ctx  + self.ant_contrast(ami_ctx  - asmi_ctx)

        imi  = self.imi_head(imi_ref)
        asmi = self.asmi_head(asmi_ref)
        ilmi = self.ilmi_head(ilmi_ref)
        ami  = self.ami_head(ami_ref)
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)

    def _contrast_v2_refs(self, signal: torch.Tensor):
        """Region-context vectors (after gated sibling contrast) for the 4 MI labels.

        Shared by contrast_v2 and dyn_anat_v2 so the joint cross-lead encoders are
        computed identically.
        """
        inferior   = self.inferior_encoder(signal[:, INFERIOR_LEADS, :])
        reciprocal = self.reciprocal_proj(self.reciprocal_encoder(signal[:, RECIPROCAL_LEADS, :]))
        lateral    = self.lateral_encoder(signal[:, LATERAL_LEADS, :])
        anterior   = self.anterior_encoder(signal[:, ANTERIOR_LEADS, :])
        anterior6  = self.anterior6_encoder(signal[:, ANTERIOR6_LEADS, :])
        inferolat  = self.inferolateral_encoder(signal[:, self.INFEROLATERAL_LEADS, :])

        # Region tokens include a dedicated joint token for the extended sibling.
        inf_tokens = torch.stack([inferior, reciprocal, lateral, inferolat], dim=1)
        inf_tokens = self.inferior_region_tf(inf_tokens)
        ant_tokens = torch.stack([anterior, lateral, anterior6], dim=1)
        ant_tokens = self.anterior_region_tf(ant_tokens)

        imi_ctx  = self.imi_pool(inf_tokens)
        ilmi_ctx = self.ilmi_pool(inf_tokens)
        asmi_ctx = self.asmi_pool(ant_tokens)
        ami_ctx  = self.ami_pool(ant_tokens)

        # Gated contrast (identity at init -> ramps up only where helpful).
        imi_ref  = self.inf_contrast(imi_ctx,  ilmi_ctx)
        ilmi_ref = self.inf_contrast(ilmi_ctx, imi_ctx)
        asmi_ref = self.ant_contrast(asmi_ctx, ami_ctx)
        ami_ref  = self.ant_contrast(ami_ctx,  asmi_ctx)
        return imi_ref, asmi_ref, ilmi_ref, ami_ref

    def _forward_contrast_v2(self, signal: torch.Tensor) -> torch.Tensor:
        imi_ref, asmi_ref, ilmi_ref, ami_ref = self._contrast_v2_refs(signal)
        imi  = self.imi_head(imi_ref)
        asmi = self.asmi_head(asmi_ref)
        ilmi = self.ilmi_head(ilmi_ref)
        ami  = self.ami_head(ami_ref)
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)

    def _forward_dyn_anat_v2(self, signal: torch.Tensor) -> torch.Tensor:
        # Joint cross-lead morphology (preserved from contrast_v2)...
        imi_ref, asmi_ref, ilmi_ref, ami_ref = self._contrast_v2_refs(signal)
        # ...plus a gated per-patient per-lead dynamic attention term (no-op at init).
        adds = self.lead_attn(signal)
        imi  = self.imi_head(imi_ref  + adds["imi"])
        asmi = self.asmi_head(asmi_ref + adds["asmi"])
        ilmi = self.ilmi_head(ilmi_ref + adds["ilmi"])
        ami  = self.ami_head(ami_ref  + adds["ami"])
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)


class HardRoutingMIHead(_GraphTransformerHead):
    """E07 hard-routing: raw signal graph, non-overlapping lead pools per label."""

    IMI_LEADS  = INFERIOR_LEADS + RECIPROCAL_LEADS
    ASMI_LEADS = ANTERIOR_LEADS
    ILMI_LEADS = INFERIOR_LEADS + RECIPROCAL_LEADS + LATERAL_LEADS
    AMI_LEADS  = ANTERIOR6_LEADS

    def __init__(self, shared_dim: int, branch_dim: int, dropout: float, use_cross_attention: bool = False) -> None:
        super().__init__(branch_dim, dropout)
        self.imi_node_pool  = TaskTokenPooling(branch_dim, dropout)
        self.asmi_node_pool = TaskTokenPooling(branch_dim, dropout)
        self.ilmi_node_pool = TaskTokenPooling(branch_dim, dropout)
        self.ami_node_pool  = TaskTokenPooling(branch_dim, dropout)
        self.imi_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.asmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ilmi_head = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)
        self.ami_head  = MLPHead(branch_dim, 1, hidden_dim=branch_dim, dropout=dropout)

    def forward(
        self,
        signal: torch.Tensor,
        shared_features: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        graph_nodes = self.encode_graph(signal)
        imi_pooled  = self.imi_node_pool(graph_nodes[:, self.IMI_LEADS, :])
        asmi_pooled = self.asmi_node_pool(graph_nodes[:, self.ASMI_LEADS, :])
        ilmi_pooled = self.ilmi_node_pool(graph_nodes[:, self.ILMI_LEADS, :])
        ami_pooled  = self.ami_node_pool(graph_nodes[:, self.AMI_LEADS, :])
        imi  = self.imi_head(imi_pooled)
        asmi = self.asmi_head(asmi_pooled)
        ilmi = self.ilmi_head(ilmi_pooled)
        ami  = self.ami_head(ami_pooled)
        return torch.cat([imi, asmi, ilmi, ami], dim=-1)


class ConductionHead(nn.Module):
    """Soft-routing conduction head: cond_expert + ConductionLeadGroupEncoder fusion."""

    def __init__(
        self,
        shared_dim: int,
        num_conduction_labels: int,
        hidden_dim: int,
        dropout: float,
        cd_lead_dim: int = 0,
    ) -> None:
        super().__init__()
        self.token_pool = TaskTokenPooling(shared_dim, dropout)
        in_features = shared_dim * 2 + cd_lead_dim
        self.head = MLPHead(
            in_features=in_features,
            out_features=num_conduction_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.cd_lead_dim = cd_lead_dim

    def forward(
        self,
        global_features: torch.Tensor,
        sequence_features: torch.Tensor,
        cd_lead_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.token_pool(sequence_features)
        parts = [global_features, tokens]
        if self.cd_lead_dim > 0 and cd_lead_feats is not None:
            parts.append(cd_lead_feats)
        return self.head(torch.cat(parts, dim=-1))


class HardRoutingConductionHead(_GraphTransformerHead):
    """E07 hard-routing conduction head on raw signal graph."""

    def __init__(
        self,
        shared_dim: int,
        num_conduction_labels: int,
        hidden_dim: int,
        dropout: float,
        cd_lead_dim: int = 0,
    ) -> None:
        super().__init__(hidden_dim, dropout)
        self.node_pool = TaskTokenPooling(hidden_dim, dropout)
        self.head = MLPHead(
            in_features=hidden_dim,
            out_features=num_conduction_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        signal: torch.Tensor,
        global_features: torch.Tensor,
        sequence_features: torch.Tensor,
        cd_lead_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph_nodes = self.encode_graph(signal)
        return self.head(self.node_pool(graph_nodes))


class ECGMultiTaskModel(nn.Module):
    """Unified multitask model with selectable MI/conduction routing mode."""

    def __init__(
        self,
        backbone: nn.Module,
        num_arrhythmia_labels: int,
        num_mi_labels: int,
        num_conduction_labels: int,
        hrv_enabled: bool,
        num_hrv_targets: int,
        head_hidden_dim: int,
        mi_branch_dim: int,
        dropout: float,
        hrv_detach: bool = False,
        mi_gradient_scale: float = 1.0,
        use_cross_attention: bool = False,
        routing_mode: str = "soft",
        mi_head_mode: str = "shared",
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.hrv_enabled = hrv_enabled
        self.hrv_detach = hrv_detach
        self.mi_gradient_scale = mi_gradient_scale
        self.routing_mode = routing_mode
        self._hard_routing = routing_mode == "hard_graph"
        self._decoupled = routing_mode == "decoupled"
        self._is_multi_branch = isinstance(backbone, (MultiBranchTransformerBackbone, DecoupledMultiTaskBackbone))
        self._has_mi = num_mi_labels > 0
        shared_dim = backbone.output_dim

        self.sequence_pool = AttentionPooling(shared_dim)
        self.arrhythmia_pool = TaskTokenPooling(shared_dim, dropout)

        self.arrhythmia_head = MLPHead(
            shared_dim * 2,
            num_arrhythmia_labels,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        cd_lead_dim = 0
        if isinstance(backbone, (MultiBranchTransformerBackbone, DecoupledMultiTaskBackbone)):
            cd_lead_dim = backbone.cd_lead_out_dim

        if routing_mode == "hard_graph":
            self.mi_head = HardRoutingMIHead(
                shared_dim=shared_dim, branch_dim=mi_branch_dim,
                dropout=dropout, use_cross_attention=use_cross_attention,
            )
            self.conduction_head = HardRoutingConductionHead(
                shared_dim=shared_dim,
                num_conduction_labels=num_conduction_labels,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                cd_lead_dim=cd_lead_dim,
            )
        elif routing_mode == "decoupled":
            self.mi_head = DecoupledSubsetMIHead(
                branch_dim=mi_branch_dim, dropout=dropout, head_mode=mi_head_mode,
            )
            self.conduction_head = ConductionHead(
                shared_dim=shared_dim,
                num_conduction_labels=num_conduction_labels,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                cd_lead_dim=cd_lead_dim,
            )
        elif routing_mode == "subset_lead":
            self.mi_head = SubsetLeadMIHead(
                shared_dim=shared_dim, branch_dim=mi_branch_dim, dropout=dropout,
            )
            self.conduction_head = ConductionHead(
                shared_dim=shared_dim,
                num_conduction_labels=num_conduction_labels,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                cd_lead_dim=cd_lead_dim,
            )
        else:
            self.mi_head = MIHead(
                shared_dim=shared_dim, branch_dim=mi_branch_dim,
                dropout=dropout, use_cross_attention=use_cross_attention,
            )
            self.conduction_head = ConductionHead(
                shared_dim=shared_dim,
                num_conduction_labels=num_conduction_labels,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                cd_lead_dim=cd_lead_dim,
            )

        if hrv_enabled:
            self.hrv_head = MLPHead(
                shared_dim * 2,
                num_hrv_targets,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
            )

    @staticmethod
    def _scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
        return tensor * scale + tensor.detach() * (1.0 - scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)

        if self._is_multi_branch:
            arrhy_global = backbone_out["arrhy_global"]
            arrhy_seq    = backbone_out["arrhy_seq"]
            cond_global  = backbone_out["cond_global"]
            cond_seq     = backbone_out["cond_seq"]

            arrhythmia_tokens = self.arrhythmia_pool(arrhy_seq)
            rhythm_features   = torch.cat([arrhy_global, arrhythmia_tokens], dim=-1)

            if not self._decoupled:
                mi_global    = backbone_out["mi_global"]
                mi_seq       = backbone_out["mi_seq"]
                mi_shared    = mi_global + self.sequence_pool(mi_seq)
                mi_seq_feats = mi_seq
        else:
            global_features   = backbone_out["global_features"]
            sequence_features = backbone_out["sequence_features"]
            pooled_sequence   = self.sequence_pool(sequence_features)
            arrhythmia_tokens = self.arrhythmia_pool(sequence_features)
            rhythm_features   = torch.cat([global_features, arrhythmia_tokens], dim=-1)
            shared_features   = global_features + pooled_sequence

            if self.mi_gradient_scale < 1.0:
                mi_shared    = self._scale_gradient(shared_features, self.mi_gradient_scale)
                mi_seq_feats = self._scale_gradient(sequence_features, self.mi_gradient_scale)
            else:
                mi_shared    = shared_features
                mi_seq_feats = sequence_features

            cond_global = shared_features
            cond_seq    = sequence_features

        cd_feats = backbone_out.get("cond_lead_feats") if self._is_multi_branch else None

        if self._decoupled:
            outputs = {
                "arrhythmia": self.arrhythmia_head(rhythm_features),
                "mi": self.mi_head(x),
                "conduction": self.conduction_head(cond_global, cond_seq, cd_feats),
            }
        elif self._hard_routing:
            outputs = {
                "arrhythmia": self.arrhythmia_head(rhythm_features),
                "mi": self.mi_head(x, mi_shared, mi_seq_feats),
                "conduction": self.conduction_head(x, cond_global, cond_seq, cd_feats),
            }
        else:
            outputs = {
                "arrhythmia": self.arrhythmia_head(rhythm_features),
                "mi": self.mi_head(x, mi_shared, mi_seq_feats),
                "conduction": self.conduction_head(cond_global, cond_seq, cd_feats),
            }

        if self.hrv_enabled:
            if self._is_multi_branch:
                hrv_features = torch.cat([backbone_out["arrhy_global"], backbone_out["mi_global"]], dim=-1)
            else:
                hrv_features = torch.cat([global_features, pooled_sequence], dim=-1)
            if self.hrv_detach:
                hrv_features = hrv_features.detach()
            outputs["hrv"] = self.hrv_head(hrv_features)

        return outputs

    def freeze_for_phase2(self) -> None:
        self._phase2_active = True
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.arrhythmia_pool.parameters():
            param.requires_grad = False
        for param in self.arrhythmia_head.parameters():
            param.requires_grad = False
        for param in self.sequence_pool.parameters():
            param.requires_grad = False
        if self.hrv_enabled:
            for param in self.hrv_head.parameters():
                param.requires_grad = False
        self.mi_gradient_scale = 1.0
        self._apply_phase2_eval()

    def _apply_phase2_eval(self) -> None:
        self.backbone.eval()
        self.arrhythmia_pool.eval()
        self.arrhythmia_head.eval()
        self.sequence_pool.eval()
        if self.hrv_enabled:
            self.hrv_head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_phase2_active", False):
            self._apply_phase2_eval()
        return self


class ECGMIModel(nn.Module):
    """E11: MI-only single-task model — decoupled subset-lead head, no shared backbone."""

    def __init__(
        self,
        num_mi_labels: int,
        mi_branch_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if num_mi_labels != 4:
            raise ValueError(f"ECGMIModel expects 4 MI labels, got {num_mi_labels}")
        self.mi_head = DecoupledSubsetMIHead(branch_dim=mi_branch_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"mi": self.mi_head(x)}
