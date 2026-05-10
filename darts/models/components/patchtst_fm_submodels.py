# Copyright contributors to the TSFM project (IBM)
# Adapted for Darts from https://github.com/ibm-granite/granite-tsfm
# Licensed under the Apache License, Version 2.0
#
# Reference:
#   Yunshi Wen, Wesley M. Gifford, Chandra Reddy, Lam M. Nguyen, Jayant Kalagnanam,
#   and Anak Agung Julius, "Revisiting the Generic Transformer: Deconstructing a Strong
#   Baseline for Time Series Foundation Models," arXiv:2602.06909, 2026.
"""PatchTST-FM submodel components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RevIN(nn.Module):
    """Reversible Instance Normalization with optional asinh transform."""

    def __init__(self, dim: int = -1, std_min: float = 1e-5, use_sinh: bool = True):
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.use_sinh = use_sinh
        self.mean: torch.Tensor
        self.std: torch.Tensor

    def fit_transform(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            self._get_statistics(x, mask)
            return self.transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            x = (x - self.mean) / self.std
            if self.use_sinh:
                x = torch.asinh(x)
            return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if self.use_sinh:
                x = torch.sinh(x)
            if x.ndim != self.mean.ndim:
                x = x * self.std.unsqueeze(1) + self.mean.unsqueeze(1)
            else:
                x = x * self.std + self.mean
            return x

    def _get_statistics(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            self.mean = x.mean(dim=self.dim, keepdim=True)
            std = x.std(dim=self.dim, keepdim=True)
            self.std = torch.where(std > self.std_min, std, torch.ones_like(std))
        else:
            mask = mask.bool()
            unmask = (~mask).float()
            count = unmask.sum(dim=self.dim, keepdim=True).clamp(min=1)
            x_mean = (x * unmask).sum(dim=self.dim, keepdim=True) / count
            x_std = (((x - x_mean) * unmask) ** 2).sum(
                dim=self.dim, keepdim=True
            ) / count
            x_std = x_std.sqrt()
            x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
            self.mean = x_mean
            self.std = x_std


class _ResidualBlock(nn.Module):
    """MLP with skip connection for input/output projections."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_out)
        self.residual = nn.Linear(d_in, d_out)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x))) + self.residual(x)


class _LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding added to patch embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(-2), device=x.device).unsqueeze(0)
        pe = self.embedding(positions)
        if x.ndim == 4:
            pe = pe.unsqueeze(1)
        return x + pe


class _MLP(nn.Module):
    """Feed-forward network used in transformer blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Identity())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Identity())
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _Attention(nn.Module):
    """Multi-head self-attention with scaled dot-product attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with self-attention and MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.attn = _Attention(
            d_model, num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.mlp = _MLP(
            in_dim=d_model,
            out_dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


def _make_attn_mask(query_pad: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """Build additive attention mask from padding masks.

    Masked positions get -inf, valid positions get 0.0.
    """
    q_pad = query_pad.bool()
    k_pad = key_pad.bool()
    pad = q_pad.unsqueeze(-1) | k_pad.unsqueeze(-2)
    attn_mask = torch.zeros_like(pad, dtype=torch.float32)
    attn_mask.masked_fill_(pad, float("-inf"))
    return attn_mask
