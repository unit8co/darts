# Copyright contributors to the TSFM project (IBM)
# Adapted for Darts from https://github.com/ibm-granite/granite-tsfm
# Licensed under the Apache License, Version 2.0
#
# Reference:
#   Yunshi Wen, Wesley M. Gifford, Chandra Reddy, Lam M. Nguyen, Jayant Kalagnanam,
#   and Anak Agung Julius, "Revisiting the Generic Transformer: Deconstructing a Strong
#   Baseline for Time Series Foundation Models," arXiv:2602.06909, 2026.
"""PatchTST-FM submodel components.

Faithful port of the submodules from ``ibm-granite/granite-tsfm`` (branch
``patchtst-fm``), keeping parameter names and constructor signatures so that
pre-trained weights load directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import raise_log


class _RevIN(nn.Module):
    """Reversible Instance Normalization with optional asinh transform."""

    def __init__(
        self,
        dim: int = -1,
        std_min: float = 1e-5,
        max_val: float = 100,
        use_sinh: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.max_val = max_val
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

    def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std

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

    def __init__(self, d_model: int, max_len: int = 5000, kind: str = "add"):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        if kind not in ["add", "mul"]:
            raise_log(
                ValueError(f"Invalid `kind`: {kind}"),
            )
        self.kind = kind

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            offset, offset + x.size(-2), device=x.device
        ).unsqueeze(0)
        pe = self.embedding(positions)
        if x.ndim == 4:
            pe = pe.unsqueeze(1)
        if self.kind == "add":
            return x + pe
        else:  # "mul"
            return x * pe


class _MLP(nn.Module):
    """Feed-forward network used in transformer blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        norm: bool = False,
        activation: nn.Module | None = None,
        output_activation: nn.Module | None = None,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        if output_activation is None:
            output_activation = nn.Identity()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(norm_layer(hidden_dim) if norm else nn.Identity())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(nn.Dropout(dropout))
        layers.append(norm_layer(hidden_dim) if norm else nn.Identity())
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(output_activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward network (alternative to MLP in transformer blocks)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 384,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = round(hidden_dim * 2 / 3)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(in_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * self.activation(self.fc2(x))
        return self.dropout(self.fc3(x))


class _Attention(nn.Module):
    """Multi-head self-attention with scaled dot-product attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
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
    """Standard Transformer block with self-attention and MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        norm_first: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_type: str = "mlp",
    ):
        super().__init__()
        self.norm_first = norm_first
        self.norm1 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        self.attn = _Attention(
            d_model, num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout
        )
        self.norm2 = norm_layer(d_model, elementwise_affine=True, eps=1e-6)
        if mlp_type == "swiglu":
            self.mlp = _SwiGLU(
                d_model,
                d_model,
                hidden_dim=int(mlp_ratio * d_model),
                dropout=dropout,
            )
        elif mlp_type == "mlp":
            self.mlp = _MLP(
                in_dim=d_model,
                out_dim=d_model,
                hidden_dim=int(mlp_ratio * d_model),
                dropout=dropout,
            )
        else:
            raise_log(
                ValueError(f"Unsupported `mlp_type`: {mlp_type}"),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn(self.norm1(x), attn_mask)
            x = x + self.dropout(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(x + self.attn(x, attn_mask))
            x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


def _make_attn_mask(query_pad: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """Build an additive attention mask of shape (B, Q, K) from query/key padding masks.

    Parameters
    ----------
    query_pad
        (B, Q) bool or 0/1 tensor. 1/True = padded query position.
    key_pad
        (B, K) bool or 0/1 tensor. 1/True = padded key position.

    Returns
    -------
    attn_mask
        (B, Q, K) float tensor, where masked positions are -inf and valid positions are 0.0
        (for use with SDPA).
    """
    q_pad = query_pad.bool()
    k_pad = key_pad.bool()
    pad = q_pad.unsqueeze(-1) | k_pad.unsqueeze(-2)
    attn_mask = torch.zeros_like(pad, dtype=torch.float32)
    attn_mask.masked_fill_(pad, float("-inf"))
    return attn_mask
