"""
TimesFM 2.5 Submodels
---------------------

---
title: TimesFM 2.5 Submodels
summary: This module contains the submodules used in the TimesFM 2.5 model.
---

# License and Attribution

Apache-2.0 License from https://github.com/google-research/timesfm/blob/master/LICENSE,
accessed on 23 December 2025:

Copyright 2025 Google LLC
SPDX-License-Identifier: Apache-2.0

Ported from https://github.com/google-research/timesfm/commit/6bd8044275f8b76cdc9554f2fecccac5f31a156c
on 23 December 2025.

# Modifications for Darts

Adapted for Darts with custom `PLForecastingModule` and `FoundationModel` integration:
- TODO
"""

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ForecastConfig:
    """Options for forecasting.

    Attributes:
      max_context: The maximum context length. This is used by the complied decode
        function at inference time during batched inference. Any input time series
        with length less than max_context will be padded with zeros, and with
        length greater than max_context will be truncated.
      max_horizon: The maximum horizon length. This is used by the complied decode
        function at inference time during batched inference. The compiled cached
        decoding function will by default forecast till max_horizon.
      normalize_inputs: Whether to normalize the inputs. This is useful when the
        raw inputs are of extremely large or small magnitudes which may result in
        numerical issues.
      window_size: The window size for decomposed forecasting.
        TODO(siriuz42):implement it.
      per_core_batch_size: The batch size per core. Used at inference time during
        batched inference when multiple GPU / TPU devices are used.
      use_continuous_quantile_head: Whether to use a separate continuous quantile
        head to avoid quantile collapsing.
      force_flip_invariance: Whether to force flip invariance. TimesFM guarantees
        that TimesFM(aX + b) = a * TimesFM(x) + b for a >= 0 by default. This flag
        extends it to a < 0 as well.
      infer_is_positive: Whether to guarantee nonnegativity of the output if the
        input is nonnegative.
      fix_quantile_crossing: Whether to fix quantile crossing.
      return_backcast: Whether to return backcast.
    """

    max_context: int = 0
    max_horizon: int = 0
    normalize_inputs: bool = False
    window_size: int = 0
    per_core_batch_size: int = 1
    use_continuous_quantile_head: bool = False
    force_flip_invariance: bool = True
    infer_is_positive: bool = True
    fix_quantile_crossing: bool = False
    return_backcast: bool = False


@dataclass(frozen=True)
class ResidualBlockConfig:
    """Framework-agnostic config for a residual block."""

    input_dims: int
    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: Literal["relu", "swish", "none"]


@dataclass(frozen=True)
class TransformerConfig:
    """Framework-agnostic config for a transformer."""

    model_dims: int
    hidden_dims: int
    num_heads: int
    attention_norm: Literal["rms"]
    feedforward_norm: Literal["rms"]
    qk_norm: Literal["rms", "none"]
    use_bias: bool
    use_rotary_position_embeddings: bool
    ff_activation: Literal["relu", "swish", "none"]
    fuse_qkv: bool


@dataclass(frozen=True)
class StackedTransformersConfig:
    """Framework-agnostic config for a stacked transformers."""

    num_layers: int
    transformer: TransformerConfig


class RMSNorm(nn.Module):
    """RMS normalization."""

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features))
        self.num_features = num_features
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and a linear residual connection."""

    def __init__(self, config: ResidualBlockConfig):
        super().__init__()
        self.config = config
        self.hidden_layer = nn.Linear(
            in_features=config.input_dims,
            out_features=config.hidden_dims,
            bias=config.use_bias,
        )
        self.output_layer = nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
        )
        self.residual_layer = nn.Linear(
            in_features=config.input_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
        )
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()
        elif config.activation == "none":
            self.activation = nn.Identity()
        else:
            raise_log(
                ValueError(f"Activation: {config.activation} not supported."),
                logger,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(
            self.activation(self.hidden_layer(x))
        ) + self.residual_layer(x)


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    kv_length: int = 0,
) -> torch.Tensor:
    """Makes attention mask."""
    if kv_length == 0:
        kv_length = query_length

    q_index = torch.arange(query_length, device=num_all_masked_kv.device)[
        None, None, :, None
    ]
    kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[
        None, None, None, :
    ]
    return torch.logical_and(
        q_index >= kv_index,
        kv_index >= num_all_masked_kv[:, None, None],
    )


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(
        self,
        inputs: torch.Tensor,
        position: torch.Tensor,
    ):
        """Generates a JTensor of sinusoids with different frequencies."""
        half_embedding_dim = self.embedding_dims // 2
        fraction = (
            2
            * torch.arange(0, half_embedding_dim, device=inputs.device)
            / self.embedding_dims
        )
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        ).to(inputs.device)

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        else:
            raise_log(
                ValueError("Inputs must be of rank 4."),
                logger,
            )

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


def _torch_dot_product_attention(query, key, value, mask=None):
    """
    Performs the exact same (unscaled) attention as the above function,
    but using the fast and fused F.scaled_dot_product_attention kernel.
    """

    # 1. Permute inputs from (B, L, H, D) to the expected (B, H, L, D)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # 2. Call the fused attention kernel
    #    - Pass the mask to `attn_mask`.
    #    - Set `scale=1.0` to disable the default 1/sqrt(d_k) scaling.
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=1.0
    )

    # 3. Permute the output back to the original (B, L, H, D) layout
    output = output.permute(0, 2, 1, 3)

    return output


class PerDimScale(nn.Module):
    """Per-dimension scaling."""

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = (
            1.442695041 / math.sqrt(self.num_dims) * F.softplus(self.per_dim_scale)
        )
        return x * scale_factor


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        use_bias: bool = False,
        attention_fn: Callable[..., torch.Tensor] = _torch_dot_product_attention,
        qk_norm: str = "rms",
        fuse_qkv: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.qk_norm = qk_norm
        self.fuse_qkv = fuse_qkv

        if self.in_features % self.num_heads != 0:
            raise_log(
                ValueError(
                    f"Memory dimension ({self.in_features}) must be divisible by "
                    f"'num_heads' heads ({self.num_heads})."
                ),
                logger,
            )

        if self.fuse_qkv:
            self.qkv_proj = nn.Linear(
                self.in_features, 3 * self.in_features, bias=use_bias
            )
        else:
            self.query = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.key = nn.Linear(self.in_features, self.in_features, bias=use_bias)
            self.value = nn.Linear(self.in_features, self.in_features, bias=use_bias)
        self.out = nn.Linear(self.in_features, self.in_features, bias=use_bias)

        if self.qk_norm == "rms":
            self.query_ln = RMSNorm(self.head_dim)
            self.key_ln = RMSNorm(self.head_dim)
        else:
            self.query_ln = nn.Identity()
            self.key_ln = nn.Identity()

        self.use_rotary_position_embeddings = use_rotary_position_embeddings
        if self.use_rotary_position_embeddings:
            self.rotary_position_embedding = RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
            )

        self.use_per_dim_scale = use_per_dim_scale
        if use_per_dim_scale:
            self.per_dim_scale = PerDimScale(num_dims=self.head_dim)

    def forward(
        self,
        inputs_q: torch.Tensor,
        *,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n_patches, _ = inputs_q.shape

        if self.fuse_qkv:
            qkv = self.qkv_proj(inputs_q)
            query, key, value = torch.chunk(qkv, 3, dim=-1)
            query = query.view(b, n_patches, self.num_heads, self.head_dim)
            key = key.view(b, n_patches, self.num_heads, self.head_dim)
            value = value.view(b, n_patches, self.num_heads, self.head_dim)
        else:
            query = self.query(inputs_q).view(
                b, n_patches, self.num_heads, self.head_dim
            )
            key = self.key(inputs_q).view(b, n_patches, self.num_heads, self.head_dim)
            value = self.value(inputs_q).view(
                b, n_patches, self.num_heads, self.head_dim
            )

        num_masked = torch.sum(patch_mask.to(torch.int32), dim=-1, keepdim=True)

        if self.use_rotary_position_embeddings:
            position = (
                torch.arange(n_patches, device=inputs_q.device)[None, :] - num_masked
            )
            query = self.rotary_position_embedding(query, position)
            key = self.rotary_position_embedding(key, position)

        query = self.query_ln(query)
        key = self.key_ln(key)

        if self.use_per_dim_scale:
            query = self.per_dim_scale(query)

        attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)

        x = self.attention_fn(
            query,
            key,
            value,
            mask=attn_mask,
        )

        x = x.reshape(b, n_patches, self.in_features)
        out = self.out(x)
        return out


class Transformer(nn.Module):
    """Classic Transformer used in TimesFM."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        if config.attention_norm == "rms":
            self.pre_attn_ln = RMSNorm(num_features=config.model_dims)
            self.post_attn_ln = RMSNorm(num_features=config.model_dims)
        else:
            raise_log(
                ValueError(f"Layer norm: {config.attention_norm} not supported."),
                logger,
            )

        self.attn = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.model_dims,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=config.use_rotary_position_embeddings,
            qk_norm=config.qk_norm,
            fuse_qkv=config.fuse_qkv,
        )

        if config.feedforward_norm == "rms":
            self.pre_ff_ln = RMSNorm(num_features=config.model_dims)
            self.post_ff_ln = RMSNorm(num_features=config.model_dims)
        else:
            raise_log(
                ValueError(f"Layer norm: {config.feedforward_norm} not supported."),
                logger,
            )

        self.ff0 = nn.Linear(
            in_features=config.model_dims,
            out_features=config.hidden_dims,
            bias=config.use_bias,
        )
        self.ff1 = nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.model_dims,
            bias=config.use_bias,
        )
        if config.ff_activation == "relu":
            self.activation = nn.ReLU()
        elif config.ff_activation == "swish":
            self.activation = nn.SiLU()
        elif config.ff_activation == "none":
            self.activation = nn.Identity()
        else:
            raise_log(
                ValueError(f"Activation: {config.ff_activation} not supported."),
                logger,
            )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.attn(
            inputs_q=self.pre_attn_ln(input_embeddings),
            patch_mask=patch_mask,
        )
        attn_output = self.post_attn_ln(attn_output) + input_embeddings
        output_embeddings = (
            self.post_ff_ln(
                self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output))))
            )
            + attn_output
        )
        return output_embeddings


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Updates the running stats."""
    is_legit = torch.logical_not(mask).to(x.dtype)
    inc_n = torch.sum(is_legit, dim=-1)

    inc_mu_numerator = torch.nansum(x, dim=-1)
    inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
    inc_mu = inc_mu_numerator / inc_n_safe
    inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

    inc_var_numerator = torch.nansum((x - inc_mu.unsqueeze(-1)) ** 2, dim=-1)
    inc_var = inc_var_numerator / inc_n_safe
    inc_var = torch.where(inc_n == 0, 0.0, inc_var)

    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, 1.0, new_n)

    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, 0.0, new_mu)

    term1 = n * sigma.pow(2)
    term2 = inc_n * inc_var
    term3 = n * (mu - new_mu).pow(2)
    term4 = inc_n * (inc_mu - new_mu).pow(2)

    new_var = (term1 + term2 + term3 + term4) / new_n_safe
    new_var = torch.where(new_n == 0, 0.0, new_var)
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

    return (w := (new_n, new_mu, new_sigma), w)


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
    """Reversible instance normalization."""
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]

    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)
