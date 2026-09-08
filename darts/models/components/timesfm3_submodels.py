"""
TimesFM 3.0 Submodels
---------------------

---
title: TimesFM 3.0 Submodels
summary: This module contains the submodules used in the TimesFM 3.0 model.
---

# License and Attribution

Apache-2.0 License from https://github.com/google-research/timesfm/blob/master/LICENSE,
accessed on 8 September 2026:

Copyright 2026 Google LLC
SPDX-License-Identifier: Apache-2.0

Ported from https://github.com/google-research/timesfm/commit/9de33f6f487baf8adc26eb757f29b6a7a557b823
on 8 September 2026.

# Modifications for Darts

Adapted for Darts with custom `PLForecastingModule` and `FoundationModel` integration:
- Remove the `PyTorchModelHubMixin` integration: model config and weights are loaded from
  HuggingFace Hub using `HuggingFaceConnector` instead. Attribute names are kept identical
  to the original implementation, so the HuggingFace checkpoint weights can be loaded with
  `load_state_dict()` without key remapping.
- Remove `DecodeCache` and cache-related logic used for auto-regressive decoding:
  auto-regressive forecasting is handled by `TorchForecastingModel` by calling the model
  repeatedly.
- Remove `segment_ids` / `segment_pos` / `initial_stats` support for multi-segment inputs,
  as well as the manual (non-SDPA) attention path: only the `scaled_dot_product_attention`
  path (used by the released checkpoint) is kept.
- Replace `torch.nn.RMSNorm` (requires `torch>=2.4`) with a custom implementation compatible
  with `torch>=2.0`, numerically matching the original on `torch>=2.4` via
  `torch.nn.functional.rms_norm`.
- Remove the linear interpolation of missing values applied by the upstream
  `TimesFM3Forecaster`: missing values are handled through the masking logic of `decode()`.
- Only the `identity` input transformation is supported (the one used by the released
  checkpoint).
- Configurations can be built directly from the dictionaries stored in the HuggingFace
  `config.json` using `build_residual_block_config()` and `build_stacked_transformers_config()`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import raise_log

_TOLERANCE = 1e-6
_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041

# `torch.nn.functional.rms_norm` is only available from torch>=2.4
_HAS_F_RMS_NORM = hasattr(F, "rms_norm")


@dataclass(frozen=True)
class _ResidualBlockConfig:
    """Framework-agnostic config for a residual block."""

    hidden_dims: int
    output_dims: int
    use_bias: bool
    activation: Literal["relu", "swish", "none"]
    dropout: float = 0.0
    identity_skip: bool = False
    prenorm: Literal["rms", "none"] = "none"


@dataclass(frozen=True)
class _TransformerConfig:
    """Framework-agnostic config for a transformer."""

    model_dims: int
    hidden_dims: int
    num_heads: int
    attention_norm: Literal["rms"]
    feedforward_norm: Literal["rms"]
    qk_norm: Literal["rms", "none"]
    use_bias: bool
    use_rope_seq: bool
    use_rope_var: bool
    ff_activation: Literal["relu", "swish", "none"]
    deterministic: bool
    v_norm: Literal["rms", "none"] = "none"
    causal_attention: bool = True
    debug_no_masking: bool = False
    training: bool = True
    use_memory_efficient_attention: bool = True
    paired_token_skip_second: bool = False
    max_variates: int = 32
    # PyTorch-only: when True uses F.scaled_dot_product_attention.
    use_sdpa: bool = True


@dataclass(frozen=True)
class _StackedTransformersConfig:
    """Framework-agnostic config for a stacked transformers."""

    num_layers: int
    transformer: _TransformerConfig
    use_remat: bool = True


def build_residual_block_config(
    config: _ResidualBlockConfig | dict[str, Any],
) -> _ResidualBlockConfig:
    """Builds a `_ResidualBlockConfig` from a dataclass or a raw dictionary
    (e.g. the `residual_block_config` entry of the HuggingFace `config.json`)."""
    if isinstance(config, _ResidualBlockConfig):
        return config
    return _ResidualBlockConfig(**config)


def build_stacked_transformers_config(
    config: _StackedTransformersConfig | dict[str, Any],
) -> _StackedTransformersConfig:
    """Builds a `_StackedTransformersConfig` from a dataclass or a raw dictionary
    (e.g. the `transformer_config` entry of the HuggingFace `config.json`)."""
    if isinstance(config, _StackedTransformersConfig):
        return config
    transformer = config.get("transformer", {})
    if not isinstance(transformer, _TransformerConfig):
        transformer = _TransformerConfig(**transformer)
    config = dict(config)
    config["transformer"] = transformer
    return _StackedTransformersConfig(**config)


class _RMSNorm(nn.Module):
    """RMS normalization matching ``torch.nn.RMSNorm`` (available from torch>=2.4).

    When ``elementwise_affine=True``, the learned ``weight`` is initialized to ones
    (unlike the Flax-style scale parameter used by the TimesFM 2.5 port).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float | None = None,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if _HAS_F_RMS_NORM:
            return F.rms_norm(inputs, self.normalized_shape, self.weight, self.eps)
        eps = self.eps if self.eps is not None else torch.finfo(inputs.dtype).eps
        variance = inputs.pow(2).mean(-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(variance + eps)
        if self.weight is not None:
            normed_inputs = normed_inputs * self.weight
        return normed_inputs


class _PerDimScale(nn.Module):
    """Per-dimension scaling (Pax-style).

    Replaces the standard 1/sqrt(d) query scaling with a learnable:
        x * RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims) * softplus(per_dim_scale)

    The per_dim_scale parameter is initialized to zeros, so at init time
    softplus(0) ≈ 0.693..., and the net scale is close to 1/sqrt(d).
    """

    def __init__(self, num_dims: int):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies per-dim scaling to the last dimension of x."""
        return (
            x
            * _RECIPROCAL_OF_SOFTPLUS_0
            / math.sqrt(self.num_dims)
            * F.softplus(self.per_dim_scale)
        )


def _make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    kv_length: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """Makes attention mask. True = attend, False = mask.

    Parameters
    ----------
    query_length
        Number of query positions.
    num_all_masked_kv
        Shape ``(batch,)``. Number of leading masked KV positions.
    kv_length
        Length of KV sequence. Defaults to ``query_length``.
    causal
        Whether to apply causal masking.

    Returns
    -------
        Boolean mask of shape ``(batch, 1, query_length, kv_length)``. True = attend.
    """
    if kv_length == 0:
        kv_length = query_length

    device = num_all_masked_kv.device
    q_index = torch.arange(query_length, device=device).view(1, 1, -1, 1)
    kv_index = torch.arange(kv_length, device=device).view(1, 1, 1, -1)
    mask = kv_index >= num_all_masked_kv.view(-1, 1, 1, 1)
    if causal:
        return (q_index >= kv_index) & mask
    return mask


class _RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Stateless module — no learnable parameters.
    Supports 3D ``(batch, n, d)`` and 4D ``(batch, n, h, hd)`` inputs.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

        half_dim = embedding_dims // 2
        fraction = 2.0 * torch.arange(half_dim, dtype=torch.float32) / embedding_dims
        timescale = min_timescale * (max_timescale / min_timescale) ** fraction
        self.register_buffer("timescale", timescale, persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies rotary positional embeddings.

        Parameters
        ----------
        inputs
            Shape ``(batch, n, d)`` or ``(batch, n, h, hd)``.
        position
            Shape ``(batch, n)``. If ``None``, uses ``arange(n)``.

        Returns
        -------
            Tensor with same shape as inputs, with RoPE applied.
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise_log(
                ValueError(
                    "The embedding dims of the rotary position embedding "
                    "must match the hidden dimension of the inputs."
                ),
            )
        timescale = self.timescale.to(inputs.device)

        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(
                seq_length, device=inputs.device, dtype=torch.float32
            ).unsqueeze(0)

        if inputs.dim() == 4:
            # (b, n) -> (b, n, 1, 1) for broadcasting with (b, n, h, hd)
            pos = position.unsqueeze(-1).unsqueeze(-1)
            ts = timescale.view(1, 1, 1, -1)
        elif inputs.dim() == 3:
            pos = position.unsqueeze(-1)
            ts = timescale.view(1, 1, -1)
        else:
            raise_log(ValueError("Inputs must be of rank 3 or 4."))

        sinusoid_inp = pos.float() / ts
        sin_val = torch.sin(sinusoid_inp)
        cos_val = torch.cos(sinusoid_inp)
        first_half, second_half = inputs.chunk(2, dim=-1)
        first_part = first_half * cos_val - second_half * sin_val
        second_part = second_half * cos_val + first_half * sin_val
        return torch.cat([first_part, second_part], dim=-1)


class _MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE, QK-norm, and PerDimScale.

    This matches the original ``MultiHeadAttention`` exactly, including the
    pre-multiplication of query by sqrt(head_dim) which cancels with the
    standard 1/sqrt(d) scaling in dot-product attention.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        causal_attention: bool = True,
        use_bias: bool = False,
        qk_norm: str = "rms",
        v_norm: str = "none",
        rescale_logits: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.causal_attention = causal_attention
        self.head_dim = in_features // num_heads
        # rescale_logits=False → MEA=True behaviour: Q is pre-multiplied by √d,
        #   no internal division. Matches Flax
        #   memory_efficient_attention(rescale_logits=False).
        # rescale_logits=True  → MEA=False behaviour: Q*√d is passed but divided
        #   by √d internally, so they cancel (net scale = 1.0). Matches Flax
        #   nn.dot_product_attention.
        self.rescale_logits = rescale_logits

        # Q, K, V projections: Linear(in, heads*hd)
        # We'll reshape the output to (b, n, heads, hd)
        self.query_proj = nn.Linear(in_features, in_features, bias=use_bias)
        self.key_proj = nn.Linear(in_features, in_features, bias=use_bias)
        self.value_proj = nn.Linear(in_features, in_features, bias=use_bias)

        # Output projection
        self.out_proj = nn.Linear(in_features, in_features, bias=use_bias)

        # QK normalization
        if qk_norm == "rms":
            self.query_ln = _RMSNorm(self.head_dim)
            self.key_ln = _RMSNorm(self.head_dim)
        else:
            self.query_ln = None
            self.key_ln = None

        # V normalization
        if v_norm == "rms":
            self.value_ln = _RMSNorm(self.head_dim, elementwise_affine=False)
        else:
            self.value_ln = None

        # RoPE
        if use_rotary_position_embeddings:
            self.rotary_position_embedding = _RotaryPositionalEmbedding(
                embedding_dims=self.head_dim
            )
        else:
            self.rotary_position_embedding = None

        # PerDimScale
        if use_per_dim_scale:
            self.per_dim_scale = _PerDimScale(num_dims=self.head_dim)
        else:
            self.per_dim_scale = None

    def forward(
        self,
        inputs_q: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies multi-head attention.

        Parameters
        ----------
        inputs_q
            Shape ``(batch, n_patches, d)``.
        patch_mask
            Shape ``(batch, n_patches)``. True = masked patch.

        Returns
        -------
            Output of shape ``(batch, n_patches, d)``.
        """
        batch_size, n_patches, _ = inputs_q.shape
        device = inputs_q.device

        if patch_mask is None:
            patch_mask = torch.zeros(
                batch_size, n_patches, dtype=torch.bool, device=device
            )

        # Project Q, K, V and reshape to (b, n, h, hd)
        query = self.query_proj(inputs_q).view(
            batch_size, n_patches, self.num_heads, self.head_dim
        )
        key = self.key_proj(inputs_q).view(
            batch_size, n_patches, self.num_heads, self.head_dim
        )
        value = self.value_proj(inputs_q).view(
            batch_size, n_patches, self.num_heads, self.head_dim
        )

        # Apply RoPE
        if self.rotary_position_embedding is not None:
            query = self.rotary_position_embedding(query)
            key = self.rotary_position_embedding(key)

        # QK normalization
        if self.query_ln is not None:
            query = self.query_ln(query)
        if self.key_ln is not None:
            key = self.key_ln(key)

        # PerDimScale
        if self.per_dim_scale is not None:
            query = self.per_dim_scale(query)

        # V normalization
        if self.value_ln is not None:
            value = self.value_ln(value)

        # Full-sequence mode: no positions are front-masked; masked patches are
        # removed from the K/V positions
        attn_mask = _make_attn_mask(
            query_length=n_patches,
            num_all_masked_kv=torch.zeros(batch_size, dtype=torch.int32, device=device),
            causal=self.causal_attention,
        )
        attn_mask = attn_mask & (~patch_mask[:, None, None, :])

        # Transpose for attention: (b, n, h, d) -> (b, h, n, d)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # SDPA computes: softmax(Q @ K^T * scale) @ V.
        if self.rescale_logits:
            # MEA=False equivalent: Flax passes Q*√d to nn.dot_product_attention
            # which divides by √d internally → net scale = 1.0.
            attn_scale = 1.0
        else:
            # MEA=True equivalent: Flax passes Q*√d with rescale_logits=False
            # → no internal division → net scale = √d.
            attn_scale = math.sqrt(self.head_dim)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask.expand(-1, self.num_heads, -1, -1),
            scale=attn_scale,
        )

        # Transpose back: (b, h, n, d) -> (b, n, h, d)
        x = x.transpose(1, 2).contiguous()

        # Reshape and project: (b, n, h, d) -> (b, n, h*d)
        x = x.view(batch_size, n_patches, self.in_features)

        return self.out_proj(x)


class _MixingTransformer(nn.Module):
    """Transformer with sequential sequence and variate attention.

    Attention is applied first across the sequence dimension 'n', then
    across the variate dimension 'v' for inputs of shape 'b v n d'.

    Architecture per layer:
        1. Sequence attention: pre_ln -> reshape(bv,n,d) -> MHA -> post_ln +
        residual
        2. Variate attention: pre_ln -> reshape(bn,v,d) -> MHA -> post_ln + residual
        3. FFN: pre_ln -> ff0 -> activation -> ff1 -> post_ln + residual
    """

    def __init__(
        self,
        config: _TransformerConfig,
        use_variate_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.use_variate_attention = use_variate_attention

        # Sequence attention norms + module
        self.pre_seq_attn_ln = _RMSNorm(config.model_dims)
        self.post_seq_attn_ln = _RMSNorm(config.model_dims)
        # rescale_logits mirrors Flax: use_memory_efficient_attention=True →
        #   MEA=True (rescale_logits=False, scale=√d);
        #   use_memory_efficient_attention=False → MEA=False (rescale_logits=True,
        #   net scale=1.0).
        rescale_logits = not config.use_memory_efficient_attention
        self.seq_attn = _MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.model_dims,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=config.use_rope_seq,
            qk_norm=config.qk_norm,
            v_norm=config.v_norm,
            causal_attention=config.causal_attention,
            use_bias=config.use_bias,
            rescale_logits=rescale_logits,
        )

        # Variate attention norms + module
        if use_variate_attention:
            self.pre_var_attn_ln = _RMSNorm(config.model_dims)
            self.post_var_attn_ln = _RMSNorm(config.model_dims)
            self.var_attn = _MultiHeadAttention(
                num_heads=config.num_heads,
                in_features=config.model_dims,
                use_per_dim_scale=True,
                use_rotary_position_embeddings=config.use_rope_var,
                qk_norm=config.qk_norm,
                v_norm=config.v_norm,
                causal_attention=False,
                use_bias=config.use_bias,
                rescale_logits=rescale_logits,
            )

        # FFN norms + layers
        self.pre_ff_ln = _RMSNorm(config.model_dims)
        self.post_ff_ln = _RMSNorm(config.model_dims)
        self.ff0 = nn.Linear(
            config.model_dims, config.hidden_dims, bias=config.use_bias
        )
        self.ff1 = nn.Linear(
            config.hidden_dims, config.model_dims, bias=config.use_bias
        )
        self.activation = _get_activation_fn(config.ff_activation)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_embeddings
            Shape ``(batch, v, n, d)``.
        patch_mask
            Shape ``(batch, v, n)``. True = masked.

        Returns
        -------
            Output embeddings of shape ``(batch, v, n, d)``.
        """
        b, v, n, d = input_embeddings.shape

        # --- Sequence Attention ---
        seq_attn_in = self.pre_seq_attn_ln(input_embeddings)
        # (b, v, n, d) -> (b*v, n, d)
        seq_attn_in_flat = seq_attn_in.reshape(b * v, n, d)
        patch_mask_flat = patch_mask.reshape(b * v, n)

        seq_attn_out_flat = self.seq_attn(
            seq_attn_in_flat,
            patch_mask=patch_mask_flat,
        )
        seq_attn_out = seq_attn_out_flat.view(b, v, n, d)
        h1 = self.post_seq_attn_ln(seq_attn_out) + input_embeddings

        # --- Variate Attention ---
        if self.use_variate_attention:
            var_attn_in = self.pre_var_attn_ln(h1)
            # (b, v, n, d) -> (b*n, v, d)
            var_attn_in_flat = var_attn_in.permute(0, 2, 1, 3).reshape(b * n, v, d)
            # Mask: (b, v, n) -> (b, n, v) -> (b*n, v)
            var_patch_mask = patch_mask.permute(0, 2, 1).reshape(b * n, v)

            var_attn_out_flat = self.var_attn(
                var_attn_in_flat,
                patch_mask=var_patch_mask,
            )
            # (b*n, v, d) -> (b, n, v, d) -> (b, v, n, d)
            var_attn_out = var_attn_out_flat.view(b, n, v, d).permute(0, 2, 1, 3)
            h2 = self.post_var_attn_ln(var_attn_out) + h1
        else:
            h2 = h1

        # --- FeedForward ---
        ff_out = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
        return self.post_ff_ln(ff_out) + h2


class _StackedMixingTransformer(nn.Module):
    """Stacked MixingTransformer layers."""

    def __init__(
        self,
        config: _StackedTransformersConfig,
        use_variate_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            _MixingTransformer(
                config=config.transformer,
                use_variate_attention=use_variate_attention,
            )
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all layers.

        Parameters
        ----------
        input_embeddings
            Shape ``(batch, v, n, d)``.
        patch_mask
            Shape ``(batch, v, n)``.

        Returns
        -------
            Output embeddings of shape ``(batch, v, n, d)``.
        """
        output = input_embeddings
        for layer in self.layers:
            output = layer(output, patch_mask)
        return output


class _ResidualBlock(nn.Module):
    """Residual block with two linear layers and a linear residual connection.

    Architecture:
        if prenorm == "rms": x_norm = RMSNorm(x)  else: x_norm = x
        hidden = activation(hidden_layer(x_norm))
        output = output_layer(hidden) + residual_layer(x)  [or + x if identity_skip]
    """

    def __init__(self, config: _ResidualBlockConfig):
        super().__init__()
        self.config = config

        # Defining placeholder layers in __init__ ensures PyTorch registers them as
        # submodules. This is required for standard parameter tracking, printing/
        # debugging, and correctly propagating device/dtype moves applied to the
        # parent module (e.g. `model.to(device)`) before any forward handles them.
        # They are safely re-initialized/overwritten with correct dimensions during
        # the first forward pass (using `set_input_dims()`) before the matrix
        # multiplication is evaluated, avoiding shape mismatch errors.
        self.hidden_layer = nn.Linear(
            in_features=config.hidden_dims,  # placeholder, set in first forward
            out_features=config.hidden_dims,
            bias=config.use_bias,
        )
        self.output_layer = nn.Linear(
            in_features=config.hidden_dims,
            out_features=config.output_dims,
            bias=config.use_bias,
        )

        if config.identity_skip:
            self.residual_layer = None
        else:
            self.residual_layer = nn.Linear(
                in_features=config.hidden_dims,  # placeholder
                out_features=config.output_dims,
                bias=config.use_bias,
            )

        self.activation = _get_activation_fn(config.activation)

        if config.prenorm == "rms":
            self.pre_norm = _RMSNorm(config.hidden_dims)
        else:
            self.pre_norm = None

        # Mark layers as lazy so input dim gets set on first use
        self._input_dim_set = False

    def set_input_dims(self, input_dim: int) -> None:
        """Reinitialize linear layers with the correct input dimension."""
        if self._input_dim_set:
            return
        device = self.hidden_layer.weight.device
        dtype = self.hidden_layer.weight.dtype

        self.hidden_layer = nn.Linear(
            input_dim, self.config.hidden_dims, bias=self.config.use_bias
        ).to(device=device, dtype=dtype)
        self.output_layer = nn.Linear(
            self.config.hidden_dims,
            self.config.output_dims,
            bias=self.config.use_bias,
        ).to(device=device, dtype=dtype)
        if self.residual_layer is not None:
            self.residual_layer = nn.Linear(
                input_dim, self.config.output_dims, bias=self.config.use_bias
            ).to(device=device, dtype=dtype)
        if self.pre_norm is not None:
            self.pre_norm = _RMSNorm(input_dim).to(device=device, dtype=dtype)
        self._input_dim_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (b, ..., input_dim)."""
        if not self._input_dim_set:
            self.set_input_dims(x.shape[-1])

        if self.pre_norm is not None:
            hidden_input = self.pre_norm(x)
        else:
            hidden_input = x

        hidden_output = self.activation(self.hidden_layer(hidden_input))

        if self.residual_layer is not None:
            return self.output_layer(hidden_output) + self.residual_layer(x)
        else:
            return self.output_layer(hidden_output) + x


def _make_safe_for_division(values: torch.Tensor) -> torch.Tensor:
    """Handles near zero values."""
    return torch.where(values < _TOLERANCE, 1.0, values)


def _update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Updates running stats with a new patch of data.

    Parameters
    ----------
    n
        Count of seen non-masked elements. Shape: ``(batch, v)``.
    mu
        Running mean. Shape: ``(batch, v)``.
    sigma
        Running std. Shape: ``(batch, v)``.
    x
        New data patch. Shape: ``(batch, v, p)``.
    mask
        Boolean mask where True = masked/invalid. Shape: ``(batch, v, p)``.

    Returns
    -------
        Tuple of ``(new_n, new_mu, new_sigma)``, each of shape ``(batch, v)``.
    """
    is_legit = ~mask
    is_legit_f = is_legit.float()
    inc_n = is_legit_f.sum(dim=-1)

    # mean of valid elements in patch
    x_masked = torch.where(is_legit, x, torch.zeros_like(x))
    inc_sum = x_masked.sum(dim=-1)
    inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_sum), inc_sum / inc_n)

    # std of valid elements in patch
    x_diff_sq = torch.where(
        is_legit, (x - inc_mu.unsqueeze(-1)) ** 2, torch.zeros_like(x)
    )
    inc_var = torch.where(
        inc_n == 0,
        torch.zeros_like(inc_sum),
        x_diff_sq.sum(dim=-1) / inc_n,
    )
    inc_sigma = torch.sqrt(inc_var)

    new_n = n + inc_n
    new_mu = torch.where(
        new_n == 0,
        torch.zeros_like(mu),
        (n * mu + inc_mu * inc_n) / new_n,
    )
    new_sigma = torch.sqrt(
        torch.where(
            new_n == 0,
            torch.zeros_like(sigma),
            (
                n * sigma * sigma
                + inc_n * inc_sigma * inc_sigma
                + n * (mu - new_mu) * (mu - new_mu)
                + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
            )
            / new_n,
        )
    )
    return new_n, new_mu, new_sigma


def _get_running_stats(
    values: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes cumulative running statistics patch-by-patch.

    For each variate, patch `i` gets stats computed from all unmasked values
    in patches 0 through i (inclusive).

    Parameters
    ----------
    values
        Input values. Shape: ``(batch, v, n, p)``.
    masks
        Boolean mask (True=masked). Shape: ``(batch, v, n, p)``.

    Returns
    -------
        Tuple of ``(running_n, running_mu, running_sigma)``, each shape ``(batch, v, n)``.
    """
    b, v, n, _ = values.shape
    device = values.device

    init_n = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
    init_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)

    all_n = []
    all_mu = []
    all_sigma = []
    cur_n, cur_mu, cur_sigma = init_n, init_mu, init_sigma

    for i in range(n):
        cur_n, cur_mu, cur_sigma = _update_running_stats(
            cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
        )
        all_n.append(cur_n)
        all_mu.append(cur_mu)
        all_sigma.append(cur_sigma)

    return (
        torch.stack(all_n, dim=2),
        torch.stack(all_mu, dim=2),
        torch.stack(all_sigma, dim=2),
    )


def _revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """Reversible per-instance normalization.

    Automatically expands mu/sigma dims to match x.

    Parameters
    ----------
    x
        Input tensor. Shape: ``(batch, ..., d)``.
    mu
        Mean tensor. Shape: ``(batch, ...)`` with 1 or 2 fewer dims than x.
    sigma
        Std tensor. Same shape as mu.
    reverse
        If True, applies reverse normalization (denormalize).

    Returns
    -------
        Normalized or denormalized tensor, same shape as x.
    """
    if mu.dim() == x.dim() - 1:
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    elif mu.dim() == x.dim() - 2:
        mu = mu.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)
    else:
        raise_log(
            ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}."),
        )
    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / _make_safe_for_division(sigma)


def _get_output_patch_via_roll(
    x: torch.Tensor, rolls: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates labels of output_patch length by rolling the patched inputs.

    Takes patched input (b, v, n, p) and creates output patches of length
    p*rolls by concatenating shifted views of the patches.

    Parameters
    ----------
    x
        Patched inputs. Shape: ``(batch, v, n, p)``.
    rolls
        Number of rolls (= output_patch_len / patch_len).

    Returns
    -------
    tuple
        Tuple of:
            - Rolled output. Shape: ``(batch, v, n, p * rolls)``.
            - Wrap-around mask. Shape: ``(1, 1, n, p * rolls)`` bool.
    """
    b, v, n, p = x.shape
    device = x.device
    rolling_mat = torch.zeros(b, v, n, rolls + 1, p, device=device, dtype=x.dtype)
    rolling_mat[:, :, :, 0, :] = x

    for i in range(rolls):
        rolling_mat[:, :, :, i + 1, :] = torch.roll(
            rolling_mat[:, :, :, i, :], shifts=-1, dims=2
        )

    # Take [1:] along the roll axis and flatten
    result = rolling_mat[:, :, :, 1:, :].reshape(b, v, n, rolls * p)

    # Build wrap-around mask
    patch_idx = torch.arange(n, device=device)
    point_idx = torch.arange(rolls * p, device=device)
    source_patch = patch_idx[:, None] + 1 + point_idx[None, :] // p
    wrap_mask = (source_patch >= n).unsqueeze(0).unsqueeze(0)

    return result, wrap_mask


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "none": lambda x: x,
}


def _get_activation_fn(
    activation_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the activation function for the given name."""
    try:
        return _ACTIVATIONS[activation_name]
    except KeyError:
        raise_log(
            ValueError(
                f"Activation: {activation_name} not supported. Supported "
                f"activations: {list(_ACTIVATIONS.keys())}"
            ),
        )


def _stitch_patches(
    patch_preds: torch.Tensor,
    patch_len: int,
) -> torch.Tensor:
    """Stitches overlapping patch predictions.

    Each patch predicts patch_len + overlap timepoints, where
    overlap = patch_preds.shape[3] - patch_len is inferred from the input.
    Consecutive patches share overlap timepoints, which are linearly stitched.

    Parameters
    ----------
    patch_preds
        Predictions of shape ``(batch, variates, num_patches, patch_len + overlap,
        num_quantiles)``.
    patch_len
        The patch length.

    Returns
    -------
        Stitched predictions of shape ``(batch, variates,
        num_patches * patch_len + overlap, num_quantiles)``.
    """
    b, v, num_patches, total_len, q = patch_preds.shape
    overlap = total_len - patch_len

    if num_patches == 1:
        return patch_preds[:, :, 0, :, :]

    stitch_weights = torch.linspace(
        1.0, 0.0, overlap, device=patch_preds.device, dtype=patch_preds.dtype
    )
    stitch_weights = stitch_weights[None, None, None, :, None]

    first_chunk = patch_preds[:, :, 0, :patch_len, :]

    prev_patches = patch_preds[:, :, :-1, :, :]
    next_patches = patch_preds[:, :, 1:, :, :]

    prev_overlaps = prev_patches[:, :, :, patch_len:, :]
    next_overlaps = next_patches[:, :, :, :overlap, :]

    stitched_overlaps = (
        stitch_weights * prev_overlaps + (1.0 - stitch_weights) * next_overlaps
    )

    middles = next_patches[:, :, :, overlap:patch_len, :]

    output_chunks = torch.cat([stitched_overlaps, middles], dim=3)

    mid = output_chunks.reshape(b, v, (num_patches - 1) * patch_len, q)

    tail = patch_preds[:, :, -1, patch_len:, :]

    return torch.cat([first_chunk, mid, tail], dim=2)


def _cpm_iterative_revin_refine(
    raw_logits: torch.Tensor,
    revin_n: torch.Tensor,
    revin_mu: torch.Tensor,
    revin_sigma: torch.Tensor,
    patch_cpm_mask: torch.Tensor,
    median_q_idx: int,
    rolls: int,
    patch_len: int,
    num_quantiles: int,
    value_clip: float = 1e9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refines RevIN stats at CPM-masked patches via iterative estimation.

    For each CPM-masked position p the currently frozen stats (from the last
    observed patch before the CPM region) are replaced with stats that also
    incorporate model-estimated values for all CPM patches that precede p.

    Parameters
    ----------
    raw_logits
        Output of the output head, shape ``(b, v, n, output_patch_len *
        num_quantiles)``. In RevIN-normalised space, before reverse RevIN.
    revin_n
        Count of valid (unmasked) values accumulated at each patch
        position, shape ``(b, v, n)``.
    revin_mu
        Running mean per position, ``(b, v, n)``.
    revin_sigma
        Running std per position, ``(b, v, n)``.
    patch_cpm_mask
        Boolean mask, True = CPM-masked patch, shape ``(b, n)``.
    median_q_idx
        Index into quantiles selecting the median quantile used as
        point estimate (typically ``num_quantiles // 2``).
    rolls
        Number of output patches per input patch (``output_patch_len // patch_len``).
    patch_len
        Length of each input patch.
    num_quantiles
        Total number of quantile heads.
    value_clip
        Absolute bound for clamping estimated values after reverse RevIN.

    Returns
    -------
    tuple
        Tuple ``(refined_mu, refined_sigma)``, each shape ``(b, v, n)``.
        Non-CPM positions are identical to ``revin_mu`` / ``revin_sigma``.
        CPM positions incorporate estimates of all preceding CPM patches in
        the same block (and all estimates from earlier blocks in the segment).
    """
    b, v, n_patches, _ = raw_logits.shape
    device = raw_logits.device

    # Reshape and slice raw_logits to keep only the median quantile.
    # (b, v, n, oq) -> (b, v, n, rolls, patch_len, num_quantiles)
    # -> (b, v, n, rolls, patch_len)
    median_logits = raw_logits.reshape(
        b, v, n_patches, rolls, patch_len, num_quantiles
    )[:, :, :, :, :, median_q_idx]

    # Initialise carry with zeros.
    carry_n = torch.zeros((b, v), dtype=torch.float32, device=device)
    carry_mu = torch.zeros((b, v), dtype=torch.float32, device=device)
    carry_sigma = torch.zeros((b, v), dtype=torch.float32, device=device)
    anchor_predicted_values = torch.zeros(
        (b, v, rolls, patch_len), dtype=torch.float32, device=device
    )
    block_offset = torch.zeros((b,), dtype=torch.long, device=device)

    refined_mu_list = []
    refined_sigma_list = []

    step_masks = torch.zeros((b, v, patch_len), dtype=torch.bool, device=device)

    for i in range(n_patches):
        actual_n = revin_n[:, :, i]
        actual_mu = revin_mu[:, :, i]
        actual_sigma = revin_sigma[:, :, i]
        current_step_logits = median_logits[:, :, i]
        is_cpm = patch_cpm_mask[:, i : i + 1]  # (b, 1)

        # Select the block_offset[b]-th patch for each batch element
        offset_onehot = torch.eq(
            torch.arange(rolls, device=device).unsqueeze(0),
            block_offset.unsqueeze(1),
        ).float()
        predicted_values_step = torch.einsum(
            "br,bvrp->bvp", offset_onehot, anchor_predicted_values
        )

        # Update running stats with the estimated patch.
        new_n, new_mu, new_sigma = _update_running_stats(
            carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
        )

        out_n = torch.where(is_cpm, new_n, actual_n)
        out_mu = torch.where(is_cpm, new_mu, actual_mu)
        out_sigma = torch.where(is_cpm, new_sigma, actual_sigma)

        # Advance block_offset: +1 (mod rolls) for CPM, reset to 0 for non-CPM.
        new_block_offset = torch.where(
            is_cpm.squeeze(-1),
            (block_offset + 1) % rolls,
            torch.zeros_like(block_offset),
        )

        should_update_anchor = torch.eq(new_block_offset, 0)

        # Pre-calculate predicted values for the new anchor.
        step_predicted_values = _revin(
            current_step_logits, out_mu, out_sigma, reverse=True
        )
        step_predicted_values = torch.clamp(
            step_predicted_values, -value_clip, value_clip
        )

        new_anchor_predicted_values = torch.where(
            should_update_anchor.view(b, 1, 1, 1),
            step_predicted_values,
            anchor_predicted_values,
        )

        carry_n = out_n
        carry_mu = out_mu
        carry_sigma = out_sigma
        anchor_predicted_values = new_anchor_predicted_values
        block_offset = new_block_offset

        refined_mu_list.append(out_mu)
        refined_sigma_list.append(out_sigma)

    refined_mu = torch.stack(refined_mu_list, dim=2)
    refined_sigma = torch.stack(refined_sigma_list, dim=2)
    return refined_mu, refined_sigma
