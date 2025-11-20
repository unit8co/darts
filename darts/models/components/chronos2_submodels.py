"""
Chronos-2 Submodels
-------------------

---
title: Chronos-2 Submodels
summary: This module contains the submodules used in the Chronos-2 model.
---

# License and Attribution

Apache-2.0 License from https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE,
accessed on 4 November 2025:

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

Ported from https://github.com/amazon-science/chronos-forecasting/commit/c23d34cd887b889c302ca7b6df3fa0bca96d78a9
on 4 November 2025.

# Modifications for Darts

Adapted for Darts with custom `PLForecastingModule` and `FoundationModel` integration:
- Remove dependencies on `transformers` and `einops` libraries.
- Load model config and weights from HuggingFace Hub using `HuggingFaceConnector`.
- Remove `output_attentions` option from forward pass.
- Integrate likelihood model and loss computation with Darts `QuantileRegression`, and
    remove original loss computation in forward pass.
- Replace `*Output` return type with direct `torch.Tensor` to comply with Darts
    `PLForecastingModule` interface.
- Replace `einops` rearrange operations with native PyTorch tensor operations.
"""

from typing import Literal

import torch
from torch import nn

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


class _Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x


class _InstanceNorm(nn.Module):
    """
    Apply standardization along the last dimension and optionally apply arcsinh after standardization.
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
            )
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        if self.use_arcsinh:
            x = torch.sinh(x)

        x = x * scale + loc

        return x.to(orig_dtype)


class _ResidualBlock(nn.Module):
    """A generic residual block which can be used for input and output embedding layers"""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            raise_log(
                ValueError("Layer norm should not be used in ResidualBlock."),
                logger,
            )

        if not act_fn_name == "relu":
            raise_log(
                ValueError("ReLU should be used in ResidualBlock"),
                logger,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        return out


class _RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example,
                note that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then,
                if q and k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k
                have the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (_RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (_RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class _MHA(nn.Module):
    """Multi-head Attention Layer"""

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        rope_theta: float,
        attn_implementation: Literal["eager", "sdpa"],
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.kv_proj_dim: int = d_kv
        self.n_heads: int = num_heads
        self.dropout: float = dropout_rate
        self.inner_dim: int = self.n_heads * self.kv_proj_dim
        self.attn_implementation = attn_implementation

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = _RoPE(dim=self.kv_proj_dim, base=rope_theta)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager attention implementation using manual matmul.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len]

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: [batch, n_heads, q_len, kv_len]
        """
        # Compute attention weights (no scaling - this is the original Chronos-2 implementation)
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # "bnqd,bnkd->bnqk"
        scores += mask
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """SDPA attention implementation using torch.nn.functional.scaled_dot_product_attention.

        Args:
            query_states: [batch, n_heads, seq_len, kv_proj_dim]
            key_states: [batch, n_heads, seq_len, kv_proj_dim]
            value_states: [batch, n_heads, seq_len, kv_proj_dim]
            mask: [batch, n_heads, q_len, kv_len] - additive mask (0 for valid, -inf for invalid)

        Returns:
            attn_output: [batch, n_heads, seq_len, kv_proj_dim]
            attn_weights: None (SDPA doesn't return weights)
        """
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # Match eager implementation (no scaling)
        )

        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head attention forward pass.

        Args:
            hidden_states : Input tensor of shape [batch_size, seq_len, d_model]
            mask : Attention mask tensor of shape [batch_size, num_heads, q_len, kv_len]
            position_ids : Position IDs for RoPE. Defaults to None.

        Returns:
            AttentionOutput: Contains:
                - hidden_states : Output tensor of shape [batch_size, seq_len, d_model]
        """
        if self.use_rope:
            assert position_ids is not None, (
                "position_ids must be provided when self.use_rope=True"
            )

        seq_length = hidden_states.shape[1]

        def shape(states: torch.Tensor) -> torch.Tensor:
            """(batch, seq_len, inner_dim) -> (batch, n_heads, seq_len, kv_proj_dim)"""
            return states.view(
                -1,
                seq_length,
                self.n_heads,
                self.kv_proj_dim,
            ).permute(0, 2, 1, 3)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            """(batch, n_heads, seq_len, kv_proj_dim) -> (batch, seq_len, inner_dim)"""
            return states.permute(0, 2, 1, 3).reshape(
                -1,
                seq_length,
                self.inner_dim,
            )

        # Construct query states
        query_states = shape(self.q(hidden_states))

        # Construct key/value states
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        if self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = _RoPE.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if self.attn_implementation == "sdpa":
            attn_output, _ = self._sdpa_attention(
                query_states, key_states, value_states, mask
            )
        else:  # eager
            attn_output, _ = self._eager_attention(
                query_states, key_states, value_states, mask
            )

        # Project attention output
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)

        return attn_output


class _Chronos2LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class _TimeSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        rope_theta: float,
        attn_implementation: Literal["eager", "sdpa"],
        layer_norm_epsilon: float,
    ):
        super().__init__()
        self.self_attention = _MHA(
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rope_theta=rope_theta,
            attn_implementation=attn_implementation,
            use_rope=True,
        )
        self.layer_norm = _Chronos2LayerNorm(d_model, layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attention(
            normed_hidden_states,
            position_ids=position_ids,
            mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states


class _GroupSelfAttention(nn.Module):
    """Self-attention applied along the batch axis masked by the group attention mask"""

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float,
        rope_theta: float,
        attn_implementation: Literal["eager", "sdpa"],
        layer_norm_epsilon: float,
    ):
        super().__init__()
        # we don't use RoPE here because there's no natural ordering along the batch axis
        self.self_attention = _MHA(
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            rope_theta=rope_theta,
            attn_implementation=attn_implementation,
            use_rope=False,
        )
        self.layer_norm = _Chronos2LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # flip time and batch axes because attention operates along dim=-2
        hidden_states = hidden_states.permute(1, 0, 2)
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        # flip time and batch axes back to their original position
        hidden_states = hidden_states.permute(1, 0, 2)

        return hidden_states


class _MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        dense_act_fn: str,
    ):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

        if dense_act_fn != "relu":
            raise_log(
                ValueError("ReLU should be used in ResidualBlock"),
                logger,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class _FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        dense_act_fn: str,
        is_gated_act: bool,
        layer_norm_epsilon: float,
    ):
        super().__init__()

        self.mlp: nn.Module = _MLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            dense_act_fn=dense_act_fn,
        )
        self.layer_norm = _Chronos2LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

        if is_gated_act:
            raise_log(
                ValueError("Gated activations are unsupported in FeedForward"),
                logger,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class _Chronos2EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float,
        rope_theta: float,
        attn_implementation: Literal["eager", "sdpa"],
        dense_act_fn: str,
        layer_norm_epsilon: float,
        is_gated_act: bool,
    ):
        super().__init__()

        self.layer = nn.ModuleList([
            _TimeSelfAttention(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                rope_theta=rope_theta,
                attn_implementation=attn_implementation,
                layer_norm_epsilon=layer_norm_epsilon,
            ),
            _GroupSelfAttention(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                rope_theta=rope_theta,
                attn_implementation=attn_implementation,
                layer_norm_epsilon=layer_norm_epsilon,
            ),
            _FeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                dense_act_fn=dense_act_fn,
                is_gated_act=is_gated_act,
                layer_norm_epsilon=layer_norm_epsilon,
            ),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        # apply time attention
        time_hidden_states: torch.Tensor = self.layer[0](
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # apply group attention
        group_hidden_states = self.layer[1](
            time_hidden_states,
            attention_mask=group_time_mask,
        )

        # apply feed forward layer
        hidden_states = self.layer[2](group_hidden_states)

        return hidden_states


class _Chronos2Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float,
        rope_theta: float,
        attn_implementation: Literal["eager", "sdpa"],
        dense_act_fn: str,
        layer_norm_epsilon: float,
        is_gated_act: bool,
        num_layers: int,
    ):
        super().__init__()

        self.block = nn.ModuleList([
            _Chronos2EncoderBlock(
                d_model=d_model,
                d_kv=d_kv,
                d_ff=d_ff,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                rope_theta=rope_theta,
                attn_implementation=attn_implementation,
                dense_act_fn=dense_act_fn,
                layer_norm_epsilon=layer_norm_epsilon,
                is_gated_act=is_gated_act,
            )
            for i in range(num_layers)
        ])
        self.final_layer_norm = _Chronos2LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(
        attention_mask: torch.Tensor, floating_type: torch.dtype
    ) -> torch.Tensor:
        assert attention_mask.ndim == 2, (
            "attention_mask must have shape (batch, seq_len)"
        )

        # Add new dims for attention heads and q_len
        attention_mask = attention_mask[:, None, None, :]

        # Invert binary mask to float mask which can be added to attention scores
        attention_mask = attention_mask.to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    @staticmethod
    def _construct_and_invert_group_time_mask(
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        floating_type: torch.dtype,
    ) -> torch.Tensor:
        # construct group_mask (batch, batch) from group ids
        # a cell is True if both row and col had the same group id
        group_mask = group_ids[:, None] == group_ids[None, :]
        # outer product of group_mask and attention_mask (time_mask)
        # group_time_mask combines group and time masks to ensure that attention only uses
        # tokens from the same group which are also not masked in time
        group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)

        if torch.is_floating_point(group_time_mask):
            # this ensures that mixed precision training does not overflow
            floating_type = group_time_mask.dtype

        # reshape mask to shape of attention scores
        group_time_mask = group_time_mask.permute(2, 0, 1).unsqueeze(1)
        group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min

        return group_time_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = inputs_embeds.size(1)

        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        # make the time attention mask broadcastable to attention scores (batch, n_heads, q_len, kv_len) and invert
        extended_attention_mask = self._expand_and_invert_time_attention_mask(
            attention_mask, inputs_embeds.dtype
        )

        # construct group time mask
        group_time_mask = self._construct_and_invert_group_time_mask(
            group_ids, attention_mask, inputs_embeds.dtype
        )

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module) in enumerate(self.block):
            hidden_states = layer_module(
                hidden_states,
                position_ids=position_ids,
                attention_mask=extended_attention_mask,
                group_time_mask=group_time_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
