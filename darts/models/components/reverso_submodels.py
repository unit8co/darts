"""
Reverso Submodels
-----------------

---
title: Reverso Submodels
summary: This module contains the submodules used in the Reverso model.
---

# License and Attribution

MIT License from https://github.com/shinfxh/reverso/blob/main/LICENSE

Copyright (c) 2026 Xinghong Fu, Yanhong Li, Georgios Papaioannou, Yoon Kim

Ported from https://github.com/shinfxh/reverso (reverso_torch/model.py and reverso/model.py).

# Modifications for Darts

Adapted for Darts with custom `PLForecastingModule` and `FoundationModel` integration:
- Remove autoregressive rollout logic (handled by `PLForecastingModule`).
- Replace FlashFFTConv with FFT-based circular convolution (pure PyTorch).
- Replace fla.layers.DeltaNet with pure-PyTorch delta-rule linear attention.
- Prefix all class names with `_` for internal use.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import get_logger

logger = get_logger(__name__)


class _RMSNorm(nn.Module):
    """RMS normalization matching fla.modules.layernorm.RMSNorm weight layout."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms * self.weight.float()).to(dtype)


class _PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for the output decoder head."""

    def __init__(self, d_model: int, max_len: int = 6500):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class _Gating(nn.Module):
    """Gated short convolution block."""

    def __init__(self, channels: int, temporal_kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=temporal_kernel,
                padding=temporal_kernel // 2,
                groups=channels,
            ),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class _MLPBlock(nn.Module):
    """Feed-forward block with skip connection and LayerNorm."""

    def __init__(self, d_in: int, d_out: int, d_intermediate: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(d_out)
        if d_intermediate and d_intermediate > 0:
            self.linear = nn.Linear(d_in, d_intermediate)
            self.linear_final = nn.Linear(d_intermediate, d_out)
        else:
            self.linear = nn.Linear(d_in, d_out)
            self.linear_final = nn.Identity()
        self.activation = nn.ReLU()
        self.skip_linear = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
        residual = self.skip_linear(x)
        y = self.linear(x)
        y = self.activation(y)
        y = self.linear_final(y)
        y = self.norm(y)
        y = residual + y
        if y.ndim == 3:
            y = y.permute(0, 2, 1)
        return y


class _CNNBlock(nn.Module):
    """Long convolution via FFT (replaces FlashFFTConv)."""

    def __init__(self, channels: int, seq_len: int, gating_kernel_size: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.k = nn.Parameter(torch.randn(channels, seq_len, dtype=torch.float32))
        self.pregate = _Gating(channels, gating_kernel_size)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_conv = x.contiguous().to(torch.bfloat16)
        pregate = self.pregate(x_conv.float()).to(x_conv.dtype)
        x_gated = (pregate * x_conv).float()

        # Circular convolution via FFT (matches FlashFFTConv behaviour)
        X = torch.fft.rfft(x_gated, n=self.seq_len, dim=-1)
        K = torch.fft.rfft(self.k.float(), n=self.seq_len, dim=-1)
        out = torch.fft.irfft(X * K.unsqueeze(0), n=self.seq_len, dim=-1)

        out = self.activation(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = out + residual
        return out


class _TorchDeltaNet(nn.Module):
    """Pure-PyTorch delta-rule linear attention.

    Weight-compatible with ``fla.layers.DeltaNet`` (same parameter names and shapes)
    so that pre-trained checkpoints load directly.
    """

    def __init__(
        self,
        d_model: int | None = None,
        hidden_size: int = 1024,
        mode: str = "chunk",
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.use_beta = use_beta
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        # projections (match fla naming)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = nn.Conv1d(
                self.key_dim,
                self.key_dim,
                conv_size,
                padding=conv_size - 1,
                groups=self.key_dim,
                bias=conv_bias,
            )
            self.k_conv1d = nn.Conv1d(
                self.key_dim,
                self.key_dim,
                conv_size,
                padding=conv_size - 1,
                groups=self.key_dim,
                bias=conv_bias,
            )
            self.v_conv1d = nn.Conv1d(
                self.value_dim,
                self.value_dim,
                conv_size,
                padding=conv_size - 1,
                groups=self.value_dim,
                bias=conv_bias,
            )

        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.o_norm = _RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _causal_conv1d(
        self, x: torch.Tensor, conv: nn.Conv1d, apply_silu: bool = True
    ) -> torch.Tensor:
        """Causal depthwise conv1d: (B, L, D) -> (B, L, D)."""
        y = conv(x.transpose(1, 2))  # (B, D, L + pad)
        y = y[..., : x.shape[1]].transpose(1, 2)  # truncate to causal
        if apply_silu:
            y = F.silu(y)
        return y

    @staticmethod
    def _delta_rule_recurrent(
        q: torch.Tensor,  # (B, H, L, K)
        k: torch.Tensor,  # (B, H, L, K)
        v: torch.Tensor,  # (B, H, L, V)
        beta: torch.Tensor,  # (B, H, L)
    ) -> torch.Tensor:
        """Chunked parallel-scan delta rule.

        Replaces the naive step-by-step recurrence with a Hillis-Steele
        parallel prefix scan *within* fixed-size chunks, while propagating
        the recurrent state sequentially *across* chunks.  This is
        numerically equivalent (max diff ~4e-7 in float32) but 2-9x faster
        on CPU because each scan step is a single batched matmul instead of
        a Python for-loop over time steps.

        The chunk size is chosen automatically based on K (head dimension)
        to balance the O(K^2) scan matmul cost against Python-loop overhead.
        """
        B, H, L, K = q.shape
        V = v.shape[-1]
        device, dtype = q.device, q.dtype

        # Larger K → smaller optimal chunk (scan matmuls are O(K^2) each).
        # Values determined empirically across reverso-nano/small/full.
        if K <= 8:
            chunk_size = 512
        elif K <= 16:
            chunk_size = 128
        else:
            chunk_size = 64

        eye = torch.eye(K, device=device, dtype=dtype)
        o = torch.empty(B, H, L, V, device=device, dtype=dtype)
        h = q.new_zeros(B, H, K, V)  # inter-chunk recurrent state

        num_chunks = (L + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, L)
            clen = end - start

            q_c = q[:, :, start:end]
            k_c = k[:, :, start:end]
            v_c = v[:, :, start:end]
            b_c = beta[:, :, start:end]

            # Build per-step transition matrices and bias vectors:
            #   A_t = I - β_t k_t k_t^T   (B, H, clen, K, K)
            #   b_t = β_t k_t v_t^T        (B, H, clen, K, V)
            beta_exp = b_c.unsqueeze(-1).unsqueeze(-1)
            As = eye - beta_exp * (k_c.unsqueeze(-1) * k_c.unsqueeze(-2))
            bs = beta_exp * (k_c.unsqueeze(-1) * v_c.unsqueeze(-2))

            # Hillis-Steele inclusive prefix scan within chunk.
            # After ceil(log2(clen)) steps of batched matmul:
            #   As[t] = A_t @ A_{t-1} @ ... @ A_0  (cumulative transition)
            #   bs[t] = cumulative bias such that state = As[t] @ h_prev + bs[t]
            scan_steps = int(math.ceil(math.log2(clen)))
            for d in range(scan_steps):
                stride = 2**d
                if stride >= clen:
                    break
                new_A = torch.matmul(As[:, :, stride:], As[:, :, : clen - stride])
                new_b = (
                    torch.matmul(As[:, :, stride:], bs[:, :, : clen - stride])
                    + bs[:, :, stride:]
                )
                As = torch.cat([As[:, :, :stride], new_A], dim=2)
                bs = torch.cat([bs[:, :, :stride], new_b], dim=2)

            # Materialize all states: h_t = As[t] @ h_prev + bs[t]
            states = torch.matmul(As, h.unsqueeze(2).expand(-1, -1, clen, -1, -1)) + bs

            # Readout: o_t = q_t^T @ h_t
            o[:, :, start:end] = torch.einsum("bhlk,bhlkv->bhlv", q_c, states)

            # Propagate state to next chunk
            h = As[:, :, -1] @ h + bs[:, :, -1]

        return o  # (B, H, L, V)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask=None, **kwargs
    ) -> tuple[torch.Tensor, None, None]:
        B, L, _ = hidden_states.shape

        if self.use_short_conv:
            q = self._causal_conv1d(
                self.q_proj(hidden_states),
                self.q_conv1d,
                apply_silu=(self.qk_activation == "silu"),
            )
            k = self._causal_conv1d(
                self.k_proj(hidden_states),
                self.k_conv1d,
                apply_silu=(self.qk_activation == "silu"),
            )
            v = self._causal_conv1d(
                self.v_proj(hidden_states), self.v_conv1d, apply_silu=True
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.qk_activation == "silu":
                q = F.silu(q)
                k = F.silu(k)
            v = F.silu(v)

        # reshape to multi-head: (B, L, H, D)
        q = q.view(B, L, self.num_heads, self.head_k_dim)
        k = k.view(B, L, self.num_heads, self.head_k_dim)
        v = v.view(B, L, self.num_heads, self.head_v_dim)

        # L2 normalization per head
        if self.qk_norm == "l2":
            q = q / (q.norm(2, dim=-1, keepdim=True).pow(2).add(1e-6)).sqrt()
            k = k / (k.norm(2, dim=-1, keepdim=True).pow(2).add(1e-6)).sqrt()

        # beta
        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()  # (B, L, H)
        else:
            beta = q.new_ones(B, L, self.num_heads)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # -> (B, H, L, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        beta = beta.permute(0, 2, 1)  # (B, H, L)

        q = q * (self.head_k_dim**-0.5)

        o = self._delta_rule_recurrent(q, k, v, beta)  # (B, H, L, V)

        # -> (B, L, H, V) then RMSNorm per head
        o = o.permute(0, 2, 1, 3)
        o = self.o_norm(o)

        # merge heads and project
        o = o.reshape(B, L, self.value_dim)
        o = self.o_proj(o)
        return o, None, None


class _AttentionBlock(nn.Module):
    """DeltaNet attention block with optional state weaving."""

    def __init__(
        self,
        d_model: int,
        expand_v: float,
        state_weaving: bool = False,
        is_intermediate: bool = False,
    ):
        super().__init__()
        self.state_weaving = state_weaving
        self.is_intermediate = is_intermediate
        self.attention = _TorchDeltaNet(
            mode="chunk",
            d_model=d_model,
            expand_k=1.0,
            expand_v=expand_v,
            num_heads=4,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            conv_size=4,
            allow_neg_eigval=False,
            qk_activation="silu",
            qk_norm="l2",
            layer_idx=0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        residual = x_t
        if self.state_weaving and self.is_intermediate:
            x_t = x_t.clone()
            x_t[:, 0:1, :] = x_t[:, 0:1, :] + x_t[:, -1:, :]
        attn_out = self.attention(hidden_states=x_t, attention_mask=None)
        if isinstance(attn_out, tuple):
            out = attn_out[0]
        else:
            out = attn_out
        out = self.norm(out)
        out = out + residual
        out = out.transpose(1, 2)
        return out
