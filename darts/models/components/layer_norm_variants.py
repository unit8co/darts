"""
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """An alternate to layer normalization, without mean centering and the learned bias [1]

    References
    ----------
    .. [1] Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information
           Processing Systems 32 (2019).
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, input_size, **kwargs):
        super().__init__(input_size, elementwise_affine=False, **kwargs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, input_size, **kwargs) -> None:
        super().__init__(input_size, **kwargs)


class ReversibleInstanceNorm(nn.Module):
    """Reversible Instance Normalization based on [1]

    References
    ----------
    .. [1] Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
            Distribution Shift" International Conference on Learning Representations (2022)
    """

    def __init__(self, axis, input_dim, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x, mode, target_slice=None):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=self.axis, keepdim=True)
        self.std = torch.sqrt(torch.var(x, dim=self.axis, keepdim=True) + self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.std
        if self.affine:

            massage_shape = not (
                (self.axis != -1) and (self.axis != x.ndim - 1)
            )

            # if axis isn't the last dimension, swap it to the last dimension
            if massage_shape:
                x = x.swapaxes(-2, self.axis)

            x = x * self.affine_weight
            x = x + self.affine_bias

            # swap axis back
            if massage_shape:
                x = x.swapaxes(-2, self.axis)

        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:

            massage_shape = not (
                (self.axis is not -1) and (self.axis is not x.ndim - 1)
            )

            # if axis isn't the last dimension, swap it to the last dimension
            if massage_shape:
                x = x.swapaxes(-2, self.axis)

            x = x - self.affine_bias[target_slice]
            x = x / self.affine_weight[target_slice]

            # swap axis back
            if massage_shape:
                x = x.swapaxes(-2, self.axis)

        x = x * self.std
        x = x + self.mean
        return x
