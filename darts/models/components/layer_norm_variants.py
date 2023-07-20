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


class RINorm(nn.Module):
    def __init__(self, input_dim: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization based on [1]

        Parameters
        ----------
        input_dim
            The dimension of the input axis being normalized
        eps
            The epsilon value for numerical stability
        affine
            Whether to apply an affine transformation after normalization

        References
        ----------
        .. [1] Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift" International Conference on Learning Representations (2022)
        """

        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor):
        # at the beginning of `PLForecastingModule.forward()`, `x` has shape
        # (batch_size, input_chunk_length, n_targets).
        # select all dimensions except batch and input_dim (0, -1)
        # TL;DR: calculate mean and variance over all dimensions except batch and input_dim
        calc_dims = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def inverse(self, x: torch.Tensor):
        # x is assumed to be the output of PLForecastingModule.forward(), and has shape
        # (batch_size, output_chunk_length, n_targets, nr_params). we ha
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (
                self.affine_weight.view(self.affine_weight.shape + (1,))
                + self.eps * self.eps
            )
        x = x * self.stdev.view(self.stdev.shape + (1,))
        x = x + self.mean.view(self.mean.shape + (1,))
        return x
