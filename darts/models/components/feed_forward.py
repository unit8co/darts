r"""
The MIT License (MIT)

Copyright (c) 2020 Varuna Jayasiri

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


---
title: Position-wise Feed-Forward Network (FFN)
summary: Documented reusable implementation of the position wise feedforward network.
---

# Position-wise Feed-Forward Network (FFN)
This is a [PyTorch](https://pytorch.org)  implementation
of position-wise feedforward network used in transformer.

FFN consists of two fully connected layers.
Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
four times that of the token embedding $d_{model}$.
So it is sometime also called the expand-and-contract network.

There is an activation at the hidden layer, which is
usually set to ReLU (Rectified Linear Unit) activation, $$\max(0, x)$$

That is, the FFN function is,
$$FFN(x, W_1, W_2, b_1, b_2) = \max(0, x W_1 + b_1) W_2 + b_2$$
where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.

Sometimes the
GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
$$x \Phi(x)$$ where $\Phi(x) = P(X \le x), X \sim \mathcal{N}(0,1)$
### Gated Linear Units

This is a generic implementation that supports different variants including
[Gated Linear Units](https://papers.labml.ai/paper/2002.05202) (GLU).
"""

import torch
from torch import nn as nn

from darts.utils.torch import MonteCarloDropout


class FeedForward(nn.Module):
    """
    ## FFN module

    source
    [FeedForward network](https://arxiv.org/abs/2002.05202)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer,
           compatible with Monte Carlo dropout at inference time
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = MonteCarloDropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)
