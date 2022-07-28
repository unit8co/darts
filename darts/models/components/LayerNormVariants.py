import torch
from torch import nn


class ScaleNorm(nn.Module):
    """An alternate to Layer normalization which is both effective and simpler [1]

    References
    ----------
    .. [1] Nguyen, Toan Q., and Julian Salazar. "Transformers without tears: Improving the normalization of
           self-attention." arXiv preprint arXiv:1910.05895 (2019).
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


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
