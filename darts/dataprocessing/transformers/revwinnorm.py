import torch
import torch.nn as nn
from typing import Optional


class RevWinNorm(nn.Module):
    def __init__(self, num_features: int, norm_affine: bool=False , eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params(norm_affine=norm_affine)

    def forward(self, x, mode: str, scaler: Optional[str], norm_type: str = 'instance'):
        if mode == "norm":
            self._get_statistics(x, norm_type=norm_type)
            x = self._normalize(x, scaler=scaler)
        elif mode == "denorm":
            x = self._denormalize(x, scaler=scaler)
        else:
            raise NotImplementedError
        return x

    def _init_params(self, norm_affine: bool=False):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        # enable learnable
        if norm_affine is True:
            self.affine_weight.requires_grad = True
            self.affine_bias.requires_grad = True
        else:
            self.affine_weight.requires_grad = False
            self.affine_bias.requires_grad = False

    def _get_statistics(self, x, norm_type: str='instance'):
        if norm_type == 'instance':
            dim2reduce = (1,)  
        elif norm_type == 'batch':
            dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, scaler: Optional[str] ='standard'):
        if scaler == 'standard':
            x = x - self.mean
            x = x / self.stdev
        elif scaler == 'mean':
            x = x / self.mean

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, norm_type: str='instance', scaler: Optional[str]='standard'):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        if scaler == 'standard':
            x = x * self.stdev
            x = x + self.mean
        elif scaler == 'mean':
            x = x * self.mean
        return x
