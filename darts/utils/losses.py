"""
PyTorch Loss Functions
----------------------
"""
# Inspiration: https://github.com/ElementAI/N-BEATS/blob/master/common/torch/losses.py

import numpy as np
import torch
import torch.nn as nn


def _divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0
    result[result == np.inf] = 0.0
    return result


class SmapeLoss(nn.Module):
    def __init__(self, block_denom_grad: bool = True):
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
        both of length :math:`T`, it is computed as

        .. math::
            \\frac{1}{T}
            \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}
                                  {\\left| y_t \\right| + \\left| \\hat{y}_t \\right|} }.

        The results of divisions yielding NaN or Inf are replaced by 0.

        Parameters
        ----------
        block_denom_grad
            Whether to stop the gradient in the denomitator
        """
        super().__init__()
        self.block_denom_grad = block_denom_grad

    def forward(self, inpt, tgt):
        num = torch.abs(tgt - inpt)
        if self.block_denom_grad:
            with torch.no_grad():
                denom = torch.abs(tgt) + torch.abs(inpt)
        else:
            denom = torch.abs(tgt) + torch.abs(inpt)
        return torch.mean(_divide_no_nan(num, denom))


class MapeLoss(nn.Module):
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """

    def __init__(self):
        super().__init__()

    def forward(self, inpt, tgt):
        return torch.mean(torch.abs(inpt - tgt))
