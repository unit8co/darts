"""
Likelihood Models
-----------------
"""

from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple
import torch
import torch.nn as nn

from ..timeseries import TimeSeries


class LikelihoodModel(ABC):

    def __init__(self):
        """
        Abstract class for a `Likelihood` model. It contains all the logic to compute the loss
        and to sample the distribution, given the parameters of the distribution
        """
        pass

    @abstractmethod
    def _compute_loss(self, output: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _sample(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def _num_parameters(self) -> int:
        pass


class GaussianLikelihoodModel(LikelihoodModel):

    def __init__(self):
        self.loss = nn.GaussianNLLLoss(reduction='sum')

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_means, output_vars = self._means_and_vars_from_output(output)
        return self.loss(output_means, target, output_vars)

    def _sample(self, output: torch.Tensor) -> torch.Tensor:
        output_means, output_vars = self._means_and_vars_from_output(output)
        return torch.normal(output_means, output_vars)

    @property
    def _num_parameters(self) -> int:
        return 2

    def _means_and_vars_from_output(self, output):
        softplus_activation = nn.Softplus()
        output_size = output.shape[-1]
        output_means = output[:, :, :output_size // 2]
        output_vars = softplus_activation(output[:, :, output_size // 2:])
        return output_means, output_vars
