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
        Abstract class for a likelihood model. It contains all the logic to compute the loss
        and to sample the distribution, given the parameters of the distribution
        """
        pass

    @abstractmethod
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss from a model `output`, which represents the parameters of a given probability
        distribution for every ground truth value in `target`, and the `target` itself.
        """
        pass

    @abstractmethod
    def _sample(self, output: torch.Tensor) -> torch.Tensor:
        """
        Samples a prediction from the probability distributions defined by the specific likelihood model
        and the parameters given in `output`.
        """
        pass

    @property
    @abstractmethod
    def _num_parameters(self) -> int:
        """
        Returns the number of parameters that define the probability distribution for one single
        target value.
        """
        pass


class GaussianLikelihoodModel(LikelihoodModel):
    """
    Gaussian Likelihood
    """

    def __init__(self):
        self.loss = nn.GaussianNLLLoss(reduction='mean')
        self.softplus = nn.Softplus()
        super().__init__()        

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
        output_size = output.shape[-1]
        output_means = output[:, :, :output_size // 2]
        output_vars = self.softplus(output[:, :, output_size // 2:])
        return output_means, output_vars


class PoissonLikelihoodModel(LikelihoodModel):
    """
    Poisson Likelihood; can typically be used to model event counts in fixed intervals
    https://en.wikipedia.org/wiki/Poisson_distribution
    """

    def __init__(self):
        self.loss = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()
        super().__init__()

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        model_output = self._lambda_from_output(output)
        return self.loss(model_output, target)

    def _sample(self, output: torch.Tensor) -> torch.Tensor:
        output_lambda = self._lambda_from_output(output)
        return torch.poisson(output_lambda)

    @property
    def _num_parameters(self) -> int:
        return 1

    def _lambda_from_output(self, output):
        return self.softplus(output)
