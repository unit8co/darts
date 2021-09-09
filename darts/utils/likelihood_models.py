"""
Likelihood Models
-----------------
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


class Likelihood(ABC):

    def __init__(self):
        """
        Abstract class for a likelihood model. It contains all the logic to compute the loss
        and to sample the distribution, given the parameters of the distribution
        """
        pass

    @abstractmethod
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss from a `model_output`, which represents the parameters of a given probability
        distribution for every ground truth value in `target`, and the `target` itself.
        """
        pass

    @abstractmethod
    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Samples a prediction from the probability distributions defined by the specific likelihood model
        and the parameters given in `model_output`.
        """
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """
        Returns the number of parameters that define the probability distribution for one single
        target value.
        """
        pass


class GaussianLikelihood(Likelihood):
    """
    Univariate Gaussian Likelihood
    Components are modeled by separate univariate distributions, with optional time-independent priors.
    """
    def __init__(self, prior_mu: Optional[float] = None, prior_sigma: Optional[float] = None, beta=1.):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.use_prior = self.prior_mu is not None or self.prior_sigma is not None
        self.beta = beta

        self.nllloss = nn.GaussianNLLLoss(reduction='mean', full=True)
        self.softplus = nn.Softplus()

        super().__init__()

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_means, out_sigmas = self._means_and_vars_from_model_output(model_output)
        loss = self.nllloss(out_means.contiguous(),  # TODO: can remove contiguous?
                            target.contiguous(),
                            out_sigmas.contiguous())

        if self.use_prior:
            out_distr = Normal(out_means, out_sigmas)

            prior_mu = torch.tensor(self.prior_mu).to(out_means.device) if self.prior_mu is not None else out_means
            prior_sigma = torch.tensor(self.prior_sigma).to(out_means.device) if self.prior_sigma is not None else out_sigmas
            prior_distr = Normal(prior_mu, prior_sigma)

            # add KL term
            loss += self.beta * torch.mean(kl_divergence(prior_distr, out_distr))

        return loss

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_output_means, model_output_vars = self._means_and_vars_from_model_output(model_output)
        return torch.normal(model_output_means, model_output_vars)

    @property
    def num_parameters(self) -> int:
        return 2

    def _means_and_vars_from_model_output(self, model_output):
        output_size = model_output.shape[-1]
        output_means = model_output[:, :, :output_size // 2]
        output_vars = self.softplus(model_output[:, :, output_size // 2:])
        return output_means, output_vars


class PoissonLikelihood(Likelihood):
    """
    Poisson Likelihood; can typically be used to model event counts in fixed intervals
    https://en.wikipedia.org/wiki/Poisson_distribution
    """

    def __init__(self):
        self.loss = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()
        super().__init__()

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        model_output = self._lambda_from_output(model_output)
        return self.loss(model_output, target)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_lambda = self._lambda_from_output(model_output)
        return torch.poisson(model_lambda)

    @property
    def num_parameters(self) -> int:
        return 1

    def _lambda_from_output(self, model_output):
        return self.softplus(model_output)
