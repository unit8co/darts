"""
Likelihood Models
-----------------
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

# TODO: rename internally to avoid exports
from torch.distributions import Normal, Poisson, NegativeBinomial
from torch.distributions.kl import kl_divergence


class Likelihood(ABC):
    def __init__(self):
        """
        Abstract class for a likelihood model. It contains all the logic to compute the loss
        and to sample the distribution, given the parameters of the distribution.
        It also allows for users to specify "prior" beliefs about the distribution parameters.
        In such cases, the a KL-divergence term is added to the loss in order to regularise it in the
        direction of the prior distribution. The parameter `beta` controls the strength of the regularisation.
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
    def __init__(self, prior_mu: Optional[float] = None, prior_sigma: Optional[float] = None, beta=1.):
        """
        Univariate Gaussian Likelihood
        Components are modeled by separate univariate distributions, with optional time-independent priors.

        It is possible to specify a prior on mu or sigma only. Leaving both to `None` won't be using a prior,
        and corresponds to doing maximum likelihood.

        Parameters
        ----------
        prior_mu
            mean of the prior Gaussian distribution (default: None)
        prior_sigma
            standard deviation (or scale) of the prior Gaussian distribution (default: None)
        beta
            strength of the loss regularisation induced by the prior
        """
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
    def __init__(self, prior_lambda: Optional[float] = None, beta=1.):
        """
        Poisson Likelihood; can typically be used to model event counts during time intervals, when the events
        happen independently of the time since the last event.
        https://en.wikipedia.org/wiki/Poisson_distribution

        It is possible to specify a time-independent prior rate `lambda` to capture a-priori
        knowledge about the process. Leaving it to `None` won't be using a prior,
        and corresponds to doing maximum likelihood.

        Parameters
        ----------
        prior_lambda
            rate of the prior Poisson distribution (default: None)
        beta
            strength of the loss regularisation induced by the prior
        """
        self.prior_lambda = prior_lambda
        self.beta = beta

        self.nllloss = nn.PoissonNLLLoss(log_input=False, full=True)
        self.softplus = nn.Softplus()
        super().__init__()

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lambda_out = self._lambda_from_output(model_output)
        loss = self.nllloss(lambda_out, target)
        if self.prior_lambda is not None:
            out_distr = Poisson(lambda_out)
            prior_distr = Poisson(torch.tensor(self.prior_lambda).to(lambda_out.device))
            loss += self.beta * torch.mean(kl_divergence(prior_distr, out_distr))
        return loss

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_lambda = self._lambda_from_output(model_output)
        return torch.poisson(model_lambda)

    @property
    def num_parameters(self) -> int:
        return 1

    def _lambda_from_output(self, model_output):
        return self.softplus(model_output)


class NegativeBinomialLikelihood(Likelihood):
    def __init__(self, prior_mu: Optional[float] = None, prior_alpha: Optional[float] = None, beta=1.):
        """
        Negative Binomial Likelihood
        """
        self.prior_mu = prior_mu
        self.prior_alpha = prior_alpha
        self.beta = beta
        self.use_prior = self.prior_mu is not None or self.prior_alpha is not None

        self.softplus = nn.Softplus()
        super().__init__()

    @staticmethod
    def _get_r_and_p_from_mu_and_alpha(mu, alpha):
        # See https://en.wikipedia.org/wiki/Negative_binomial_distribution for the different parametrizations
        r = 1. / alpha
        p = r / (mu + r)
        return r, p

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        mu_out, alpha_out = self._means_and_alphas_from_model_output(model_output)

        r_out, p_out = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu_out, alpha_out)
        out_distr = NegativeBinomial(r_out, p_out)

        # take negative log likelihood as loss
        loss = - out_distr.log_prob(target).mean()
        if self.use_prior:
            prior_mu = torch.tensor(self.prior_mu).to(mu_out.device) if self.prior_mu is not None else mu_out
            prior_alpha = torch.tensor(self.prior_alpha).to(mu_out.device) if self.prior_alpha is not None else alpha_out
            prior_r, prior_p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(prior_mu, prior_alpha)
            prior_distr = NegativeBinomial(prior_r, prior_p)
            loss += self.beta * torch.mean(kl_divergence(prior_distr, out_distr))
        return loss

    def sample(self, model_output: torch.Tensor):
        mu, alpha = self._means_and_alphas_from_model_output(model_output)
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        distr = NegativeBinomial(r, p)
        return distr.sample()

    def _means_and_alphas_from_model_output(self, model_output):
        output_size = model_output.shape[-1]
        output_means = self.softplus(model_output[:, :, :output_size // 2])
        output_alphas = self.softplus(model_output[:, :, output_size // 2:])
        return output_means, output_alphas

    @property
    def num_parameters(self) -> int:
        return 2
