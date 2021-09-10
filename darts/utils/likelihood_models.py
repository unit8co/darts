"""
Likelihood Models
-----------------
"""

# TODO: Table on README listing distribution, possible priors and wiki article
from darts.utils.utils import raise_if_not

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import (Normal as _Normal,
                                 Poisson as _Poisson,
                                 NegativeBinomial as _NegativeBinomial,
                                 Bernoulli as _Bernoulli,
                                 Laplace as _Laplace,
                                 Beta as _Beta,
                                 Exponential as _Exponential,
                                 MultivariateNormal as _MultivariateNormal,
                                 Dirichlet as _Dirichlet,
                                 Geometric as _Geometric,
                                 Binomial as _Binomial,
                                 Cauchy as _Cauchy,
                                 ContinuousBernoulli as _ContinuousBernoulli,
                                 HalfNormal as _HalfNormal,
                                 LogNormal as _LogNormal,
                                 LowRankMultivariateNormal as _LowRankMultivariateNormal,  # scales?
                                 Pareto as _Pareto,
                                 Uniform as _Uniform,
                                 Weibull as _Weibull
                                 )


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

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor, ):
        """
        Computes a loss from a `model_output`, which represents the parameters of a given probability
        distribution for every ground truth value in `target`, and the `target` itself.
        """
        params_out = self._params_from_output(model_output)
        # out_distr = self._distr_from_params(params_out)
        # loss = -out_distr.log_prob(target).mean()
        loss = self._nllloss(params_out, target)

        prior_params = self._prior_params
        if prior_params is not None:
            out_distr = self._distr_from_params(params_out)
            device = params_out[0].device
            prior_params = tuple([
                # use model output as "prior" for parameters not specified as prior
                torch.tensor(prior_params[i]).to(device) if prior_params[i] is not None else params_out[i]
                for i in range(len(prior_params))
            ])
            prior_distr = self._distr_from_params(prior_params)

            # Loss regularization using the prior distribution
            loss += self.beta * torch.mean(kl_divergence(prior_distr, out_distr))

        return loss

    def _nllloss(self, params_out, target):
        """
        This is the basic way to compute the NLL loss. It can be overwritten by likelihoods for which
        PyTorch proposes a numerically better NLL loss.
        """
        out_distr = self._distr_from_params(params_out)
        return -out_distr.log_prob(target).mean()

    @property
    def _prior_params(self):
        """
        Has to be overwritten by the Likelihood objects supporting specifying a prior distribution on the
        outputs. If it returns None, no prior will be used and the model will be trained with plain maximum likelihood.
        """
        return None

    @abstractmethod
    def _distr_from_params(self, params: Tuple) -> torch.distributions.Distribution:
        """
        Returns a torch distribution built with the specified params
        """
        pass

    @abstractmethod
    def _params_from_output(self, model_output: torch.Tensor):
        """
        Returns the distribution parameters, obtained from the raw model outputs
        (e.g. applies softplus or sigmoids to get parameters in the expected domains).
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


class BernoulliLikelihood(Likelihood):
    def __init__(self, prior_p: Optional[float] = None, beta=1.):
        """
        Bernoulli Likelihood; can be used to model binary events in {0, 1}
        https://en.wikipedia.org/wiki/Bernoulli_distribution

        It is possible to specify a time-independent prior on the probability parameter `p` to capture a-priori
        knowledge about the process. Leaving it to `None` won't be using a prior,
        and corresponds to doing maximum likelihood.

        Parameters
        ----------
        prior_p
            probability `p` of the prior Bernoulli distribution, in (0, 1) (default: None)
        beta
            strength of the loss regularisation induced by the prior
        """
        self.prior_p = prior_p
        if self.prior_p is not None:
            raise_if_not(0 < self.prior_p < 1., 'The parameter p must be in the open interval (0, 1)')
        self.beta = beta

        self.sigmoid = nn.Sigmoid()
        super().__init__()

    @property
    def _prior_params(self):
        return (self.prior_p, ) if self.prior_p is not None else None

    def _distr_from_params(self, params):
        p = params[0]
        return _Bernoulli(p)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_p = self._params_from_output(model_output)
        return torch.bernoulli(model_p)

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        p = self.sigmoid(model_output)
        return p


class GaussianLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_sigma=None, beta=1.):
        """
        Univariate Gaussian Likelihood
        Components are modeled by separate univariate distributions, with optional time-independent priors.

        It is possible to specify a prior on mu or sigma only. Leaving both to `None` won't be using a prior,
        and corresponds to doing maximum likelihood.

        For the prior parameters: if a scalar value is provided, one value will be used as prior for all components,
        and if an array-like is provided, one value can be specified per component.

        Parameters
        ----------
        prior_mu
            mean of the prior Gaussian distribution (default: None).
        prior_sigma
            standard deviation (or scale) of the prior Gaussian distribution (default: None)
        beta
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.beta = beta

        self.nllloss = nn.GaussianNLLLoss(reduction='mean', full=True)
        self.softplus = nn.Softplus()

        super().__init__()

    def _nllloss(self, params_out, target):
        means_out, sigmas_out = params_out
        return self.nllloss(means_out.contiguous(), target.contiguous(), sigmas_out.contiguous())

    @property
    def _prior_params(self):
        # return None if no prior
        if self.prior_mu is None and self.prior_sigma is None:
            return None
        else:
            return self.prior_mu, self.prior_sigma

    def _distr_from_params(self, params):
        mu, sigma = params
        return _Normal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        return torch.normal(mu, sigma)

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        output_size = model_output.shape[-1]
        mu = model_output[:, :, :output_size // 2]
        sigma = self.softplus(model_output[:, :, output_size // 2:])
        return mu, sigma


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
            out_distr = _Poisson(lambda_out)
            prior_distr = _Poisson(torch.tensor(self.prior_lambda).to(lambda_out.device))
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
    def __init__(self):
        """
        Negative Binomial Likelihood.
        https://en.wikipedia.org/wiki/Negative_binomial_distribution

        It does not support priors.
        """
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
        out_distr = _NegativeBinomial(r_out, p_out)

        # take negative log likelihood as loss
        loss = -out_distr.log_prob(target).mean()
        return loss

    def sample(self, model_output: torch.Tensor):
        mu, alpha = self._means_and_alphas_from_model_output(model_output)
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        distr = _NegativeBinomial(r, p)
        return distr.sample()

    def _means_and_alphas_from_model_output(self, model_output):
        output_size = model_output.shape[-1]
        output_means = self.softplus(model_output[:, :, :output_size // 2])
        output_alphas = self.softplus(model_output[:, :, output_size // 2:])
        return output_means, output_alphas

    @property
    def num_parameters(self) -> int:
        return 2



