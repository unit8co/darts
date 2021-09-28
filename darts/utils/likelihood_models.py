"""
Likelihood Models
-----------------

The likelihood models contain all the logic needed to train and use Darts' neural network models in
a probabilistic way. This essentially means computing an appropriate training loss and sample from the
distribution, given the parameters of the distribution.

By default, all versions will be trained using their negative log likelihood as a loss function
(hence performing maximum likelihood estimation when training the model).
However, most likelihoods also optionally support specifying time-independent "prior"
beliefs about the distribution parameters.
In such cases, the a KL-divergence term is added to the loss in order to regularise it in the
direction of the specified prior distribution. (Note that this is technically not purely
a Bayesian approach as the priors are actual parameters values, and not distributions).
The parameter `prior_strength` controls the strength of the "prior" regularisation on the loss.

Some distributions (such as ``GaussianLikelihood``, and ``PoissonLikelihood``) are univariate,
in which case they are applied to model each component of multivariate series independently.
Some other distributions (such as ``DirichletLikelihood``) are multivariate,
in which case they will model all components of multivariate time series jointly.

Univariate likelihoods accept either scalar or array-like values for the optional prior parameters.
If a scalar is provided, it is used as a prior for all components of the series. If an array-like is provided,
the i-th value will be used as a prior for the i-th component of the series. Multivariate likelihoods
require array-like objects when specifying priors.

The target series used for training must always lie within the distribution's support, otherwise
errors will be raised during training. You can refer to the individual likelihoods' documentation
to see what is the support. Similarly, the prior parameters also have to lie in some pre-defined domains.
"""

# TODO: Table on README listing distribution, possible priors and wiki article
from darts.utils.utils import raise_if_not

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import collections

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import (Normal as _Normal,
                                 Poisson as _Poisson,
                                 NegativeBinomial as _NegativeBinomial,
                                 Bernoulli as _Bernoulli,
                                 Gamma as _Gamma,
                                 Gumbel as _Gumbel,
                                 Laplace as _Laplace,
                                 Beta as _Beta,
                                 Exponential as _Exponential,
                                 MultivariateNormal as _MultivariateNormal,
                                 Dirichlet as _Dirichlet,
                                 Geometric as _Geometric,
                                 Cauchy as _Cauchy,
                                 ContinuousBernoulli as _ContinuousBernoulli,
                                 HalfNormal as _HalfNormal,
                                 LogNormal as _LogNormal,
                                 Weibull as _Weibull
                                 )

MIN_CAUCHY_GAMMA_SAMPLING = 1e-100

# Some utils for checking parameters' domains
def _check(param, predicate, param_name, condition_str):
    if param is None: return
    if isinstance(param, (collections.Sequence, np.ndarray)):
        raise_if_not(all(predicate(p) for p in param),
                     'All provided parameters {} must be {}.'.format(param_name, condition_str))
    else:
        raise_if_not(predicate(param), 'The parameter {} must be {}.'.format(param_name, condition_str))


def _check_strict_positive(param, param_name=''):
    _check(param, lambda p: p > 0, param_name, 'strictly positive')


def _check_in_open_0_1_intvl(param, param_name=''):
    _check(param, lambda p: 0 < p < 1, param_name, 'in the open interval (0, 1)')


class Likelihood(ABC):
    def __init__(self, prior_strength=1.):
        """
        Abstract class for a likelihood model.
        """
        self.prior_strength = prior_strength

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        """
        Computes a loss from a `model_output`, which represents the parameters of a given probability
        distribution for every ground truth value in `target`, and the `target` itself.
        """
        params_out = self._params_from_output(model_output)
        loss = self._nllloss(params_out, target)

        prior_params = self._prior_params
        use_prior = prior_params is not None and any(p is not None for p in prior_params)
        if use_prior:
            out_distr = self._distr_from_params(params_out)
            device = params_out[0].device
            prior_params = tuple([
                # use model output as "prior" for parameters not specified as prior
                torch.tensor(prior_params[i]).to(device) if prior_params[i] is not None else params_out[i]
                for i in range(len(prior_params))
            ])
            prior_distr = self._distr_from_params(prior_params)

            # Loss regularization using the prior distribution
            loss += self.prior_strength * torch.mean(kl_divergence(prior_distr, out_distr))

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


class GaussianLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.):
        """
        Univariate Gaussian distribution.

        https://en.wikipedia.org/wiki/Normal_distribution

        - Univariate continuous distribution.
        - Support: :math:`\mathbb{R}`.
        - Parameters: mean :math:`\\mu \in \mathbb{R}`, standard deviation :math:`\\sigma > 0`.

        Parameters
        ----------
        prior_mu
            mean of the prior Gaussian distribution (default: None).
        prior_sigma
            standard deviation (or scale) of the prior Gaussian distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, 'sigma')

        self.nllloss = nn.GaussianNLLLoss(reduction='mean', full=True)
        self.softplus = nn.Softplus()

        super().__init__(prior_strength)

    def _nllloss(self, params_out, target):
        means_out, sigmas_out = params_out
        return self.nllloss(means_out.contiguous(), target.contiguous(), sigmas_out.contiguous())

    @property
    def _prior_params(self):
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
    def __init__(self, prior_lambda=None, prior_strength=1.):
        """
        Poisson distribution. Can typically be used to model event counts during time intervals, when the events
        happen independently of the time since the last event.

        https://en.wikipedia.org/wiki/Poisson_distribution

        - Univariate discrete distribution
        - Support: :math:`\mathbb{N}_0` (natural numbers including 0).
        - Parameter: rate :math:`\\lambda > 0`.

        Parameters
        ----------
        prior_lambda
            rate :math:`\\lambda` of the prior Poisson distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_lambda = prior_lambda
        _check_strict_positive(self.prior_lambda, 'lambda')

        self.nllloss = nn.PoissonNLLLoss(log_input=False, full=True)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    def _nllloss(self, params_out, target):
        lambda_out = params_out
        return self.nllloss(lambda_out, target)

    @property
    def _prior_params(self):
        return self.prior_lambda,

    def _distr_from_params(self, params):
        lmbda = params[0]
        return _Poisson(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_lambda = self._params_from_output(model_output)
        return torch.poisson(model_lambda)

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output):
        lmbda = self.softplus(model_output)
        return lmbda


class NegativeBinomialLikelihood(Likelihood):
    def __init__(self):
        """
        Negative Binomial distribution.

        https://en.wikipedia.org/wiki/Negative_binomial_distribution

        It does not support priors.

        - Univariate discrete distribution.
        - Support: :math:`\mathbb{N}_0` (natural numbers including 0).
        - Parameters: number of failures :math:`r > 0`, success probability :math:`p \in (0, 1)`.

        Behind the scenes the distribution is reparameterized so that the actual outputs of the
        network are in terms of the mean :math:`\\mu` and shape :math:`\\alpha`.
        """
        self.softplus = nn.Softplus()
        super().__init__()

    @property
    def _prior_params(self):
        return None

    @staticmethod
    def _get_r_and_p_from_mu_and_alpha(mu, alpha):
        # See https://en.wikipedia.org/wiki/Negative_binomial_distribution for the different parametrizations
        r = 1. / alpha
        p = r / (mu + r)
        return r, p

    def _distr_from_params(self, params):
        mu, alpha = params
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        return _NegativeBinomial(r, p)

    def sample(self, model_output: torch.Tensor):
        mu, alpha = self._params_from_output(model_output)
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        distr = _NegativeBinomial(r, p)
        return distr.sample()

    def _params_from_output(self, model_output):
        output_size = model_output.shape[-1]
        mu = self.softplus(model_output[:, :, :output_size // 2])
        alpha = self.softplus(model_output[:, :, output_size // 2:])
        return mu, alpha

    @property
    def num_parameters(self) -> int:
        return 2


class BernoulliLikelihood(Likelihood):
    def __init__(self, prior_p=None, prior_strength=1.):
        """
        Bernoulli distribution.

        https://en.wikipedia.org/wiki/Bernoulli_distribution

        - Univariate discrete distribution.
        - Support: :math:`\{0, 1\}`.
        - Parameter: probability :math:`p \in (0, 1)`.

        Parameters
        ----------
        prior_p
            probability :math:`p` of the prior Bernoulli distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_p = prior_p
        _check_in_open_0_1_intvl(self.prior_p, 'p')

        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_p,

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


class BetaLikelihood(Likelihood):
    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.):
        """
        Beta distribution.

        https://en.wikipedia.org/wiki/Beta_distribution

        - Univariate continuous distribution.
        - Support: open interval :math:`(0,1)`
        - Parameters: shape parameters :math:`\\alpha > 0` and :math:`\\beta > 0`.

        Parameters
        ----------
        prior_alpha
            shape parameter :math:`\\alpha` of the Beta distribution, strictly positive (default: None)
        prior_beta
            shape parameter :math:`\\beta` distribution, strictly positive (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, 'alpha')
        _check_strict_positive(self.prior_beta, 'beta')

        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_alpha, self.prior_beta

    def _distr_from_params(self, params):
        alpha, beta = params
        return _Beta(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._params_from_output(model_output)
        distr = _Beta(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        output_size = model_output.shape[-1]
        alpha = self.softplus(model_output[:, :, :output_size // 2])
        beta = self.softplus(model_output[:, :, output_size // 2:])
        return alpha, beta


class CauchyLikelihood(Likelihood):
    def __init__(self, prior_xzero=None, prior_gamma=None, prior_strength=1.):
        """
        Cauchy Distribution.

        https://en.wikipedia.org/wiki/Cauchy_distribution

        - Univariate continuous distribution.
        - Support: :math:`\mathbb{R}`.
        - Parameters: location :math:`x_0 \in \mathbb{R}`, scale :math:`\gamma > 0`.

        Due to its fat tails, this distribution is typically harder to estimate,
        and your mileage may vary. Also be aware that it typically
        requires a large value for `num_samples` for sampling predictions.

        Parameters
        ----------
        prior_xzero
            location parameter :math:`x_0` of the Cauchy distribution (default: None)
        prior_gamma
            scale parameter :math:`\\gamma` of the Cauchy distribution, strictly positive (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_xzero = prior_xzero
        self.prior_gamma = prior_gamma
        _check_strict_positive(self.prior_gamma, 'gamma')

        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_xzero, self.prior_gamma

    def _distr_from_params(self, params):
        xzero, gamma = params
        return _Cauchy(xzero, gamma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        xzero, gamma = self._params_from_output(model_output)

        # We need this hack as sometimes the output of the softplus is 0 in practice for Cauchy...
        gamma[gamma < MIN_CAUCHY_GAMMA_SAMPLING] = MIN_CAUCHY_GAMMA_SAMPLING

        distr = _Cauchy(xzero, gamma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        output_size = model_output.shape[-1]
        xzero = model_output[:, :, :output_size // 2]
        gamma = self.softplus(model_output[:, :, output_size // 2:])
        return xzero, gamma


class ContinuousBernoulliLikelihood(Likelihood):
    def __init__(self, prior_lambda=None, prior_strength=1.):
        """
        Continuous Bernoulli distribution.

        https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution

        - Univariate continuous distribution.
        - Support: open interval :math:`(0, 1)`.
        - Parameter: shape :math:`\\lambda \in (0,1)`

        Parameters
        ----------
        prior_lambda
            shape :math:`\\lambda` of the prior Continuous Bernoulli distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_lambda = prior_lambda
        _check_in_open_0_1_intvl(self.prior_lambda, 'lambda')

        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_lambda,

    def _distr_from_params(self, params):
        lmbda = params[0]
        return _ContinuousBernoulli(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_lmbda = self._params_from_output(model_output)
        distr = _ContinuousBernoulli(model_lmbda)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        lmbda = self.sigmoid(model_output)
        return lmbda


class DirichletLikelihood(Likelihood):
    def __init__(self, prior_alphas=None, prior_strength=1.):
        """
        Dirichlet distribution.

        https://en.wikipedia.org/wiki/Dirichlet_distribution

        - Multivariate continuous distribution, modeling all components of a time series jointly.
        - Support: The :math:`K`-dimensional simplex for series of dimension :math:`K`, i.e.,
          :math:`x_1, ..., x_K \\text{ with } x_i \in (0,1),\\; \\sum_i^K{x_i}=1`.
        - Parameter: concentrations :math:`\\alpha_1, ..., \\alpha_K` with :math:`\\alpha_i > 0`.

        Parameters
        ----------
        prior_alphas
            concentrations parameters :math:`\\alpha` of the prior Dirichlet distribution.
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_alphas = prior_alphas
        _check_strict_positive(self.prior_alphas)
        self.softmax = nn.Softmax(dim=2)
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_alphas,

    def _distr_from_params(self, params: Tuple):
        alphas = params[0]
        return _Dirichlet(alphas)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        alphas = self._params_from_output(model_output)
        distr = _Dirichlet(alphas)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1  # 1 parameter per component

    def _params_from_output(self, model_output):
        alphas = self.softmax(model_output)  # take softmax over components
        return alphas


class ExponentialLikelihood(Likelihood):
    def __init__(self, prior_lambda=None, prior_strength=1.):
        """
        Exponential distribution.

        https://en.wikipedia.org/wiki/Exponential_distribution

        - Univariate continuous distribution.
        - Support: :math:`\mathbb{R}_{>0}`.
        - Parameter: rate :math:`\\lambda > 0`.

        Parameters
        ----------
        prior_lambda
            rate :math:`\\lambda` of the prior exponential distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_lambda = prior_lambda
        _check_strict_positive(self.prior_lambda, 'lambda')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_lambda,

    def _distr_from_params(self, params: Tuple):
        lmbda = params[0]
        return _Exponential(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        lmbda = self._params_from_output(model_output)
        distr = _Exponential(lmbda)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        lmbda = self.softplus(model_output)
        return lmbda


class GammaLikelihood(Likelihood):
    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.):
        """
        Gamma distribution.

        https://en.wikipedia.org/wiki/Gamma_distribution

        - Univariate continuous distribution
        - Support: :math:`\mathbb{R}_{>0}`.
        - Parameters: shape :math:`\\alpha > 0` and rate :math:`\\beta > 0`.

        Parameters
        ----------
        prior_alpha
            shape :math:`\\alpha` of the prior gamma distribution (default: None).
        prior_beta
            rate :math:`\\beta` of the prior gamma distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, 'alpha')
        _check_strict_positive(self.prior_beta, 'beta')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_alpha, self.prior_beta

    def _distr_from_params(self, params: Tuple):
        alpha, beta = params
        return _Gamma(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._params_from_output(model_output)
        distr = _Gamma(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        output_size = model_output.shape[-1]
        alpha = self.softplus(model_output[:, :, :output_size // 2])
        beta = self.softplus(model_output[:, :, output_size // 2:])
        return alpha, beta


class GeometricLikelihood(Likelihood):
    def __init__(self, prior_p=None, prior_strength=1.):
        """
        Geometric distribution.

        https://en.wikipedia.org/wiki/Geometric_distribution

        - Univariate discrete distribution
        - Support: :math:`\mathbb{N}_0` (natural numbers including 0).
        - Parameter: success probability :math:`p \in (0, 1)`.

        Parameters
        ----------
        prior_p
            success probability :math:`p` of the prior geometric distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_p = prior_p
        _check_in_open_0_1_intvl(self.prior_p, 'p')
        self.sigmoid = nn.Sigmoid()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_p,

    def _distr_from_params(self, params: Tuple):
        p = params[0]
        return _Geometric(p)

    def sample(self, model_output):
        p = self._params_from_output(model_output)
        distr = _Geometric(p)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        p = self.sigmoid(model_output)
        return p


class GumbelLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_beta=None, prior_strength=1.):
        """
        Gumbel distribution.

        https://en.wikipedia.org/wiki/Gumbel_distribution

        - Univariate continuous distribution
        - Support: :math:`\mathbb{R}`.
        - Parameters: location :math:`\\mu \in \mathbb{R}` and scale :math:`\\beta > 0`.

        Parameters
        ----------
        prior_mu
            location :math:`\\mu` of the prior Gumbel distribution (default: None).
        prior_beta
            scale :math:`\\beta` of the prior Gumbel distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_beta)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_beta

    def _distr_from_params(self, params: Tuple):
        mu, beta = params
        return _Gumbel(mu, beta)

    def sample(self, model_output):
        mu, beta = self._params_from_output(model_output)
        distr = _Gumbel(mu, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        output_size = model_output.shape[-1]
        mu = model_output[:, :, :output_size // 2]
        beta = self.softplus(model_output[:, :, output_size // 2:])
        return mu, beta


class HalfNormalLikelihood(Likelihood):
    def __init__(self, prior_sigma=None, prior_strength=1.):
        """
        Half-normal distribution.

        https://en.wikipedia.org/wiki/Half-normal_distribution

        - Univariate continuous distribution.
        - Support: :math:`\mathbb{R}_{>0}`.
        - Parameter: rate :math:`\\sigma > 0`.

        Parameters
        ----------
        prior_sigma
            standard deviation :math:`\\sigma` of the prior half-normal distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, 'sigma')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_sigma,

    def _distr_from_params(self, params: Tuple):
        sigma = params[0]
        return _HalfNormal(sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        sigma = self._params_from_output(model_output)
        distr = _HalfNormal(sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        sigma = self.softplus(model_output)
        return sigma


class LaplaceLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_b=None, prior_strength=1.):
        """
        Laplace distribution.

        https://en.wikipedia.org/wiki/Laplace_distribution

        - Univariate continuous distribution
        - Support: :math:`\mathbb{R}`.
        - Parameters: location :math:`\\mu \in \mathbb{R}` and scale :math:`b > 0`.

        Parameters
        ----------
        prior_mu
            location :math:`\\mu` of the prior Laplace distribution (default: None).
        prior_b
            scale :math:`b` of the prior Laplace distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_b = prior_b
        _check_strict_positive(self.prior_b)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_b

    def _distr_from_params(self, params: Tuple):
        mu, b = params
        return _Laplace(mu, b)

    def sample(self, model_output):
        mu, b = self._params_from_output(model_output)
        distr = _Laplace(mu, b)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        output_size = model_output.shape[-1]
        mu = model_output[:, :, :output_size // 2]
        b = self.softplus(model_output[:, :, output_size // 2:])
        return mu, b


class LogNormalLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.):
        """
        Log-normal distribution.

        https://en.wikipedia.org/wiki/Log-normal_distribution

        - Univariate continuous distribution.
        - Support: :math:`\mathbb{R}_{>0}`.
        - Parameters: :math:`\\mu \in \mathbb{R}` and :math:`\\sigma > 0`.

        Parameters
        ----------
        prior_mu
            parameter :math:`\\mu` of the prior log-normal distribution (default: None).
        prior_sigma
            parameter :math:`\\sigma` of the prior log-normal distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, 'sigma')
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_sigma

    def _distr_from_params(self, params):
        mu, sigma = params
        return _LogNormal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        distr = _LogNormal(mu, sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        output_size = model_output.shape[-1]
        mu = model_output[:, :, :output_size // 2]
        sigma = self.softplus(model_output[:, :, output_size // 2:])
        return mu, sigma


class WeibullLikelihood(Likelihood):
    def __init__(self, prior_strength=1.):
        """
        Weibull distribution.

        https://en.wikipedia.org/wiki/Weibull_distribution

        - Univariate continuous distribution
        - Support: :math:`\mathbb{R}_{>0}`.
        - Parameters: scale :math:`\\lambda > 0` and concentration :math:`k > 0`.

        It does not support priors.

        Parameters
        ----------
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return None

    def _distr_from_params(self, params: Tuple):
        lmba, k = params
        return _Weibull(lmba, k)

    def sample(self, model_output):
        lmbda, k = self._params_from_output(model_output)
        distr = _Weibull(lmbda, k)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        output_size = model_output.shape[-1]
        lmbda = self.softplus(model_output[:, :, :output_size // 2])
        k = self.softplus(model_output[:, :, output_size // 2:])
        return lmbda, k


if False:
    """ TODO
        To make it work, we'll have to change our models so they optionally accept an absolute
        number of parameters, instead of num_perameters per component.
    """

    class MultivariateNormal(Likelihood):
        def __init__(self, dim: int, prior_mu=None, prior_covmat=None, prior_strength=1.):
            self.dim = dim
            self.prior_mu = prior_mu
            self.prior_covmat = prior_covmat
            if self.prior_mu is not None:
                raise_if_not(len(self.prior_mu) == self.dim, 'The provided prior_mu must have a size matching the '
                                                             'provided dimension.')
            if self.prior_covmat is not None:
                raise_if_not(self.prior_covmat.shape == (self.dim, self.dim), 'The provided prior on the covariaance '
                                                                              'matrix must have size (dim, dim).')
                _check_strict_positive(self.prior_covmat.flatten(), 'covariance matrix')

            self.softplus = nn.Softplus()
            super().__init__(prior_strength)

        @property
        def _prior_params(self):
            return self.prior_mu, self.prior_covmat

        def _distr_from_params(self, params: Tuple):
            mu, covmat = params
            return _MultivariateNormal(mu, covmat)

        def sample(self, model_output: torch.Tensor):
            mu, covmat = self._params_from_output(model_output)
            distr = _MultivariateNormal(mu, covmat)
            return distr.sample()

        @property
        def num_parameters(self) -> int:
            return int(self.dim + (self.dim**2 - self.dim) / 2)

        def _params_from_output(self, model_output: torch.Tensor):
            device = model_output.device
            mu = model_output[:, :, :self.dim]
            covmat_coefs = self.softplus(model_output[:, :, self.dim:])

            print('model output: {}'.format(model_output.shape))

            # build covariance matrix
            covmat = torch.zeros((model_output.shape[0], model_output.shape[1], self.dim, self.dim)).to(device)
            tril_indices = torch.tril_indices(row=self.dim, col=self.dim, offset=1, device=device)
            covmat[tril_indices[0], tril_indices[1]] = covmat_coefs
            covmat[tril_indices[1], tril_indices[0]] = covmat_coefs
            covmat[range(self.dim), range(self.dim)] = 1.

