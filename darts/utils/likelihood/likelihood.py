from enum import Enum

from darts import TimeSeries


class LikelihoodType(Enum):
    Gaussian = "gaussian"
    Poisson = "poisson"
    NegativeBinomial = "negativebinomial"
    Bernoulli = "bernoulli"
    Beta = "beta"
    Cauchy = "cauchy"
    ContinuousBernoulli = "continuousbernoulli"
    Dirichlet = "dirichlet"
    Exponential = "exponential"
    Gamma = "gamma"
    Geometric = "geometric"
    Gumbel = "gumbel"
    HalfNormal = "halfnormal"
    Laplace = "laplace"
    LogNormal = "lognormal"
    Weibull = "weibull"
    Quantile = "quantile"


class BaseLikelihood:
    def __init__(
        self,
        likelihood_type: LikelihoodType,
        parameter_names: list[str],
    ):
        """
        Base class for all likelihoods.

        * torch `Likelihood`
        * simple likelihoods (e.g. for regression)
        """
        self._likelihood_type = likelihood_type
        self._parameter_names = parameter_names

    def likelihood_components_names(self, input_series: TimeSeries) -> list[str]:
        """Generates names for the parameters of the Likelihood."""
        return [
            f"{tgt_name}_{param_n}"
            for tgt_name in input_series.components
            for param_n in self.parameter_names
        ]

    @property
    def likelihood_type(self) -> LikelihoodType:
        """Returns the likelihood type."""
        return self._likelihood_type

    @property
    def parameter_names(self) -> list[str]:
        """Returns the likelihood parameter names."""
        return self._parameter_names

    @property
    def num_parameters(self) -> int:
        """Returns the number of distribution parameters for a single target value."""
        return len(self.parameter_names)

    @property
    def simplified_name(self) -> str:
        """Returns the simplified likelihood name."""
        return self._likelihood_type.value

    @property
    def supports_parameter_autoregression(self) -> bool:
        return self.likelihood_type is LikelihoodType.Quantile
