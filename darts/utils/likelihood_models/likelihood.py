import inspect
from collections.abc import Sequence
from enum import Enum
from typing import Optional, Union

import pandas as pd

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

        Parameters
        ----------
        likelihood_type
            A pre-defined `LikelihoodType`.
        parameter_names
            The likelihood (distribution) parameter names.
        """
        self._likelihood_type = likelihood_type
        self._parameter_names = parameter_names

    def likelihood_components_names(self, input_series: TimeSeries) -> list[str]:
        """Generates names for the parameters of the Likelihood."""
        return likelihood_component_names(
            components=input_series.components, parameter_names=self.parameter_names
        )

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

    def __repr__(self) -> str:
        """Return the class and parameters of the instance in a nice format"""
        cls_name = self.__class__.__name__
        # only display the constructor parameters as user cannot change the other attributes
        init_signature = inspect.signature(self.__class__.__init__)
        params_string = ", ".join([
            f"{str(v)}"
            for _, v in init_signature.parameters.items()
            if str(v) != "self"
        ])
        return f"{cls_name}({params_string})"


def likelihood_component_names(
    components: Union[pd.Index, list[str]], parameter_names: list[str]
):
    """Generates formatted likelihood parameter names for components and parameter names.

    The order of the returned names is: `[comp1_param_1, ... comp1_param_n, ..., comp_n_param_n]`.

    Parameters
    ----------
    components
        A sequence of component names to add to the beginning of the returned names.
    parameter_names
        A sequence of likelihood parameter names to add to the end of the returned names.
    """
    return [
        f"{tgt_name}_{param_n}"
        for tgt_name in components
        for param_n in parameter_names
    ]


def quantile_names(q: Union[float, list[float]], component: Optional[str] = None):
    """Generates formatted quantile names, optionally added to a component name.

    Parameters
    ----------
    q
        A float or list of floats with the quantiles to generate the names for.
    component
        Optionally, a component name to add to the beginning of the quantile names.
    """
    # predicted quantile text format
    comp = f"{component}_" if component is not None else ""
    if isinstance(q, float):
        return f"{comp}q{q:.2f}"
    else:
        return [f"{comp}q{q_i:.2f}" for q_i in q]


def quantile_interval_names(
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]],
    component: Optional[str] = None,
):
    """Generates formatted quantile interval names, optionally added to a component name.

    Parameters
    ----------
    q_interval
        A tuple or multiple tuples with the (lower bound, upper bound) of the quantile intervals.
    component
        Optionally, a component name to add to the beginning of the quantile names.
    """
    # predicted quantile text format
    comp = f"{component}_" if component is not None else ""
    if isinstance(q_interval, tuple):
        return f"{comp}q{q_interval[0]:.2f}_q{q_interval[1]:.2f}"
    else:
        return [f"{comp}q{q_lo:.2f}_q{q_hi:.2f}" for q_lo, q_hi in q_interval]
