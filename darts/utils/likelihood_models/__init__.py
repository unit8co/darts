"""
Likelihood Models
-----------------

Likelihood models for producing probabilistic forecasts.
"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.utils.likelihood_models.torch import (
        BernoulliLikelihood as BernoulliLikelihood,
    )
    from darts.utils.likelihood_models.torch import BetaLikelihood as BetaLikelihood
    from darts.utils.likelihood_models.torch import CauchyLikelihood as CauchyLikelihood
    from darts.utils.likelihood_models.torch import (
        ContinuousBernoulliLikelihood as ContinuousBernoulliLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        DirichletLikelihood as DirichletLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        ExponentialLikelihood as ExponentialLikelihood,
    )
    from darts.utils.likelihood_models.torch import GammaLikelihood as GammaLikelihood
    from darts.utils.likelihood_models.torch import (
        GaussianLikelihood as GaussianLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        GeometricLikelihood as GeometricLikelihood,
    )
    from darts.utils.likelihood_models.torch import GumbelLikelihood as GumbelLikelihood
    from darts.utils.likelihood_models.torch import (
        HalfNormalLikelihood as HalfNormalLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        LaplaceLikelihood as LaplaceLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        LogNormalLikelihood as LogNormalLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        NegativeBinomialLikelihood as NegativeBinomialLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        PoissonLikelihood as PoissonLikelihood,
    )
    from darts.utils.likelihood_models.torch import (
        QuantileRegression as QuantileRegression,
    )
    from darts.utils.likelihood_models.torch import (
        WeibullLikelihood as WeibullLikelihood,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "BernoulliLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "BetaLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "CauchyLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "ContinuousBernoulliLikelihood": (
        "darts.utils.likelihood_models.torch",
        "(Py)Torch",
    ),
    "DirichletLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "ExponentialLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "GammaLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "GaussianLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "GeometricLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "GumbelLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "HalfNormalLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "LaplaceLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "LogNormalLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "NegativeBinomialLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "PoissonLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "QuantileRegression": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
    "WeibullLikelihood": ("darts.utils.likelihood_models.torch", "(Py)Torch"),
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
