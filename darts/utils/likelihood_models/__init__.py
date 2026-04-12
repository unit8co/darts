"""
Likelihood Models
-----------------

Likelihood models for producing probabilistic forecasts.
"""

from darts.utils._lazy import setup_lazy_imports

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
