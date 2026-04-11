"""
Likelihood Models
-----------------

Likelihood models for producing probabilistic forecasts.
"""

import importlib

from darts.logging import get_logger
from darts.utils.utils import NotImportedModule

logger = get_logger(__name__)

_TORCH_LIKELIHOOD_NAMES = [
    "BernoulliLikelihood",
    "BetaLikelihood",
    "CauchyLikelihood",
    "ContinuousBernoulliLikelihood",
    "DirichletLikelihood",
    "ExponentialLikelihood",
    "GammaLikelihood",
    "GaussianLikelihood",
    "GeometricLikelihood",
    "GumbelLikelihood",
    "HalfNormalLikelihood",
    "LaplaceLikelihood",
    "LogNormalLikelihood",
    "NegativeBinomialLikelihood",
    "PoissonLikelihood",
    "QuantileRegression",
    "WeibullLikelihood",
]

__all__ = list(_TORCH_LIKELIHOOD_NAMES)


def __getattr__(name: str):
    if name in _TORCH_LIKELIHOOD_NAMES:
        try:
            mod = importlib.import_module("darts.utils.likelihood_models.torch")
            return getattr(mod, name)
        except ModuleNotFoundError:
            logger.warning(
                "Support for PyTorch based likelihood models not available. "
                'To enable them, install "darts[torch]" or "darts[all]" (with pip); '
                'or "u8darts-torch" or "u8darts-all" (with conda).'
            )
            return NotImportedModule(module_name="(Py)Torch", warn=False)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
