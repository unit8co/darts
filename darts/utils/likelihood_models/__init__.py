"""
Likelihood Models
-----------------
"""

from darts.logging import get_logger
from darts.utils.utils import NotImportedModule

logger = get_logger(__name__)

# allow direct import of torch likelihoods as they are the only ones useful to the user
try:
    from darts.utils.likelihood_models.torch import (
        BernoulliLikelihood,
        BetaLikelihood,
        CauchyLikelihood,
        ContinuousBernoulliLikelihood,
        DirichletLikelihood,
        ExponentialLikelihood,
        GammaLikelihood,
        GaussianLikelihood,
        GeometricLikelihood,
        GumbelLikelihood,
        HalfNormalLikelihood,
        LaplaceLikelihood,
        LogNormalLikelihood,
        NegativeBinomialLikelihood,
        PoissonLikelihood,
        QuantileRegression,
        WeibullLikelihood,
    )
except ModuleNotFoundError:
    logger.warning(
        "Support for PyTorch based likelihood models not available. "
        'To enable them, install "darts", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )
    BernoulliLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    BetaLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    CauchyLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    ContinuousBernoulliLikelihood = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    DirichletLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    ExponentialLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    GammaLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    GaussianLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    GeometricLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    GumbelLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    HalfNormalLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    LaplaceLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    LogNormalLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    NegativeBinomialLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    PoissonLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)
    QuantileRegression = NotImportedModule(module_name="(Py)Torch", warn=False)
    WeibullLikelihood = NotImportedModule(module_name="(Py)Torch", warn=False)


__all__ = [
    "GaussianLikelihood",
    "PoissonLikelihood",
    "NegativeBinomialLikelihood",
    "BernoulliLikelihood",
    "BetaLikelihood",
    "CauchyLikelihood",
    "ContinuousBernoulliLikelihood",
    "DirichletLikelihood",
    "ExponentialLikelihood",
    "GammaLikelihood",
    "GeometricLikelihood",
    "GumbelLikelihood",
    "HalfNormalLikelihood",
    "LaplaceLikelihood",
    "LogNormalLikelihood",
    "WeibullLikelihood",
    "QuantileRegression",
]
