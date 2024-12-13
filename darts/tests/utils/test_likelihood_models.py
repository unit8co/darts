from itertools import combinations

import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
from darts.utils.likelihood_models import (
    BetaLikelihood,
    CauchyLikelihood,
    ExponentialLikelihood,
    GaussianLikelihood,
    PoissonLikelihood,
    QuantileRegression,
    WeibullLikelihood,
)

likelihood_models = {
    "quantile": [QuantileRegression(), QuantileRegression([0.25, 0.5, 0.75])],
    "gaussian": [
        GaussianLikelihood(prior_mu=0, prior_sigma=1),
        GaussianLikelihood(prior_mu=10, prior_sigma=1),
    ],
    "exponential": [
        ExponentialLikelihood(prior_lambda=0.1),
        ExponentialLikelihood(prior_lambda=0.5),
    ],
    "poisson": [
        PoissonLikelihood(prior_lambda=2),
        PoissonLikelihood(prior_lambda=5),
    ],
    "cauchy": [
        CauchyLikelihood(prior_xzero=-0.4, prior_gamma=2),
        CauchyLikelihood(prior_xzero=3, prior_gamma=2),
    ],
    "weibull": [
        WeibullLikelihood(prior_strength=1.0),
        WeibullLikelihood(prior_strength=0.8),
    ],
    "beta": [
        BetaLikelihood(prior_alpha=0.2, prior_beta=0.4, prior_strength=0.3),
        BetaLikelihood(prior_alpha=0.2, prior_beta=0.4, prior_strength=0.6),
    ],
}


class TestLikelihoodModel:
    def test_intra_class_equality(self):
        for _, model_pair in likelihood_models.items():
            assert model_pair[0] == model_pair[0]
            assert model_pair[1] == model_pair[1]
            assert model_pair[0] != model_pair[1]

    def test_inter_class_equality(self):
        model_combinations = combinations(likelihood_models.keys(), 2)
        for first_model_name, second_model_name in model_combinations:
            assert (
                likelihood_models[first_model_name][0]
                != likelihood_models[second_model_name][0]
            )
