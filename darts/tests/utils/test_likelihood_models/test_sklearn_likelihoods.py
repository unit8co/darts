from itertools import combinations

import numpy as np
import pytest

from darts.utils.likelihood_models.sklearn import (
    GaussianLikelihood,
    MultiQuantileRegression,
    PoissonLikelihood,
    QuantileRegression,
)

# equality between likelihoods is only dependent on the main distribution parameters
likelihood_models_equal = {
    "quantile": [
        QuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9]),
        QuantileRegression(n_outputs=12, quantiles=[0.1, 0.5, 0.9]),
    ],
    "multiquantile": [
        MultiQuantileRegression(n_outputs=1, quantiles=[0.2, 0.5, 0.8]),
        MultiQuantileRegression(n_outputs=12, quantiles=[0.2, 0.5, 0.8]),
    ],
    "gaussian": [
        GaussianLikelihood(n_outputs=1),
        GaussianLikelihood(n_outputs=12),
    ],
    "poisson": [
        PoissonLikelihood(n_outputs=1),
        PoissonLikelihood(n_outputs=12),
    ],
}

likelihood_models_unequal = {
    "quantile": [
        QuantileRegression(n_outputs=1, quantiles=[0.2, 0.5, 0.8]),
        QuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9]),
    ],
    "multiquantile": [
        MultiQuantileRegression(n_outputs=1, quantiles=[0.2, 0.5, 0.8]),
        MultiQuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9]),
    ],
}


class TestSKLearnLikelihoodModel:
    @pytest.mark.parametrize("quantiles", [[0.5], [0.1, 0.5, 0.9]])
    def test_quantile_sampling(self, quantiles, monkeypatch):
        likelihood = QuantileRegression(n_outputs=3, quantiles=quantiles)
        probs = np.linspace(0, 1, 12).reshape(2, 3, 2, 1)
        monkeypatch.setattr(np.random, "uniform", lambda size: probs)
        offsets = np.arange(12).reshape(2, 3, 2, 1)
        model_output = offsets + 10 * np.array(quantiles)

        samples = likelihood.sample(model_output)

        expected = offsets.squeeze(-1) + 10 * np.clip(
            probs.squeeze(-1), quantiles[0], quantiles[-1]
        )
        assert samples.shape == (2, 3, 2)
        np.testing.assert_allclose(samples, expected)

    def test_intra_class_equality(self):
        for _, model_pair in likelihood_models_equal.items():
            assert model_pair[0] == model_pair[0]
            assert model_pair[1] == model_pair[1]
            assert model_pair[0] == model_pair[1]

        for _, model_pair in likelihood_models_unequal.items():
            assert model_pair[0] == model_pair[0]
            assert model_pair[1] == model_pair[1]
            assert model_pair[0] != model_pair[1]

    def test_inter_class_equality(self):
        model_combinations = combinations(likelihood_models_equal.keys(), 2)
        for first_model_name, second_model_name in model_combinations:
            assert (
                likelihood_models_equal[first_model_name][0]
                != likelihood_models_equal[second_model_name][0]
            )
