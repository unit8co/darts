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


class TestQuantileRegressionEstimatorPredict:
    @pytest.mark.parametrize(
        "raw_shape,quantiles",
        [
            ((4, 6, 3), [0.1, 0.5, 0.9]),
            ((2, 2, 3), [0.1, 0.5, 0.9]),
        ],
    )
    def test_estimator_predict_reshapes_multioutput_output(self, raw_shape, quantiles):
        k, n_flat, n_quantiles = raw_shape
        n_outputs = 2
        n_components = n_flat // n_outputs
        assert len(quantiles) == n_quantiles
        likelihood = QuantileRegression(n_outputs=n_outputs, quantiles=quantiles)
        raw = np.arange(k * n_flat * n_quantiles, dtype=float).reshape(
            k, n_flat, n_quantiles
        )

        class _Inner:
            def predict(self, x, **kwargs):
                return raw

        out = likelihood._estimator_predict(
            type("_M", (), {"model": _Inner()})(), np.zeros((k, 1))
        )
        expected = raw.reshape(k, n_outputs, n_components, n_quantiles)
        assert out.shape == expected.shape
        np.testing.assert_array_equal(out, expected)

    def test_estimator_predict_reshapes_2d_flat_output(self):
        likelihood = QuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9])
        k, n_targets, n_quantiles = 2, 3, 3
        raw = np.arange(k * n_targets * n_quantiles, dtype=float).reshape(k, -1)

        class _Inner:
            def predict(self, x, **kwargs):
                return raw

        out = likelihood._estimator_predict(
            type("_M", (), {"model": _Inner()})(), np.zeros((k, 1))
        )
        expected = raw.reshape(k, n_targets, n_quantiles).reshape(
            k, 1, n_targets, n_quantiles
        )
        np.testing.assert_array_equal(out, expected)

    def test_predict_likelihood_parameters_component_order(self):
        likelihood = QuantileRegression(n_outputs=2, quantiles=[0.1, 0.5, 0.9])
        k, n_times, n_components, n_quantiles = 2, 2, 2, 3
        model_output = np.arange(k * n_times * n_components * n_quantiles, dtype=float)
        model_output = model_output.reshape(k, n_times, n_components, n_quantiles)
        flat = likelihood.predict_likelihood_parameters(model_output)
        assert flat.shape == (k, n_times, n_components * n_quantiles)
        for c in range(n_components):
            np.testing.assert_array_equal(
                flat[:, :, c * n_quantiles : (c + 1) * n_quantiles],
                model_output[:, :, c, :],
            )


class TestSKLearnGaussianLikelihood:
    """The estimator (CatBoost's `RMSEWithUncertainty`) returns `[mu, variance]`,
    but the likelihood exposes the parameter as `sigma`."""

    def test_predict_likelihood_parameters_returns_standard_deviation(self):
        lkl = GaussianLikelihood(n_outputs=1)
        assert lkl.parameter_names == ["mu", "sigma"]

        # shape (n_components * output_chunk_length, n_series * n_samples, 2)
        model_output = np.array([[[2.0, 9.0], [3.0, 16.0]]])

        params = lkl.predict_likelihood_parameters(model_output)

        # the variances 9 and 16 must be reported as the deviations 3 and 4
        np.testing.assert_allclose(params, np.array([[[2.0, 3.0]], [[3.0, 4.0]]]))

    def test_predict_likelihood_parameters_agrees_with_sample(self):
        """`sample()` takes the square root before drawing; the reported `sigma`
        must be the same scale, otherwise the two paths disagree."""
        lkl = GaussianLikelihood(n_outputs=1)
        model_output = np.array([[[0.0, 25.0]]])

        reported_sigma = lkl.predict_likelihood_parameters(model_output)[0, 0, 1]

        np.random.seed(0)
        samples = lkl.sample(np.repeat(model_output, 5000, axis=0))
        np.testing.assert_allclose(samples.std(), reported_sigma, rtol=0.05)

    def test_predict_likelihood_parameters_does_not_mutate_input(self):
        lkl = GaussianLikelihood(n_outputs=1)
        model_output = np.array([[[2.0, 9.0]]])
        expected = model_output.copy()

        lkl.predict_likelihood_parameters(model_output)

        np.testing.assert_array_equal(model_output, expected)
