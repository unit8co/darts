from itertools import combinations

import numpy as np

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
        samples = np.array([lkl.sample(model_output)[0, 0, 0] for _ in range(20000)])
        np.testing.assert_allclose(samples.std(), reported_sigma, rtol=0.05)

    def test_predict_likelihood_parameters_does_not_mutate_input(self):
        lkl = GaussianLikelihood(n_outputs=1)
        model_output = np.array([[[2.0, 9.0]]])
        expected = model_output.copy()

        lkl.predict_likelihood_parameters(model_output)

        np.testing.assert_array_equal(model_output, expected)
