from itertools import combinations

from darts.utils.likelihood_models.sklearn import (
    GaussianLikelihood,
    PoissonLikelihood,
    QuantileRegression,
)

# equality between likelihoods is only dependent on the main distribution parameters
likelihood_models_equal = {
    "quantile": [
        QuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9], random_state=1),
        QuantileRegression(n_outputs=12, quantiles=[0.1, 0.5, 0.9], random_state=2),
    ],
    "gaussian": [
        GaussianLikelihood(n_outputs=1, random_state=1),
        GaussianLikelihood(n_outputs=12, random_state=2),
    ],
    "poisson": [
        PoissonLikelihood(n_outputs=1, random_state=1),
        PoissonLikelihood(n_outputs=12, random_state=2),
    ],
}

likelihood_models_unequal = {
    "quantile": [
        QuantileRegression(n_outputs=1, quantiles=[0.2, 0.5, 0.8], random_state=1),
        QuantileRegression(n_outputs=1, quantiles=[0.1, 0.5, 0.9], random_state=1),
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
