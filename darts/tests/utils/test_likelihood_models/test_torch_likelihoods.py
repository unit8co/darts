from itertools import combinations

import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import torch

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

# equality between likelihoods is only dependent on the main distribution parameters
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


class TestTorchLikelihoodModel:
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

    @pytest.mark.parametrize(
        "likelihood",
        [
            BernoulliLikelihood(),
            BernoulliLikelihood(prior_p=0.5),
            BetaLikelihood(),
            CauchyLikelihood(),
            ContinuousBernoulliLikelihood(),
            DirichletLikelihood(),
            DirichletLikelihood(prior_alphas=[1.0, 2.0]),
            ExponentialLikelihood(),
            ExponentialLikelihood(prior_lambda=0.5),
            GammaLikelihood(),
            GaussianLikelihood(),
            GaussianLikelihood(prior_mu=0.0, prior_sigma=1.0),
            GeometricLikelihood(),
            GumbelLikelihood(),
            HalfNormalLikelihood(),
            LaplaceLikelihood(),
            LogNormalLikelihood(),
            NegativeBinomialLikelihood(),
            PoissonLikelihood(),
            PoissonLikelihood(prior_lambda=2.0),
            QuantileRegression([0.1, 0.5, 0.9]),
            WeibullLikelihood(),
        ],
    )
    def test_compute_loss_uses_all_samples(self, likelihood):
        # the loss of a batch must be the average of the losses of its individual samples
        torch.manual_seed(42)
        batch_size, n_times, n_components = 4, 3, 2
        model_output = torch.randn(
            batch_size, n_times, n_components, likelihood.num_parameters
        )
        if isinstance(likelihood, QuantileRegression):
            target = torch.randn(batch_size, n_times, n_components)
        else:
            params = likelihood._as_tuple(likelihood._params_from_output(model_output))
            target = likelihood._distr_from_params(params).sample()

        loss = likelihood.compute_loss(model_output, target, None)
        loss_per_sample = torch.stack([
            likelihood.compute_loss(model_output[i : i + 1], target[i : i + 1], None)
            for i in range(batch_size)
        ])
        assert torch.allclose(loss, loss_per_sample.mean())


class TestTorchLikelihoodInputValidation:
    def test_gaussian_negative_prior_mu(self):
        with pytest.raises(ValueError, match="strictly positive"):
            GaussianLikelihood(prior_sigma=-1.0)

    def test_exponential_negative_lmbda(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ExponentialLikelihood(prior_lambda=-0.5)

    def test_beta_invalid_prior(self):
        with pytest.raises(ValueError, match="strictly positive"):
            BetaLikelihood(prior_alpha=-1.0)

    def test_gaussian_negative_prior_sigma_sequence(self):
        with pytest.raises(
            ValueError, match="All provided parameters.*strictly positive"
        ):
            GaussianLikelihood(prior_sigma=[-1.0, 1.0])

    def test_quantile_input_tensors(self):
        qs = [0.1, 0.5, 0.9]
        lkl = QuantileRegression(qs)

        output_shape = (4, 3, 2, len(qs))
        target_shape = (4, 3, 2)
        with pytest.raises(
            ValueError, match="mismatch between predicted and target shape."
        ):
            lkl.compute_loss(
                model_output=torch.zeros(output_shape[:-1]),
                target=torch.zeros(target_shape),
                sample_weight=torch.zeros(target_shape),
            )

        with pytest.raises(
            ValueError, match="mismatch between number of predicted quantiles."
        ):
            lkl.compute_loss(
                model_output=torch.zeros(output_shape[:-1] + (len(qs) - 1,)),
                target=torch.zeros(target_shape),
                sample_weight=torch.zeros(target_shape),
            )
