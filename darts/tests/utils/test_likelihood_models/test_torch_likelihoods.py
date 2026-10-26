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
            params = likelihood._params_from_output(model_output)
            target = likelihood._distr_from_params(params).sample()

        loss = likelihood.compute_loss(model_output, target, None)
        loss_per_sample = torch.stack([
            likelihood.compute_loss(model_output[i : i + 1], target[i : i + 1], None)
            for i in range(batch_size)
        ])
        assert torch.allclose(loss, loss_per_sample.mean())

    @pytest.mark.parametrize(
        "likelihood",
        [
            BernoulliLikelihood(),
            BetaLikelihood(),
            CauchyLikelihood(),
            ContinuousBernoulliLikelihood(),
            ExponentialLikelihood(),
            GammaLikelihood(),
            GaussianLikelihood(),
            GeometricLikelihood(),
            GumbelLikelihood(),
            HalfNormalLikelihood(),
            LaplaceLikelihood(),
            LogNormalLikelihood(),
            NegativeBinomialLikelihood(),
            PoissonLikelihood(),
            QuantileRegression([0.1, 0.5, 0.9]),
            WeibullLikelihood(),
        ],
    )
    def test_predict_likelihood_parameters_component_order(self, likelihood):
        # parameters must be grouped by component, in the same order as `component_names()`
        # (<comp_0>_<param_0>, <comp_0>_<param_1>, ..., <comp_1>_<param_0>, ...)
        torch.manual_seed(42)
        n_components = 3
        model_output = torch.randn(2, 4, n_components, likelihood.num_parameters)

        params = likelihood.predict_likelihood_parameters(model_output)
        params_per_component = torch.cat(
            [
                likelihood.predict_likelihood_parameters(model_output[:, :, i : i + 1])
                for i in range(n_components)
            ],
            dim=-1,
        )
        assert params.shape == (2, 4, n_components * likelihood.num_parameters)
        assert torch.allclose(params, params_per_component)

    def test_negative_binomial_mean_and_variance(self):
        # the network outputs `mu` (mean) and `alpha` (dispersion): mean = mu, var = mu + alpha * mu**2
        likelihood = NegativeBinomialLikelihood()
        mu = torch.tensor([0.5, 3.0, 50.0, 1e4], dtype=torch.float64)
        alpha = torch.tensor([0.1, 0.2, 0.5, 2.0], dtype=torch.float64)
        distr = likelihood._distr_from_params((mu, alpha))
        assert torch.allclose(distr.mean, mu)
        assert torch.allclose(distr.variance, mu + alpha * mu**2)

        # the returned parameters `r` and `p` describe the same distribution
        model_output = torch.stack([mu, alpha], dim=-1).reshape(1, 1, 4, 2)
        # inverse of softplus
        model_output = model_output + torch.log(-torch.expm1(-model_output))
        params = likelihood.predict_likelihood_parameters(model_output).reshape(4, 2)
        r, p = params[:, 0], params[:, 1]
        assert torch.allclose(r, 1 / alpha)
        assert torch.allclose(r * p / (1 - p), mu)

        # loss stays finite for large means and small dispersion
        target = torch.tensor([0.0, 3.0, 60.0, 2e4], dtype=torch.float64)
        assert torch.isfinite(distr.log_prob(target)).all()


class TestBernoulliLogitLoss:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("prior_p", [None, 0.25])
    def test_confident_incorrect_predictions(self, dtype, weighted, prior_p):
        likelihood = BernoulliLikelihood(prior_p=prior_p, prior_strength=0.3)
        output = torch.tensor([-40.0, 40.0], dtype=dtype).reshape(1, 1, 2, 1)
        output.requires_grad_()
        target = torch.tensor([1.0, 0.0], dtype=dtype).reshape(1, 1, 2)
        weights = torch.tensor([0.5, 2.0], dtype=dtype).reshape(1, 1, 2)
        sample_weight = weights if weighted else None

        loss = likelihood.compute_loss(output, target, sample_weight)
        gradient = torch.autograd.grad(loss, output)[0].squeeze(-1)
        logits = output.detach().squeeze(-1)
        expected_loss = torch.nn.functional.softplus((1 - 2 * target) * logits)
        expected_gradient = logits.sigmoid() - target
        if weighted:
            expected_loss = expected_loss * weights
            expected_gradient = expected_gradient * weights
        expected_loss = expected_loss.mean()
        expected_gradient = expected_gradient / target.numel()
        if prior_p is not None:
            prior = torch.tensor(prior_p, dtype=dtype)
            neg_entropy = prior.xlogy(prior) + (1 - prior).xlogy(1 - prior)
            cross_entropy = torch.nn.functional.softplus(logits) - prior * logits
            expected_loss += 0.3 * (cross_entropy + neg_entropy).mean()
            expected_gradient += 0.3 * (logits.sigmoid() - prior) / target.numel()
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(gradient, expected_gradient)

    @pytest.mark.parametrize("prior_p", [None, 0.25])
    def test_ordinary_predictions_match_probability_loss(self, prior_p):
        likelihood = BernoulliLikelihood(prior_p=prior_p, prior_strength=0.3)
        output = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
        output = output.reshape(1, 1, 3, 1).requires_grad_()
        target = torch.tensor([1.0, 0.0, 1.0], dtype=output.dtype).reshape(1, 1, 3)
        weights = torch.tensor([0.5, 1.0, 2.0], dtype=output.dtype).reshape(1, 1, 3)
        reference = super(BernoulliLikelihood, likelihood).compute_loss(
            output, target, weights
        )
        actual = likelihood.compute_loss(output, target, weights)
        expected_grad = torch.autograd.grad(reference, output, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, output)[0]
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_training_step_recovers_from_confident_wrong_prediction(self):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(20)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        likelihood = BernoulliLikelihood()
        inputs = torch.ones((2, 1))
        target = torch.zeros((2, 1, 1))
        before = likelihood.compute_loss(
            model(inputs).reshape(2, 1, 1, 1), target, None
        )
        optimizer.zero_grad()
        before.backward()
        optimizer.step()
        after = likelihood.compute_loss(model(inputs).reshape(2, 1, 1, 1), target, None)
        assert after < before


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
