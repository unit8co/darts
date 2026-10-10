from itertools import combinations

import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import numpy as np
import torch
from scipy import stats

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
    ZeroInflatedLikelihood,
    _ZeroInflatedDistribution,
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
            ZeroInflatedLikelihood(ExponentialLikelihood()),
            ZeroInflatedLikelihood(GaussianLikelihood()),
            ZeroInflatedLikelihood(NegativeBinomialLikelihood()),
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


def _float64(value):
    return torch.tensor(value, dtype=torch.float64)


class TestZeroInflatedDistribution:
    @pytest.mark.parametrize(
        "base_distr,ref_log_prob,values",
        [
            (
                torch.distributions.Poisson(_float64(2.5)),
                lambda y: stats.poisson.logpmf(y, 2.5),
                [0.0, 1.0, 3.0, 7.0],
            ),
            (
                # torch's `probs` is the success probability; scipy's `p` is the failure probability
                torch.distributions.NegativeBinomial(_float64(2.0), _float64(0.4)),
                lambda y: stats.nbinom.logpmf(y, 2.0, 0.6),
                [0.0, 1.0, 3.0, 7.0],
            ),
            (
                torch.distributions.Gamma(_float64(2.0), _float64(1.5)),
                lambda y: stats.gamma.logpdf(y, 2.0, scale=1 / 1.5),
                [0.0, 0.3, 1.0, 4.0],
            ),
            (
                torch.distributions.LogNormal(_float64(0.5), _float64(0.8)),
                lambda y: stats.lognorm.logpdf(y, 0.8, scale=np.exp(0.5)),
                [0.0, 0.3, 1.0, 4.0],
            ),
            (
                torch.distributions.Normal(_float64(1.0), _float64(2.0)),
                lambda y: stats.norm.logpdf(y, 1.0, 2.0),
                [0.0, -1.5, 0.3, 4.0],
            ),
        ],
    )
    def test_log_prob(self, base_distr, ref_log_prob, values):
        gate = 0.3
        distr = _ZeroInflatedDistribution(base_distr, _float64(gate))
        expected = []
        for y in values:
            if y != 0:
                expected.append(np.log(1 - gate) + ref_log_prob(y))
            elif base_distr.support.is_discrete:
                expected.append(np.log(gate + (1 - gate) * np.exp(ref_log_prob(0.0))))
            else:
                expected.append(np.log(gate))
        np.testing.assert_allclose(
            distr.log_prob(_float64(values)).numpy(), expected, rtol=1e-6
        )

    def test_mean(self):
        base = torch.distributions.Poisson(torch.tensor([2.0, 5.0]))
        distr = _ZeroInflatedDistribution(base, torch.tensor([0.25, 0.5]))
        assert torch.allclose(distr.mean, torch.tensor([1.5, 2.5]))

    def test_sample_zero_fraction(self):
        torch.manual_seed(42)
        rate = torch.full((20000,), 3.0)
        distr = _ZeroInflatedDistribution(
            torch.distributions.Poisson(rate), torch.full((20000,), 0.5)
        )
        zero_fraction = (distr.sample() == 0).float().mean()
        expected = 0.5 + 0.5 * torch.exp(torch.tensor(-3.0))
        assert abs(zero_fraction - expected) < 0.01

    def test_accepts_python_float_gate(self):
        # used by the model tests, which build distributions from plain parameter values
        distr = _ZeroInflatedDistribution(torch.distributions.Poisson(5.0), 0.5)
        assert distr.sample((3, 2)).shape == (3, 2)


class TestZeroInflatedLikelihood:
    @pytest.mark.parametrize(
        "likelihood",
        [
            QuantileRegression(),
            DirichletLikelihood(),
            BernoulliLikelihood(),
            ZeroInflatedLikelihood(PoissonLikelihood()),
            PoissonLikelihood(prior_lambda=2.0),
            GaussianLikelihood(prior_mu=0.0),
            GaussianLikelihood(beta_nll=0.5),
        ],
    )
    def test_unsupported_likelihoods(self, likelihood):
        with pytest.raises(ValueError, match="does not support"):
            ZeroInflatedLikelihood(likelihood)

    def test_parameter_names(self):
        likelihood = ZeroInflatedLikelihood(NegativeBinomialLikelihood())
        assert likelihood.num_parameters == 3
        assert likelihood.component_names(components=["a", "b"]) == [
            "a_r",
            "a_p",
            "a_zi_p",
            "b_r",
            "b_p",
            "b_zi_p",
        ]

    def test_equality(self):
        assert ZeroInflatedLikelihood(PoissonLikelihood()) == ZeroInflatedLikelihood(
            PoissonLikelihood()
        )
        assert ZeroInflatedLikelihood(PoissonLikelihood()) != ZeroInflatedLikelihood(
            GammaLikelihood()
        )
        assert ZeroInflatedLikelihood(PoissonLikelihood()) != PoissonLikelihood()

    def test_predict_likelihood_parameters(self):
        torch.manual_seed(42)
        likelihood = ZeroInflatedLikelihood(NegativeBinomialLikelihood())
        model_output = torch.randn(2, 4, 3, 3)
        params = likelihood.predict_likelihood_parameters(model_output)

        base_params = NegativeBinomialLikelihood().predict_likelihood_parameters(
            model_output[..., :2]
        )
        gate = torch.sigmoid(model_output[..., 2:])
        expected = torch.cat([base_params.reshape(2, 4, 3, 2), gate], dim=3)
        assert torch.allclose(params, expected.reshape(2, 4, 9))

    def test_sample(self):
        torch.manual_seed(42)
        likelihood = ZeroInflatedLikelihood(PoissonLikelihood())
        # gate logit 0 -> zi_p = 0.5; base rate softplus(3) ~ 3.05
        model_output = torch.zeros(20000, 1, 1, 2)
        model_output[..., 0] = 3.0
        samples = likelihood.sample(model_output)
        assert samples.shape == (20000, 1, 1)
        rate = torch.nn.functional.softplus(torch.tensor(3.0))
        expected = 0.5 + 0.5 * torch.exp(-rate)
        assert abs((samples == 0).float().mean() - expected) < 0.01

    @pytest.mark.parametrize(
        "base_likelihood",
        [
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
            WeibullLikelihood(),
        ],
    )
    def test_loss_with_zero_targets(self, base_likelihood):
        torch.manual_seed(42)
        likelihood = ZeroInflatedLikelihood(base_likelihood)
        model_output = torch.randn(
            4, 3, 2, likelihood.num_parameters, requires_grad=True
        )
        with torch.no_grad():
            params = likelihood._params_from_output(model_output)
            target = likelihood._distr_from_params(params).sample()
        target[:, 0, 0] = 0.0

        loss = likelihood.compute_loss(model_output, target, None)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(model_output.grad).all()
        # every sample of the batch contributes to the gradient
        assert (model_output.grad.abs().sum(dim=(1, 2, 3)) > 0).all()
