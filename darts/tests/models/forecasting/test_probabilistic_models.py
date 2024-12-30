import itertools
import platform

import numpy as np
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.metrics import mae
from darts.models import (
    ARIMA,
    BATS,
    TBATS,
    CatBoostModel,
    ConformalNaiveModel,
    ExponentialSmoothing,
    LightGBMModel,
    LinearRegressionModel,
    NotImportedModule,
    XGBModel,
)
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

if TORCH_AVAILABLE:
    import torch

    from darts.models import (
        BlockRNNModel,
        DLinearModel,
        NBEATSModel,
        RNNModel,
        TCNModel,
        TFTModel,
        TiDEModel,
        TransformerModel,
        TSMixerModel,
    )
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
    from darts.utils.likelihood_models import (
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

lgbm_available = not isinstance(LightGBMModel, NotImportedModule)
cb_available = not isinstance(CatBoostModel, NotImportedModule)

# conformal models require a fitted base model
# in tests below, the model is re-trained for new input series.
# using a fake trained model should allow the same API with conformal models
conformal_forecaster = LinearRegressionModel(lags=10, output_chunk_length=5)
conformal_forecaster._fit_called = True

# model_cls, model_kwargs, err_univariate, err_multivariate
models_cls_kwargs_errs = [
    (ExponentialSmoothing, {}, 0.3, None),
    (ARIMA, {"p": 1, "d": 0, "q": 1, "random_state": 42}, 0.03, None),
    (
        BATS,
        {
            "use_trend": False,
            "use_damped_trend": False,
            "use_box_cox": True,
            "use_arma_errors": False,
            "random_state": 42,
        },
        0.04,
        None,
    ),
    (
        TBATS,
        {
            "use_trend": False,
            "use_damped_trend": False,
            "use_box_cox": True,
            "use_arma_errors": False,
            "random_state": 42,
        },
        0.04,
        0.04,
    ),
    (
        ConformalNaiveModel,
        {
            "model": conformal_forecaster,
            "cal_length": 1,
            "random_state": 42,
            "quantiles": [0.1, 0.5, 0.9],
        },
        0.04,
        0.04,
    ),
]

xgb_test_params = {
    "n_estimators": 1,
    "max_depth": 1,
    "max_leaves": 1,
}
lgbm_test_params = {
    "n_estimators": 1,
    "max_depth": 1,
    "num_leaves": 2,
    "verbosity": -1,
}
cb_test_params = {
    "iterations": 1,
    "depth": 1,
    "verbose": -1,
}

if TORCH_AVAILABLE:
    models_cls_kwargs_errs += [
        (
            RNNModel,
            {
                "input_chunk_length": 2,
                "training_length": 10,
                "n_epochs": 20,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.02,
            0.04,
        ),
        (
            TCNModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 60,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.06,
            0.06,
        ),
        (
            BlockRNNModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 20,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.03,
            0.04,
        ),
        (
            TransformerModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 20,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.03,
            0.04,
        ),
        (
            NBEATSModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 10,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.15,
            0.26,
        ),
        (
            TFTModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 10,
                "random_state": 0,
                "add_relative_index": True,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.02,
            0.1,
        ),
        (
            TiDEModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 10,
                "random_state": 0,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.06,
            0.1,
        ),
        (
            TSMixerModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 5,
                "n_epochs": 100,
                "random_state": 0,
                "num_blocks": 1,
                "hidden_size": 32,
                "dropout": 0.2,
                "ff_size": 32,
                "batch_size": 8,
                "likelihood": GaussianLikelihood(),
                **tfm_kwargs,
            },
            0.06,
            0.1,
        ),
    ]


@pytest.mark.slow
class TestProbabilisticModels:
    np.random.seed(0)

    constant_ts = tg.constant_timeseries(length=200, value=0.5)
    constant_noisy_ts = constant_ts + tg.gaussian_timeseries(length=200, std=0.1)
    constant_multivar_ts = constant_ts.stack(constant_ts)
    constant_noisy_multivar_ts = constant_noisy_ts.stack(constant_noisy_ts)
    num_samples = 5

    constant_noisy_ts_short = constant_noisy_ts[:30]

    @pytest.mark.slow
    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_fit_predict_determinism(self, config):
        model_cls, model_kwargs, _, _ = config
        if TORCH_AVAILABLE and issubclass(model_cls, TorchForecastingModel):
            fit_kwargs = {"epochs": 1, "max_samples_per_ts": 3}
        else:
            fit_kwargs = {}
        # whether the first predictions of two models initiated with the same random state are the same
        model = model_cls(**model_kwargs)
        model.fit(self.constant_noisy_ts_short, **fit_kwargs)
        pred1 = model.predict(n=10, num_samples=2).values()

        model = model_cls(**model_kwargs)
        model.fit(self.constant_noisy_ts_short, **fit_kwargs)
        pred2 = model.predict(n=10, num_samples=2).values()

        assert (pred1 == pred2).all()

        # test whether the next prediction of the same model is different
        pred3 = model.predict(n=10, num_samples=2).values()
        assert (pred2 != pred3).any()

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_probabilistic_forecast_accuracy_univariate(self, config):
        """Test on univariate series"""
        model_cls, model_kwargs, err, _ = config
        model = model_cls(**model_kwargs)
        self.helper_test_probabilistic_forecast_accuracy(
            model, err, self.constant_ts, self.constant_noisy_ts
        )

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_probabilistic_forecast_accuracy_multivariate(self, config):
        """Test on multivariate series, when supported"""
        model_cls, model_kwargs, _, err = config
        model = model_cls(**model_kwargs)
        if model.supports_multivariate:
            self.helper_test_probabilistic_forecast_accuracy(
                model,
                err,
                self.constant_multivar_ts,
                self.constant_noisy_multivar_ts,
            )

    def helper_test_probabilistic_forecast_accuracy(self, model, err, ts, noisy_ts):
        model.fit(noisy_ts[:100])
        pred = model.predict(n=50, num_samples=100)

        # test accuracy of the median prediction compared to the noiseless ts
        mae_err_median = mae(ts[100:], pred)
        assert mae_err_median < err

        # test accuracy for increasing quantiles between 0.7 and 1 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.7, 0.8, 0.9, 0.99]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            assert mae_err < new_mae + 0.1
            mae_err = new_mae

        # test accuracy for decreasing quantiles between 0.3 and 0 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.3, 0.2, 0.1, 0.01]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            assert mae_err < new_mae + 0.1
            mae_err = new_mae

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [(LinearRegressionModel, False, {}), (XGBModel, False, xgb_test_params)]
            + ([(LightGBMModel, False, lgbm_test_params)] if lgbm_available else [])
            + ([(CatBoostModel, True, cb_test_params)] if cb_available else []),
            [1, 3],  # n components
            [
                "quantile",
                "poisson",
                "gaussian",
            ],  # likelihood
            [True, False],  # multi models
            [1, 2],  # horizon
        ),
    )
    def test_predict_likelihood_parameters_regression_models(self, config):
        """
        Check that the shape of the predicted likelihood parameters match expectations, for both
        univariate and multivariate series.

        Note: values are not tested as it would be too time consuming
        """
        (
            (model_cls, supports_gaussian, model_kwargs),
            n_comp,
            likelihood,
            multi_models,
            horizon,
        ) = config

        seed = 142857
        n_times, n_samples = 100, 1
        lkl = {"kwargs": {"likelihood": likelihood}}

        if likelihood == "quantile":
            lkl["kwargs"]["quantiles"] = [0.05, 0.50, 0.95]
            lkl["ts"] = TimeSeries.from_values(
                np.random.normal(loc=0, scale=1, size=(n_times, n_comp, n_samples))
            )
            lkl["expected"] = np.array([-1.67, 0, 1.67])
        elif likelihood == "poisson":
            lkl["ts"] = TimeSeries.from_values(
                np.random.poisson(lam=4, size=(n_times, n_comp, n_samples))
            )
            lkl["expected"] = np.array([4])
        elif likelihood == "gaussian":
            if not supports_gaussian:
                return

            lkl["ts"] = TimeSeries.from_values(
                np.random.normal(loc=10, scale=3, size=(n_times, n_comp, n_samples))
            )
            lkl["expected"] = np.array([10, 3])
        else:
            assert False, f"unknown likelihood {likelihood}"

        model = model_cls(
            lags=3,
            output_chunk_length=2,
            random_state=seed,
            **lkl["kwargs"],
            multi_models=multi_models,
            **model_kwargs,
        )
        model.fit(lkl["ts"])
        pred_lkl_params = model.predict(
            n=horizon, num_samples=1, predict_likelihood_parameters=True
        )
        if n_comp == 1:
            assert lkl["expected"].shape == pred_lkl_params.values()[0].shape, (
                "The shape of the predicted likelihood parameters do not match expectation "
                "for univariate series."
            )
        else:
            assert (
                horizon,
                len(lkl["expected"]) * n_comp,
                1,
            ) == pred_lkl_params.all_values().shape, (
                "The shape of the predicted likelihood parameters do not match expectation "
                "for multivariate series."
            )

    """ More likelihood tests
    """
    if TORCH_AVAILABLE:
        runs_on_m1 = platform.processor() == "arm"
        np.random.seed(42)
        torch.manual_seed(42)

        real_series = TimeSeries.from_values(np.random.randn(100, 2) + [0, 5])
        vals = real_series.all_values()

        real_pos_series = TimeSeries.from_values(np.where(vals > 0, vals, -vals))
        discrete_pos_series = TimeSeries.from_values(
            np.random.randint(low=0, high=11, size=(100, 2))
        )
        binary_series = TimeSeries.from_values(
            np.random.randint(low=0, high=2, size=(100, 2))
        )
        bounded_series = TimeSeries.from_values(np.random.beta(2, 5, size=(100, 2)))
        simplex_series = bounded_series["0"].stack(1.0 - bounded_series["0"])

        lkl_series = [
            (GaussianLikelihood(), real_series, 0.17, 3),
            (PoissonLikelihood(), discrete_pos_series, 2, 2),
            (NegativeBinomialLikelihood(), discrete_pos_series, 0.5, 0.5),
            (BernoulliLikelihood(), binary_series, 0.15, 0.15),
            (GammaLikelihood(), real_pos_series, 0.3, 0.3),
            (GumbelLikelihood(), real_series, 0.2, 3),
            (LaplaceLikelihood(), real_series, 0.3, 4),
            (BetaLikelihood(), bounded_series, 0.1, 0.1),
            (ExponentialLikelihood(), real_pos_series, 0.3, 2),
            (DirichletLikelihood(), simplex_series, 0.3, 0.3),
            (GeometricLikelihood(), discrete_pos_series, 1, 1),
            (CauchyLikelihood(), real_series, 3, 11),
            (ContinuousBernoulliLikelihood(), bounded_series, 0.1, 0.1),
            (HalfNormalLikelihood(), real_pos_series, 0.3, 8),
            (LogNormalLikelihood(), real_pos_series, 0.3, 1),
            (WeibullLikelihood(), real_pos_series, 0.2, 2.5),
            (QuantileRegression(), real_series, 0.2, 1),
        ]

        @pytest.mark.parametrize("lkl_config", lkl_series)
        def test_likelihoods_and_resulting_mean_forecasts(self, lkl_config):
            def _get_avgs(series):
                return np.mean(series.all_values()[:, 0, :]), np.mean(
                    series.all_values()[:, 1, :]
                )

            lkl, series, diff1, diff2 = lkl_config
            seed = 142857
            torch.manual_seed(seed=seed)
            kwargs = {
                "likelihood": lkl,
                "n_epochs": 50,
                "random_state": seed,
                **tfm_kwargs,
            }

            model = RNNModel(input_chunk_length=5, **kwargs)
            model.fit(series)
            pred = model.predict(n=50, num_samples=50)

            avgs_orig, avgs_pred = _get_avgs(series), _get_avgs(pred)
            assert abs(avgs_orig[0] - avgs_pred[0]) < diff1, (
                "The difference between the mean forecast and the mean series is larger "
                f"than expected on component 0 for distribution {lkl}"
            )
            assert abs(avgs_orig[1] - avgs_pred[1]) < diff2, (
                "The difference between the mean forecast and the mean series is larger "
                f"than expected on component 1 for distribution {lkl}"
            )

        @pytest.mark.parametrize(
            "lkl_config",
            [  # tuple of (likelihood, likelihood params)
                (GaussianLikelihood(), [10, 1]),
                (PoissonLikelihood(), [5]),
                (DirichletLikelihood(), [torch.Tensor([0.3, 0.3, 0.3])]),
                (
                    QuantileRegression([0.05, 0.5, 0.95]),
                    [-1.67, 0, 1.67],
                ),
                (NegativeBinomialLikelihood(), [2, 0.5]),
                (BernoulliLikelihood(), [0.8]),
                (GammaLikelihood(), [2.0, 2.0]),
                (GumbelLikelihood(), [3.0, 4.0]),
                (LaplaceLikelihood(), [0, 1]),
                (BetaLikelihood(), [0.5, 0.5]),
                (ExponentialLikelihood(), [1.0]),
                (GeometricLikelihood(), [0.3]),
                (ContinuousBernoulliLikelihood(), [0.4]),
                (HalfNormalLikelihood(), [1]),
                (LogNormalLikelihood(), [0, 0.25]),
                (WeibullLikelihood(), [1, 1.5]),
            ]
            + ([(CauchyLikelihood(), [0, 1])] if not runs_on_m1 else []),
        )
        def test_predict_likelihood_parameters_univariate_torch_models(
            self, lkl_config
        ):
            """Checking convergence of model for each metric is too time consuming, making sure that the dimensions
            of the predictions contain the proper elements for univariate input.
            """
            lkl, lkl_params = lkl_config
            # fix seed to avoid values outside of distribution's support
            seed = 142857
            torch.manual_seed(seed=seed)
            kwargs = {
                "likelihood": lkl,
                "n_epochs": 1,
                "random_state": seed,
                **tfm_kwargs,
            }

            n_times = 5
            n_comp = 1
            n_samples = 1
            # QuantileRegression is not distribution
            if isinstance(lkl, QuantileRegression):
                values = np.random.normal(
                    loc=0, scale=1, size=(n_times, n_comp, n_samples)
                )
            else:
                values = lkl._distr_from_params(lkl_params).sample((
                    n_times,
                    n_comp,
                    n_samples,
                ))

                # Dirichlet must be handled slightly differently since its multivariate
                if isinstance(lkl, DirichletLikelihood):
                    values = torch.swapaxes(values, 1, 3)
                    values = torch.squeeze(values, 3)
                    lkl_params = lkl_params[0]

            ts = TimeSeries.from_values(
                values, columns=[f"dummy_{i}" for i in range(values.shape[1])]
            )

            # [DualCovariatesModule, PastCovariatesModule, MixedCovariatesModule]
            models = [
                RNNModel(4, "RNN", training_length=4, **kwargs),
                NBEATSModel(4, 1, **kwargs),
                DLinearModel(4, 1, **kwargs),
            ]

            true_lkl_params = np.array(lkl_params)
            for model in models:
                # univariate
                model.fit(ts)
                pred_lkl_params = model.predict(
                    n=1, num_samples=1, predict_likelihood_parameters=True
                )

                # check the dimensions, values require too much training
                assert pred_lkl_params.values().shape[1] == len(true_lkl_params)

        @pytest.mark.parametrize(
            "lkl_config",
            [
                (
                    GaussianLikelihood(),
                    [10, 1],
                    ["dummy_0_mu", "dummy_0_sigma", "dummy_1_mu", "dummy_1_sigma"],
                ),
                (PoissonLikelihood(), [5], ["dummy_0_lambda", "dummy_1_lambda"]),
                (
                    QuantileRegression([0.05, 0.5, 0.95]),
                    [-1.67, 0, 1.67],
                    [
                        "dummy_0_q0.05",
                        "dummy_0_q0.50",
                        "dummy_0_q0.95",
                        "dummy_1_q0.05",
                        "dummy_1_q0.50",
                        "dummy_1_q0.95",
                    ],
                ),
            ],
        )
        def test_predict_likelihood_parameters_multivariate_torch_models(
            self, lkl_config
        ):
            """Checking convergence of model for each metric is too time consuming, making sure that the dimensions
            of the predictions contain the proper elements for multivariate inputs.
            """
            lkl, lkl_params, comp_names = lkl_config
            # fix seed to avoid values outside of distribution's support

            seed = 142857
            torch.manual_seed(seed=seed)
            kwargs = {
                "likelihood": lkl,
                "n_epochs": 1,
                "random_state": seed,
                **tfm_kwargs,
            }

            n_times = 5
            n_comp = 2
            n_samples = 1
            if isinstance(lkl, QuantileRegression):
                values = np.random.normal(
                    loc=0, scale=1, size=(n_times, n_comp, n_samples)
                )
            else:
                values = lkl._distr_from_params(lkl_params).sample((
                    n_times,
                    n_comp,
                    n_samples,
                ))
            ts = TimeSeries.from_values(
                values, columns=[f"dummy_{i}" for i in range(values.shape[1])]
            )

            # [DualCovariatesModule, PastCovariatesModule, MixedCovariatesModule]
            models = [
                RNNModel(4, "RNN", training_length=4, **kwargs),
                TCNModel(4, 1, **kwargs),
                DLinearModel(4, 1, **kwargs),
            ]

            for model in models:
                model.fit(ts)
                pred_lkl_params = model.predict(
                    n=1, num_samples=1, predict_likelihood_parameters=True
                )
                # check the dimensions
                assert pred_lkl_params.values().shape[1] == n_comp * len(lkl_params)
                # check the component names
                assert list(pred_lkl_params.components) == comp_names, (
                    f"Components names are not matching; expected {comp_names} "
                    f"but received {list(pred_lkl_params.components)}"
                )

        def test_predict_likelihood_parameters_wrong_args(self):
            # deterministic model
            model = DLinearModel(
                input_chunk_length=4,
                output_chunk_length=4,
                n_epochs=1,
                **tfm_kwargs,
            )
            model.fit(self.constant_noisy_ts)
            with pytest.raises(ValueError):
                model.predict(n=1, predict_likelihood_parameters=True)

            model = DLinearModel(
                input_chunk_length=4,
                output_chunk_length=4,
                n_epochs=1,
                likelihood=GaussianLikelihood(),
                **tfm_kwargs,
            )
            model.fit(self.constant_noisy_ts)
            # num_samples > 1
            with pytest.raises(ValueError):
                model.predict(n=1, num_samples=2, predict_likelihood_parameters=True)
            # n > output_chunk_length
            with pytest.raises(ValueError):
                model.predict(n=5, num_samples=1, predict_likelihood_parameters=True)
            model.predict(n=4, num_samples=1, predict_likelihood_parameters=True)

        def test_stochastic_inputs(self):
            model = RNNModel(input_chunk_length=5, **tfm_kwargs)
            model.fit(self.constant_ts, epochs=2)

            # build a stochastic series
            target_vals = self.constant_ts.values()
            stochastic_vals = np.random.normal(
                loc=target_vals, scale=1.0, size=(len(self.constant_ts), 100)
            )
            stochastic_vals = np.expand_dims(stochastic_vals, axis=1)
            stochastic_series = TimeSeries.from_times_and_values(
                self.constant_ts.time_index, stochastic_vals
            )

            # A deterministic model forecasting a stochastic series
            # should return stochastic samples
            preds = [model.predict(series=stochastic_series, n=10) for _ in range(2)]

            # random samples should differ
            assert not np.array_equal(preds[0].values(), preds[1].values())
