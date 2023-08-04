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
    ExponentialSmoothing,
    LightGBMModel,
    LinearRegressionModel,
    NotImportedModule,
    XGBModel,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.tests.base_test_class import DartsBaseTestClass, tfm_kwargs
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
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

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "Torch not available. Tests related to torch-based models will be skipped."
    )
    TORCH_AVAILABLE = False

lgbm_available = not isinstance(LightGBMModel, NotImportedModule)
cb_available = not isinstance(CatBoostModel, NotImportedModule)

models_cls_kwargs_errs = [
    (ExponentialSmoothing, {}, 0.3),
    (ARIMA, {"p": 1, "d": 0, "q": 1, "random_state": 42}, 0.03),
]

models_cls_kwargs_errs += [
    (
        BATS,
        {
            "use_trend": False,
            "use_damped_trend": False,
            "use_box_cox": True,
            "use_arma_errors": False,
            "random_state": 42,
        },
        0.3,
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
        0.3,
    ),
]

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
            0.08,
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
            0.2,
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
            0.3,
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
            0.05,
        ),
    ]


@pytest.mark.slow
class ProbabilisticModelsTestCase(DartsBaseTestClass):
    np.random.seed(0)

    constant_ts = tg.constant_timeseries(length=200, value=0.5)
    constant_noisy_ts = constant_ts + tg.gaussian_timeseries(length=200, std=0.1)
    constant_multivar_ts = constant_ts.stack(constant_ts)
    constant_noisy_multivar_ts = constant_noisy_ts.stack(constant_noisy_ts)
    num_samples = 5

    constant_noisy_ts_short = constant_noisy_ts[:30]

    @pytest.mark.slow
    def test_fit_predict_determinism(self):
        for model_cls, model_kwargs, _ in models_cls_kwargs_errs:
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

            self.assertTrue((pred1 == pred2).all())

            # test whether the next prediction of the same model is different
            pred3 = model.predict(n=10, num_samples=2).values()
            self.assertTrue((pred2 != pred3).any())

    def test_probabilistic_forecast_accuracy(self):
        for model_cls, model_kwargs, err in models_cls_kwargs_errs:
            self.helper_test_probabilistic_forecast_accuracy(
                model_cls, model_kwargs, err, self.constant_ts, self.constant_noisy_ts
            )
            if issubclass(model_cls, GlobalForecastingModel):
                self.helper_test_probabilistic_forecast_accuracy(
                    model_cls,
                    model_kwargs,
                    err,
                    self.constant_multivar_ts,
                    self.constant_noisy_multivar_ts,
                )

    def helper_test_probabilistic_forecast_accuracy(
        self, model_cls, model_kwargs, err, ts, noisy_ts
    ):
        model = model_cls(**model_kwargs)
        model.fit(noisy_ts[:100])
        pred = model.predict(n=100, num_samples=100)

        # test accuracy of the median prediction compared to the noiseless ts
        mae_err_median = mae(ts[100:], pred)
        self.assertLess(mae_err_median, err)

        # test accuracy for increasing quantiles between 0.7 and 1 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.7, 0.8, 0.9, 0.99]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            self.assertLess(mae_err, new_mae + 0.1)
            mae_err = new_mae

        # test accuracy for decreasing quantiles between 0.3 and 0 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.3, 0.2, 0.1, 0.01]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            self.assertLess(mae_err, new_mae + 0.1)
            mae_err = new_mae

    @pytest.mark.slow
    def test_predict_likelihood_parameters_regression_models(self):
        """
        Check that the shape of the predicted likelihood parameters match expectations, for both
        univariate and multivariate series.

        Note: values are not tested as it would be too time consuming
        """
        seed = 142857
        n_times, n_samples = 100, 1
        model_classes = [LinearRegressionModel, XGBModel]
        if lgbm_available:
            model_classes.append(LightGBMModel)
        if cb_available:
            model_classes.append(CatBoostModel)

        for n_comp in [1, 3]:
            list_lkl = [
                {
                    "kwargs": {
                        "likelihood": "quantile",
                        "quantiles": [0.05, 0.50, 0.95],
                    },
                    "ts": TimeSeries.from_values(
                        np.random.normal(
                            loc=0, scale=1, size=(n_times, n_comp, n_samples)
                        )
                    ),
                    "expected": np.array([-1.67, 0, 1.67]),
                },
                {
                    "kwargs": {"likelihood": "poisson"},
                    "ts": TimeSeries.from_values(
                        np.random.poisson(lam=4, size=(n_times, n_comp, n_samples))
                    ),
                    "expected": np.array([4]),
                },
            ]

            for model_cls in model_classes:
                # Catboost is the only regression model supporting the GaussianLikelihood
                if cb_available and issubclass(model_cls, CatBoostModel):
                    list_lkl.append(
                        {
                            "kwargs": {"likelihood": "gaussian"},
                            "ts": TimeSeries.from_values(
                                np.random.normal(
                                    loc=10, scale=3, size=(n_times, n_comp, n_samples)
                                )
                            ),
                            "expected": np.array([10, 3]),
                        }
                    )

                for lkl in list_lkl:
                    model = model_cls(lags=3, random_state=seed, **lkl["kwargs"])
                    model.fit(lkl["ts"])
                    pred_lkl_params = model.predict(
                        n=1, num_samples=1, predict_likelihood_parameters=True
                    )
                    if n_comp == 1:
                        assert (
                            lkl["expected"].shape == pred_lkl_params.values()[0].shape
                        ), (
                            "The shape of the predicted likelihood parameters do not match expectation "
                            "for univariate series."
                        )
                    else:
                        assert (
                            1,
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
            (GaussianLikelihood(), real_series, 0.1, 3),
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

        def test_likelihoods_and_resulting_mean_forecasts(self):
            def _get_avgs(series):
                return np.mean(series.all_values()[:, 0, :]), np.mean(
                    series.all_values()[:, 1, :]
                )

            for lkl, series, diff1, diff2 in self.lkl_series:
                model = RNNModel(input_chunk_length=5, likelihood=lkl, **tfm_kwargs)
                model.fit(series, epochs=50)
                pred = model.predict(n=50, num_samples=50)

                avgs_orig, avgs_pred = _get_avgs(series), _get_avgs(pred)
                self.assertLess(
                    abs(avgs_orig[0] - avgs_pred[0]),
                    diff1,
                    "The difference between the mean forecast and the mean series is larger "
                    "than expected on component 0 for distribution {}".format(lkl),
                )
                self.assertLess(
                    abs(avgs_orig[1] - avgs_pred[1]),
                    diff2,
                    "The difference between the mean forecast and the mean series is larger "
                    "than expected on component 1 for distribution {}".format(lkl),
                )

        @pytest.mark.slow
        def test_predict_likelihood_parameters_univariate_torch_models(self):
            """Checking convergence of model for each metric is too time consuming, making sure that the dimensions
            of the predictions contain the proper elements for univariate input.
            """
            # fix seed to avoid values outside of distribution's support
            seed = 142857
            torch.manual_seed(seed=seed)
            list_lkl = [
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
            if not self.runs_on_m1:
                list_lkl.append(
                    (CauchyLikelihood(), [0, 1]),
                )

            n_times = 100
            n_comp = 1
            n_samples = 1
            for lkl, lkl_params in list_lkl:
                # QuantileRegression is not distribution
                if isinstance(lkl, QuantileRegression):
                    values = np.random.normal(
                        loc=0, scale=1, size=(n_times, n_comp, n_samples)
                    )
                else:
                    values = lkl._distr_from_params(lkl_params).sample(
                        (n_times, n_comp, n_samples)
                    )

                    # Dirichlet must be handled sligthly differently since its multivariate
                    if isinstance(lkl, DirichletLikelihood):
                        values = torch.swapaxes(values, 1, 3)
                        values = torch.squeeze(values, 3)
                        lkl_params = lkl_params[0]

                ts = TimeSeries.from_values(
                    values, columns=[f"dummy_{i}" for i in range(values.shape[1])]
                )

                # [DualCovariatesModule, PastCovariatesModule, MixedCovariatesModule]
                models = [
                    RNNModel(
                        4,
                        "RNN",
                        likelihood=lkl,
                        n_epochs=1,
                        random_state=seed,
                        **tfm_kwargs,
                    ),
                    NBEATSModel(
                        4,
                        1,
                        likelihood=lkl,
                        n_epochs=1,
                        random_state=seed,
                        **tfm_kwargs,
                    ),
                    DLinearModel(
                        4,
                        1,
                        likelihood=lkl,
                        n_epochs=1,
                        random_state=seed,
                        **tfm_kwargs,
                    ),
                ]

                true_lkl_params = np.array(lkl_params)
                for model in models:
                    # univariate
                    model.fit(ts)
                    pred_lkl_params = model.predict(
                        n=1, num_samples=1, predict_likelihood_parameters=True
                    )

                    # check the dimensions, values require too much training
                    self.assertEqual(
                        pred_lkl_params.values().shape[1], len(true_lkl_params)
                    )

        @pytest.mark.slow
        def test_predict_likelihood_parameters_multivariate_torch_models(self):
            """Checking convergence of model for each metric is too time consuming, making sure that the dimensions
            of the predictions contain the proper elements for multivariate inputs.
            """
            torch.manual_seed(seed=142857)
            list_lkl = [
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
            ]

            n_times = 100
            n_comp = 2
            n_samples = 1
            for lkl, lkl_params, comp_names in list_lkl:
                if isinstance(lkl, QuantileRegression):
                    values = np.random.normal(
                        loc=0, scale=1, size=(n_times, n_comp, n_samples)
                    )
                else:
                    values = lkl._distr_from_params(lkl_params).sample(
                        (n_times, n_comp, n_samples)
                    )
                ts = TimeSeries.from_values(
                    values, columns=[f"dummy_{i}" for i in range(values.shape[1])]
                )

                # [DualCovariatesModule, PastCovariatesModule, MixedCovariatesModule]
                models = [
                    RNNModel(4, "RNN", likelihood=lkl, n_epochs=1, **tfm_kwargs),
                    TCNModel(4, 1, likelihood=lkl, n_epochs=1, **tfm_kwargs),
                    DLinearModel(4, 1, likelihood=lkl, n_epochs=1, **tfm_kwargs),
                ]

                for model in models:
                    model.fit(ts)
                    pred_lkl_params = model.predict(
                        n=1, num_samples=1, predict_likelihood_parameters=True
                    )
                    # check the dimensions
                    self.assertEqual(
                        pred_lkl_params.values().shape[1], n_comp * len(lkl_params)
                    )
                    # check the component names
                    self.assertTrue(
                        list(pred_lkl_params.components) == comp_names,
                        f"Components names are not matching; expected {comp_names} "
                        f"but received {list(pred_lkl_params.components)}",
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
            with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError):
                model.predict(n=1, num_samples=2, predict_likelihood_parameters=True)
            # n > output_chunk_length
            with self.assertRaises(ValueError):
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
            self.assertFalse(np.alltrue(preds[0].values() == preds[1].values()))
