import random
import unittest
from itertools import product

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.metrics import mape, r2_score
from darts.models import ARIMA, FFT, ExponentialSmoothing, NaiveDrift, Theta
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import gaussian_timeseries as gt
from darts.utils.timeseries_generation import linear_timeseries as lt
from darts.utils.timeseries_generation import random_walk_timeseries as rt
from darts.utils.timeseries_generation import sine_timeseries as st

logger = get_logger(__name__)


try:
    from darts.models import (
        BlockRNNModel,
        LinearRegressionModel,
        RandomForest,
        TCNModel,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "Torch models are not installed - will not be tested for backtesting"
    )
    TORCH_AVAILABLE = False


def get_dummy_series(
    ts_length: int, lt_end_value: int = 10, st_value_offset: int = 10
) -> TimeSeries:
    return (
        lt(length=ts_length, end_value=lt_end_value)
        + st(length=ts_length, value_y_offset=st_value_offset)
        + rt(length=ts_length)
    )


def compare_best_against_random(model_class, params, series, stride=1):

    # instantiate best model in expanding window mode
    np.random.seed(1)
    best_model_1, _, _ = model_class.gridsearch(
        params,
        series,
        forecast_horizon=10,
        stride=stride,
        metric=mape,
        start=series.time_index[-21],
    )

    # instantiate best model in split mode
    train, val = series.split_before(series.time_index[-10])
    best_model_2, _, _ = model_class.gridsearch(
        params, train, val_series=val, metric=mape
    )

    # instantiate model with random parameters from 'params'
    random.seed(1)
    random_param_choice = {}
    for key in params.keys():
        random_param_choice[key] = random.choice(params[key])
    random_model = model_class(**random_param_choice)

    # perform backtest forecasting on both models
    best_score_1 = best_model_1.backtest(
        series, start=series.time_index[-21], forecast_horizon=10
    )
    random_score_1 = random_model.backtest(
        series, start=series.time_index[-21], forecast_horizon=10
    )

    # perform train/val evaluation on both models
    best_model_2.fit(train)
    best_score_2 = mape(best_model_2.predict(len(val)), series)
    random_model = model_class(**random_param_choice)
    random_model.fit(train)
    random_score_2 = mape(random_model.predict(len(val)), series)

    # check whether best models are at least as good as random models
    expanding_window_ok = best_score_1 <= random_score_1
    split_ok = best_score_2 <= random_score_2

    return expanding_window_ok and split_ok


class BacktestingTestCase(DartsBaseTestClass):
    def test_backtest_forecasting(self):
        linear_series = lt(length=50)
        linear_series_int = TimeSeries.from_values(linear_series.values())
        linear_series_multi = linear_series.stack(linear_series)

        # univariate model + univariate series
        score = NaiveDrift().backtest(
            linear_series,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
        )
        self.assertEqual(score, 1.0)

        # very large train length should not affect the backtest
        score = NaiveDrift().backtest(
            linear_series,
            train_length=10000,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
        )
        self.assertEqual(score, 1.0)

        # using several metric function should not affect the backtest
        score = NaiveDrift().backtest(
            linear_series,
            train_length=10000,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=[r2_score, mape],
        )
        np.testing.assert_almost_equal(score, np.array([1.0, 0.0]))

        # window of size 2 is too small for naive drift
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(
                linear_series,
                train_length=2,
                start=pd.Timestamp("20000201"),
                forecast_horizon=3,
                metric=r2_score,
            )

        # test that it also works for time series that are not Datetime-indexed
        score = NaiveDrift().backtest(
            linear_series_int, start=0.7, forecast_horizon=3, metric=r2_score
        )
        self.assertEqual(score, 1.0)

        with self.assertRaises(ValueError):
            NaiveDrift().backtest(
                linear_series,
                start=pd.Timestamp("20000217"),
                forecast_horizon=3,
                overlap_end=False,
            )
        NaiveDrift().backtest(
            linear_series, start=pd.Timestamp("20000216"), forecast_horizon=3
        )
        NaiveDrift().backtest(
            linear_series,
            start=pd.Timestamp("20000217"),
            forecast_horizon=3,
            overlap_end=True,
        )

        # Using forecast_horizon default value
        NaiveDrift().backtest(linear_series, start=pd.Timestamp("20000216"))
        NaiveDrift().backtest(
            linear_series, start=pd.Timestamp("20000217"), overlap_end=True
        )

        # Using an int or float value for start
        NaiveDrift().backtest(linear_series, start=30)
        NaiveDrift().backtest(linear_series, start=0.7, overlap_end=True)

        # Set custom train window length
        NaiveDrift().backtest(linear_series, train_length=10, start=30)

        # Using invalid start and/or forecast_horizon values
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=0.7, forecast_horizon=-1)
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=-0.7, forecast_horizon=1)

        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=100)
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=1.2)
        with self.assertRaises(TypeError):
            NaiveDrift().backtest(linear_series, start="wrong type")
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, train_length=0, start=0.5)
        with self.assertRaises(TypeError):
            NaiveDrift().backtest(linear_series, train_length=1.2, start=0.5)
        with self.assertRaises(TypeError):
            NaiveDrift().backtest(linear_series, train_length="wrong type", start=0.5)

        with self.assertRaises(ValueError):
            NaiveDrift().backtest(
                linear_series, start=49, forecast_horizon=2, overlap_end=False
            )

        # univariate model + multivariate series
        with self.assertRaises(ValueError):
            FFT().backtest(
                linear_series_multi, start=pd.Timestamp("20000201"), forecast_horizon=3
            )

        # multivariate model + univariate series
        if TORCH_AVAILABLE:
            tcn_model = TCNModel(
                input_chunk_length=12, output_chunk_length=1, batch_size=1, n_epochs=1
            )
            pred = tcn_model.historical_forecasts(
                linear_series,
                start=pd.Timestamp("20000125"),
                forecast_horizon=3,
                verbose=False,
                last_points_only=True,
            )
            self.assertEqual(pred.width, 1)
            self.assertEqual(pred.end_time(), linear_series.end_time())

            # multivariate model + multivariate series
            with self.assertRaises(ValueError):
                tcn_model.backtest(
                    linear_series_multi,
                    start=pd.Timestamp("20000125"),
                    forecast_horizon=3,
                    verbose=False,
                )

            tcn_model = TCNModel(
                input_chunk_length=12, output_chunk_length=3, batch_size=1, n_epochs=1
            )
            pred = tcn_model.historical_forecasts(
                linear_series_multi,
                start=pd.Timestamp("20000125"),
                forecast_horizon=3,
                verbose=False,
                last_points_only=True,
            )
            self.assertEqual(pred.width, 2)
            self.assertEqual(pred.end_time(), linear_series.end_time())

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_backtest_regression(self):
        np.random.seed(4)

        gaussian_series = gt(mean=2, length=50)
        sine_series = st(length=50)
        features = gaussian_series.stack(sine_series)
        features_multivariate = (
            (gaussian_series + sine_series).stack(gaussian_series).stack(sine_series)
        )
        target = sine_series

        features = features.with_columns_renamed(
            features.components, ["Value0", "Value1"]
        )

        features_multivariate = features_multivariate.with_columns_renamed(
            features_multivariate.components, ["Value0", "Value1", "Value2"]
        )

        # univariate feature test
        score = LinearRegressionModel(
            lags=None, lags_future_covariates=[0, -1]
        ).backtest(
            series=target,
            future_covariates=features,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
            last_points_only=True,
        )
        self.assertGreater(score, 0.9)

        # univariate feature test + train length
        score = LinearRegressionModel(
            lags=None, lags_future_covariates=[0, -1]
        ).backtest(
            series=target,
            future_covariates=features,
            start=pd.Timestamp("20000201"),
            train_length=20,
            forecast_horizon=3,
            metric=r2_score,
            last_points_only=True,
        )
        self.assertGreater(score, 0.9)

        # Using an int or float value for start
        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features,
            start=30,
            forecast_horizon=3,
            metric=r2_score,
        )
        self.assertGreater(score, 0.9)

        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features,
            start=0.5,
            forecast_horizon=3,
            metric=r2_score,
        )
        self.assertGreater(score, 0.9)

        # Using a too small start value
        with self.assertRaises(ValueError):
            RandomForest(lags=12).backtest(series=target, start=0, forecast_horizon=3)

        with self.assertRaises(ValueError):
            RandomForest(lags=12).backtest(
                series=target, start=0.01, forecast_horizon=3
            )

        # Using RandomForest's start default value
        score = RandomForest(lags=12, random_state=0).backtest(
            series=target, forecast_horizon=3, start=0.5, metric=r2_score
        )
        self.assertGreater(score, 0.95)

        # multivariate feature test
        score = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
        )
        self.assertGreater(score, 0.94)

        # multivariate feature test with train window 35
        score_35 = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            train_length=35,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
        )
        logger.info(
            "Score for multivariate feature test with train window 35 is: ", score_35
        )
        self.assertGreater(score_35, 0.92)

        # multivariate feature test with train window 45
        score_45 = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            train_length=45,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
        )
        logger.info(
            "Score for multivariate feature test with train window 45 is: ", score_45
        )
        self.assertGreater(score_45, 0.94)
        self.assertGreater(score_45, score_35)

        # multivariate with stride
        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=r2_score,
            last_points_only=True,
            stride=3,
        )
        self.assertGreater(score, 0.9)

    def test_gridsearch(self):
        np.random.seed(1)

        dummy_series = get_dummy_series(ts_length=50)
        dummy_series_int_index = TimeSeries.from_values(dummy_series.values())

        theta_params = {"theta": list(range(3, 10))}
        self.assertTrue(compare_best_against_random(Theta, theta_params, dummy_series))
        self.assertTrue(
            compare_best_against_random(Theta, theta_params, dummy_series_int_index)
        )
        self.assertTrue(
            compare_best_against_random(Theta, theta_params, dummy_series, stride=2)
        )

        fft_params = {"nr_freqs_to_keep": [10, 50, 100], "trend": [None, "poly", "exp"]}
        self.assertTrue(compare_best_against_random(FFT, fft_params, dummy_series))

        es_params = {"seasonal_periods": list(range(5, 10))}
        self.assertTrue(
            compare_best_against_random(ExponentialSmoothing, es_params, dummy_series)
        )

    def test_gridsearch_metric_score(self):
        np.random.seed(1)

        model_class = Theta
        params = {"theta": list(range(3, 6))}
        dummy_series = get_dummy_series(ts_length=50)

        best_model, _, score = model_class.gridsearch(
            params,
            series=dummy_series,
            forecast_horizon=10,
            stride=1,
            start=dummy_series.time_index[-21],
        )
        recalculated_score = best_model.backtest(
            series=dummy_series,
            start=dummy_series.time_index[-21],
            forecast_horizon=10,
            stride=1,
        )

        self.assertEqual(score, recalculated_score, "The metric scores should match")

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_random_search(self):
        np.random.seed(1)

        dummy_series = get_dummy_series(ts_length=50)

        param_range = list(range(10, 20))
        params = {"lags": param_range}

        model = RandomForest(lags=1)
        result = model.gridsearch(
            params, dummy_series, forecast_horizon=1, n_random_samples=5
        )

        self.assertEqual(type(result[0]), RandomForest)
        self.assertEqual(type(result[1]["lags"]), int)
        self.assertEqual(type(result[2]), float)
        self.assertTrue(min(param_range) <= result[1]["lags"] <= max(param_range))

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_n_random_samples_bad_arguments(self):
        dummy_series = get_dummy_series(ts_length=50)

        params = {"lags": list(range(1, 11)), "past_covariates": list(range(1, 11))}

        with self.assertRaises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=-5
            )
        with self.assertRaises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=105
            )
        with self.assertRaises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=-24.56
            )
        with self.assertRaises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=1.5
            )

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_n_random_samples(self):
        np.random.seed(1)

        params = {"lags": list(range(1, 11)), "past_covariates": list(range(1, 11))}

        params_cross_product = list(product(*params.values()))

        # Test absolute sample
        absolute_sampled_result = RandomForest._sample_params(params_cross_product, 10)
        self.assertEqual(len(absolute_sampled_result), 10)

        # Test percentage sample
        percentage_sampled_result = RandomForest._sample_params(
            params_cross_product, 0.37
        )
        self.assertEqual(len(percentage_sampled_result), 37)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_n_jobs(self):
        """
        Testing that running gridsearch with multiple workers returns the same
        best_parameters as the single worker run.
        """

        np.random.seed(1)

        dummy_series = get_dummy_series(
            ts_length=100, lt_end_value=1, st_value_offset=0
        ).astype(np.float32)
        ts_train, ts_val = dummy_series.split_before(split_point=0.8)

        test_cases = [
            {
                "model": ARIMA,  # ExtendedForecastingModel
                "parameters": {"p": [18, 4], "q": [2, 3]},
            },
            {
                "model": BlockRNNModel,  # TorchForecastingModel
                "parameters": {
                    "input_chunk_length": [5, 10],
                    "output_chunk_length": [1, 3],
                    "n_epochs": [1, 5],
                    "random_state": [
                        42
                    ],  # necessary to avoid randomness among runs with same parameters
                },
            },
        ]

        for test in test_cases:

            model = test["model"]
            parameters = test["parameters"]

            np.random.seed(1)
            _, best_params1, _ = model.gridsearch(
                parameters=parameters, series=ts_train, val_series=ts_val, n_jobs=1
            )

            np.random.seed(1)
            _, best_params2, _ = model.gridsearch(
                parameters=parameters, series=ts_train, val_series=ts_val, n_jobs=-1
            )

            self.assertEqual(best_params1, best_params2)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_multi(self):
        dummy_series = st(length=40, value_y_offset=10).stack(
            lt(length=40, end_value=20)
        )
        tcn_params = {
            "input_chunk_length": [12],
            "output_chunk_length": [3],
            "n_epochs": [1],
            "batch_size": [1],
            "kernel_size": [2, 3, 4],
        }
        TCNModel.gridsearch(tcn_params, dummy_series, forecast_horizon=3, metric=mape)
