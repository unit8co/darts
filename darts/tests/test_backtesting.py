import unittest
import numpy as np
import pandas as pd
import random

from darts.metrics import mape, r2_score
from darts.utils.timeseries_generation import (
    linear_timeseries as lt,
    sine_timeseries as st,
    random_walk_timeseries as rt,
    constant_timeseries as ct,
    gaussian_timeseries as gt
)
from darts.models import (
    Theta,
    FFT,
    ExponentialSmoothing,
    NaiveSeasonal,
    StandardRegressionModel,
    NaiveDrift,
)

from .base_test_class import DartsBaseTestClass
from ..logging import get_logger
logger = get_logger(__name__)

try:
    from ..models import TCNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch models are not installed - will not be tested for backtesting')
    TORCH_AVAILABLE = False


def compare_best_against_random(model_class, params, series):

    # instantiate best model in expanding window mode
    best_model_1, _ = model_class.gridsearch(params,
                                          series,
                                          forecast_horizon=10,
                                          metric=mape,
                                          start=series.time_index()[-21])

    # instantiate best model in split mode
    train, val = series.split_before(series.time_index()[-10])
    best_model_2, _ = model_class.gridsearch(params, train, val_series=val, metric=mape)

    # intantiate model with random parameters from 'params'
    random_param_choice = {}
    for key in params.keys():
        random_param_choice[key] = random.choice(params[key])
    random_model = model_class(**random_param_choice)

    # perform backtest forecasting on both models
    best_score_1 = best_model_1.backtest(series, start=series.time_index()[-21], forecast_horizon=10)
    random_score_1 = random_model.backtest(series, start=series.time_index()[-21], forecast_horizon=10)

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
        linear_series_multi = linear_series.stack(linear_series)

        # univariate model + univariate series
        score = NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000201'),
                                      forecast_horizon=3, metric=r2_score)
        self.assertEqual(score, 1.0)

        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000217'), forecast_horizon=3)
        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000217'), forecast_horizon=3,
                                  overlap_end=False)
        NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000216'), forecast_horizon=3)
        NaiveDrift().backtest(linear_series,
                              start=pd.Timestamp('20000217'),
                              forecast_horizon=3,
                              overlap_end=True)

        # Using forecast_horizon default value
        NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000216'))
        NaiveDrift().backtest(linear_series, start=pd.Timestamp('20000217'), overlap_end=True)

        # Using an int or float value for start
        NaiveDrift().backtest(linear_series, start=30)
        NaiveDrift().backtest(linear_series, start=0.7, overlap_end=True)

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
            NaiveDrift().backtest(linear_series, start='wrong type')

        with self.assertRaises(ValueError):
            NaiveDrift().backtest(linear_series, start=49, forecast_horizon=2, overlap_end=False)

        # univariate model + multivariate series
        with self.assertRaises(AssertionError):
            NaiveDrift().backtest(linear_series_multi, start=pd.Timestamp('20000201'), forecast_horizon=3)

        # multivariate model + univariate series
        if TORCH_AVAILABLE:
            tcn_model = TCNModel(input_chunk_length=12, output_chunk_length=1, batch_size=1, n_epochs=1)
            pred = tcn_model.historical_forecasts(linear_series,
                                                  start=pd.Timestamp('20000125'),
                                                  forecast_horizon=3,
                                                  verbose=False,
                                                  last_points_only=True)
            self.assertEqual(pred.width, 1)
            self.assertEqual(pred.end_time(), linear_series.end_time())

            # multivariate model + multivariate series
            with self.assertRaises(ValueError):
                tcn_model.backtest(linear_series_multi, start=pd.Timestamp('20000125'),
                                   forecast_horizon=3, verbose=False)

            tcn_model = TCNModel(input_chunk_length=12, output_chunk_length=3, batch_size=1, n_epochs=1)
            pred = tcn_model.historical_forecasts(linear_series_multi,
                                                  start=pd.Timestamp('20000125'),
                                                  forecast_horizon=3,
                                                  verbose=False,
                                                  last_points_only=True)
            self.assertEqual(pred.width, 2)
            self.assertEqual(pred.end_time(), linear_series.end_time())

    def test_backtest_regression(self):
        gaussian_series = gt(mean=2, length=50)
        sine_series = st(length=50)
        features = [gaussian_series + sine_series, gaussian_series]
        features_multivariate = [(gaussian_series + sine_series).stack(gaussian_series), gaussian_series]
        target = st(length=50)

        # univariate feature test
        score = StandardRegressionModel(15).backtest(features, target, pd.Timestamp('20000201'), 3, metric=r2_score)
        self.assertEqual(score, 1.0)

        # Using an int or float value for start
        score = StandardRegressionModel(15).backtest(features, target, start=30, forecast_horizon=3, metric=r2_score)
        self.assertEqual(score, 1.0)

        score = StandardRegressionModel(15).backtest(features, target, start=0.5, forecast_horizon=3, metric=r2_score)
        self.assertEqual(score, 1.0)

        # Using a too small start value
        with self.assertRaises(ValueError):
            StandardRegressionModel(15).backtest(features, target, start=0, forecast_horizon=3)

        with self.assertRaises(ValueError):
            StandardRegressionModel(15).backtest(features, target, start=0.01, forecast_horizon=3)

        # Using StandardRegressionModel's start default value
        score = StandardRegressionModel(15).backtest(features, target, forecast_horizon=3, metric=r2_score)
        self.assertEqual(score, 1.0)

        # multivariate feature test
        score = StandardRegressionModel(15).backtest(features_multivariate,
                                                     target,
                                                     pd.Timestamp('20000201'),
                                                     forecast_horizon=3,
                                                     metric=r2_score)
        self.assertEqual(score, 1.0)

        # multivariate target
        score = StandardRegressionModel(15).backtest(features_multivariate,
                                                     target.stack(target),
                                                     pd.Timestamp('20000201'),
                                                     forecast_horizon=3,
                                                     metric=r2_score)
        self.assertEqual(score, 1.0)

        # multivariate target with stride
        hist = StandardRegressionModel(15).historical_forecasts(features_multivariate,
                                                                target.stack(target),
                                                                pd.Timestamp('20000201'),
                                                                forecast_horizon=3,
                                                                stride=3,
                                                                last_points_only=True)
        self.assertEqual(r2_score(target.stack(target), hist), 1.0)
        self.assertEqual((hist.time_index()[1] - hist.time_index()[0]).days, 3)

    def test_gridsearch(self):

        np.random.seed(1)
        ts_length = 50
        dummy_series = (
            lt(length=ts_length, end_value=10) + st(length=ts_length, value_y_offset=10) + rt(length=ts_length)
        )

        theta_params = {'theta': list(range(3, 10))}
        self.assertTrue(compare_best_against_random(Theta, theta_params, dummy_series))

        fft_params = {'nr_freqs_to_keep': [10, 50, 100], 'trend': [None, 'poly', 'exp']}
        self.assertTrue(compare_best_against_random(FFT, fft_params, dummy_series))

        es_params = {'seasonal_periods': list(range(5, 10))}
        self.assertTrue(compare_best_against_random(ExponentialSmoothing, es_params, dummy_series))

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_gridsearch_multi(self):
        dummy_series = st(length=40, value_y_offset=10).stack(lt(length=40, end_value=20))
        tcn_params = {
            'input_chunk_length': [12],
            'output_chunk_length': [3],
            'n_epochs': [1],
            'batch_size': [1],
            'kernel_size': [2, 3, 4]
        }
        TCNModel.gridsearch(tcn_params,
                            dummy_series,
                            forecast_horizon=3,
                            metric=mape)

    def test_forecasting_residuals(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = ct(length=20)
        residuals = model.residuals(constant_ts)
        np.testing.assert_almost_equal(residuals.univariate_values(), np.zeros(len(residuals)))

        # test constant, positive residuals
        linear_ts = lt(length=20)
        residuals = model.residuals(linear_ts)
        np.testing.assert_almost_equal(np.diff(residuals.univariate_values()), np.zeros(len(residuals) - 1))
        np.testing.assert_array_less(np.zeros(len(residuals)), residuals.univariate_values())
