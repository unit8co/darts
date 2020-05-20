import unittest
import numpy as np
import random
import logging

from ..backtesting import backtest_gridsearch, backtest_forecasting, forecasting_residuals
from ..metrics import mape
from ..utils.timeseries_generation import (
    linear_timeseries as lt,
    sine_timeseries as st,
    random_walk_timeseries as rt,
    constant_timeseries as ct
)
from ..models import Theta, FFT, ExponentialSmoothing, NaiveSeasonal


def compare_best_against_random(model_class, params, series):

    # instantiate best model in expanding window mode
    best_model_1 = backtest_gridsearch(model_class, params, series, fcast_horizon_n=10, metric=mape)

    # instantiate best model in split mode
    train, val = series.split_before(series.time_index()[-10])
    best_model_2 = backtest_gridsearch(model_class, params, train, val_series=val, metric=mape)

    # intantiate model with random parameters from 'params'
    random_param_choice = {}
    for key in params.keys():
        random_param_choice[key] = random.choice(params[key])
    random_model = model_class(**random_param_choice)

    # perform backtest forecasting on both models
    best_forecast_1 = backtest_forecasting(series, best_model_1, series.time_index()[-21], 10)
    random_forecast_1 = backtest_forecasting(series, random_model, series.time_index()[-21], 10)

    # perform train/val evaluation on both models
    best_model_2.fit(train)
    best_forecast_2 = best_model_2.predict(len(val))
    random_model = model_class(**random_param_choice)
    random_model.fit(train)
    random_forecast_2 = random_model.predict(len(val))

    # check whether best models are at least as good as random models
    expanding_window_ok = mape(best_forecast_1, series) <= mape(random_forecast_1, series)
    split_ok = mape(best_forecast_2, val) <= mape(random_forecast_2, val)

    return expanding_window_ok and split_ok


class BacktestingTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_backtest_gridsearch(self):

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

    def test_forecasting_residuals(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = ct(length=20)
        residuals = forecasting_residuals(model, constant_ts)
        np.testing.assert_almost_equal(residuals.values(), np.zeros(len(residuals)))

        # test constant, positive residuals
        linear_ts = lt(length=20)
        residuals = forecasting_residuals(model, linear_ts)
        np.testing.assert_almost_equal(np.diff(residuals.values()), np.zeros(len(residuals) - 1))
        np.testing.assert_array_less(np.zeros(len(residuals)), residuals.values())
