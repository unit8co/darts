import unittest
import numpy as np
import random

from ..backtesting import backtest_gridsearch, backtest_forecasting
from ..metrics import mape
from ..utils.timeseries_generation import linear_timeseries as lt, sine_timeseries as st, random_walk_timeseries as rt
from ..models import Theta, FFT, ExponentialSmoothing


def compare_best_against_random(model_class, params, series):

    # instantiate best model given parameters 'params'
    best_model = backtest_gridsearch(model_class, params, series, 10)

    # intantiate model with random parameters from 'params'
    random_param_choice = {}
    for key in params.keys():
        random_param_choice[key] = random.choice(params[key])
    random_model = model_class(**random_param_choice)

    # perform backtest forecasting on both models
    best_forecast = backtest_forecasting(series, best_model, series.time_index()[-20], 10)
    random_forecast = backtest_forecasting(series, random_model, series.time_index()[-20], 10)
    return mape(best_forecast, series) <= mape(random_forecast, series)


class BacktestingTestCase(unittest.TestCase):

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
