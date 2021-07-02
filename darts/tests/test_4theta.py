import unittest
import numpy as np
import random

from .base_test_class import DartsBaseTestClass
from ..utils.utils import SeasonalityMode, TrendMode, ModelMode
from ..models import Theta, FourTheta
from ..metrics import mape
from ..utils.timeseries_generation import (
    linear_timeseries as lt,
    sine_timeseries as st,
    random_walk_timeseries as rt,
)


class FourThetaTestCase(DartsBaseTestClass):
    def test_input(self):
        with self.assertRaises(ValueError):
            FourTheta(model_mode=SeasonalityMode.ADDITIVE)
        with self.assertRaises(ValueError):
            FourTheta(season_mode=ModelMode.ADDITIVE)
        with self.assertRaises((ValueError, TypeError)):
            FourTheta(trend_mode='linear')

    def test_negative_series(self):
        sine_series = st(length=50)
        model = FourTheta(model_mode=ModelMode.MULTIPLICATIVE, trend_mode=TrendMode.EXPONENTIAL,
                          season_mode=SeasonalityMode.ADDITIVE, normalization=False)
        model.fit(sine_series)
        self.assertTrue(model.model_mode is ModelMode.ADDITIVE and model.trend_mode is TrendMode.LINEAR)

    def test_zero_mean(self):
        sine_series = st(length=50)
        with self.assertRaises(ValueError):
            model = FourTheta(model_mode=ModelMode.MULTIPLICATIVE, trend_mode=TrendMode.EXPONENTIAL)
            model.fit(sine_series)

    def test_theta(self):
        np.random.seed(1)
        series = rt(length=50, mean=100)
        theta_param = np.random.randint(1, 5)
        theta = Theta(theta_param)
        fourtheta = FourTheta(theta_param, normalization=False)
        theta.fit(series)
        fourtheta.fit(series)
        forecast_theta = theta.predict(20)
        forecast_fourtheta = fourtheta.predict(20)
        weighted_delta = (forecast_theta - forecast_fourtheta)/forecast_theta
        self.assertTrue((weighted_delta <= 3e-5).all().item())

    def test_best_model(self):
        random.seed(1)
        sine_series = st(length=50, value_y_offset=50)
        linear_series = lt(length=50)
        series = sine_series + linear_series
        train_series, val_series = series.split_before(series.time_index[-10])
        thetas = np.linspace(-3, 3, 30)
        best_model, _ = FourTheta.select_best_model(train_series, thetas)
        model = FourTheta(random.choice(thetas), model_mode=random.choice(list(ModelMode)),
                          trend_mode=random.choice(list(TrendMode)), season_mode=random.choice(list(SeasonalityMode)))
        model.fit(train_series)
        best_model.fit(train_series)
        forecast_random = model.predict(10)
        forecast_best = best_model.predict(10)
        self.assertTrue(mape(val_series, forecast_best) <= mape(val_series, forecast_random))
