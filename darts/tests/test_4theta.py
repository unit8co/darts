import unittest
import logging
import numpy as np
import random

from .. import Season, Trend, Model
from ..models import Theta, FourTheta
from ..metrics import mape
from ..utils.timeseries_generation import (
    linear_timeseries as lt,
    sine_timeseries as st,
    random_walk_timeseries as rt,
)


class FourThetaTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_input(self):
        with self.assertRaises(ValueError):
            FourTheta(model_mode=Season.ADDITIVE)
        with self.assertRaises(ValueError):
            FourTheta(season_mode=Model.ADDITIVE)
        with self.assertRaises((ValueError, TypeError)):
            FourTheta(trend_mode='linear')

    def test_negative_series(self):
        sine_series = st(length=50)
        model = FourTheta(model_mode=Model.MULTIPLICATIVE, trend_mode=Trend.EXPONENTIAL)
        model.fit(sine_series)
        self.assertTrue(model.model_mode is Model.ADDITIVE and model.trend_mode is Trend.LINEAR)

    def test_theta(self):
        random.seed(1)
        series = rt(length=50, mean=100)
        theta_param = random.randrange(-5, 5)
        theta = Theta(theta_param)
        fourtheta = FourTheta(2 - theta_param)
        theta.fit(series)
        fourtheta.fit(series)
        forecast_theta = theta.predict(20)
        forecast_fourtheta = fourtheta.predict(20)
        self.assertTrue((forecast_theta - forecast_fourtheta <= 1e-12).all()[0])

    def test_best_model(self):
        random.seed(1)
        sine_series = st(length=50, value_y_offset=50)
        linear_series = lt(length=50)
        series = sine_series + linear_series
        train_series, val_series = series.split_before(series.time_index()[-10])
        thetas = np.linspace(-3, 3, 30)
        best_model = FourTheta.select_best_model(train_series, thetas)
        model = FourTheta(random.choice(thetas), model_mode=random.choice(list(Model)),
                          trend_mode=random.choice(list(Trend)), season_mode=random.choice(list(Season)))
        model.fit(train_series)
        best_model.fit(train_series)
        forecast_random = model.predict(10)
        forecast_best = model.predict(10)
        self.assertTrue(mape(val_series, forecast_best) <= mape(val_series, forecast_random))
