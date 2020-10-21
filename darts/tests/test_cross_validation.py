import unittest
import logging
from ..utils.cross_validation import generalized_rolling_origin_evaluation as groe
from ..models import ExponentialSmoothing, NaiveSeasonal
from ..utils.timeseries_generation import (random_walk_timeseries as rt, constant_timeseries as ct,
                                           sine_timeseries as st, linear_timeseries as lt)
from ..metrics import mape, mae, mase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CrossValidationTestCase(unittest.TestCase):
    series0 = ct(value=0, length=50)
    series1 = rt(length=50)
    series2 = st(length=50)
    series3 = lt(length=50)
    model1 = NaiveSeasonal(K=10)
    model2 = ExponentialSmoothing(seasonal_periods=10)

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_groe_input_metric(self):
        groe(self.series1, self.model1, metric=mape, stride=5)
        groe(self.series1, self.model1, metric='mape', stride=5)
        groe(self.series1, self.model1,
             metric=lambda ts1, ts2: 0.5 * mape(ts1, ts2) + 0.5 * mae(ts1, ts2), stride=5)
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric='plop', stride=5)

    def test_groe_input_params(self):
        groe(self.series1, self.model1, metric=mape, stride=5, n_prediction_steps=9)
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape)
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape, stride=-5, n_prediction_steps=10)

    def test_groe_input_origin(self):
        # small time series
        groe(self.series1, self.model1, metric=mape, stride=1, first_origin=2)
        groe(self.series1, self.model1, metric=mape, stride=1, first_origin=48)
        # impossible values
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape, stride=1, first_origin=52)
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape, stride=1, first_origin=0)

    def test_groe_input_timestamp(self):
        # small time series
        groe(self.series1, self.model1, metric=mape, stride=1, first_origin=pd.Timestamp('2000-01-03'))
        groe(self.series1, self.model1, metric=mape, stride=1, first_origin=pd.Timestamp('2000-02-18'))
        # impossible values
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape, stride=1, first_origin=pd.Timestamp('2000-02-25'))
        with self.assertRaises(ValueError):
            groe(self.series1, self.model1, metric=mape, stride=1, first_origin=pd.Timestamp('2000-01-01'))

    def test_groe_inf_output(self):
        # metric
        self.assertEqual(np.inf, groe(self.series0, self.model1, metric=mape, stride=1))
        # model
        self.assertEqual(np.inf, groe(self.series2, self.model2, metric=mape, first_origin=5, stride=1))

    def test_groe_ouput(self):
        # test 1
        value = groe(self.series2, self.model1, first_origin=35, n_prediction_steps=2, forecast_horizon=10)
        self.assertAlmostEqual(value, 0)

        # test 2
        combined = (self.series1 + self.series2) * 10
        value = groe(combined, self.model1, first_origin=35, n_prediction_steps=2, forecast_horizon=10, stride=5, metric=mase)

        comb1, rest1 = combined[:35], combined[35:45]
        comb2, rest2 = combined[:40], combined[40:50]

        self.model1.fit(comb1)
        fcast1 = self.model1.predict(10)
        err1 = mase(rest1, fcast1, comb1) * 10
        
        self.model1.fit(comb2)
        fcast2 = self.model1.predict(10)
        err2 = mase(rest2, fcast2, comb2) * 10

        self.assertEqual(value, err1 + err2)

        # test 3
        value = groe(combined, self.model1, first_origin=37, n_prediction_steps=2, forecast_horizon=10, stride=5, metric=mase)

        comb1, rest1 = combined[:37], combined[37:47]
        comb2, rest2 = combined[:42], combined[42:50]

        self.model1.fit(comb1)
        fcast1 = self.model1.predict(10)
        err1 = mase(rest1, fcast1, comb1) * 10
        
        self.model1.fit(comb2)
        fcast2 = self.model1.predict(8)
        err2 = mase(rest2, fcast2, comb2) * 8

        self.assertEqual(value, err1 + err2)

if __name__ == '__main__':
    unittest.main()
