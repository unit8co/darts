import logging
import unittest

import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape, overall_percentage_error, mase
from u8timeseries import Prophet, KthValueAgoBaseline, ExponentialSmoothing, TimeSeries, Arima, AutoArima
from u8timeseries.models.theta import Theta
from ..models.fft import FFT
from ..models.RNN_model import RNNModel

class AutoregressionModelsTestCase(unittest.TestCase):

    # forecasting horizon used in runnability tests
    forecasting_horizon = 5

    # dummy timeseries for runnability tests
    np.random.seed(1)
    ts_gaussian = tg.gaussian_timeseries(length=100, mean=50)

    # real timeseries for functionality tests
    df = pd.read_csv('examples/AirPassengers.csv', delimiter=",")
    ts_passengers = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp('19570101'))

    # autoregressive models - maximum error tuples
    models = [
        (ExponentialSmoothing(), 4.8),
        (Prophet(), 13.5),
        (Arima(0, 1, 1), 17.1),
        (Arima(1, 1, 1), 14.2),
        (AutoArima(), 13.7),
        (Theta(), 11.3),
        (Theta(1), 20.2),
        (Theta(3), 9.8),
        (FFT(trend='poly'), 11.4),
        (KthValueAgoBaseline(), 32.4),
    ]

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_models_runnability(self):
        for model, _ in self.models:
            model.fit(self.ts_gaussian)
            prediction = model.predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in self.models:
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            current_mape = mape(prediction, self.ts_pass_val)
            self.assertTrue(current_mape < max_mape, "{} model exceeded the maximum MAPE of {}." \
                            "with a MAPE of {}".format(str(model), max_mape, current_mape))


            
