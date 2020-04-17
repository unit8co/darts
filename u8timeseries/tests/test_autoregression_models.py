import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from u8timeseries import Prophet, KthValueAgoBaseline, ExponentialSmoothing, TimeSeries, Arima, AutoArima
from u8timeseries.models.theta import Theta

class ModelsTestCase(unittest.TestCase):

    # forecasting horizon used in predictions
    forecasting_horizon = 5

    # dummy timeseries for autoregression forecasting
    times = pd.date_range('20000101', '20000130')
    values = np.sin(range(len(times))) + np.array([2.0] * len(times))
    ts: TimeSeries = TimeSeries.from_times_and_values(times, values)


    def test_autoregressive_models_runnability(self):
        models = [
            ExponentialSmoothing(), 
            Prophet(),
            Arima(1, 1, 1),
            AutoArima(),
            KthValueAgoBaseline(),
            Theta()
        ]

        for model in models:
            model.fit(self.ts)
            prediction = model.predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)
