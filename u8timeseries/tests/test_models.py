import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from u8timeseries import Prophet, KthValueAgoBaseline, ExponentialSmoothing, TimeSeries, Arima, AutoArima
from u8timeseries import StandardRegressiveModel
from u8timeseries.models.theta import Theta

class ModelsTestCase(unittest.TestCase):
    __test__ = True

    # general prediction / training parameters
    forecasting_horizon = 5
    regression_window = 5

    # dummy timeseries for autoregression forecasting and target for regression
    times = pd.date_range('20000101', '20000130')
    values = np.sin(range(len(times))) + np.array([2.0] * len(times))
    ts: TimeSeries = TimeSeries.from_times_and_values(times, values)

    # dummy feature timeseries for regression
    feature_values = np.array(range(len(times))).astype(float)
    feature_ts: TimeSeries = TimeSeries.from_times_and_values(times, feature_values)

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

    
    def test_regressive_models_runnability(self):
        models = [
            StandardRegressiveModel(self.regression_window)
        ]

        for model in models:
            # training and predicting on same features, since only runnability is tested
            model.fit([self.feature_ts], self.ts)
            prediction = model.predict([self.feature_ts])
            self.assertTrue(len(prediction) == len(self.feature_ts))

if __name__ == "__main__":
    unittest.main()