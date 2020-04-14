import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from u8timeseries import StandardRegressiveModel

class RegressionModelsTestCase(unittest.TestCase):

    # number of data points used for training
    regression_window = 5

    # dummy feature and target timeseries for regression
    times = pd.date_range('20000101', '20000130')
    target_values = np.sin(range(len(times))) + np.array([2.0] * len(times))
    feature_values = np.array(range(len(times)), dtype=float)
    target_ts: TimeSeries = TimeSeries.from_times_and_values(times, target_values)
    feature_ts: TimeSeries = TimeSeries.from_times_and_values(times, feature_values)

    
    def test_regressive_models_runnability(self):
        models = [
            StandardRegressiveModel(self.regression_window)
        ]

        for model in models:
            # training and predicting on same features, since only runnability is tested
            model.fit([self.feature_ts], self.target_ts)
            prediction = model.predict([self.feature_ts])
            self.assertTrue(len(prediction) == len(self.feature_ts))
