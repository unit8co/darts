import unittest
import pandas as pd
import numpy as np
import logging

from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape, overall_percentage_error, mase
from u8timeseries import Prophet, KthValueAgoBaseline, ExponentialSmoothing, TimeSeries, Arima, AutoArima
from u8timeseries.models.theta import Theta

class AutoregressionModelsTestCase(unittest.TestCase):

    # forecasting horizon used in runnability tests
    forecasting_horizon = 5

    # dummy timeseries for runnability tests
    np.random.seed(0)
    ts_gaussian = tg.gaussian_timeseries(length=100, mean=50)

    # real timeseries for functionality tests
    df = pd.read_csv('examples/AirPassengers.csv', delimiter=",")
    ts_passengers = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp('19570101'))

    # default autoregressive models
    models = [
        ExponentialSmoothing(), 
        Prophet(),
        Arima(1, 1, 1),
        AutoArima(),
        Theta()
    ]
    baseline_models = [
        KthValueAgoBaseline()
    ]

    # maximum error values for baselines
    max_mape_baseline = 40

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_models_runnability(self):
        for model in (self.models + self.baseline_models):
            model.fit(self.ts_gaussian)
            prediction = model.predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_baseline_models(self):
        # for every baseline model, check whether its errors do not exceed the given bounds
        for baseline in self.baseline_models:
            baseline.fit(self.ts_pass_train)
            prediction = baseline.predict(len(self.ts_pass_val))
            self.assertTrue(mape(prediction, self.ts_pass_val) < self.max_mape_baseline, 
                            "{} baseline model exceeded the maximum MAPE of {}.".format(str(baseline), self.max_mape_baseline))

    def test_models_against_baselines(self):
        # iterate through all baseline models and save best scores
        best_mape = 100
        for baseline in self.baseline_models:
            baseline.fit(self.ts_pass_train)
            prediction = baseline.predict(len(self.ts_pass_val))
            best_mape = min(mape(prediction, self.ts_pass_val), best_mape)

        # iterate through all models and check if they are at least as good as the baselines
        for model in self.models:
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            self.assertTrue(mape(prediction, self.ts_pass_val) < best_mape, 
                            "{} model performed worse than baseline.".format(str(model)))

            
