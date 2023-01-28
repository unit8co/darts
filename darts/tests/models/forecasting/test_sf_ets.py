import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.datasets import AirPassengersDataset

# from darts.metrics import mape
from darts.models import StatsForecastETS
from darts.tests.base_test_class import DartsBaseTestClass


class StatsForecastETSTestCase(DartsBaseTestClass):
    # real timeseries for functionality tests
    ts_passengers = AirPassengersDataset().load()
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp("19570101"))

    # as future covariates we want a trend
    trend_values = np.arange(start=1, stop=len(ts_passengers) + 1)
    trend_times = ts_passengers.time_index
    ts_trend = TimeSeries.from_times_and_values(
        times=trend_times, values=trend_values, columns=["trend"]
    )
    ts_trend_train, ts_trend_val = ts_trend.split_after(pd.Timestamp("19570101"))

    def test_fit_on_residuals(self):
        model = StatsForecastETS(season_length=12, model="ZZN")

        # test if we are indeed fitting the AutoETS on the residuals of the linear regression
        model.fit(series=self.ts_pass_train, future_covariates=self.ts_trend_train)

        # check if linear regression was fit
        self.assertIsNotNone(model._linreg)
        self.assertTrue(model._linreg._fit_called)

        # create the residuals from the linear regression
        fitted_values = model._linreg.model.predict(
            X=self.ts_trend_train.values(copy=False)
        )
        fitted_values_ts = TimeSeries.from_times_and_values(
            times=self.ts_pass_train.time_index, values=fitted_values
        )
        resids = self.ts_pass_train - fitted_values_ts

        # now make in-sample predictions with the AutoETS model
        in_sample_preds = model.model.predict_in_sample()["fitted"]
        ts_in_sample_preds = TimeSeries.from_times_and_values(
            times=self.ts_pass_train.time_index, values=in_sample_preds
        )

        # compare in-sample predictions to the residuals they have supposedly been fitted on
        # current_mape = mape(resids, ts_in_sample_preds)

        return resids, ts_in_sample_preds
