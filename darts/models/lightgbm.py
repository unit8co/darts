"""
LightGBM
----------------------
"""

from lightgbm import LGBMRegressor
import numpy as np

from darts.models.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries
from darts.logging import get_logger

logger = get_logger(__name__)


class LightGBM(ForecastingModel):
    def __init__(self, *lightgbm_args, **lightgbm_kwargs):
        """ LightGBM

        This implementation is a thin wrapper around lightgbm Python library

        Parameters
        ----------
        lightgbm_args
            Positional arguments for the lightgbm model
        lightgbm_kwargs
            Keyword arguments for the lightgbm model
        """

        super().__init__()
        self.model = LGBMRegressor(*lightgbm_args, **lightgbm_kwargs)
        self.last_tick = None

    def __str__(self):
        return 'LightGBM'

    def fit(self, series: TimeSeries):
        super().fit(series)
        series = self.training_series  # defined in super()

        x_train = series.time_index()
        y = series.univariate_values()
        self.last_tick = x_train[-1]

        # Reshape data
        x_train = np.array(x_train).reshape(x_train.shape[0], 1)

        self.model.fit(x_train, y)

    def predict(self, n):
        super().predict(n)
        x_test = super()._generate_new_dates(n)

        # Reshape data
        x_test = np.array(x_test).reshape(x_test.shape[0], 1)
        forecast = self.model.predict(x_test)
        return self._build_forecast_series(forecast)

    def predict_time_series(self, val: TimeSeries):
        super().predict(len(val))
        x_test = val.time_index()

        # Reshape data
        x_test = np.array(x_test).reshape(x_test.shape[0], 1)
        forecast = self.model.predict(x_test)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('AirPassengers.csv', delimiter=",")
    series = TimeSeries.from_dataframe(df, 'Month', ['#Passengers'])

    train, val = series.split_before(pd.Timestamp('19540101'))

    model = LightGBM()
    model.fit(train)
    naive_forecast = model.predict_time_series(val)

    series.plot(label='actual')
    naive_forecast.plot(label='naive forecast (K=1)')
