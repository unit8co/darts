"""
Moving Average
-------------------------------
"""
import numpy as np

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class MovingAverage(ForecastingModel):
    def __init__(self, window: int):
        """Moving Average Model

        Parameters
        ----------
        window
            The length of the window over which to average values.
        """
        super().__init__()
        self.window = window
        self.mean_val = None

    def __str__(self):
        return "Moving average predictor model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.mean_val = series[-self.window :].univariate_values().mean()
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)
