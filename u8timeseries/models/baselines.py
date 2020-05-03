"""
Baseline Models
---------------

A collection of simple benchmark models.
"""

from .forecasting_model import ForecastingModel
from ..timeseries import TimeSeries
from ..logging import raise_if_not, time_log, get_logger
import numpy as np

logger = get_logger(__name__)


class NaiveMean(ForecastingModel):
    def __init__(self):
        super().__init__()
        self.mean_val = None

    def __str__(self):
        return 'Naive mean predictor model'

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.mean_val = np.mean(series.values())

    def predict(self, n: int):
        super().predict(n)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)


class NaiveSeasonal(ForecastingModel):
    """
    A baseline model that always predict value of `k` time steps ago.


    More precisely, at last know time value t, the prediction for t + i is given by the value at time t - k + i.

    :param k: An integer, determines how far to fetch the prediction value.
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.last_k_vals = None
        self.K = k

    def __str__(self):
        return 'Naive seasonal model, with K={}'.format(self.K)

    @time_log(logger=logger)
    def fit(self, series: TimeSeries):
        super().fit(series)
        raise_if_not(len(series) >= self.K, 'The time series has to contain at least K={} points'.format(self.K), logger)
        self.last_k_vals = series.values()[-self.K:]

    def predict(self, n: int):
        super().predict(n)
        forecast = np.array([self.last_k_vals[i % self.K] for i in range(n)])
        return self._build_forecast_series(forecast)


class NaiveDrift(ForecastingModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Naive drift model'

    def fit(self, series: TimeSeries):
        super().fit(series)

    def predict(self, n: int):
        super().predict(n)
        first, last = self.training_series.first_value(), self.training_series.last_value()
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n)
        return self._build_forecast_series(forecast)
