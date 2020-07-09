"""
Baseline Models
---------------

A collection of simple benchmark models.
"""

from typing import Optional
import numpy as np

from .forecasting_model import UnivariateForecastingModel
from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger

logger = get_logger(__name__)


class NaiveMean(UnivariateForecastingModel):
    def __init__(self):
        """ Naive Mean Model

            This model has no parameter, and always predicts the
            mean value of the training series.
        """
        super().__init__()
        self.mean_val = None

    def __str__(self):
        return 'Naive mean predictor model'

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        self.mean_val = np.mean(series.univariate_values())

    def predict(self, n: int):
        super().predict(n)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)


class NaiveSeasonal(UnivariateForecastingModel):
    def __init__(self, K: int = 1):
        """ Naive Seasonal Model

        This model always predicts the value of `K` time steps ago.
        When :math:`K=1`, this model predicts the last value of the training set.
        When :math:`K>1`, it repeats the last :math:`K` values of the training set.

        Parameters
        ----------
        K
            the number of last time steps of the training set to repeat
        """
        super().__init__()
        self.last_k_vals = None
        self.K = K

    @property
    def min_train_series_length(self):
        return max(self.K, 3)

    def __str__(self):
        return 'Naive seasonal model, with K={}'.format(self.K)

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        raise_if_not(len(series) >= self.K, 'The time series requires at least K={} points'.format(self.K), logger)
        self.last_k_vals = series.univariate_values()[-self.K:]

    def predict(self, n: int):
        super().predict(n)
        forecast = np.array([self.last_k_vals[i % self.K] for i in range(n)])
        return self._build_forecast_series(forecast)


class NaiveDrift(UnivariateForecastingModel):
    def __init__(self):
        """ Naive Drift Model

            This model fits a line between the first and last point of the training series,
            and extends it in the future. For a training series of length :math:`T`, we have:

            .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)
        """
        super().__init__()

    def __str__(self):
        return 'Naive drift model'

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        series = self.training_series

    def predict(self, n: int):
        super().predict(n)
        first, last = self.training_series.first_value(), self.training_series.last_value()
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n)
        return self._build_forecast_series(forecast)
