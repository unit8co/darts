"""
Implementation of baseline model.
-------------------------------------------
"""

from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, time_log, get_logger
import numpy as np

logger = get_logger(__name__)

class KthValueAgoBaseline(AutoRegressiveModel):
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
        return '{} value ago baseline'.format(self.K)

    @time_log(logger=logger)
    def fit(self, series: 'TimeSeries'):
        super().fit(series)
        raise_if_not(len(series) >= self.K, 'The time series has to contain at least K={} points'.format(self.K), logger)
        self.last_k_vals = series.values()[-self.K:]

    def predict(self, n: int):
        super().predict(n)
        forecast = np.array([self.last_k_vals[i % self.K] for i in range(n)])
        return self._build_forecast_series(forecast)
