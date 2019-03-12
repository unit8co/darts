from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
import numpy as np


class KthValueAgoBaseline(AutoRegressiveModel):
    """
    A baseline model that always predict value of K time steps ago
    E.g., always predict last observed value when K=1, or last year value with K=12 (on monthly data)
    """

    def __init__(self, K=1):
        super().__init__()
        self.last_k_vals = None
        self.K = K

    def __str__(self):
        return '{} value ago baseline'.format(self.K)

    def fit(self, series: TimeSeries):
        super().fit(series)
        assert len(series) >= self.K, 'The time series has to contain at least K={} points'.format(self.K)
        self.last_k_vals = series.values()[-self.K:]

    def predict(self, n):
        super().predict(n)
        forecast = np.array([self.last_k_vals[i % self.K] for i in range(n)])
        return self._build_forecast_series(forecast)
