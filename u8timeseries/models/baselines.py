from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
import numpy as np


class KthValueAgoBaseline(AutoRegressiveModel):
    """
    A baseline model that always predict value of K time steps ago
    E.g., always predict last observed value when K=1, or last year value with K=12 (on monthly data)
    """

    def __init__(self, K=1):
        super(KthValueAgoBaseline, self).__init__()
        self.kth_value_ago = None
        self.K = K

    def __str__(self):
        return '{} value ago baseline'.format(self.K)

    def fit(self, series: TimeSeries):
        super(KthValueAgoBaseline, self).fit(series)
        assert len(series) >= self.K, 'The time series has to contain at least K={} points'.format(self.K)
        self.kth_value_ago = series.values()[-self.K]

    def predict(self, n):
        forecast = np.array([self.kth_value_ago] * n)
        return self._build_forecast_series(forecast)
