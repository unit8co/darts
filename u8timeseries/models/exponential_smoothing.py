"""
Implementation of an Simple Exponential Smoothing model.
--------------------------------------------------------
"""

from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import time_log, get_logger
import statsmodels.tsa.holtwinters as hw

logger = get_logger(__name__)

class ExponentialSmoothing(AutoRegressiveModel):
    """
    Implementation of a Simple Exponential Smoothing.

    Currently just a wrapper around the statsmodels holtwinter implementation.

    :param trend: A string for the type of trend to consider: either `additive` (default) or `multiplicative`.
    :param seasonal: A string for the type of seasonality to consider: either `additive` (default) or `multiplicative`.
    :param seasonal_periods: An integer, the order of seasonality to consider.
    """

    def __init__(self, trend: str = 'additive', seasonal: str = 'additive', seasonal_periods: int = 12):
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def __str__(self):
        return 'Exponential smoothing'

    @time_log(logger=logger)
    def fit(self, series: TimeSeries):
        super().fit(series)
        self.model = hw.ExponentialSmoothing(series.values(),
                                             trend=self.trend,
                                             seasonal=self.seasonal,
                                             seasonal_periods=self.seasonal_periods).fit()

    def predict(self, n):
        super().predict(n)
        forecast = self.model.forecast(n)
        return self._build_forecast_series(forecast)
