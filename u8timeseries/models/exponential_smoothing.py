"""
Exponential Smoothing
---------------------
"""

from .forecasting_model import ForecastingModel
from ..timeseries import TimeSeries
from ..custom_logging import time_log, get_logger
import statsmodels.tsa.holtwinters as hw

logger = get_logger(__name__)

class ExponentialSmoothing(ForecastingModel):
    def __init__(self, trend: str = 'additive', seasonal: str = 'additive', seasonal_periods: int = 12):
        """ Exponential Smoothing

        Parameters
        ----------
        trend
        seasonal
        seasonal_periods
        """
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
