"""
Exponential Smoothing
---------------------
"""

from typing import Optional
import statsmodels.tsa.holtwinters as hw

from .forecasting_model import UnivariateForecastingModel
from ..logging import get_logger
from ..timeseries import TimeSeries

logger = get_logger(__name__)


class ExponentialSmoothing(UnivariateForecastingModel):
    def __init__(self,
                 trend: Optional[str] = 'additive',
                 damped: Optional[bool] = False,
                 seasonal: Optional[str] = 'additive',
                 seasonal_periods: Optional[int] = 12,
                 **fit_kwargs):
        """ Exponential Smoothing

        This is a wrapper around
        `statsmodels  Holt-Winters' Exponential Smoothing
        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html>`_.

        We refer to this link for the original and more complete documentation of the parameters.

        Parameters
        ----------
        trend
            Type of trend component
        damped
            Should the trend component be damped.
        seasonal
            Type of seasonal component
        seasonal_periods
            The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily
            data with a weekly cycle.
        fit_kwargs
            Some optional keyword arguments that will be used to call
            `statsmodels.tsa.holtwinters.ExponentialSmoothing.fit()`.
            See `the documentation
            <https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html>`_.
        """
        super().__init__()
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fit_kwargs = fit_kwargs
        self.model = None

    def __str__(self):
        return 'Exponential smoothing'

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        series = self.training_series
        hw_model = hw.ExponentialSmoothing(series.values(),
                                           trend=self.trend,
                                           damped=self.damped,
                                           seasonal=self.seasonal,
                                           seasonal_periods=self.seasonal_periods)

        hw_results = hw_model.fit(**self.fit_kwargs)
        self.model = hw_results

    def predict(self, n):
        super().predict(n)
        forecast = self.model.forecast(n)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        if (self.seasonal_periods is not None and self.seasonal_periods > 1):
            return 2 * self.seasonal_periods
        return 3
