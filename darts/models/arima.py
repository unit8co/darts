"""
ARIMA
-----

Models for ARIMA (Autoregressive integrated moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
"""

from statsmodels.tsa.arima.model import ARIMA as staARIMA
from typing import Optional, Tuple

from .forecasting_model import ExtendedForecastingModel
from ..timeseries import TimeSeries
from ..logging import get_logger
logger = get_logger(__name__)


class ARIMA(ExtendedForecastingModel):
    def __init__(self,
                 p: int = 12, d: int = 1, q: int = 0,
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: Optional[str] = None):
        """ ARIMA
        ARIMA-type models extensible with exogenous variables and seasonal components.

        Parameters
        ----------
        p : int
            Order (number of time lags) of the autoregressive model (AR)
        d : int
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted. (I)
        q : int
            The size of the moving average window (MA).
        seasonal_order: Tuple[int, int, int, int]
            The (P,D,Q,s) order of the seasonal component for the AR parameters,
            differences, MA parameters and periodicity
        trend: str
            Parameter controlling the deterministic trend. ‘n‘ indicates no trend,
            ‘c’ a constant term, ‘t’ linear trend in time, and ‘ct’ includes both.
             Default is ‘c’ for models without integration, and no trend for models with integration.
        """
        super().__init__()
        self.order = p, d, q
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None


    def __str__(self):
        if self.seasonal_order == (0, 0, 0, 0):
            return f'ARIMA{self.order}'
        return f'SARIMA{self.order}x{self.seasonal_order}'

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None):
        super().fit(series, exog)
        m = staARIMA(
            self.training_series.values(),
            exog=exog.values() if exog else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        )
        self.model = m.fit()

    def predict(self, n: int, exog: Optional[TimeSeries] = None):
        super().predict(n, exog)
        forecast = self.model.forecast(steps=n,
                                       exog=exog.values() if exog else None)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30
