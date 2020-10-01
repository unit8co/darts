"""
ARIMA
-----

Models for ARIMA (Autoregressive integrated moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
"""

from statsmodels.tsa.arima_model import ARMA as staARMA
from statsmodels.tsa.arima_model import ARIMA as staARIMA

from .forecasting_model import UnivariateForecastingModel
from ..timeseries import TimeSeries
from ..logging import get_logger

logger = get_logger(__name__)


class ARIMA(UnivariateForecastingModel):
    def __init__(self, p: int = 12, d: int = 1, q: int = 0):
        """ ARIMA

        Parameters
        ----------
        p : int
            Order (number of time lags) of the autoregressive model (AR)
        d : int
            The order of differentiation; i.e., the number of times the data have had past values subtracted. (I)
        q : int
            The size of the moving average window (MA).
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def __str__(self):
        return 'ARIMA({},{},{})'.format(self.p, self.d, self.q)

    def fit(self, series: TimeSeries):
        super().fit(series)
        series = self.training_series
        m = staARIMA(series.values(),
                     order=(self.p, self.d, self.q)) if self.d > 0 else staARMA(series.values(), order=(self.p, self.q))
        self.model = m.fit(disp=0)

    def predict(self, n):
        super().predict(n)
        forecast = self.model.forecast(steps=n)[0]
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30
