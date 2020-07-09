"""
ARIMA and Auto-ARIMA
--------------------

Models for ARIMA (Autoregressive integrated moving average) and auto-ARIMA [1]_.
The implementations are wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_
and `pmdarima <https://github.com/alkaline-ml/pmdarima>`_.

References
----------
.. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
"""

from typing import Optional
from statsmodels.tsa.arima_model import ARMA as staARMA
from statsmodels.tsa.arima_model import ARIMA as staARIMA
from pmdarima import AutoARIMA as PmdAutoARIMA

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

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
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


class AutoARIMA(UnivariateForecastingModel):

    def __init__(self, *autoarima_args, **autoarima_kwargs):
        """ Auto-ARIMA

        This implementation is a thin wrapper around
        `pmdarima AutoARIMA model <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_,
        which provides functionality similar
        to R's `auto.arima <https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima>`_.

        This model supports the same parameters as the pmdarima AutoARIMA model.
        See `pmdarima documentation <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html>`_
        for an extensive documentation and a list of supported parameters.

        Parameters
        ----------
        autoarima_args
            Positional arguments for the pmdarima AutoARIMA model
        autoarima_kwargs
            Keyword arguments for the pmdarima AutoARIMA model
        """

        super().__init__()
        self.model = PmdAutoARIMA(*autoarima_args, **autoarima_kwargs)

    def __str__(self):
        return 'Auto-ARIMA'

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        series = self.training_series
        self.model.fit(series.values())

    def predict(self, n):
        super().predict(n)
        forecast = self.model.predict(n_periods=n)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30
