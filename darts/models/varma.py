"""
VARMA
-----

Models for VARMA (Vector Autoregressive moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Vector_autoregression
"""

from statsmodels.tsa.api import VARMAX as staVARMA
from typing import Optional

from .forecasting_model import ExtendedForecastingModel
from ..timeseries import TimeSeries
from ..logging import get_logger

logger = get_logger(__name__)


class VARMA(ExtendedForecastingModel):
    def __init__(self, p: int = 1, q: int = 0):
        """ VARMA

        Parameters
        ----------
        p : int
            Order (number of time lags) of the autoregressive model (AR)
        q : int
            The size of the moving average window (MA).
        """
        super().__init__()
        self.p = p
        self.q = q
        self.model = None

    def __str__(self):
        return 'VARMA({},{})'.format(self.p, self.q)

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None):
        super().fit(series, exog)
        series = self.training_series
        exog = exog.values() if exog else None
        m = staVARMA(endog=series._df, exog=exog, order=(self.p, self.q), trend="n")
        self.model = m.fit(disp=0)

    def predict(self, n: int, exog: Optional[TimeSeries] = None):
        super().predict(n, exog)
        forecast = self.model.forecast(steps=n,
                                       exog=exog.values() if exog else None)
        return self._build_forecast_series(forecast)

    @property
    def min_train_series_length(self) -> int:
        return 30