"""
VARIMA
-----

Models for VARIMA (Vector Autoregressive moving average) [1]_.
The implementations is wrapped around `statsmodels <https://github.com/statsmodels/statsmodels>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Vector_autoregression
"""
import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR as staVAR
from statsmodels.tsa.api import VARMAX as staVARMA
from typing import Optional

from .forecasting_model import ExtendedForecastingModel
from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if

logger = get_logger(__name__)


class VARIMA(ExtendedForecastingModel):
    def __init__(self, p: int = 1, d: int = 0, q: int = 0, trend: Optional[str] = None):
        """ VARIMA

        Parameters
        ----------
        p : int
            Order (number of time lags) of the autoregressive model (AR)
        d : int
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted. (I) Note that Darts only supports d <= 1 because for
            d > 1 the optimizer often does not result in stable predictions. If results are not stable
            for d = 1 try to set d = 0 and enable the trend parameter
            to account for possible non-stationarity.
        q : int
            The size of the moving average window (MA).
        trend: str
            Parameter controlling the deterministic trend. ‘n‘ indicates no trend,
            ‘c’ a constant term, ‘t’ linear trend in time, and ‘ct’ includes both.
            Default is ‘c’ for models without integration, and no trend for models with integration.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.trend = trend
        self.model = None

        assert d <= 1, "d > 1 not supported."

    def __str__(self):
        if self.d == 0:
            return 'VARMA({},{})'.format(self.p, self.q)
        return 'VARIMA({},{},{})'.format(self.p, self.d, self.q)

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None):
        self._last_values = series.last_values()  # needed for back-transformation when d=1
        for _ in range(self.d):
            series = TimeSeries.from_dataframe(series.pd_dataframe(copy=False).diff().dropna())

        super().fit(series, exog)
        series = self.training_series
        exog = exog.values() if exog else None
        m = staVARMA(endog=series.pd_dataframe(copy=False), exog=exog, order=(self.p, self.q), trend=self.trend)
        self.model = m.fit(disp=0)

    def predict(self,
                n: int,
                exog: Optional[TimeSeries] = None,
                num_samples: int = 1):
        super().predict(n, exog, num_samples)
        forecast = self.model.forecast(steps=n, exog=exog.values() if exog else None)
        forecast = self._invert_transformation(forecast)
        return self._build_forecast_series(np.array(forecast))

    def _invert_transformation(self, series_df: pd.DataFrame):
        if self.d == 0:
            return series_df
        series_df = self._last_values + series_df.cumsum(axis=0)
        return series_df

    @property
    def min_train_series_length(self) -> int:
        return 30

    def _supports_range_index(self) -> bool:
        raise_if(self.trend and self.trend != "c",
            "'trend' is not None. Range indexing is not supported in that case.",
            logger
        )
        return True
