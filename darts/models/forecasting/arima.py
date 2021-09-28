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
import numpy as np

from darts.models.forecasting.forecasting_model import DualCovariatesForecastingModel
from darts.timeseries import TimeSeries
from darts.logging import get_logger
logger = get_logger(__name__)


class ARIMA(DualCovariatesForecastingModel):
    def __init__(self,
                 p: int = 12, d: int = 1, q: int = 0,
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: Optional[str] = None,
                 random_state: int = 0):
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
        np.random.seed(random_state)

    def __str__(self):
        if self.seasonal_order == (0, 0, 0, 0):
            return f'ARIMA{self.order}'
        return f'SARIMA{self.order}x{self.seasonal_order}'

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super().fit(series, future_covariates)
        m = staARIMA(
            self.training_series.values(),
            exog=future_covariates.values() if future_covariates else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend
        )
        self.model = m.fit()

    def predict(self, n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1):

        if num_samples > 1 and self.trend:
            logger.warn('Trends are not well supported yet for getting probabilistic forecasts with ARIMA.'
                        'If you run into issues, try calling fit() with num_samples=1 or removing the trend from'
                        'your model.')

        super().predict(n, future_covariates, num_samples)

        if num_samples == 1:
            forecast = self.model.forecast(steps=n,
                                           exog=future_covariates.values() if future_covariates else None)
        else:
            forecast = self.model.simulate(nsimulations=n,
                                           repetitions=num_samples,
                                           initial_state=self.model.states.predicted[-1, :],
                                           exog=future_covariates.values() if future_covariates else None)

        return self._build_forecast_series(forecast)

    def _is_probabilistic(self) -> bool:
        return True

    @property
    def min_train_series_length(self) -> int:
        return 30
