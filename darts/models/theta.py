"""
Theta Method
------------
"""

import math
from typing import Optional

import numpy as np
import statsmodels.tsa.holtwinters as hw

from ..utils.statistics import check_seasonality, extract_trend_and_seasonality, remove_seasonality
from .forecasting_model import UnivariateForecastingModel
from ..logging import raise_log, get_logger
from ..timeseries import TimeSeries

logger = get_logger(__name__)


class Theta(UnivariateForecastingModel):
    # .. todo: Implement OTM: Optimized Theta Method (https://arxiv.org/pdf/1503.03529.pdf)
    # .. todo: From the OTM, set theta_2 = 2-theta_1 to recover our generalization - but we have an explicit formula.
    # .. todo: Try with something different than SES? They do that in the paper.
    def __init__(self,
                 theta: int = 0,
                 seasonality_period: Optional[int] = None,
                 mode: str = 'multiplicative'):
        """
        An implementation of the Theta method with configurable `theta` parameter.

        See `Unmasking the Theta method <https://robjhyndman.com/papers/Theta.pdf>`_ paper.

        The training time series is de-seasonalized according to `seasonality_period`,
        or an inferred seasonality period.

        Parameters
        ----------
        theta
            Value of the theta parameter. Defaults to 0. Cannot be set to 2.0.
            If theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        seasonality_period
            User-defined seasonality period. If not set, will be tentatively inferred from the training series upon
            calling `fit()`.
        mode
            Type of seasonality. Either "additive" or "multiplicative".
        """

        super().__init__()

        self.model = None
        self.coef = 1
        self.alpha = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.seasonality_period = seasonality_period
        self.mode = mode

        # Remark on the values of the theta parameter:
        # - if theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        # - if theta = 2, the formula for self.coef below fails, hence it is forbidden.

        if self.theta == 2:
            raise_log(ValueError('The parameter theta cannot be equal to 2.'), logger)

    def fit(self, series: TimeSeries, component_index: Optional[int] = None):
        super().fit(series, component_index)
        ts = self.training_series

        self.length = len(ts)
        self.season_period = self.seasonality_period

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_period is None:
            max_lag = len(ts) // 2
            self.is_seasonal, self.season_period = check_seasonality(ts, self.season_period, max_lag=max_lag)
            logger.info('Theta model inferred seasonality of training series: {}'.format(self.season_period))
        else:
            self.is_seasonal = True  # force the user-defined seasonality to be considered as a true seasonal period.

        new_ts = ts

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(ts, self.season_period, model=self.mode)
            new_ts = remove_seasonality(ts, self.season_period, model=self.mode)

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(new_ts.values()).fit()

        # Linear Regression part of the decomposition. We select the degree one coefficient.
        b_theta = np.polyfit(np.array([i for i in range(0, self.length)]), (1.0 - self.theta) * new_ts.values(), 1)[0]

        # Normalization of the coefficient b_theta.
        self.coef = b_theta / (2.0 - self.theta)

        self.alpha = self.model.params["smoothing_level"]

    def predict(self, n: int) -> 'TimeSeries':
        super().predict(n)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.coef * np.array([i + (1 - (1 - self.alpha) ** self.length) / self.alpha for i in range(0, n)])

        # Combining the two forecasts
        forecast += drift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:

            replicated_seasonality = np.tile(self.seasonality.pd_series()[-self.season_period:],
                                             math.ceil(n / self.season_period))[:n]
            if self.mode in ['multiplicative', 'mul']:
                forecast *= replicated_seasonality
            elif self.mode in ['additive', 'add']:
                forecast += replicated_seasonality
            else:
                raise ValueError("mode cannot be {}".format(self.mode))

        return self._build_forecast_series(forecast)

    def __str__(self):
        return 'Theta({})'.format(self.theta)
