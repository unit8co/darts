"""
Theta Method
------------
"""

import math
from typing import Optional, List

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
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1

        new_ts = ts

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(ts, self.season_period, model=self.mode)
            new_ts = remove_seasonality(ts, self.season_period, model=self.mode)

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(new_ts.values()).fit(initial_level=0.2)

        # Linear Regression part of the decomposition. We select the degree one coefficient.
        b_theta = np.polyfit(np.array([i for i in range(0, self.length)]), (1.0 - self.theta) * new_ts.values(), 1)[0]

        # Normalization of the coefficient b_theta.
        self.coef = b_theta / (2.0 - self.theta)  # change to b_theta / (-self.theta) if classical theta

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


class FourTheta(UnivariateForecastingModel):
    def __init__(self,
                 theta: int = 0,
                 seasonality_period: Optional[int] = None,
                 model_mode: str = 'multiplicative',
                 season_mode: str = 'multiplicative',
                 trend_mode: str = 'linear'):
        """
        An implementation of the 4Theta method with configurable `theta` parameter.

        See M4 competition `solution <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`_.

        The training time series is de-seasonalized according to `seasonality_period`,
        or an inferred seasonality period.

        This model is similar to Theta, with theta=2-theta, `mode`=additive and `trend`=linear.

        Parameters
        ----------
        theta
            Value of the theta parameter. Defaults to 2.
            If theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        seasonality_period
            User-defined seasonality period. If not set, will be tentatively inferred from the training series upon
            calling `fit()`.
        model_mode
            Type of model combining the Theta lines. Either "additive" or "multiplicative".
        season_mode
            Type of seasonality. Either "additive" or "multiplicative".
        trend_mode
            Type of trend to fit. Either "linear" or "exponential".
        """

        super().__init__()

        self.model = None
        self.drift = None
        self.coef = 1
        self.mean = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.season_period = seasonality_period
        self.model_mode = model_mode
        self.season_mode = season_mode
        self.trend_mode = trend_mode
        self.wses = 0 if self.theta == 0 else (1 / theta)
        self.wdrift = 1 - self.wses
        self.fitted_values = None

        # Remark on the values of the theta parameter:
        # - if theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        # - if theta = 0, then the theta method restricts to a simple `trend_mode` regression.

    def fit(self, ts, component_index: Optional[int] = None):
        super().fit(ts, component_index)

        self.length = len(ts)
        # normalization of data
        self.mean = ts.mean()
        new_ts = ts / self.mean

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_period is None:
            max_lag = len(ts) // 2
            self.is_seasonal, self.season_period = check_seasonality(ts, self.season_period, max_lag=max_lag)
            logger.info('Theta model inferred seasonality of training series: {}'.format(self.season_period))
        else:
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(new_ts, self.season_period, model=self.season_mode)
            new_ts = remove_seasonality(new_ts, self.season_period, model=self.season_mode)

        if (new_ts <= 0).values.any():
            self.model_mode = 'additive'
            self.trend_mode = 'linear'
            logger.warn("Time series has negative values. Fallback to additive and linear model")

        # Drift part of the decomposition
        if self.trend_mode == 'linear':
            linreg = new_ts.values()
        elif self.trend_mode == 'exponential':
            linreg = np.log(new_ts.values())
        else:
            self.trend_mode = 'linear'
            linreg = new_ts.values()
            logger.warn("Unknown value for trend. Fallback to linear.")
        self.drift = np.poly1d(np.polyfit(np.arange(self.length), linreg, 1))
        theta0_in = self.drift(np.arange(self.length))
        if self.trend_mode == 'exponential':
            theta0_in = np.exp(theta0_in)

        if self.model_mode == 'additive':
            theta_t = self.theta * new_ts.values() + (1 - self.theta) * theta0_in
        elif self.model_mode == 'multiplicative' and (theta0_in > 0).all():
            theta_t = (new_ts.values() ** self.theta) * (theta0_in ** (1 - self.theta))
        else:
            self.model_mode = 'additive'
            theta_t = self.theta * new_ts.values() + (1 - self.theta) * theta0_in
            logger.warn("Negative Theta line. Fallback to additive model")

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(theta_t).fit()
        theta2_in = self.model.fittedvalues

        if self.model_mode == 'additive':
            self.fitted_values = self.wses * theta2_in + self.wdrift * theta0_in
        elif self.model_mode == 'multiplicative' and (theta2_in > 0).all():
            self.fitted_values = theta2_in**self.wses * theta0_in**self.wdrift
        else:
            # Fallback to additive model
            self.model_mode = 'additive'
            theta_t = self.theta * new_ts.values() + (1 - self.theta) * theta0_in
            self.model = hw.SimpleExpSmoothing(theta_t).fit()
            theta2_in = self.model.fittedvalues
            self.fitted_values = self.wses * theta2_in + self.wdrift * theta0_in
            logger.warn("Negative Theta line. Fallback to additive model")

        if self.is_seasonal:
            if self.season_mode == 'additive':
                self.fitted_values += self.seasonality.values()
            elif self.season_mode == 'multiplicative':
                self.fitted_values *= self.seasonality.values()
        self.fitted_values *= self.mean
        # takes too much time to create a time series for fitted_values
        # self.fitted_values = TimeSeries.from_times_and_values(ts.time_index(), self.fitted_values)

    def predict(self, n: int) -> 'TimeSeries':
        super().predict(n)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.drift(np.arange(self.length, self.length + n))
        if self.trend_mode == 'exponential':
            drift = np.exp(drift)

        if self.model_mode == 'additive':
            forecast = self.wses * forecast + self.wdrift * drift
        elif self.model_mode == 'multiplicative':
            forecast = forecast**self.wses * drift**self.wdrift
        else:
            raise_log(ValueError("model_mode cannot be {}".format(self.model_mode)))

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:

            replicated_seasonality = np.tile(self.seasonality.pd_series()[-self.season_period:],
                                             math.ceil(n / self.season_period))[:n]
            if self.season_mode in ['multiplicative', 'mul']:
                forecast *= replicated_seasonality
            elif self.season_mode in ['additive', 'add']:
                forecast += replicated_seasonality
            else:
                raise_log(ValueError("season_mode cannot be {}".format(self.season_mode)))

        forecast *= self.mean

        return self._build_forecast_series(forecast)

    @staticmethod
    def select_best_model(ts: TimeSeries, thetas: List[int] = None, m: Optional[int] = None):
        """
        Performs a grid search over all hyper parameters to select the best model.

        Parameters
        ----------
        ts
            The TimeSeries on which the model will be tested.
        thetas
            A list of thetas to loop over.
        m
            Optionally, the season used to decompose the time series.
        Returns
        -------
        theta
            The best performing model on the time series.
        """
        # Only import if needed
        from ..backtesting.backtesting import backtest_gridsearch
        from sklearn.metrics import mean_absolute_error as mae
        if thetas is None:
            thetas = [1, 2, 3]
        elif isinstance(thetas, int):
            thetas = [thetas]
        season_mode = ["additive", "multiplicative"]
        model_mode = ["additive", "multiplicative"]
        drift_mode = ["linear", "exponential"]
        if (ts.values() <= 0).any():
            drift_mode = ["linear"]
            model_mode = ["additive"]
            season_mode = ["additive"]

        theta = backtest_gridsearch(FourTheta,
                                    {"theta": thetas,
                                     "model_mode": model_mode,
                                     "season_mode": season_mode,
                                     "trend_mode": drift_mode,
                                     "seasonality_period": [m]
                                     },
                                    ts, val_series='train', metric=mae)
        return theta

    def __str__(self):
        return '4Theta(theta:{}, curve:{}, model:{}, seasonality:{})'.format(self.theta, self.trend_mode,
                                                                             self.model_mode, self.season_mode)
