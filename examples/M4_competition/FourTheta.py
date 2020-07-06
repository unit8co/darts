import math
from typing import Optional, List

import numpy as np
import statsmodels.tsa.holtwinters as hw

from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality, remove_seasonality
from darts.models.forecasting_model import ForecastingModel
from darts.logging import raise_log, get_logger
from darts.timeseries import TimeSeries

from darts.metrics import mae
from darts.models import AutoARIMA
from darts.backtesting import backtest_gridsearch

logger = get_logger(__name__)


class FourTheta(ForecastingModel):
    def __init__(self,
                 theta: int = 2,
                 seasonality_period: Optional[int] = None,
                 mode: str = 'multiplicative',
                 season_mode: str = 'multiplicative',
                 trend: str = 'linear'):
        """
        An implementation of the 4Theta method with configurable `theta` parameter.

        See M4 competition `https://github.com/Mcompetitions/M4-methods/blob/master/005%20-%20vangspiliot/Method-Description-4Theta.pdf`.

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
        self.drift = None
        self.coef = 1
        self.mean = 1
        self.length = 0
        self.theta = theta
        self.is_seasonal = False
        self.seasonality = None
        self.season_period = seasonality_period
        self.mode = mode
        self.season_mode = season_mode
        self.trend = trend
        self.wses = 0 if self.theta == 0 else 1/theta
        self.wdrift = 1 - self.wses
        self.fitted_values = None

        # Remark on the values of the theta parameter:
        # - if theta = 1, then the theta method restricts to a simple exponential smoothing (SES)
        # - if theta = 2, the formula for self.coef below fails, hence it is forbidden.

    def fit(self, ts):
        super().fit(ts)

        self.length = len(ts)
        self.mean = ts.mean()
        new_ts = ts/self.mean

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_period is None:
            max_lag = len(ts) // 2
            self.is_seasonal, self.season_period = check_seasonality(ts, self.season_period, max_lag=max_lag)
            logger.info('Theta model inferred seasonality of training series: {}'.format(self.season_period))
        else:
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1
            # todo: check season?

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(new_ts, self.season_period, model=self.season_mode)
            new_ts = remove_seasonality(new_ts, self.season_period, model=self.season_mode)

        if (new_ts <= 0).values.any():
            self.mode = 'additive'
            self.trend = 'linear'

        if self.trend == 'linear':
            linreg = new_ts.values()
        elif self.trend == 'exponential':
            linreg = np.log(new_ts.values())
        else:
            self.trend = 'linear'
            linreg = new_ts.values()
            print("unknown value for trend. Fallback to linear.")
        self.drift = np.poly1d(np.polyfit(np.arange(self.length), linreg, 1))
        Theta0In = self.drift(np.arange(self.length))
        if self.trend == 'exponential':
            Theta0In = np.exp(Theta0In)

        if self.mode == 'additive':
            ThetaT = self.theta*new_ts.values() + (1 - self.theta) * Theta0In
        elif self.mode == 'multiplicative' and (Theta0In > 0).all():
            ThetaT = (new_ts.values()**self.theta) * (Theta0In**(1-self.theta))
        else:
            self.mode = 'additive'
            ThetaT = self.theta * new_ts.values() + (1 - self.theta) * Theta0In
            #print("fallback to additive mode")

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(ThetaT).fit()
        Theta2In = self.model.fittedvalues

        if self.mode == 'additive':
            self.fitted_values = self.wses * Theta2In + self.wdrift * Theta0In
        elif self.mode == 'multiplicative' and (Theta2In>0).all():
            self.fitted_values = Theta2In**self.wses * Theta0In**self.wdrift
        else:
            # Fallback to additive model
            self.mode = 'additive'
            ThetaT = self.theta * new_ts.values() + (1 - self.theta) * Theta0In
            self.model = hw.SimpleExpSmoothing(ThetaT).fit()
            Theta2In = self.model.fittedvalues
            self.fitted_values = self.wses * Theta2In + self.wdrift * Theta0In
            #print("fallback to additive mode")

        if self.is_seasonal:
            if self.season_mode == 'additive':
                self.fitted_values += self.seasonality.values()
            elif self.season_mode == 'multiplicative':
                self.fitted_values *= self.seasonality.values()
        self.fitted_values *= self.mean
        # self.fittedvalues = TimeSeries.from_times_and_values(ts.time_index(), self.fittedvalues)


    def predict(self, n: int) -> 'TimeSeries':
        super().predict(n)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.drift(np.arange(self.length, self.length + n))
        if self.trend == 'exponential':
            drift = np.exp(drift)

        if self.mode == 'additive':
            forecast = self.wses * forecast + self.wdrift * drift
        else:
            forecast = forecast**self.wses * drift**self.wdrift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:

            replicated_seasonality = np.tile(self.seasonality.pd_series()[-self.season_period:],
                                             math.ceil(n / self.season_period))[:n]
            if self.season_mode in ['multiplicative', 'mul']:
                forecast *= replicated_seasonality
            elif self.season_mode in ['additive', 'add']:
                forecast += replicated_seasonality
            else:
                raise ValueError("mode cannot be {}".format(self.mode))

        forecast *= self.mean

        return self._build_forecast_series(forecast)

    def __str__(self):
        return '4Theta(theta:{}, curve:{}, model:{}, seasonality:{})'.format(self.theta, self.trend,
                                                                             self.mode, self.season_mode)


class FourThetaARMA(ForecastingModel):
    def __init__(self,
                 theta: List[int] = [2],
                 seasonality_period: Optional[int] = None,
                 mode: List[str] = ['multiplicative'],
                 season_mode: List[str] = ['multiplicative'],
                 trend: List[str] = ['linear'],
                 normalize: bool = True):
        
        super().__init__()
        
        self.fourtheta = None
        self.arma = None
        self.theta = theta
        self.seasonality_period = seasonality_period
        self.mode = mode
        self.season_mode = season_mode
        self.trend = trend
        self.normalize = normalize
        self.error = None
        
    def fit(self, ts):
        super().fit(ts)
        self.fourtheta = backtest_gridsearch(FourTheta,
                                       {"theta": self.theta,
                                        "mode": self.mode,
                                        "season_mode": self.season_mode,
                                        "trend": self.trend,
                                        "seasonality_period": [self.seasonality_period]
                                       },
                                       ts,
                                       val_series='train',
                                       metric=mae)
        self.fourtheta.fit(ts)
        if len(ts)>=30:
            self.error = ts - TimeSeries.from_times_and_values(ts.time_index(), self.fourtheta.fittedvalues)
            self.arma = AutoARIMA(d=0, D=0, stationary=True)
            self.arma.fit(self.error)
        
    def predict(self, n: int):
        super().predict(n)
        predictions = self.fourtheta.predict(n)
        if self.arma is not None:
            predictions = predictions + self.arma.predict(n)
        return predictions
