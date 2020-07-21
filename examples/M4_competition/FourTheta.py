import math
from typing import Optional, List, Callable

import numpy as np
import statsmodels.tsa.holtwinters as hw
from itertools import product

from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality, remove_from_series
from darts.utils import _build_tqdm_iterator
from darts.models.forecasting_model import UnivariateForecastingModel
from darts.logging import raise_if_not, get_logger
from darts import TimeSeries, TrendMode, SeasonalityMode, ModelMode

from sklearn.metrics import mean_absolute_error as mae
from darts.models import AutoARIMA

logger = get_logger(__name__)


def backtest_gridsearch(model_class: type,
                        parameters: dict,
                        train_series: TimeSeries,
                        metric: Callable[[np.ndarray, np.ndarray], float] = mae,
                        verbose=False):

    model = model_class()
    raise_if_not(hasattr(model, "fitted_values"), "The model must have a fitted_values attribute"
                                                  " to compare with the train TimeSeries", logger)

    min_error = float('inf')
    best_param_combination = {}

    # compute all hyperparameter combinations from selection
    params_cross_product = list(product(*parameters.values()))

    # iterate through all combinations of the provided parameters and choose the best one
    iterator = _build_tqdm_iterator(params_cross_product, verbose)
    for param_combination in iterator:
        param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
        model = model_class(**param_combination_dict)
        model.fit(train_series)
        # Takes too much time to create a TimeSeries
        # Overhead: 2-10 ms in average
        # fitted_values = TimeSeries.from_times_and_values(train_series.time_index(), model.fitted_values)
        error = metric(model.fitted_values, train_series.univariate_values())
        if error < min_error:
            min_error = error
            best_param_combination = param_combination_dict
    logger.info('Chosen parameters: ' + str(best_param_combination))
    return model_class(**best_param_combination)


class FourTheta(UnivariateForecastingModel):
    def __init__(self,
                 theta: int = 2,
                 seasonality_period: Optional[int] = None,
                 season_mode: SeasonalityMode = SeasonalityMode.MULTIPLICATIVE,
                 model_mode: ModelMode = ModelMode.ADDITIVE,
                 trend_mode: TrendMode = TrendMode.LINEAR,
                 normalization: bool = True):
        """
        An implementation of the 4Theta method with configurable `theta` parameter.

        See M4 competition `solution <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`_.

        The training time series is de-seasonalized according to `seasonality_period`,
        or an inferred seasonality period.

        `season_mode` must be a SeasonalityMode Enum member.
        `model_mode` must be a ModelMode Enum member.
        `trend_mode` must be a TrendMode Enum member.
        You can access the different Enums with `from darts import SeasonalityMode, TrendMode, ModelMode`.

        When called with `theta = X`, `model_mode = Model.ADDITIVE` and `trend_mode = Trend.LINEAR`,
        this model is equivalent to calling `Theta(theta=X)`.
        Even though this model is an improvement of `Theta`, `FourTheta` is a naive implementation of the algorithm.
        Thus, a difference in performance can be observed.
        `Theta` is recommended if the model is good enough for the application.

        Parameters
        ----------
        theta
            Value of the theta parameter. Defaults to 2.
            If theta = 1, then the fourtheta method restricts to a simple exponential smoothing (SES).
            If theta = 0, then the fourtheta method restricts to a simple `trend_mode` regression.
        seasonality_period
            User-defined seasonality period. If not set, will be tentatively inferred from the training series upon
            calling `fit()`.
        model_mode
            Type of model combining the Theta lines. Either ModelMode.ADDITIVE or ModelMode.MULTIPLICATIVE.
            Defaults to `ModelMode.ADDITIVE`.
        season_mode
            Type of seasonality.
            Either SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.ADDITIVE or SeasonalityMode.NONE.
            Defaults to `SeasonalityMode.MULTIPLICATIVE`.
        trend_mode
            Type of trend to fit. Either TrendMode.LINEAR or TrendMode.EXPONENTIAL.
            Defaults to `TrendMode.LINEAR`.
        normalization
            If `True`, the data is normalized so that the mean is 1. Defaults to `True`.
        """

        super().__init__()

        self.model = None
        self.drift = None
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
        self.normalization = normalization

        raise_if_not(model_mode in ModelMode,
                     "Unknown value for model_mode: {}.".format(model_mode), logger)
        raise_if_not(trend_mode in TrendMode,
                     "Unknown value for trend_mode: {}.".format(trend_mode), logger)
        raise_if_not(season_mode in SeasonalityMode,
                     "Unknown value for season_mode: {}.".format(season_mode), logger)

    def fit(self, ts, component_index: Optional[int] = None):
        super().fit(ts, component_index)
        # Check univariate time series
        ts._assert_univariate()

        self.length = len(ts)
        # normalization of data
        if self.normalization:
            self.mean = ts.mean().mean()
            raise_if_not(not np.isclose(self.mean, 0),
                         "The mean value of the provided series is too close to zero to perform normalization", logger)
            new_ts = ts / self.mean
        else:
            new_ts = ts

        # Check for statistical significance of user-defined season period
        # or infers season_period from the TimeSeries itself.
        if self.season_mode is SeasonalityMode.NONE:
            self.season_period = 1
        if self.season_period is None:
            max_lag = len(ts) // 2
            self.is_seasonal, self.season_period = check_seasonality(ts, self.season_period, max_lag=max_lag)
            logger.info('FourTheta model inferred seasonality of training series: {}'.format(self.season_period))
        else:
            # force the user-defined seasonality to be considered as a true seasonal period.
            self.is_seasonal = self.season_period > 1

        # Store and remove seasonality effect if there is any.
        if self.is_seasonal:
            _, self.seasonality = extract_trend_and_seasonality(new_ts, self.season_period,
                                                                model=self.season_mode)
            new_ts = remove_from_series(new_ts, self.seasonality, model=self.season_mode)

        ts_values = new_ts.univariate_values()
        if (ts_values <= 0).any():
            self.model_mode = ModelMode.ADDITIVE
            self.trend_mode = TrendMode.LINEAR
            logger.warning("Time series has negative values. Fallback to additive and linear model")

        # Drift part of the decomposition
        if self.trend_mode is TrendMode.LINEAR:
            linreg = ts_values
        else:
            linreg = np.log(ts_values)
        self.drift = np.poly1d(np.polyfit(np.arange(self.length), linreg, 1))
        theta0_in = self.drift(np.arange(self.length))
        if self.trend_mode is TrendMode.EXPONENTIAL:
            theta0_in = np.exp(theta0_in)

        if (theta0_in > 0).all() and self.model_mode is ModelMode.MULTIPLICATIVE:
            theta_t = (ts_values ** self.theta) * (theta0_in ** (1 - self.theta))
        else:
            if self.model_mode is ModelMode.MULTIPLICATIVE:
                logger.warning("Negative Theta line. Fallback to additive model")
                self.model_mode = ModelMode.ADDITIVE
            theta_t = self.theta * ts_values + (1 - self.theta) * theta0_in

        # SES part of the decomposition.
        self.model = hw.SimpleExpSmoothing(theta_t).fit()
        theta2_in = self.model.fittedvalues

        if (theta2_in > 0).all() and self.model_mode is ModelMode.MULTIPLICATIVE:
            self.fitted_values = theta2_in**self.wses * theta0_in**self.wdrift
        else:
            if self.model_mode is ModelMode.MULTIPLICATIVE:
                self.model_mode = ModelMode.ADDITIVE
                logger.warning("Negative Theta line. Fallback to additive model")
                theta_t = self.theta * ts_values + (1 - self.theta) * theta0_in
                self.model = hw.SimpleExpSmoothing(theta_t).fit()
                theta2_in = self.model.fittedvalues
            self.fitted_values = self.wses * theta2_in + self.wdrift * theta0_in
        if self.is_seasonal:
            if self.season_mode is SeasonalityMode.ADDITIVE:
                self.fitted_values += self.seasonality.univariate_values()
            elif self.season_mode is SeasonalityMode.MULTIPLICATIVE:
                self.fitted_values *= self.seasonality.univariate_values()
        # Fitted values are the results of the fit of the model on the train series. A good fit of the model
        # will lead to fitted_values similar to ts. But one cannot see if it overfits.
        if self.normalization:
            self.fitted_values *= self.mean

    def predict(self, n: int) -> 'TimeSeries':
        super().predict(n)

        # Forecast of the SES part.
        forecast = self.model.forecast(n)

        # Forecast of the Linear Regression part.
        drift = self.drift(np.arange(self.length, self.length + n))
        if self.trend_mode is TrendMode.EXPONENTIAL:
            drift = np.exp(drift)

        if self.model_mode is ModelMode.ADDITIVE:
            forecast = self.wses * forecast + self.wdrift * drift
        else:
            forecast = forecast**self.wses * drift**self.wdrift

        # Re-apply the seasonal trend of the TimeSeries
        if self.is_seasonal:

            replicated_seasonality = np.tile(self.seasonality.pd_series()[-self.season_period:],
                                             math.ceil(n / self.season_period))[:n]
            if self.season_mode is SeasonalityMode.MULTIPLICATIVE:
                forecast *= replicated_seasonality
            else:
                forecast += replicated_seasonality

        if self.normalization:
            forecast *= self.mean

        return self._build_forecast_series(forecast)

    @staticmethod
    def select_best_model(ts: TimeSeries, thetas: Optional[List[int]] = None,
                          m: Optional[int] = None, normalization: bool = True) -> 'FourTheta':
        """
        Performs a grid search over all hyper parameters to select the best model,
        using the fitted values on the training series `ts`.


        Uses 'backtesting.backtest_gridsearch' with 'use_fitted_values=True' and 'metric=metrics.mae`.

        Parameters
        ----------
        ts
            The TimeSeries on which the model will be tested.
        thetas
            A list of thetas to loop over. Defaults to [1, 2, 3].
        m
            Optionally, the season used to decompose the time series.
        normalization
            If `True`, the data is normalized so that the mean is 1. Defaults to `True`.
        Returns
        -------
        FourTheta
            The best performing model on the time series.
        """
        if thetas is None:
            thetas = [1, 2, 3]
        if (ts.values() <= 0).any():
            drift_mode = [TrendMode.LINEAR]
            model_mode = [ModelMode.ADDITIVE]
            season_mode = [SeasonalityMode.ADDITIVE]
            logger.warning("The given TimeSeries has negative values. The method will only test "
                           "linear trend and additive modes.")
        else:
            season_mode = [season for season in SeasonalityMode]
            model_mode = [model for model in ModelMode]
            drift_mode = [trend for trend in TrendMode]

        theta = backtest_gridsearch(FourTheta,
                                    {"theta": thetas,
                                     "model_mode": model_mode,
                                     "season_mode": season_mode,
                                     "trend_mode": drift_mode,
                                     "seasonality_period": [m],
                                     "normalization": [normalization]
                                     },
                                    ts)
        return theta

    def __str__(self):
        return '4Theta(theta:{}, curve:{}, model:{}, seasonality:{})'.format(self.theta, self.trend_mode,
                                                                             self.model_mode, self.season_mode)


class FourThetaARMA(UnivariateForecastingModel):
    def __init__(self,
                 theta: List[int] = None,
                 seasonality_period: Optional[int] = None,
                 mode: List[str] = None,
                 season_mode: List[str] = None,
                 trend: List[str] = None):
        
        super().__init__()

        if theta is None:
            theta = [1, 2, 3]
        if mode is None:
            mode = ['additive']
        if season_mode is None:
            season_mode = ['multiplicative']
        if trend is None:
            trend = ['linear']
        self.fourtheta = None
        self.arma = None
        self.theta = theta
        self.seasonality_period = seasonality_period
        self.mode = mode
        self.season_mode = season_mode
        self.trend = trend
        self.error = None
        
    def fit(self, ts, component_index: Optional[int] = None):
        super().fit(ts, component_index)
        self.fourtheta = backtest_gridsearch(FourTheta,
                                             {"theta": self.theta,
                                              "mode": self.mode,
                                              "season_mode": self.season_mode,
                                              "trend": self.trend,
                                              "seasonality_period": [self.seasonality_period]
                                              },
                                             ts)
        self.fourtheta.fit(ts)
        if len(ts) >= 30:
            self.error = ts - TimeSeries.from_times_and_values(ts.time_index(), self.fourtheta.fitted_values)
            self.arma = AutoARIMA(d=0, D=0, stationary=True)
            self.arma.fit(self.error)
        
    def predict(self, n: int):
        super().predict(n)
        predictions = self.fourtheta.predict(n)
        if self.arma is not None:
            predictions = predictions + self.arma.predict(n)
        return predictions
