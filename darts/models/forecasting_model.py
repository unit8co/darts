"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for all forecasting models.

    All implementations of forecasting have to implement the `fit()` and `predict()` methods defined below.
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the model on the provided series

        Parameters
        ----------
        series
            the training time series on which to fit the model
        """
        raise_if_not(len(series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """ Predicts values for a certain number of time steps after the end of the training series

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points, starting after the end of the training time series
        """

        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3

    def _generate_new_dates(self, n: int) -> pd.DatetimeIndex:
        """
        Generates `n` new dates after the end of the training set
        """
        new_dates = [
            (self.training_series.time_index()[-1] + (i * self.training_series.freq())) for i in range(1, n + 1)
        ]
        return pd.DatetimeIndex(new_dates, freq=self.training_series.freq_str())

    def _build_forecast_series(self,
                               points_preds: np.ndarray) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index.
        """

        time_index = self._generate_new_dates(len(points_preds))

        return TimeSeries.from_times_and_values(time_index, points_preds, freq=self.training_series.freq())

    def backtest(self,
                 series: TimeSeries,
                 training_window_initial_size: Union[int, float] = 0.5,
                 training_window_stride: int = 0,
                 forcast_horizon: int = 1,
                 verbose: bool = False) -> (TimeSeries, TimeSeries):
        """ Use the model to forecast values of `series` on a sliding window.
        
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the `model` on the
        training set, emits a (point) prediction for a fixed window, and then moves the end of the training set forward
        by `stride` time step. The resulting predictions and residuals (diff between `series` and the prediction)
        are then returned.

        This always re-trains the models on the entire training window. If the `training_window_stride` is 0, the 
        training window is expanding, if the `training_stride` is 1 then the training window is sliding as follow:

        iteration 1:

             x            o
           /   \          ^
         x       x - x ...|
        |_________|_______|
         training  forcast
          window   horizon

        iteration 2:

             x            o
           /   \            \  
         x       x - x ...    o
        |___|_________|_______^
        stride training forcast
               window   horizon
        
        ...

        legend:
            - x: point from provided `series`.
            - o: point forcasted by the model trained on the `training_window` data.

        ..warning:: if the `training_window_stride` is above 1 the training window size will have to shrink.

        Parameters
        ----------
        series
            The univariate time series on which to backtest.
        training_window_initial_size
            A percentage or number of steps corresponding to the intial training window size starting from timestep 0.
        training_window_stride
            A value that will be used to offset the training window at each iteration (if above 1 the training window
            will shrink).
        forcast_horizon
            Delay in time steps between the training window and the forcasted value.
        verbose
            Show backtesting iterations progress.
        
        Returns
        -------
        serie_forcasted
            A TimeSeries corresponding to each point prediction occuring during backtest.
        residuals
            A TimeSeries corresponding to the difference between `serie_forcasted` and the provided `serie`.
        """ # noqa : W605

        # sanity checks
        series._assert_univariate()

        if isinstance(training_window_initial_size, float):
            training_window_initial_size = round(len(series) * training_window_initial_size)
        raise_if(training_window_initial_size + forcast_horizon > len(series),
                 'The initial training window size and forcast horizon combined must be smaller than `series` length.',
                 logger)

        raise_if_not(forcast_horizon > 0, 'The provided forecasting horizon must be a positive integer.', logger)

        raise_if(series.freq() is None, 'The frequency of the provided `series` must be defined.')

        values = []
        times = []

        for i in range(training_window_initial_size + forcast_horizon, len(series) + 1):
            # build the training series
            training_window_start = (i - (training_window_initial_size + forcast_horizon)) * training_window_stride
            training_window_end = i - forcast_horizon
            train = series[training_window_start:training_window_end]

            self.fit(train)
            pred = self.predict(forcast_horizon)
            values.append(pred.last_value())
            times.append(pred.end_time())

        return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))


class UnivariateForecastingModel(ForecastingModel):
    """ The base class for univariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Parameters
        ----------
        series
            A **univariate** training time series on which to fit the model.
        """
        series._assert_univariate()
        super().fit(series)


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """

    @abstractmethod
    def fit(self, covariate_series: TimeSeries, target_series: TimeSeries) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        covariate_series
            The training time series on which to fit the model (can be multivariate or univariate).
        target_series
            The target values used ad dependent variables when training the model
        """
        raise_if_not(len(covariate_series) == len(target_series), "covariate_series and target_series musth have same "
                     "length.")
        super().fit(covariate_series)
