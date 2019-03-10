from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import math
from u8timeseries.backtesting import backtest
from ..timeseries import TimeSeries
from typing import Optional


class AutoRegressiveModel(ABC):
    """
    This is a base class for various implementation of uni-variate time series forecasting models
    These models predict future values of one time series using no other data.
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: Optional[TimeSeries] = None

        # state
        self.fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries):
        self.training_series = series
        self.fit_called = True

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """
        :return: A TimeSeries containing the n next points, starting after the end of the training time series.
        """
        pass

    def predict_interval(self, interval: pd.Timedelta) -> TimeSeries:
        """
        Generates a predicted time series lasting at least [interval] and starting after the end of the
        training time series.
        """
        assert self.training_series is not None, 'You must first call fit() with a well-defined TimeSeries'

        nr_steps = int(math.ceil(interval / self.training_series.freq()))
        return self.predict(nr_steps)

    def _generate_new_dates(self, n: int):
        """
        Generate n new dates after the end of the training set
        """
        return pd.date_range(start=self.training_series.time_index()[-1],
                             periods=n+1,
                             freq=self.training_series.time_index().freq)[-n:]

    def _build_forecast_series(self, points_preds: np.ndarray,
                               lower_bound: Optional[np.ndarray] = None,
                               upper_bound: Optional[np.ndarray] = None):

        time_index = self._generate_new_dates(len(points_preds))

        return TimeSeries.from_times_and_values(time_index, points_preds, lower_bound, upper_bound)


    # TODO: Could we just have 1 backtest function for all models?
    """
    def backtest(self, series: TimeSeries,
                 start_dt, n, eval_fun, nr_steps_iter=1, predict_nth_only=False):

        # Prepare generic fit() and predict() calls to be used from backtest()
        def fit_fn(*args):
            return self.fit(*args)

        def predict_fn(_, _n):
            return self.predict(_n)

        return backtest(series, start_dt, n, eval_fun, fit_fn,
                        predict_fn, nr_steps_iter, predict_nth_only)
    """

    """
    def _get_new_dates(self, n):
        # This function creates a list of the n new dates (after the end of training set)
        # :param n: number of dates after training set to generate
        return [add_time_delta_to_datetime(self.training_dates[-1], i, self.stepduration_str)
                for i in range(1, n + 1)]
    """

    # def _build_forecast_df(self, point_preds, lower_bound=None, upper_bound=None):
    #     """
    #     Builds the pandas DataFrame to be returned by predict() method
    #     The column names are inspired from Prophet
    #
    #     :param point_preds: a list or array of n point-predictions
    #     :param lower_bound: optionally, a list or array of lower bounds
    #     :param upper_bound:optionally, a list or array of upper bounds
    #     :return: a dataframe nicely formatted
    #     """
    #
    #     columns = {
    #         'yhat': pd.Series(point_preds)
    #     }
    #
    #     if self.time_column is not None:
    #         n = len(point_preds)
    #         new_dates = self._get_new_dates(n)
    #         columns[self.time_column] = pd.Series(new_dates)
    #
    #     if lower_bound is not None:
    #         assert len(point_preds) == len(lower_bound), 'bounds should be same size as point predictions'
    #         columns['yhat_lower'] = lower_bound
    #
    #     if upper_bound is not None:
    #         assert len(point_preds) == len(upper_bound), 'bounds should be same size as point predictions'
    #         columns['yhat_upper'] = upper_bound
    #
    #     return pd.DataFrame(columns)