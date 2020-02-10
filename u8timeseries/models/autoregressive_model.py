from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ..timeseries import TimeSeries
from typing import Optional


class AutoRegressiveModel(ABC):
    """
    Base class for implementation of Auto-regressive models.

    This is a base class for various implementation of uni-variate time series forecasting models.
    These models predict future values of one time series using no other data.
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        self.training_series = series
        self._fit_called = True

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """
        :return: A TimeSeries containing the `n` next points, starting after the end of the training time series.
        """
        assert self._fit_called, 'fit() must be called before predict()'

    def _generate_new_dates(self, n: int):
        """
        Generate `n` new dates after the end of the training set
        """
        new_dates = [self.training_series.time_index()[-1] + (i * self.training_series.freq()) for i in range(1, n+1)]
        return pd.DatetimeIndex(new_dates, freq=self.training_series.freq_str())

    def _build_forecast_series(self, points_preds: np.ndarray,
                               lower_bound: Optional[np.ndarray] = None,
                               upper_bound: Optional[np.ndarray] = None):

        time_index = self._generate_new_dates(len(points_preds))

        return TimeSeries.from_times_and_values(time_index, points_preds, lower_bound, upper_bound)
