"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Optional, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not

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

        return TimeSeries.from_times_and_values(time_index, points_preds, self.training_series.freq())


class UnivariateForecastingModel(ForecastingModel):
    """ The base class for univariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries, component_index: Optional[int] = None) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Parameters
        ----------
        series
            The training time series on which to fit the model.
        component_index
            Optionally, a zero-indexed integer indicating the component to use if a multivariate
            time series is passed.
        """

        raise_if_not(series.width == 1 or (component_index is not None), "If a multivariate series is given"
                     "as input to this univariate model, please provide a `component_index` integer indicating"
                     " which component to use.", logger)

        if series.width == 1:
            super().fit(series)
        else:
            super().fit(series.univariate_component(component_index))


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries, target_indices: Optional[List[int]] = None) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        series
            The training time series on which to fit the model.
        target_indices
            A list of integers indicating which component(s) of the time series should be used
            as targets for forecasting.
        """

        if series.width == 1:
            target_indices = [0]

        raise_if_not(target_indices is not None and len(target_indices) > 0,
                     "If a multivariate series is given as input to this multivariate model, please"
                     " provide a list of integer indices `target_indices`"
                     " that indicate which components of this series should be predicted", logger)

        raise_if_not(all(idx >= 0 and idx < series.width for idx in target_indices), "The target indices "
                     "must all be between 0 and the width of the TimeSeries instance used for fitting - 1.", logger)

        self.target_indices = target_indices
        super().fit(series)
