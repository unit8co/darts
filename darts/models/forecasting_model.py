"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for all forecasting models.

    All implementations of forecasting have to implement the `fit()` and `predict()` methods defined below.

    Attributes
    ----------
    training_series
        reference to the `TimeSeries` used for training the model in univariate cases.
    covariate_series
        reference to the `TimeSeries` used as covariate to train the model in multivariate cases.
    target_series
        reference to the `TimeSeries` used as target to train the model in multivariate cases.
    """
    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None
        self.covariate_series: TimeSeries = None
        self.target_series: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self) -> None:
        """ Fits/trains the model on the provided series

        Implements behavior that should happen when calling the `fit` method of every forcasting model regardless of
        wether they are univariate or multivariate.
        """
        for series in (self.training_series, self.covariate_series, self.target_series):
            if series is not None:
                raise_if_not(len(series) >= self.min_train_series_length,
                             "Train series only contains {} elements but {} model requires at least {} entries"
                             .format(len(series), str(self), self.min_train_series_length))
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


class UnivariateForecastingModel(ForecastingModel):
    """The base class for univariate forecasting models."""
    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Implements behavior specific to calling the `fit` method on `UnivariateForecastingModel`.

        Parameters
        ----------
        series
            A **univariate** timeseries on which to fit the model.
        """
        series._assert_univariate()
        self.training_series = series
        super().fit()


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """
    def _make_fitable_series(self,
                             covariate_series: TimeSeries,
                             target_series: Optional[TimeSeries] = None) -> Tuple[TimeSeries, TimeSeries]:
        """Perform checks and returns ready to be used covariate and target series"""
        if target_series is None:
            target_series = covariate_series

        # general checks on covariate / target series
        raise_if_not(all(covariate_series.time_index() == target_series.time_index()), "Covariate and target "
                     "timeseries must have same time indices.")

        return covariate_series, target_series

    @abstractmethod
    def fit(self, covariate_series: TimeSeries, target_series: Optional[TimeSeries] = None) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        covariate_series
            The training time series on which to fit the model (can be multivariate or univariate).
        target_series
            The target values used as dependent variables when training the model
        """
        covariate_series, target_series = self._make_fitable_series(covariate_series, target_series)

        self.covariate_series = covariate_series
        self.target_series = target_series
        super().fit()
