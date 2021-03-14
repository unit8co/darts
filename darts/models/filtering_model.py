"""
Filtering Model Base Class
------------------------------

A filtering model captures the measured values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_{t+1}, y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' filtered value(s) at time :math:`t`.

The main functions are `fit()` and `predict()`. `fit()` learns the function `f()`, over the history of
one or several time series. The function `predict()` applies `f()` on one or several time series in order
to obtain more accurate predictions of the measured values(s)
"""

from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from itertools import product
from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not

logger = get_logger(__name__)


class FilteringModel(ABC):

    """ The base class for filterin models. It defines the *minimal* behavior that all filtering models have to support.
        The signatures in this base class are for "local" models handling only one series and no covariates.
        Sub-classes can handle more complex cases.
    """
    @abstractmethod
    def __init__(self):
        # The series used for training the model through the `fit()` function.
        # This is only used if the model has been fit on one time series.
        self.training_series: Optional[TimeSeries] = None

        # state; whether the model has been fit (on a single time series)
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Trains the model on the provided series

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.
        """
        raise_if_not(len(series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True

    @abstractmethod
    def filter(self) -> TimeSeries:
        """ Predicts filtered values from train TimeSeries

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.

        Returns
        -------
        TimeSeries
            A time series containing the filtered values.
        """
        if not self._fit_called:
            raise_log(ValueError('The model must be fit before calling `predict()`.'
                                 'For global models, if `predict()` is called without specifying a series,'
                                 'the model must have been fit on a single training series.'), logger)

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3


class MovingAverage(FilteringModel, ABC):
    
    """ Moving average class which implements Filtering Model and shows
    a simple example of filtering dara using moving average.
    """

    def __init__(self, window):
        self.values = [.0 for _ in range(window)]


    def fit(self, series: TimeSeries) -> None:
        super().fit(series)


    def filter(self):
        return self.training_series.map(self._ma_iteration)


    def _ma_iteration(self, observation):
        self.values.pop(0)
        self.values.append(observation)
        return sum(self.values) / len(self.values)
