"""
Filtering Model Base Class
------------------------------

A filtering model applies a state dependent function on current value and optionaly 
past values and/or their effects on the state
Using the current value and historic values as follows:

.. math:: y_{t+1} = f(y_{t+1}, st(y_t), st(y_{t-1}), ..., st(y_1)),

where :math:`y_t` represents the time series' smoothed value(s) at time :math:`t`.
      :math:`st` represents the state function of a past measurement.

The main function is `filter()` - learns the function `f()`, over the history of
one time series. The function `filter()` applies `f()` on one or several time series in order
to obtain more accurate predictions of the measured values(s)
"""

from typing import Optional
from itertools import product
from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not

logger = get_logger(__name__)


class FilteringModel(ABC):

    """ The base class for filtering models. It defines the *minimal* behavior that all filtering models have to support.
        The signatures in this base class are for "local" models handling only one series and no covariates.
        Sub-classes can handle more complex cases.
    """
    @abstractmethod
    def __init__(self):
        # The series used for filtering the model through the `filter()` function.
        self.training_series: Optional[TimeSeries] = None


    @abstractmethod
    def filter(self,  series: TimeSeries) -> TimeSeries:
        """ Filters values from train TimeSeries

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.

        Returns
        -------
        TimeSeries
            A time series containing the filtered values.
        """
        self.training_series = series


    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3


class MovingAverage(FilteringModel, ABC):
    
    """ Moving average class which implements Filtering Model and shows
    a simple example of a data filtering using moving average.
    """

    def __init__(self, window):
        self.window = window


    def filter(self):
        self.values = [.0 for _ in range(self.window)]
        return self.training_series.map(self._ma_iteration)


    def _ma_iteration(self, observation):
        self.values.pop(0)
        self.values.append(observation)
        return sum(self.values) / len(self.values)
