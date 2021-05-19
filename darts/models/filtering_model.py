"""
Filtering Model Base Class
--------------------------

A filtering model applies a state dependent function on current value and optionally
past values and/or their effects on the state
Using the current value and historic values as follows:

.. math:: y_{t+1} = f(y_{t+1}, st(y_t), st(y_{t-1}), ..., st(y_1)),

where :math:`y_t` represents the time series' filtered value(s) at time :math:`t`.
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
        pass

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
        pass

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the series.
        This function/property should be overridden if a value different than 3 is required.
        """
        return 3


class MovingAverage(FilteringModel, ABC):
    """ Moving average filter, implementing a FilteringModel.
    """

    def __init__(self, window):
        super().__init__()
        self.window = window

    def filter(self, series):
        values = [.0 for _ in range(self.window)]

        def _ma_iteration(observation: float) -> float:
            values.pop(0)
            values.append(observation)
            return sum(values) / len(values)

        return series.map(_ma_iteration)
