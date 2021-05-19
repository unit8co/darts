"""
Filtering Model Base Class
--------------------------

Filtering models all have a `filter(series)` function, which
returns a `TimeSeries` that is a filtered version of `series`.
"""

from abc import ABC, abstractmethod

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not

logger = get_logger(__name__)


class FilteringModel(ABC):
    """ The base class for filtering models. It defines the *minimal* behavior that all filtering models
        have to support. The filtering models are all "local" models; meaning they act on one time series alone.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def filter(self,  series: TimeSeries) -> TimeSeries:
        """ Filters a given series

        Parameters
        ----------
        series
            The series to filter.

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
