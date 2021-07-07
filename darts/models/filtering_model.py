"""
Filtering Model Base Class
--------------------------

Filtering models all have a `filter(series)` function, which
returns a `TimeSeries` that is a filtered version of `series`.
"""

from abc import ABC, abstractmethod

from ..timeseries import TimeSeries
from ..logging import get_logger

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

    def __init__(self,
                 window: int,
                 centered: bool = True):
        """
        Parameters
        ----------
        window
            The length of the window over which to average values
        centered
            Set the labels at the center of the window. If not set, the averaged values are lagging after the
            the original values.
        """
        super().__init__()
        self.window = window
        self.centered = centered

    def filter(self, series):
        """
        Computes a moving average of this series' values and returns a new TimeSeries.
        The returned series has the same length and time axis as `series`. (Note that this might create border effects).

        Behind the scenes the moving average is computed using `pandas.DataFrame.rolling()` on the underlying
        DataFrame.

        Parameters
        ----------
        series
            The series to average

         Returns
        -------
        TimeSeries
            A time series containing the average values
        """
        filtered_df = series.pd_dataframe(copy=False).rolling(window=self.window,
                                                              min_periods=1,
                                                              center=self.centered).mean()
        return TimeSeries.from_dataframe(filtered_df)
