"""
Filtering Model Base Class

Filtering models all have a `filter(series)` function, which
returns a `TimeSeries` that is a filtered version of `series`.
"""

from abc import ABC, abstractmethod

from darts.logging import get_logger, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class FilteringModel(ABC):
    """The base class for filtering models. It defines the *minimal* behavior that all filtering models
    have to support. The filtering models are all "local" models; meaning they act on one time series alone.
    """

    @abstractmethod
    def __init__(self):
        self._expect_covariates = False
        pass

    @abstractmethod
    def filter(self, series: TimeSeries) -> TimeSeries:
        """Filters a given series

        Parameters
        ----------
        series
            The series to filter.

        Returns
        -------
        TimeSeries
            A time series containing the filtered values.
        """
        raise_if_not(
            series.is_deterministic,
            "The input series must be deterministic (observations).",
        )
