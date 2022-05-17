"""
Posthoc
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from darts.timeseries import TimeSeries
from darts.utils.utils import raise_if_not


class ForecastReconciliator(ABC):
    """
    Super class for all forecast reconciliators.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reconcile(
        self,
        series: TimeSeries,
        grouping: Optional[Dict],
        forecast_errors: Optional[
            np.array
        ],  # Forecast errors for each of the components
    ) -> TimeSeries:
        pass

    # TODO:
    # def get_forecast_errors(forecasts: Union[TimeSeries, Sequence[TimeSeries]],)


class LinearForecastReconciliator(ForecastReconciliator, ABC):
    """
    Super class for all linear forecast reconciliators (bottom-up, top-down, GLS, MinT, ...)

    TODO: Actually not a super class. A moedl where users can give a desired P or get_P method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_projection_matrix():
        """
        Defines the kind of reconciliation being applied
        """
        pass

    def get_summation_matrix(series: TimeSeries):
        raise_if_not(
            series.has_grouping,
            message="The provided series must have a grouping defined.",
        )

    @abstractmethod
    def reconcile(
        self,
        series: TimeSeries,
    ) -> TimeSeries:
        pass
        # S = self.get_summation_matrix(series)
        # P = self.get_projection_matrix()
