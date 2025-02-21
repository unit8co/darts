"""
Moving Average
--------------
"""

from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries


class MovingAverageFilter(FilteringModel):
    """
    A simple moving average filter. Works on deterministic and stochastic series.
    """

    def __init__(self, window: int, centered: bool = True):
        """
        Parameters
        ----------
        window
            The length of the window over which to average values
        centered
            Set the labels at the center of the window. If not set, the averaged values are lagging after
            the original values.
        """
        super().__init__()
        self.window = window
        self.centered = centered

    def filter(self, series: TimeSeries):
        """
        Computes a moving average of this series' values and returns a new TimeSeries.
        The returned series has the same length and time axis as `series`. (Note that this might create border effects).

        Parameters
        ----------
        series
            The a deterministic series to average

        Returns
        -------
        TimeSeries
            A time series containing the average values
        """
        transformation = {
            "function": "mean",
            "mode": "rolling",
            "window": self.window,
            "center": self.centered,
            "min_periods": 1,
        }

        return series.window_transform(
            transforms=transformation, forecasting_safe=False
        )
