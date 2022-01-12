"""
Moving Average
--------------
"""

from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries


class MovingAverage(FilteringModel):
    """
    A simple moving average filter. Works on deterministic series (made of 1 sample).
    """

    def __init__(self, window: int, centered: bool = True):
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

    def filter(self, series: TimeSeries):
        """
        Computes a moving average of this series' values and returns a new TimeSeries.
        The returned series has the same length and time axis as `series`. (Note that this might create border effects).

        Behind the scenes the moving average is computed using :func:`pandas.DataFrame.rolling()` on the underlying
        DataFrame.

        Parameters
        ----------
        series
            The a deterministic series to average

        Returns
        -------
        TimeSeries
            A time series containing the average values
        """
        filtered_df = (
            series.pd_dataframe(copy=False)
            .rolling(window=self.window, min_periods=1, center=self.centered)
            .mean()
        )
        return TimeSeries.from_dataframe(filtered_df)
