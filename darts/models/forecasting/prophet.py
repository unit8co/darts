"""
Prophet
-------
"""

import pandas as pd

# from darts.timeseries import TimeSeries
from darts import TimeSeries
from darts.models.forecasting.arima import ARIMA


class Prophet(ARIMA):
    def __init__(self, df: pd.DataFrame, **kwargs):
        """Prophet model implementation.

        Parameters
        ----------
        df
            Dataframe containing the time series data
        **kwargs
            Dictionary of parameters to pass to the `Prophet` class
        """

    def fit(self, series: TimeSeries):
        """Fit the Prophet model.

        Parameters
        ----------
        series
            The series containing the time series data.
        """

    def predict(self, **kwargs) -> TimeSeries:
        """Fit the Prophet model."""
