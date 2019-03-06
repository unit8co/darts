import pandas as pd
import numpy as np


class TimeSeries:
    def __init__(self, series: pd.Series, confidence_lo: pd.Series = None, confidence_hi: pd.Series = None):
        """

        :param series:
        :param confidence_lo:
        :param confidence_hi:
        """
        assert len(series) >= 1, 'Time series must have at least one value'
        assert isinstance(series.index, pd.DatetimeIndex), 'Series must be indexed with a DatetimeIndex'

        self._series: pd.Series = series.sort_index()  # Sort by time
        self._freq: str = self._series.index.inferred_freq

        # TODO: optionally fill holes (including missing dates) - for now we assume no missing dates
        assert self._freq is not None, 'Could not infer frequency. Are some dates missing?'

        # TODO: handle confidence intervals same way
        self._confidence_lo = confidence_lo.copy() if confidence_lo is not None else None
        self._confidence_hi = confidence_hi.copy() if confidence_hi is not None else None

    def get_series(self) -> pd.Series:
        return self._series.copy()

    def get_values(self) -> np.ndarray:
        return self._series.values

    def get_time_index(self) -> pd.DatetimeIndex:
        return self._series.index

    def get_freq(self) -> str:
        return self._freq

    @staticmethod
    def from_dataframe(df: pd.DataFrame, time_col: str, value_col: str,
                       conf_lo_col: str = None, conf_hi_col: str = None):
        """
        Returns a TimeSeries built from some DataFrame columns (ignoring DataFrame index)
        :param df: the DataFrame
        :param time_col: the time column name (mandatory)
        :param value_col: the value column name (mandatory)
        :param conf_lo_col: the lower confidence interval column name (optional)
        :param conf_hi_col: the upper confidence interval column name (optional)
        """

        times: pd.Series = pd.to_datetime(df[time_col], errors='raise')
        series: pd.Series = pd.Series(df[value_col], index=times)

        conf_lo = pd.Series(df[conf_lo_col], index=times) if conf_lo_col is not None else None
        conf_hi = pd.Series(df[conf_hi_col], index=times) if conf_hi_col is not None else None

        return TimeSeries(series, conf_lo, conf_hi)

    @staticmethod
    def from_times_and_values(times: pd.DatetimeIndex,
                              values: np.ndarray,
                              confidence_lo: np.ndarray = None,
                              confidence_hi: np.ndarray = None):
        series = pd.Series(values, index=times)
        series_lo = pd.Series(confidence_lo, index=times) if confidence_lo is not None else None
        series_hi = pd.Series(confidence_hi, index=times) if confidence_hi is not None else None

        return TimeSeries(series, series_lo, series_hi)

    def __len__(self):
        return len(self._series)
