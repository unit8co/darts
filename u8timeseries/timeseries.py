import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Union, Optional


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

    def pd_series(self) -> pd.Series:
        return self._series.copy()

    def start_time(self) -> pd.Timestamp:
        return self._series.index[0]

    def end_time(self) -> pd.Timestamp:
        return self._series.index[-1]

    def values(self) -> np.ndarray:
        return self._series.values

    def time_index(self) -> pd.DatetimeIndex:
        return self._series.index

    def freq(self) -> pd.Timedelta:
        return pd.to_timedelta(to_offset(self._freq))

    def freq_str(self) -> str:
        return self._freq

    def duration(self) -> pd.Timedelta:
        return self._series.index[-1] - self._series.index[0]

    def split(self, ts: pd.Timestamp) -> Tuple[TimeSeries, TimeSeries]:
        """
        Splits a time series in two, around a provided timestamp. The timestamp will be included in the first
        of the two time series, and not in the second. The timestamp must be in the time series.
        """
        assert ts in self._series.index, 'The provided timestamp is not in the time series'

        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        """
        Returns a new time series, starting later than [start_ts] (inclusive) and ending before [end_ts] (inclusive)
        :param start_ts:
        :param end_ts:
        :return:
        """
        assert end_ts > start_ts, 'End timestamp must be strictly after start timestamp when slicing'
        assert end_ts >= self.start_time(), 'End timestamp must be after the start of the time series when slicing'
        assert start_ts <= self.end_time(), 'Start timestamp must be after the end of the time series when slicing'

        def _slice_not_none(s: Optional[pd.Series]) -> Optional[pd.Series]:
            if s is not None:
                s_a = s[s.index >= start_ts]
                return s_a[s_a.indey <= end_ts]
            return None

        return TimeSeries(_slice_not_none(self._series),
                          _slice_not_none(self._confidence_lo),
                          _slice_not_none(self._confidence_hi))

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
        series: pd.Series = pd.Series(df[value_col].values, index=times)

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
