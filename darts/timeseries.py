"""
Timeseries
----------

`TimeSeries` is the main class in `darts`. It represents a univariate or multivariate time series.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any, List, Union
from inspect import signature

from .logging import raise_log, raise_if_not, raise_if, get_logger

logger = get_logger(__name__)


class TimeSeries:
    def __init__(self,
                 df: pd.DataFrame,
                 freq: Optional[str] = None,
                 fill_missing_dates: Optional[bool] = True):
        """
        A TimeSeries is an object representing a univariate or multivariate time series.

        TimeSeries are meant to be immutable.

        Parameters
        ----------
        df
            The actual time series, as a pandas DataFrame with a proper time index.
        freq
            Optionally, a Pandas offset alias representing the frequency of the DataFrame.
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            When creating a TimeSeries instance with a length smaller than 3, this argument must be passed.
            Furthermore, this argument can be used to override the automatic frequency detection if the
            index is incomplete.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.
        """

        raise_if_not(isinstance(df, pd.DataFrame), "Data must be provided in form of a pandas.DataFrame instance",
                     logger)
        raise_if_not(len(df) > 0 and df.shape[1] > 0, 'Time series must not be empty.', logger)
        raise_if_not(isinstance(df.index, pd.DatetimeIndex), 'Time series must be indexed with a DatetimeIndex.',
                     logger)
        raise_if_not(df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), 'Time series must'
                     ' contain only numerical values.', logger)
        raise_if_not(len(df) >= 3 or freq is not None, 'Time series must have at least 3 values if the "freq" argument'
                     ' is not passed', logger)

        self._df = df.sort_index()  # Sort by time **returns a copy**
        self._df.columns = self._clean_df_columns(df.columns)

        if (len(df) < 3):
            self._freq: str = freq
        else:
            if not df.index.inferred_freq:
                if fill_missing_dates:
                    self._df = self._fill_missing_dates(self._df, freq)
                else:
                    raise_if_not(False, 'Could not infer frequency. Are some dates missing? '
                                        'Try specifying `fill_missing_dates=True`.', logger)
            self._freq: str = self._df.index.inferred_freq  # Infer frequency
            if (freq is not None and self._freq != freq):
                logger.warning('The inferred frequency does not match the value of the "freq" argument.')

        self._df.index.freq = self._freq  # Set the inferred frequency in the Pandas dataframe

        # The actual values
        self._values: np.ndarray = self._df.values

    def _clean_df_columns(self, columns: pd._typing.Axes) -> pd.Index:
        """clean pandas dataFrame columns for usage with TimeSeries"""
        # convert everything to str
        columns_list = columns.to_list()
        for i, column in enumerate(columns_list):
            if not isinstance(column, str):
                columns_list[i] = str(column)

        columns = pd.Index(columns_list)

        if isinstance(columns, pd.RangeIndex) or not columns.is_unique:
            columns = pd.Index([str(i) for i in range(len(columns))])  # we make sure columns are str for indexing.

        return columns

    def _assert_univariate(self):
        """
        Raises an error if the current TimeSeries instance is not univariate.
        """
        if (self._df.shape[1] != 1):
            raise_log(AssertionError('Only univariate TimeSeries instances support this method'), logger)

    def pd_series(self, copy=True) -> pd.Series:
        """
        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.

        Returns
        -------
        pandas.Series
            A Pandas Series representation of this univariate time series.
        """
        self._assert_univariate()
        if copy:
            return self._df.iloc[:, 0].copy()
        else:
            return self._df.iloc[:, 0]

    def pd_dataframe(self, copy=True) -> pd.DataFrame:
        """
        Parameters
        ----------
        copy
            Whether to return a copy of the dataframe. Leave it to True unless you know what you are doing.

        Returns
        -------
        pandas.DataFrame
            The Pandas Dataframe underlying this time series
        """
        if copy:
            return self._df.copy()
        else:
            return self._df

    def start_time(self) -> pd.Timestamp:
        """
        Returns
        -------
        pandas.Timestamp
            A timestamp containing the first time of the TimeSeries.
        """
        return self._df.index[0]

    def end_time(self) -> pd.Timestamp:
        """
        Returns
        -------
        pandas.Timestamp
            A timestamp containing the last time of the TimeSeries.
        """
        return self._df.index[-1]

    def first_value(self) -> float:
        """
        Returns
        -------
        float
            The first value of this univariate time series
        """
        self._assert_univariate()
        return self._values[0][0]

    def last_value(self) -> float:
        """
        Returns
        -------
        float
            The last value of this univariate time series
        """
        self._assert_univariate()
        return self._values[-1][0]

    def first_values(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The first values of every component of this time series
        """
        return self._values[0]

    def last_values(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The last values of every component of this time series
        """
        return self._values[-1]

    def values(self, copy=True) -> np.ndarray:
        """
        Parameters
        ----------
        copy
            Whether to return a copy of the values. Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        if copy:
            return np.copy(self._values)
        else:
            return self._values

    def univariate_values(self, copy=True) -> np.ndarray:
        """
        Parameters
        ----------
        copy
            Whether to return a copy of the values. Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing the time series guaranteed to be univariate.
        """
        self._assert_univariate()
        if copy:
            return np.copy(self._df.iloc[:, 0].values)
        else:
            return self._df.iloc[:, 0].values

    def time_index(self) -> pd.DatetimeIndex:
        """
        Returns
        -------
        pandas.DatetimeIndex
            The time index of this time series.
        """
        return deepcopy(self._df.index)

    def freq(self) -> pd.DateOffset:
        """
        Returns
        -------
        pandas.DateOffset
            The frequency of this time series
        """
        return to_offset(self._freq)

    def freq_str(self) -> str:
        """
        Returns
        -------
        str
            A string representation of the frequency of this time series
        """
        return self._freq

    def duration(self) -> pd.Timedelta:
        """
        Returns
        -------
        pandas.Timedelta
            The duration of this time series.
        """
        return self._df.index[-1] - self._df.index[0]

    def gaps(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame containing a row for every gap (rows with all-NaN values in underlying DataFrame)
            in this time series. The DataFrame contains three columns that include the start and end time stamps
            of the gap and the integer length of the gap (in `self.freq()` units).
        """

        is_nan_series = self._df.isna().all(axis=1).astype(int)
        diff = pd.Series(np.diff(is_nan_series.values), index=is_nan_series.index[:-1])
        gap_starts = diff[diff == 1].index + self.freq()
        gap_ends = diff[diff == -1].index

        if is_nan_series.iloc[0] == 1:
            gap_starts = gap_starts.insert(0, self.start_time())
        if is_nan_series.iloc[-1] == 1:
            gap_ends = gap_ends.insert(len(gap_ends), self.end_time())

        gap_df = pd.DataFrame()
        gap_df['gap_start'] = gap_starts
        gap_df['gap_end'] = gap_ends
        gap_df['gap_size'] = gap_df.apply(
            lambda row: pd.date_range(start=row.gap_start, end=row.gap_end, freq=self.freq()).size, axis=1
        )

        return gap_df

    def copy(self, deep: bool = True) -> 'TimeSeries':
        """
        Make a copy of this time series object

        Parameters
        ----------
        deep
            Make a deep copy. If False, the underlying pandas DataFrame will be the same

        Returns
        -------
        TimeSeries
            A copy of this time series.
        """
        if deep:
            return TimeSeries(self.pd_dataframe(), self.freq_str())
        else:
            return TimeSeries(self._df, self.freq_str())

    def _raise_if_not_within(self, ts: pd.Timestamp):
        if (ts < self.start_time()) or (ts > self.end_time()):
            raise_log(ValueError('Timestamp must be between {} and {}'.format(self.start_time(),
                                                                              self.end_time())), logger)

    def split_after(self, split_point: Union[pd.Timestamp, float, int]) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, after a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the dataset to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position after
            which the split is performed. If timestamp, it will be contained in the first TimeSeries, but not
            in the second one. The timestamp may not appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples up to the `split_point`,
            and the second contains the remaining ones.
        """
        ts = self.get_timestamp_at_point(split_point) if isinstance(split_point, (int, float)) else split_point
        self._raise_if_not_within(ts)
        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)
        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def split_before(self, split_point: Union[pd.Timestamp, float, int]) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits a TimeSeries in two, before a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the dataset to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position before
            which the split is performed. If timestamp, it will be contained in the second TimeSeries, but not
            in the first one. The timestamp may not appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples before the `split_point`,
            and the second contains the remaining ones.
        """
        ts = self.get_timestamp_at_point(split_point) if isinstance(split_point, (int, float)) else split_point
        self._raise_if_not_within(ts)
        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)
        end_first_series: pd.Timestamp = ts - self.freq()  # second series does not include ts
        return self.slice(self.start_time(), end_first_series), self.slice(ts, self.end_time())

    def drop_after(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything after the provided timestamp `ts`, included.
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        Parameters
        ----------
        ts
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, before `ts`.
        """
        self._raise_if_not_within(ts)
        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)
        end_series: pd.Timestamp = ts - self.freq()  # new series does not include ts
        return self.slice(self.start_time(), end_series)

    def drop_before(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything before the provided timestamp `ts`, included.
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        Parameters
        ----------
        ts
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, after `ts`.
        """
        self._raise_if_not_within(ts)
        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)
        start_series: pd.Timestamp = ts + self.freq()  # new series does not include ts
        return self.slice(start_series, self.end_time())

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp, copy=True) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than `start_ts` and ending before `end_ts`, inclusive on both ends.
        The timestamps don't have to be in the series.

        Parameters
        ----------
        start_ts
            The timestamp that indicates the left cut-off.
        end_ts
            The timestamp that indicates the right cut-off.
        copy
            If True, the returned series will contain a copy of this serie's dataframe, otherwise a view

        Returns
        -------
        TimeSeries
            A new series, with indices greater or equal than `start_ts` and smaller or equal than `end_ts`.
        """
        raise_if_not(end_ts >= start_ts, 'End timestamp must be after start timestamp when slicing.', logger)
        raise_if_not(end_ts >= self.start_time(),
                     'End timestamp must be after the start of the time series when slicing.', logger)
        raise_if_not(start_ts <= self.end_time(),
                     'Start timestamp must be after the end of the time series when slicing.', logger)

        def _slice_not_none(s: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if s is not None:
                s_a = s[s.index >= start_ts]
                return s_a[s_a.index <= end_ts]
            return None
        return TimeSeries(_slice_not_none(self.pd_dataframe(copy=copy)), self.freq_str())

    def slice_n_points_after(self, start_ts: pd.Timestamp, n: int, copy=True) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than `start_ts` (included) and having at most `n` points.

        The timestamp may not be in the time series. If it is, it will be included in the new TimeSeries.

        Parameters
        ----------
        start_ts
            The timestamp that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.
        copy
            If True, the returned series will contain a copy of this serie's dataframe, otherwise a view

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n` and indices greater or equal than `start_ts`.
        """
        raise_if_not(n >= 0, 'n should be a positive integer.', logger)  # TODO: logically raise if n<3, cf. init
        if not isinstance(n, int):
            logger.warning(f"Converted n to int from {n} to {int(n)}")
            n = int(n)
        self._raise_if_not_within(start_ts)
        start_ts = self.time_index()[self.time_index() >= start_ts][0]  # closest index after start_ts (new start_ts)
        end_ts: pd.Timestamp = start_ts + (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts, copy)

    def slice_n_points_before(self, end_ts: pd.Timestamp, n: int, copy=True) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending before `end_ts` (included) and having at most `n` points.

        The timestamp may not be in the TimeSeries. If it is, it will be included in the new TimeSeries.

        Parameters
        ----------
        end_ts
            The timestamp that indicates the splitting time.
        n
            The maximal length of the new time series.
        copy
            If True, the returned series will contain a copy of this serie's dataframe, otherwise a view

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n` and indices smaller or equal than `end_ts`.
        """
        raise_if_not(n >= 0, 'n should be a positive integer.', logger)
        if not isinstance(n, int):
            logger.warning(f"Converted n to int from {n} to {int(n)}")
            n = int(n)
        self._raise_if_not_within(end_ts)
        end_ts = self.time_index()[self.time_index() <= end_ts][-1]
        start_ts: pd.Timestamp = end_ts - (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts, copy)

    def slice_intersect(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Returns a TimeSeries slice of this time series, where the time index has been intersected with the one
        provided in argument. Note that this method is in general *not* symmetric.

        Parameters
        ----------
        other
            the other time series

        Returns
        -------
        TimeSeries
            a new series, containing the values of this series, over the time-span common to both time series.
        """
        time_index = self.time_index().intersection(other.time_index())
        return self.__getitem__(time_index)

    def get_timestamp_at_point(self, point: Union[pd.Timestamp, float, int]) -> pd.Timestamp:
        """
        Converts a point into a pandas.Timestamp in the time series

        Parameters
        ----------
        point
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In case of a `float`, the parameter will be treated as the proportion of the time series
            that should lie before the point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series`. Will raise a ValueError if not a valid index in `series`
            In case of a `pandas.Timestamp`, point will be returned as is provided that the timestamp
            is present in the series time index, otherwise will raise a ValueError.
        series
            The time series to index in
        """
        if isinstance(point, float):
            raise_if_not(point >= 0.0 and point < 1.0, 'point (float) should be between 0.0 and 1.0.', logger)
            point_index = int((len(self) - 1) * point)
            timestamp = self._df.index[point_index]
        elif isinstance(point, int):
            raise_if(point not in range(len(self)), "point (int) should be a valid index in series", logger)
            timestamp = self._df.index[point]
        elif isinstance(point, pd.Timestamp):
            raise_if(point not in self,
                     'point (pandas.Timestamp) must be an entry in the time series\' time index',
                     logger)
            timestamp = point
        else:
            raise_log(TypeError("`point` needs to be either `float`, `int` or `pd.Timestamp`"), logger)
        return timestamp

    def get_index_at_point(self, point: Union[pd.Timestamp, float, int]) -> int:
        """
        Converts a point into the corresponding index in the time series

        Parameters
        ----------
        point
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In case of a `float`, the parameter will be treated as the proportion of the time series
            that should lie before the point.
            In case of a `pandas.Timestamp`, will return the index corresponding to the timestamp in the series
            if the timestamp is present in the series time index, otherwise will raise a ValueError.
            In case of an `int`, the parameter will be returned as is provided that it is a valid index,
            otherwise will raise a ValueError.
        series
            The time series to index in
        """
        timestamp = self.get_timestamp_at_point(point)
        return self._df.index.get_loc(timestamp)

    def strip(self) -> 'TimeSeries':
        """
        Returns a TimeSeries slice of this time series, where NaN-only entries at the beginning and the end of the
        series are removed. No entries after (and including) the first non-NaN entry and before (and including) the
        last non-NaN entry are removed.

        Returns
        -------
        TimeSeries
            a new series based on the original where NaN-only entries at start and end have been removed
        """

        new_start_idx = self._df.first_valid_index()
        new_end_idx = self._df.last_valid_index()
        new_series = self._df.loc[new_start_idx:new_end_idx]

        return TimeSeries(new_series, self.freq_str())

    def longest_contiguous_slice(self, max_gap_size: int = 0) -> 'TimeSeries':
        """
        Returns the largest TimeSeries slice of this time series that contains no gaps (contigouse all-NaN rows)
        larger than `max_gap_size`.

        Returns
        -------
        TimeSeries
            a new series constituting the largest slice of the original with no or bounded gaps
        """
        if self._df.isna().sum().sum() == 0:
            return self.copy()
        stripped_series = self.strip()
        gaps = stripped_series.gaps()
        relevant_gaps = gaps[gaps['gap_size'] > max_gap_size]

        curr_slice_start = stripped_series.start_time()
        max_size = pd.Timedelta(days=0)
        max_slice_start = None
        max_slice_end = None
        for index, row in relevant_gaps.iterrows():
            size = row['gap_start'] - curr_slice_start - self.freq()
            if size > max_size:
                max_size = size
                max_slice_start = curr_slice_start
                max_slice_end = row['gap_start'] - self.freq()
            curr_slice_start = row['gap_end'] + self.freq()

        if stripped_series.end_time() - curr_slice_start > max_size:
            max_slice_start = curr_slice_start
            max_slice_end = self.end_time()

        return stripped_series[max_slice_start:max_slice_end]

    # TODO: other rescale? such as giving a ratio, or a specific position? Can be the same function
    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Returns a new TimeSeries, which is a multiple of this TimeSeries such that
        the first value is `value_at_first_step`.
        (Note: numerical errors can appear with `value_at_first_step > 1e+24`).

        Parameters
        ----------
        value_at_first_step
            The new value for the first entry of the TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, where the first value is `value_at_first_step` and other values
            have been scaled accordingly.
        """

        raise_if_not((self.values()[0] != 0).all(), 'Cannot rescale with first value 0.', logger)

        coef = value_at_first_step / self.values()[0]  # TODO: should the new TimeSeries have the same dtype?
        new_series = coef * self._df
        return TimeSeries(new_series, self.freq_str())

    def shift(self, n: int) -> 'TimeSeries':
        """
        Shifts the time axis of this TimeSeries by `n` time steps.

        If :math:`n > 0`, shifts in the future. If :math:`n < 0`, shifts in the past.

        For example, with :math:`n=2` and `freq='M'`, March 2013 becomes May 2013.
        With :math:`n=-2`, March 2013 becomes Jan 2013.

        Parameters
        ----------
        n
            The signed number of time steps to shift by.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with a shifted index.
        """
        if not isinstance(n, int):
            logger.warning(f"Converted n to int from {n} to {int(n)}")
            n = int(n)
        try:
            self.time_index()[-1] + n * self.freq()
        except pd.errors.OutOfBoundsDatetime:
            raise_log(OverflowError("the add operation between {} and {} will "
                                    "overflow".format(n * self.freq(), self.time_index()[-1])), logger)
        new_time_index = self._df.index.map(lambda ts: ts + n * self.freq())
        new_series = self._df.copy()
        new_series.index = new_time_index
        return TimeSeries(new_series, self.freq_str())

    def diff(self,
             n: Optional[int] = 1,
             periods: Optional[int] = 1,
             dropna: Optional[bool] = True) -> 'TimeSeries':
        """
        Returns a differenced time series. This is often used to make a time series stationary.

        Parameters
        ----------
        n
            Optionally, a signed integer indicating the number of differencing steps.
        periods
            Optionally, periods to shift for calculating difference.
        dropna
            Optionally, a boolean value indicating whether to drop the missing values created by
            the pandas.DataFrame.diff method.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed after differencing.
        """
        if not isinstance(n, int) or n < 1:
             raise_log(ValueError("'n' must be a positive integer >= 1."))
        if not isinstance(periods, int):
             raise_log(ValueError("'periods' must be an integer."))

        diff_df = self._df.diff(periods=periods)
        for _ in range(n-1):
            diff_df = diff_df.diff(periods=periods)
        if dropna:
            diff_df.dropna(inplace=True)
        return TimeSeries(diff_df, freq=None, fill_missing_dates=False)

    @staticmethod
    def from_series(pd_series: pd.Series,
                    freq: Optional[str] = None,
                    fill_missing_dates: Optional[bool] = True) -> 'TimeSeries':
        """
        Returns a TimeSeries built from a pandas Series.

        Parameters
        ----------
        pd_series
            The pandas Series instance.
        freq
            Optionally, a string representing the frequency of the Pandas Series.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.
        """
        return TimeSeries(pd.DataFrame(pd_series), freq, fill_missing_dates)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       time_col: Optional[str] = None,
                       value_cols: Optional[Union[List[str], str]] = None,
                       freq: Optional[str] = None,
                       fill_missing_dates: Optional[bool] = True) -> 'TimeSeries':
        """
        Returns a TimeSeries instance built from a selection of columns of a DataFrame.
        One column (or the DataFrame index) has to represent the time,
        and a list of columns `value_cols` has to represent the values for this time series.

        Parameters
        ----------
        df
            The DataFrame
        time_col
            The time column name (mandatory). If set to `None`, the DataFrame index will be used.
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the DataFrame. If set to
            `None` use the whole DataFrame.
        freq
            Optionally, a string representing the frequency of the Pandas DataFrame.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.

        Returns
        -------
        TimeSeries
            A univariate or multivariate TimeSeries constructed from the inputs.
        """
        if value_cols is None:
            series_df = df.loc[:, df.columns != time_col]
        else:
            if isinstance(value_cols, str):
                value_cols = [value_cols]

            series_df = df[value_cols]

        if time_col is None:
            series_df.index = pd.to_datetime(df.index, errors='raise')
        else:
            series_df.index = pd.to_datetime(df[time_col], errors='raise')

        return TimeSeries(series_df, freq, fill_missing_dates)

    @staticmethod
    def from_times_and_values(times: pd.DatetimeIndex,
                              values: Union[np.ndarray, pd.DataFrame],
                              freq: Optional[str] = None,
                              fill_missing_dates: Optional[bool] = True,
                              columns: Optional[pd._typing.Axes] = None) -> 'TimeSeries':
        """
        Returns TimeSeries built from an index and values.

        Parameters
        ----------
        times
            A `pandas.DateTimeIndex` representing the time axis for the time series.
        values
            An array of values for the TimeSeries.
        freq
            Optionally, a string representing the frequency of the time series.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.
        columns
            Columns to be used by the underlying pandas DataFrame.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.
        """
        df = pd.DataFrame(values, index=times)
        if columns is not None:
            df.columns = columns
        return TimeSeries(df, freq, fill_missing_dates)

    def plot(self,
             new_plot: bool = False,
             *args,
             **kwargs):
        """
        A wrapper method around `pandas.Series.plot()`.

        Parameters
        ----------
        new_plot
            whether to spawn a new Figure
        args
            some positional arguments for the `plot()` method
        kwargs
            some keyword arguments for the `plot()` method
        """
        raise_if(self.width > 15, "Current TimeSeries instance contains too many components to plot.", logger)
        fig = (plt.figure() if new_plot else (kwargs['figure'] if 'figure' in kwargs else plt.gcf()))
        kwargs['figure'] = fig
        if 'label' in kwargs:
            label = kwargs['label']
        for i in range(self.width):
            if i > 0:
                kwargs['figure'] = plt.gcf()
                if 'label' in kwargs:
                    kwargs['label'] = label + '_' + str(i)
            self.univariate_component(i).pd_series().plot(*args, **kwargs)
        x_label = self.time_index().name
        if x_label is not None and len(x_label) > 0:
            plt.xlabel(x_label)

    def has_same_time_as(self, other: 'TimeSeries') -> bool:
        """
        Checks whether this TimeSeries and another one have the same index.

        Parameters
        ----------
        other
            the other series

        Returns
        -------
        bool
            True if both TimeSeries have the same index, False otherwise.
        """
        if self.__len__() != len(other):
            return False
        return (other.time_index() == self.time_index()).all()

    def append(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Appends another TimeSeries to this TimeSeries.

        Parameters
        ----------
        other
            A second TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, obtained by appending the second TimeSeries to the first.
        """
        raise_if_not(other.start_time() == self.end_time() + self.freq(),
                     'Appended TimeSeries must start one time step after current one.', logger)
        # TODO additional check?
        raise_if_not(other.freq() == self.freq(),
                     'Appended TimeSeries must have the same frequency as the current one', logger)

        series = self._df.append(other.pd_dataframe())
        return TimeSeries(series, self.freq_str())

    def append_values(self,
                      values: np.ndarray,
                      index: pd.DatetimeIndex = None) -> 'TimeSeries':
        """
        Appends values to current TimeSeries, to the given indices.

        If no index is provided, assumes that it follows the original data.
        Does not update value if already existing indices are provided.

        Parameters
        ----------
        values
            An array with the values to append.
        index
            A `pandas.DateTimeIndex` for the new values (optional)

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values appended
        """
        if len(values) < 1:
            return self
        if isinstance(values, list):
            values = np.array(values)
        if index is None:
            index = pd.DatetimeIndex([self.end_time() + i * self.freq() for i in range(1, 1 + len(values))])
        raise_if_not(isinstance(index, pd.DatetimeIndex), 'Values must be indexed with a DatetimeIndex.', logger)
        raise_if_not(len(index) == len(values), 'Values and index must have same length.', logger)
        raise_if_not(self.time_index().intersection(index).empty, "Cannot add already present time index.", logger)
        new_indices = index.argsort()
        index = index[new_indices]
        # TODO do we really want that?
        raise_if_not(index[0] == self.end_time() + self.freq(),
                     'Appended index must start one time step after current one.', logger)
        if len(index) > 2:
            raise_if_not(index.inferred_freq == self.freq_str(),
                         'Appended index must have the same frequency as the current one.', logger)
        elif len(index) == 2:
            raise_if_not(index[-1] == index[0] + self.freq(),
                         'Appended index must have the same frequency as the current one.', logger)
        values = values[new_indices]
        new_series = pd.DataFrame(values, index=index)
        new_series.columns = self._clean_df_columns(new_series.columns)
        series = self._df.append(new_series)

        return TimeSeries(series, self.freq_str())

    def update(self,
               index: pd.DatetimeIndex,
               values: np.ndarray = None) -> 'TimeSeries':
        """
        Updates the TimeSeries instance with the new values provided.
        If indices are not in original TimeSeries, they will be discarded.
        Use `numpy.nan` to ignore a specific index in a series.

        Parameters
        ----------
        index
            A `pandas.DateTimeIndex` containing the indices to replace.
        values
            An array containing the values to replace (optional).

        Returns
        -------
        TimeSeries
            A new TimeSeries with updated values.
        """

        if isinstance(values, (list, range)):
            values = np.array(values)

        if values is not None and len(values.shape) == 1:
            values = values.reshape((len(values), 1))

        raise_if(values is None, "'values' parameter should not be None.", logger)
        raise_if(index is None, "Index must be filled.")
        if (values is not None):
            raise_if_not(len(values) == len(index), "The number of values must correspond "
                                                    "to the number of indices: {} != {}".format(len(values),
                                                                                                len(index)), logger)
            raise_if_not(self._df.shape[1] == values.shape[1], "The number of columns in values must correspond "
                                                               "to the number of columns in the current TimeSeries"
                                                               "instance: {} != {}".format(self._df.shape[1],
                                                                                           values.shape[1]))

        ignored_indices = [index.get_loc(ind) for ind in (set(index) - set(self.time_index()))]
        index = index.delete(ignored_indices)  # only contains indices that are present in the TimeSeries instance
        if values is None:
            series = values
        else:
            df = pd.DataFrame(np.delete(values, ignored_indices, axis=0), index=index)
            df.columns = self._clean_df_columns(df.columns)
            series = df

        raise_if_not(len(index) > 0, "Must give at least one correct index.", logger)

        new_series = self.pd_dataframe()
        if series is not None:
            new_series.update(series)
            new_series = new_series.astype(self._df.dtypes)
        return TimeSeries(new_series, self.freq_str())

    def stack(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Stacks another univariate or multivariate TimeSeries with the same index on top of
        the current one and returns the newly formed multivariate TimeSeries that includes
        all the components of `self` and of `other`.

        Parameters
        ----------
        other
            A TimeSeries instance with the same index as the current one.

        Returns
        -------
        TimeSeries
            A new multivariate TimeSeries instance.
        """
        raise_if_not((self.time_index() == other.time_index()).all(), 'The indices of the two TimeSeries instances '
                     'must be equal', logger)

        new_dataframe = pd.concat([self.pd_dataframe(), other.pd_dataframe()], axis=1)
        return TimeSeries(new_dataframe, self.freq_str())

    @property
    def width(self) -> int:
        """
        Returns
        -------
        int
            The number of components (univariate time series) of the current TimeSeries instance.
        """
        return self._df.shape[1]

    def univariate_component(self, index: int) -> 'TimeSeries':
        """
        Retrieves one of the components of the current TimeSeries instance
        and returns it as new univariate TimeSeries instance.

        Parameters
        ----------
        index
            An zero-indexed integer indicating which component to retrieve.

        Returns
        -------
        TimeSeries
            A new univariate TimeSeries instance.
        """

        raise_if_not(index >= 0 and index < self.width, 'The index must be between 0 and the number of components '
                     'of the current TimeSeries instance - 1, {}'.format(self.width - 1), logger)

        return TimeSeries.from_series(self.pd_dataframe().iloc[:, index], freq=self.freq_str())

    def add_datetime_attribute(self, attribute: str, one_hot: bool = False) -> 'TimeSeries':
        """
        Returns a new TimeSeries instance with one (or more) additional dimension(s) that contain an attribute
        of the time index of the current series specified with `component`, such as 'weekday', 'day' or 'month'.

        Parameters
        ----------
        attribute
            A pd.DatatimeIndex attribute which will serve as the basis of the new column(s).
        one_hot
            Boolean value indicating whether to add the specified attribute as a one hot encoding
            (results in more columns).

        Returns
        -------
        TimeSeries
            New TimeSeries instance enhanced by `attribute`.
        """
        from .utils import timeseries_generation as tg
        return self.stack(tg.datetime_attribute_timeseries(self.time_index(), attribute, one_hot))

    def add_holidays(self,
                     country_code: str,
                     prov: str = None,
                     state: str = None) -> 'TimeSeries':
        """
        Adds a binary univariate TimeSeries to the current one that equals 1 at every index that
        corresponds to selected country's holiday, and 0 otherwise. The frequency of the TimeSeries is daily.

        Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`_.

        Parameters
        ----------
        country_code
            The country ISO code
        prov
            The province
        state
            The state

        Returns
        -------
        TimeSeries
            TimeSeries instance enhanced with binary holiday column.
        """
        from .utils import timeseries_generation as tg
        return self.stack(tg.holidays_timeseries(self.time_index(), country_code, prov, state))

    def resample(self, freq: str, method: str = 'pad') -> 'TimeSeries':
        """
        Creates an reindexed time series with a given frequency.
        Provided method is used to fill holes in reindexed TimeSeries, by default 'pad'.

        Parameters
        ----------
        freq
            The new time difference between two adjacent entries in the returned TimeSeries.
            A DateOffset alias is expected.
        method:
            Method to fill holes in reindexed TimeSeries (note this does not fill NaNs that already were present):

            ‘pad’: propagate last valid observation forward to next valid

            ‘backfill’: use NEXT valid observation to fill.
        Returns
        -------
        TimeSeries
            A reindexed TimeSeries with given frequency.
        """

        new_df = self.pd_dataframe().asfreq(freq, method=method)

        return TimeSeries(new_df, freq)

    def is_within_range(self,
                        ts: pd.Timestamp) -> bool:
        """
        Check whether a given timestamp is withing the time interval of this time series

        Parameters
        ----------
        ts
            The `pandas.Timestamp` to check

        Returns
        -------
        bool
            Whether the timestamp is contained within the time interval of this time series.
            Note that the timestamp does not need to be *in* the time series.
        """
        index = self.time_index()
        return index[0] <= ts <= index[-1]

    def map(self,
            fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]]) -> 'TimeSeries':  # noqa: E501
        """
        Applies the function `fn` elementwise to all values in this TimeSeries, or, to only those
        values in the columns specified by the optional argument `cols`. Returns a new
        TimeSeries instance.

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y

        Returns
        -------
        TimeSeries
            A new TimeSeries instance
        """
        if not isinstance(fn, Callable):
            raise_log(TypeError("fn should be callable"), logger)

        if isinstance(fn, np.ufunc):
            if fn.nin == 1 and fn.nout == 1:
                num_args = 1
            elif fn.nin == 2 and fn.nout == 1:
                num_args = 2
            else:
                raise_log(ValueError("fn must have either one or two arguments and return a single value"), logger)
        else:
            try:
                num_args = len(signature(fn).parameters)
            except ValueError:
                raise_log(ValueError("inspect.signature(fn) failed. Try wrapping fn in a lambda, e.g. lambda x: fn(x)"),
                          logger)

        if num_args == 1:  # simple map function f(x)
            new_dataframe = self.pd_dataframe().applymap(fn)
        elif num_args == 2:  # map function uses timestamp f(timestamp, x)
            def apply_fn_wrapper(row):
                timestamp = row.name
                return row.map(lambda x: fn(timestamp, x))

            new_dataframe = self.pd_dataframe().apply(apply_fn_wrapper, axis=1)
        else:
            raise_log(ValueError("fn must have either one or two arguments"), logger)

        return TimeSeries(new_dataframe, self.freq_str())

    def to_json(self) -> str:
        """
        Converts the `TimeSeries` object to a JSON String

        Returns
        -------
        str
            A JSON String representing the time series
        """
        return self._df.to_json(orient='split', date_format='iso')

    @staticmethod
    def from_json(json_str: str) -> 'TimeSeries':
        """
        Converts the JSON String representation of a `TimeSeries` object (produced using `TimeSeries.to_json()`)
        into a `TimeSeries` object

        Parameters
        ----------
        json_str
            The JSON String to convert

        Returns
        -------
        TimeSeries
            The time series object converted from the JSON String
        """

        df = pd.read_json(json_str, orient='split')
        return TimeSeries.from_times_and_values(df.index, df)

    @staticmethod
    def _combine_or_none(df_a: Optional[pd.DataFrame],
                         df_b: Optional[pd.DataFrame],
                         combine_fn: Callable[[pd.DataFrame, pd.DataFrame], Any]) -> Optional[pd.DataFrame]:
        """
        Combines two Pandas DataFrames `df_a and `df_b` of the same shape using `combine_fn` if neither is `None`.

        Parameters
        ----------
        df_a
            the first DataFrame
        df_b
            the second DataFrame
        combine_fn
            An operation with input two Pandas DataFrames and output one Pandas DataFrame.

        Returns
        -------
        Optional[pandas.DataFrame]
            A new Pandas DataFrame, the result of [combine_fn], or None.
        """
        if df_a is not None and df_b is not None:
            return combine_fn(df_a, df_b)
        return None

    @staticmethod
    def _op_or_none(df: Optional[pd.DataFrame], op: Callable[[pd.DataFrame], Any]):
        return op(df) if df is not None else None

    def _combine_from_pd_ops(self, other: 'TimeSeries',
                             combine_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]) -> 'TimeSeries':
        """
        Combines this TimeSeries with another one, using the `combine_fn` on the underlying Pandas DataFrame.

        Parameters
        ----------
        other
            A second TimeSeries.
        combine_fn
            An operation with input two Pandas DataFrames and output one Pandas DataFrame.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with underlying Pandas DataFrame the series obtained with `combine_fn`.
        """
        raise_if_not(self.has_same_time_as(other), 'The two TimeSeries must have the same time index.', logger)
        raise_if_not(self._df.shape == other._df.shape, 'The two TimeSeries must have the same shape.', logger)

        df_a = self._df
        df_b = other._df

        # univariate case
        if df_a.shape[1] == 1:
            pd_serie_a = df_a.iloc[:, 0]
            pd_serie_b = df_b.iloc[:, 0]
            pd_serie = combine_fn(pd_serie_a, pd_serie_b)
            series_df = pd_serie.to_frame()
        # multivariate case
        else:
            raise_if(len(set(df_a.columns.to_list() + df_b.columns.to_list())) != len(df_a.columns),
                     "Column name in each TimeSeries must match one to one.")
            series_df = combine_fn(df_a, df_b)
        return TimeSeries(series_df, self.freq_str())

    @staticmethod
    def _fill_missing_dates(series: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
        """
        Tries to fill missing dates in series with NaN.
        If no value for the `freq` argument is provided, the method is successful only when explicit frequency
        can be determined from all consecutive triple timestamps.
        If a value for `freq` is given, this value will be used to determine the new frequency.

        Parameters
        ----------
        series
            The actual time series, as a pandas DataFrame with a proper time index.
        freq
            Optionally, the desired frequency of the TimeSeries instance.

        Returns
        -------
        pandas.Series
            A new Pandas DataFrame without missing dates.
        """

        if not freq:
            date_axis = series.index
            samples_size = 3
            observed_frequencies = [
                date_axis[x:x + samples_size].inferred_freq
                for x
                in range(len(date_axis) - samples_size + 1)]

            observed_frequencies = set(filter(None.__ne__, observed_frequencies))

            raise_if_not(
                len(observed_frequencies) == 1,
                "Could not infer explicit frequency. Observed frequencies: "
                + ('none' if len(observed_frequencies) == 0 else str(observed_frequencies))
                + ". Is Series too short (n=2)?",
                logger)

            freq = observed_frequencies.pop()

        return series.resample(freq).asfreq(fill_value=None)

    """
    Definition of some useful statistical methods.
    These methods rely on the Pandas implementation.
    """

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.mean(axis, skipna, level, numeric_only, **kwargs)

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._df.var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._df.std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.skew(axis, skipna, level, numeric_only, **kwargs)

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.kurtosis(axis, skipna, level, numeric_only, **kwargs)

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.min(axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.max(axis, skipna, level, numeric_only, **kwargs)

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs) -> float:
        return self._df.sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._df.median(axis, skipna, level, numeric_only, **kwargs)

    def autocorr(self, lag=1) -> float:
        return self._df.autocorr(lag)

    def describe(self, percentiles=None, include=None, exclude=None) -> pd.DataFrame:
        return self._df.describe(percentiles, include, exclude)

    """
    Definition of some dunder methods
    """

    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            if not self._df.equals(other.pd_dataframe()):
                return False
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._df)

    def __add__(self, other):
        if isinstance(other, (int, float, np.integer, np.float)):
            new_series = self._df + other
            return TimeSeries(new_series, self.freq_str())
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 + s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for + or add(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float, np.integer, np.float)):
            new_series = self._df - other
            return TimeSeries(new_series, self.freq_str())
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 - s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for - or sub(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer, np.float)):
            new_series = self._df * other
            return TimeSeries(new_series, self.freq_str())
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 * s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for * or mul(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float, np.integer, np.float)):
            if n < 0 and not all(self.values() != 0):
                raise_log(ZeroDivisionError('Cannot divide by a TimeSeries with a value 0.'), logger)

            new_series = self._df ** float(n)
            return TimeSeries(new_series, self.freq_str())
        else:
            raise_log(TypeError('unsupported operand type(s) for ** or pow(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(n).__name__)), logger)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer, np.float)):
            if (other == 0):
                raise_log(ZeroDivisionError('Cannot divide by 0.'), logger)

            new_series = self._df / other
            return TimeSeries(new_series, self.freq_str())

        elif isinstance(other, TimeSeries):
            if (not all(other.values() != 0)):
                raise_log(ZeroDivisionError('Cannot divide by a TimeSeries with a value 0.'), logger)

            return self._combine_from_pd_ops(other, lambda s1, s2: s1 / s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for / or truediv(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        series = abs(self._df)
        return TimeSeries(series, self.freq_str())

    def __neg__(self):
        series = -self._df
        return TimeSeries(series, self.freq_str())

    def __contains__(self, ts: pd.Timestamp) -> bool:
        return ts in self._df.index

    def __round__(self, n=None):
        series = self._df.round(n)
        return TimeSeries(series, self.freq_str())

    def __lt__(self, other):
        if isinstance(other, (int, float, np.integer, np.float, np.ndarray)):
            series = self._df < other
        elif isinstance(other, TimeSeries):
            series = self._df < other.pd_dataframe()
        else:
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series  # TODO should we return only the ndarray, the pd series, or our timeseries?

    def __gt__(self, other):
        if isinstance(other, (int, float, np.integer, np.float, np.ndarray)):
            series = self._df > other
        elif isinstance(other, TimeSeries):
            series = self._df > other.pd_dataframe()
        else:
            raise_log(TypeError('unsupported operand type(s) for > : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __le__(self, other):
        if isinstance(other, (int, float, np.integer, np.float, np.ndarray)):
            series = self._df <= other
        elif isinstance(other, TimeSeries):
            series = self._df <= other.pd_dataframe()
        else:
            raise_log(TypeError('unsupported operand type(s) for <= : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __ge__(self, other):
        if isinstance(other, (int, float, np.integer, np.float, np.ndarray)):
            series = self._df >= other
        elif isinstance(other, TimeSeries):
            series = self._df >= other.pd_dataframe()
        else:
            raise_log(TypeError('unsupported operand type(s) for >= : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __str__(self):
        return str(self._df) + '\nFreq: {}'.format(self.freq_str())

    def __repr__(self):
        return self.__str__()

    def __copy__(self, deep: bool = True):
        return self.copy(deep=deep)

    def __deepcopy__(self):
        return self.copy(deep=True)

    def __getitem__(self, key: Union[pd.DatetimeIndex, List[str], List[int], List[pd.Timestamp], str, int,
                                     pd.Timestamp, Any]) -> 'TimeSeries':
        """Allow indexing on darts TimeSeries.

        The supported index types are the following base types as a single value, a list or a slice:
        - pd.Timestamp -> return a TimeSeries corresponding to the value(s) at the given timestamp(s).
        - str -> return a TimeSeries including the column(s) specified as str.
        - int -> return a TimeSeries with the value(s) at the given row index.

        `pd.DatetimeIndex` is also supported and will return the corresponding value(s) at the provided time indices.

        .. warning::
            slices use pandas convention of including both ends of the slice.

        """
        def use_iloc(key: Any) -> TimeSeries:
            """return a new TimeSeries from a pd.DataFrame using iloc indexing."""
            return TimeSeries.from_dataframe(self._df.iloc[key], freq=self.freq_str())

        def use_loc(key: Any, col_indexing: Optional[bool] = False) -> TimeSeries:
            """return a new TimeSeries from a pd.DataFrame using loc indexing."""
            if col_indexing:
                return TimeSeries.from_dataframe(self._df.loc[:, key], freq=self.freq_str())
            else:
                return TimeSeries.from_dataframe(self._df.loc[key], freq=self.freq_str())

        if isinstance(key, pd.DatetimeIndex):
            check = np.array([elem in self.time_index() for elem in key])
            if not np.all(check):
                raise_log(IndexError("None of {} in the index".format(key[~check])), logger)
            return use_loc(key)
        elif isinstance(key, slice):
            if isinstance(key.start, str) or isinstance(key.stop, str):
                return use_loc(key, col_indexing=True)
            elif isinstance(key.start, int) or isinstance(key.stop, int):
                return use_iloc(key)
            elif isinstance(key.start, pd.Timestamp) or isinstance(key.stop, pd.Timestamp):
                return use_loc(key)
        elif isinstance(key, list):
            if all(isinstance(s, str) for s in key):
                return use_loc(key, col_indexing=True)
            elif all(isinstance(i, int) for i in key):
                return use_iloc(key)
            elif all(isinstance(t, pd.Timestamp) for t in key):
                return use_loc(key)
        else:
            if isinstance(key, str):
                return use_loc([key], col_indexing=True)
            elif isinstance(key, int):
                return use_iloc([key])
            elif isinstance(key, pd.Timestamp):
                return use_loc([key])

        raise_log(IndexError("The type of your index was not matched."), logger)
