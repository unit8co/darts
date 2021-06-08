"""
Timeseries
----------

`TimeSeries` is the main class in `darts`. It represents a univariate or multivariate time series.
It can represent a stochastic time series by storing several samples (trajectories).
The sub-class `SampleTimeSeries` contains one sample (and is thus not stochastic).
"""

import pandas as pd
import numpy as np
import xarray as xr

from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any, List, Union
from inspect import signature

from .logging import raise_log, raise_if_not, raise_if, get_logger

logger = get_logger(__name__)

# dimension names in the DataArray
# the "time" one can be different, if it has a name in the underlying Series/DataFrame.
DIMS = ('time', 'component', 'sample')


class TimeSeries:
    def __init__(self, xa: xr.DataArray):
        """
        Wrapper around a (well formed) DataArray. Use the static factory methods to build instances unless
        you know what you are doing.
        """
        raise_if_not(isinstance(xa, xr.DataArray), 'Data must be provided as an xarray DataArray instance.')
        raise_if_not(len(xa.shape) == 3, 'TimeSeries require DataArray of dimensionality 3 ({}).'.format(DIMS))
        raise_if_not(xa.size > 0, 'The time series array must not be empty.')
        raise_if_not(np.issubdtype(xa.values.dtype, np.number), 'The time series must contain numerical values only.')

        if xa.dims[-2:] != DIMS[-2:]:
            # The first dimension represents the time and may be named differently.
            raise_log(ValueError('The last two dimensions of the DataArray must be named {}'.format(DIMS[-2:])))

        self._time_dim = xa.dims[0]  # how the time dimension is named
        self._time_index = xa.get_index(self._time_dim)

        if not isinstance(self._time_index, pd.DatetimeIndex) and not isinstance(self._time_index, pd.RangeIndex):
            raise_log(ValueError('The time dimension of the DataArray must be indexed either with a DatetimeIndex,'
                                 'or with a RangeIndex.'))

        self._xa: xr.DataArray = xa.sortby(self._time_dim)  # returns a copy

        self._has_datetime_index = isinstance(self._time_index, pd.DatetimeIndex)

        if self._has_datetime_index:
            self._freq: pd.DateOffset = self._time_index.freq
            self._freq_str: str = self._time_index.inferred_freq
        else:
            self._freq = 1
            self._freq_str = None

    """ 
    Factory Methods
    ===============
    """

    @staticmethod
    def from_xarray(xa: xr.DataArray,
                    fill_missing_dates: Optional[bool] = True,
                    freq: Optional[str] = None) -> 'TimeSeries':

        # optionally fill missing dates; do it only when there is a DatetimeIndex (and not a RangeIndex)
        time_index = xa.get_index(xa.dims[0])
        if fill_missing_dates and isinstance(time_index, pd.DatetimeIndex):
            if not freq:
                samples_size = 3
                observed_frequencies = [
                    time_index[x:x + samples_size].inferred_freq
                    for x
                    in range(len(time_index) - samples_size + 1)]

                observed_frequencies = set(filter(None.__ne__, observed_frequencies))

                raise_if_not(
                    len(observed_frequencies) == 1,
                    "Could not infer explicit frequency. Observed frequencies: "
                    + ('none' if len(observed_frequencies) == 0 else str(observed_frequencies))
                    + ". Is Series too short (n=2)?",
                    logger)

                freq = observed_frequencies.pop()

            # TODO: test this
            xa_ = xa.resample(freq).asfreq(fill_value=None)
        else:
            xa_ = xa

        return TimeSeries(xa_)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       time_col: Optional[str] = None,
                       value_cols: Optional[Union[List[str], str]] = None,
                       fill_missing_dates: Optional[bool] = True,
                       freq: Optional[str] = None,) -> 'TimeSeries':
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
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.
        freq
            Optionally, a string representing the frequency of the Pandas DataFrame. This is useful in order to fill
            in missing values if some dates are missing.

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.
        """

        def _get_column_list(columns: pd._typing.Axes) -> pd.Index:
            # return a list of string containing column names
            clist = columns.to_list()
            for i, column in enumerate(clist):
                if not isinstance(column, str):
                    clist[i] = str(column)
            return clist

        # get values
        if value_cols is None:
            series_df = df.loc[:, df.columns != time_col]
        else:
            if isinstance(value_cols, str):
                value_cols = [value_cols]
            series_df = df[value_cols]

        # get time index
        if time_col:
            time_index = pd.DatetimeIndex(df[time_col])
        else:
            raise_if_not(isinstance(df.index, pd.RangeIndex) or isinstance(df.index, pd.DatetimeIndex),
                         'If time_col is not specified, the DataFrame must be indexed either with'
                         'a DatetimeIndex, or with a RangeIndex.')
            time_index = df.index

        if not time_index.name:
            time_index.name = DIMS[0]

        # get columns' names
        columns_list = _get_column_list(series_df.columns)

        xa = xr.DataArray(series_df.values[:, :, np.newaxis],
                          dims=(time_index.name,) + DIMS[-2:],
                          coords={'time': time_index, 'component': columns_list})

        return TimeSeries.from_xarray(xa=xa, fill_missing_dates=fill_missing_dates, freq=freq)

    @staticmethod
    def from_series() -> 'TimeSeries':
        # create
        pass

    @staticmethod
    def from_times_and_values() -> 'TimeSeries':
        pass

    @staticmethod
    def from_values() -> 'TimeSeries':
        pass

    """ 
    Properties
    ==========
    """

    @property
    def n_samples(self):
        return len(self._xa.sample)

    @property
    def n_components(self):
        return len(self._xa.component)

    @property
    def n_timesteps(self):
        return len(self._xa.time)

    @property
    def is_deterministic(self):
        return self.n_samples == 1

    @property
    def is_stochastic(self):
        return not self.is_deterministic

    @property
    def is_univariate(self):
        return self.n_components == 1

    @property
    def freq(self):
        return self._freq

    @property
    def freq_str(self):
        return self._freq_str

    @property
    def components(self):
        """
        The names of the components (equivalent to DataFrame columns) as a Pandas Index
        """
        return self._xa.get_index('component').copy()

    @property
    def columns(self):
        """
        Same as `components` property
        """
        return self.components

    @property
    def time_index(self) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """
        Returns
        -------
        Union[pd.DatetimeIndex, pd.RangeIndex]
            The time index of this time series.
        """
        return self._time_index.copy()

    @property
    def duration(self) -> Union[pd.Timedelta, int]:
        """
        Returns
        -------
        Union[pandas.Timedelta, int]
            The duration of this time series; as a Timedelta if the series is indexed by a Datetimeindex,
            and int otherwise.
        """
        return self._time_index[-1] - self._time_index[0]

    """ 
    Some asserts
    =============
    """
    # TODO: put at the bottom

    def _assert_univariate(self):
        if not self.is_univariate:
            raise_log(AssertionError('Only univariate TimeSeries instances support this method'), logger)

    def _assert_deterministic(self):
        if not self.is_deterministic:
            raise_log(AssertionError('Only deterministic TimeSeries (with 1 sample) instances support this method'),
                      logger)

    def _assert_stochastic(self):
        if not self.is_stochastic:
            raise_log(AssertionError('Only non-deterministic TimeSeries (with more than 1 samples) '
                                     'instances support this method'),
                      logger)

    def _raise_if_not_within(self, ts: Union[pd.Timestamp, int]):
        if isinstance(ts, pd.Timestamp):
            raise_if_not(self._has_datetime_index, 'Function called with a timestamp, but series not time-indexed.')
        elif isinstance(ts, int):
            raise_if(self._has_datetime_index, 'Function called with an integer, but series is time-indexed.')
        if (ts < self.start_time()) or (ts > self.end_time()):
            raise_log(ValueError('Timestamp must be between {} and {}'.format(self.start_time(),
                                                                              self.end_time())), logger)

    """
    Export functions
    ================
    """

    def pd_series(self, copy=True) -> pd.Series:
        """
        Returns a Pandas Series representation of this time series.
        Works only for univariate series that are deterministic (i.e., made of 1 sample).

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
        self._assert_deterministic()
        if copy:
            return pd.Series(self._xa[:, 0, 0].copy(), index=self._time_index.copy())
        else:
            return pd.Series(self._xa[:, 0, 0], index=self._time_index)

    def pd_dataframe(self, copy=True) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame representation of this time series.
        Each of the series components will appear as a column in the DataFrame.
        Works only for deterministic series (i.e., made of 1 sample).

        Parameters
        ----------
        copy
            Whether to return a copy of the dataframe. Leave it to True unless you know what you are doing.

        Returns
        -------
        pandas.DataFrame
            The Pandas DataFrame representation of this time series
        """
        self._assert_deterministic()
        if copy:
            return pd.DataFrame(self._xa[:, :, 0].values.copy(),
                                index=self._time_index.copy(),
                                columns=self._xa.get_index('component').copy())
        else:
            return pd.DataFrame(self._xa[:, :, 0].values,
                                index=self._time_index,
                                columns=self._xa.get_index('component'))

    def quantile_df(self, quantile=0.5) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing the single desired quantile of each component (over the samples).
        Each of the series components will appear as a column in the DataFrame. The column will be named
        "<component>_X", where "<component>" is the column name corresponding to this component, and "X"
        is the quantile value.
        The quantile columns represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantile
            The desired quantile value. The value must be represented as a fraction
            (between 0 and 1 inclusive). For instance, `0.5` will return a DataFrame
            containing the median of the (marginal) distribution of each component.

        Returns
        -------
        pandas.DataFrame
            The Pandas DataFrame containing the desired quantile for each component.
        """
        self._assert_stochastic()
        raise_if_not(0 <= quantile <= 1,
                     'The quantile values must be expressed as fraction (between 0 and 1 inclusive).')

        # column names
        cnames = list(map(lambda s: s + '_{}'.format(quantile), self.columns))

        return pd.DataFrame(self._xa.quantile(q=quantile, dim=DIMS[2]),
                            index=self._time_index,
                            columns=cnames)

    def quantile_timeseries(self, quantile=0.5) -> 'TimeSeries':
        """
        Returns a deterministic `TimeSeries` containing the single desired quantile of each component
        (over the samples) of this stochastic `TimeSeries`.
        The components in the new series are named "<component>_X", where "<component>"
        is the column name corresponding to this component, and "X" is the quantile value.
        The quantile columns represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantile
            The desired quantile value. The value must be represented as a fraction
            (between 0 and 1 inclusive). For instance, `0.5` will return a TimeSeries
            containing the median of the (marginal) distribution of each component.

        Returns
        -------
        TimeSeries
            The TimeSeries containing the desired quantile for each component.
        """
        return TimeSeries.from_dataframe(self.quantile_df(quantile))

    def quantiles_df(self, quantiles: Tuple[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame containing the desired quantiles of each component (over the samples).
        Each of the series components will appear as a column in the DataFrame. The column will be named
        "<component>_X", where "<component>" is the column name corresponding to this component, and "X"
        is the quantile value.
        The quantiles represent the marginal distributions of the components of this series.

        This works only on stochastic series (i.e., with more than 1 sample)

        Parameters
        ----------
        quantiles
            Tuple containing the desired quantiles. The values must be represented as fractions
            (between 0 and 1 inclusive). For instance, `(0.1, 0.5, 0.9)` will return a DataFrame
            containing the 10th-percentile, median and 90th-percentile of the (marginal) distribution of each component.

        Returns
        -------
        pandas.DataFrame
            The Pandas DataFrame containing the quantiles for each component.
        """
        # TODO: there might be a slightly more efficient way to do it for several quantiles at once with xarray...
        return pd.concat([self.quantile_df(quantile) for quantile in quantiles], axis=1)

    def start_time(self) -> Union[pd.Timestamp, int]:
        """
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the first time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[0]

    def end_time(self) -> Union[pd.Timestamp, int]:
        """
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the last time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[-1]

    def first_value(self) -> float:
        """
        Returns
        -------
        float
            The first value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._xa[0, 0, 0])

    def last_value(self) -> float:
        """
        Returns
        -------
        float
            The last value of this univariate deterministic time series
        """
        self._assert_univariate()
        self._assert_deterministic()
        return float(self._xa[-1, 0, 0])

    def first_values(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The first values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._xa.values[0, :, 0].copy()

    def last_values(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The last values of every component of this deterministic time series
        """
        self._assert_deterministic()
        return self._xa.values[-1, :, 0].copy()

    def values(self, copy=True, sample=0) -> np.ndarray:
        """
        Returns a 2-D Numpy array of dimension (time, component), containing this series' values for one sample.
        If this series is deterministic, it contains only one sample and only `sample=0` can be used.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        raise_if(self.is_deterministic and sample != 0, 'This series contains one sample only (deterministic),'
                                                        'so only sample=0 is accepted.')
        if copy:
            return np.copy(self._xa.values[:, :, sample])
        else:
            return self._xa.values[:, :, sample]

    def all_values(self, copy=True) -> np.ndarray:
        """
        Returns a 3-D Numpy array of dimension (time, component, sample),
        containing this series' values for all samples.

        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        if copy:
            return np.copy(self._xa.values)
        else:
            return self._xa.values

    def univariate_values(self, copy=True, sample=0) -> np.ndarray:
        """
        Returns a 1-D Numpy array of dimension (time,), containing this univariate series' values for one sample.
        If this series is deterministic, it contains only one sample and only `sample=0` can be used.

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
            return np.copy(self._xa[:, 0, sample].values)
        else:
            return self._xa[:, 0, sample].values

    """
    Other methods
    =============
    """

    def gaps(self) -> pd.DataFrame:
        """
        A function to compute and return gaps in the TimeSeries. Works only on deterministic time series (1 sample).

        Returns
        -------
        pd.DataFrame
            A pandas.DataFrame containing a row for every gap (rows with all-NaN values in underlying DataFrame)
            in this time series. The DataFrame contains three columns that include the start and end time stamps
            of the gap and the integer length of the gap (in `self.freq()` units if the series is indexed
            by a DatetimeIndex).
        """

        df = self.pd_dataframe()

        is_nan_series = df.isna().all(axis=1).astype(int)
        diff = pd.Series(np.diff(is_nan_series.values), index=is_nan_series.index[:-1])
        gap_starts = diff[diff == 1].index + self._freq
        gap_ends = diff[diff == -1].index

        if is_nan_series.iloc[0] == 1:
            gap_starts = gap_starts.insert(0, self.start_time())
        if is_nan_series.iloc[-1] == 1:
            gap_ends = gap_ends.insert(len(gap_ends), self.end_time())

        gap_df = pd.DataFrame()
        gap_df['gap_start'] = gap_starts
        gap_df['gap_end'] = gap_ends

        def intvl(start, end):
            if self._has_datetime_index:
                return pd.date_range(start=start, end=end, freq=self._freq).size
            else:
                return start - end

        gap_df['gap_size'] = gap_df.apply(
            lambda row: intvl(start=row.gap_start, end=row.gap_end).size, axis=1
        )

        return gap_df

    def copy(self) -> 'TimeSeries':
        """
        Make a copy of this time series object

        Returns
        -------
        TimeSeries
            A copy of this time series.
        """
        return TimeSeries(self._xa)  # the xarray will be copied in the TimeSeries constructor

    def get_index_at_point(self, point: Union[pd.Timestamp, float, int]) -> int:
        """
        Converts a point into an integer index

        Parameters
        ----------
        point
            This parameter supports 3 different data types: `pd.Timestamp`, `float` and `int`.

            `pd.Timestamp` work only on series that are indexed with a `pd.DatetimeIndex`. In such cases, the returned
            point will be the index of this timestamp, provided that it is present in the series time index,
            otherwise will raise a ValueError.

            In case of a `float`, the parameter will be treated as the proportion of the time series
            that should lie before the point.

            In the case of `int`, the parameter will returned as such, provided that it is in the series. Otherwise
            it will raise a ValueError.
        """
        if isinstance(point, float):
            raise_if_not(0. <= point <= 1., 'point (float) should be between 0.0 and 1.0.', logger)
            point_index = int((self.__len__() - 1) * point)
        elif isinstance(point, int):
            raise_if(point not in range(self.__len__()), "point (int) should be a valid index in series", logger)
            point_index = point
        elif isinstance(point, pd.Timestamp):
            raise_if_not(self._has_datetime_index,
                         'A Timestamp has been provided, but this series is not time-indexed.')
            raise_if(point not in self,
                     'point (pandas.Timestamp) must be an entry in the time series\' time index',
                     logger)
            point_index = self._time_index.get_loc(point)
        else:
            raise_log(TypeError("`point` needs to be either `float`, `int` or `pd.Timestamp`"), logger)
        return point_index

    def _split_at(self,
                  split_point: Union[pd.Timestamp, float, int],
                  after: bool = True) -> Tuple['TimeSeries', 'TimeSeries']:

        if isinstance(split_point, pd.Timestamp) or isinstance(split_point, int):
            self._raise_if_not_within(split_point)
        point_index = self.get_index_at_point(split_point)
        return self[:point_index+(1 if after else 0)], self[point_index+(1 if after else 0):]  # TODO Check

    def split_after(self, split_point: Union[pd.Timestamp, float, int]) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, after a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position after
            which the split is performed. A pd.Timestamp can be provided for TimeSeries that are indexed by a
            pd.DatetimeIndex. In such cases, the timestamp will be contained in the first TimeSeries, but not
            in the second one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples up to the `split_point`,
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=True)

    def split_before(self, split_point: Union[pd.Timestamp, float, int]) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, before a provided `split_point`.

        Parameters
        ----------
        split_point
            A timestamp, float or integer. If float, represents the proportion of the series to include in the
            first TimeSeries (must be between 0.0 and 1.0). If integer, represents the index position before
            which the split is performed. A pd.Timestamp can be provided for TimeSeries that are indexed by a
            pd.DatetimeIndex. In such cases, the timestamp will be contained in the second TimeSeries, but not
            in the first one. The timestamp itself does not have to appear in the original TimeSeries index.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series contains the first samples up to the `split_point`,
            and the second contains the remaining ones.
        """
        return self._split_at(split_point, after=False)

    def slice(self):
        pass
        # TODO: needed, or can be addressed using [] only?

    def slice_n_points_after(self, start_ts: Union[pd.Timestamp, int], n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting a `start_ts` and having at most `n` points.

        The provided timestamps will be included in the series.

        Parameters
        ----------
        start_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n`, starting at `start_ts`
        """
        raise_if_not(n > 0, 'n should be a positive integer.', logger)
        self._raise_if_not_within(start_ts)
        point_index = self.get_index_at_point(start_ts)
        return self[point_index:point_index+n]

    def slice_n_points_before(self, start_ts: Union[pd.Timestamp, int], n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending at `start_ts` and having at most `n` points.

        The provided timestamps will be included in the series.

        Parameters
        ----------
        start_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n`, ending at `start_ts`
        """
        raise_if_not(n > 0, 'n should be a positive integer.', logger)
        self._raise_if_not_within(start_ts)
        point_index = self.get_index_at_point(start_ts)
        return self[point_index-n+1:point_index+1]

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
        time_index = self.time_index.intersection(other.time_index)
        return self.__getitem__(time_index)

    def strip(self) -> 'TimeSeries':
        """
        Returns a TimeSeries slice of this deterministic time series, where NaN-only entries at the beginning
        and the end of the series are removed. No entries after (and including) the first non-NaN entry and
        before (and including) the last non-NaN entry are removed.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Returns
        -------
        TimeSeries
            a new series based on the original where NaN-only entries at start and end have been removed
        """

        df = self.pd_dataframe(copy=False)
        new_start_idx = df.first_valid_index()
        new_end_idx = df.last_valid_index()
        new_series = df.loc[new_start_idx:new_end_idx]
        return TimeSeries.from_dataframe(new_series)

    def longest_contiguous_slice(self, max_gap_size: int = 0) -> 'TimeSeries':
        """
        Returns the largest TimeSeries slice of this deterministic time series that contains no gaps
        (contigouse all-NaN rows) larger than `max_gap_size`.

        This method is only applicable to deterministic series (i.e., having 1 sample).

        Returns
        -------
        TimeSeries
            a new series constituting the largest slice of the original with no or bounded gaps
        """
        if not (self._xa == np.nan).any():
            return self.copy()
        stripped_series = self.strip()
        gaps = stripped_series.gaps()
        relevant_gaps = gaps[gaps['gap_size'] > max_gap_size]

        curr_slice_start = stripped_series.start_time()
        max_size = pd.Timedelta(days=0) if self._has_datetime_index else 0
        max_slice_start = None
        max_slice_end = None
        for index, row in relevant_gaps.iterrows():
            size = row['gap_start'] - curr_slice_start - self._freq
            if size > max_size:
                max_size = size
                max_slice_start = curr_slice_start
                max_slice_end = row['gap_start'] - self._freq
            curr_slice_start = row['gap_end'] + self._freq

        if stripped_series.end_time() - curr_slice_start > max_size:
            max_slice_start = curr_slice_start
            max_slice_end = self.end_time()

        return stripped_series[max_slice_start:max_slice_end]

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

        raise_if_not((self._xa[0, :, :] != 0).all(), 'Cannot rescale with first value 0.', logger)
        coef = value_at_first_step / self._xa[0, :, :]
        new_series = coef * self._xa
        return TimeSeries(new_series)

