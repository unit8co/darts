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
            xa_ = xa.resample({xa.dims[0]: freq}).asfreq()
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
                          coords={time_index.name: time_index, DIMS[1]: columns_list})

        return TimeSeries.from_xarray(xa=xa, fill_missing_dates=fill_missing_dates, freq=freq)

    @staticmethod
    def from_series() -> 'TimeSeries':
        pass

    @staticmethod
    def from_times_and_values() -> 'TimeSeries':
        pass

    @staticmethod
    def from_values() -> 'TimeSeries':
        pass

    @staticmethod
    def from_json(json_str: str) -> 'TimeSeries':
        """
        Converts the JSON String representation of a `TimeSeries` object (produced using `TimeSeries.to_json()`)
        into a `TimeSeries` object

        At the moment this only supports deterministic time series (i.e., made of 1 sample).

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
        return TimeSeries.from_dataframe(df)

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
    def width(self):
        return self.n_components

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
    def has_datetime_index(self) -> bool:
        """
        Whether this series is indexed with a DatetimeIndex (otherwise it is indexed with a RangeIndex)
        """
        return self._has_datetime_index

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

    def data_array(self, copy=True) -> xr.DataArray:
        """
        Returns the xarray DataArray representation of this time series.

        Parameters
        ----------
        copy
            Whether to return a copy of the series. Leave it to True unless you know what you are doing.

        Returns
        -------
        pandas.Series
            The xarray DataArray underlying this time series.
        """
        return self._xa.copy() if copy else self._xa

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

    def shift(self, n: int) -> 'TimeSeries':
        """
        Shifts the time axis of this TimeSeries by `n` time steps.

        If :math:`n > 0`, shifts in the future. If :math:`n < 0`, shifts in the past.

        For example, with :math:`n=2` and `freq='M'`, March 2013 becomes May 2013.
        With :math:`n=-2`, March 2013 becomes Jan 2013.

        Parameters
        ----------
        n
            The number of time steps to shift by. Can be negative.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with a shifted index.
        """
        try:
            self._time_index[-1] + n * self.freq
        except pd.errors.OutOfBoundsDatetime:
            raise_log(OverflowError("the add operation between {} and {} will "
                                    "overflow".format(n * self.freq, self.time_index[-1])), logger)
        new_time_index = self._time_index.map(lambda ts: ts + n * self.freq)
        new_xa = self._xa.assign_coords({self._xa.dims[0]: new_time_index})
        return TimeSeries(new_xa)

    def diff(self,
             n: Optional[int] = 1,
             periods: Optional[int] = 1) -> 'TimeSeries':
        """
        Returns a differenced time series. This is often used to make a time series stationary.

        Parameters
        ----------
        n
            Optionally, a positive integer indicating the number of differencing steps (default = 1).
        periods
            Optionally, periods to shift for calculating difference.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed after differencing.
        """
        if not isinstance(n, int) or n < 1:
             raise_log(ValueError("'n' must be a positive integer >= 1."))
        if not isinstance(periods, int) or periods < 1:
             raise_log(ValueError("'periods' must be an integer >= 1."))

        new_xa = self._xa.diff(dim=self._time_dim, n=periods)
        for _ in range(n-1):
            new_xa = new_xa.diff(dim=self._time_dim, n=periods)
        return TimeSeries(new_xa)

    def has_same_time_as(self, other: 'TimeSeries') -> bool:
        """
        Checks whether this TimeSeries and another one have the same time index.

        Parameters
        ----------
        other
            the other series

        Returns
        -------
        bool
            True if both TimeSeries have the same index, False otherwise.
        """
        return (other.time_index == self.time_index).all()

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
        raise_if_not(other.has_datetime_index == self.has_datetime_index,
                     'Both series must have the same type of time index (either DatetimeIndex or RangeIndex).')
        raise_if_not(other.freq == self.freq,
                     'Appended TimeSeries must have the same frequency as the current one', logger)
        if self._has_datetime_index:
            raise_if_not(other.start_time() == self.end_time() + self.freq,
                         'Appended TimeSeries must start one time step after current one.', logger)

        other_xa = other.data_array()

        # TODO: could we just remove this constraint and rename the dimension if needed
        raise_if_not(other_xa.dims[0] != self._time_dim,
                     'Both time series must have the same name for the time dimensions.')

        new_xa = xr.concat(objs=[self._xa, other_xa], dim=str(self._time_dim))
        if not self._has_datetime_index:
            new_xa = new_xa.reset_index(dims_or_levels=new_xa.dims[0])

        return TimeSeries(new_xa)

    def append_values(self,
                      values: np.ndarray,
                      index: pd.DatetimeIndex = None) -> 'TimeSeries':
        """
        Appends values to current TimeSeries, to the given indices.

        If no index is provided, assumes that it follows the original data.

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
        # TODO: needed?
        raise NotImplementedError()

    def update(self,
               index: pd.DatetimeIndex,
               values: np.ndarray = None) -> 'TimeSeries':
        """
        Updates the TimeSeries with the new values provided.
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
        # TODO: I don't think this is needed... probably better to just create a new TimeSeries
        raise NotImplementedError()

    def stack(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Stacks another univariate or multivariate TimeSeries with the same time index on top of
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
        raise_if_not(self.has_same_time_as(other), 'The indices of the two TimeSeries instances '
                     'must be equal', logger)

        new_xa = xr.concat([self._xa, other.data_array(copy=False)], dim=DIMS[1])
        return TimeSeries(new_xa)

    def univariate_component(self, index: Union[str, int]) -> 'TimeSeries':
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
        if isinstance(index, int):
            new_xa = self._xa.isel(component=index).expand_dims(DIMS[1], axis=1)
        else:
            self._xa.sel(component=index).expand_dims(DIMS[1], axis=1)
        return TimeSeries(new_xa)

    def add_datetime_attribute(self, attribute: str, one_hot: bool = False) -> 'TimeSeries':
        """
        Returns a new TimeSeries instance with one (or more) additional component(s) that contain an attribute
        of the time index of the current series specified with `attribute`, such as 'weekday', 'day' or 'month'.

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
        return self.stack(tg.datetime_attribute_timeseries(self.time_index, attribute, one_hot))

    def add_holidays(self,
                     country_code: str,
                     prov: str = None,
                     state: str = None) -> 'TimeSeries':
        """
        Adds a binary univariate component to the current series that equals 1 at every index that
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
            A new TimeSeries instance, enhanced with binary holiday component.
        """
        from .utils import timeseries_generation as tg
        return self.stack(tg.holidays_timeseries(self.time_index, country_code, prov, state))

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

        resample = self._xa.resample(freq)

        # TODO: check
        if method == 'pad':
            new_xa = resample.pad()
        elif method == 'bfill':
            new_xa = resample.backfill()
        else:
            raise_log(ValueError('Unknown method: {}'.format(method)))
        return TimeSeries(new_xa)

    def is_within_range(self, ts: Union[pd.Timestamp, int]) -> bool:
        """
        Check whether a given timestamp or integer is withing the time interval of this time series.
        If a timestamp is provided, it does not need to be *in* the time series.

        Parameters
        ----------
        ts
            The `pandas.Timestamp` or integer to check

        Returns
        -------
        bool
            Whether `ts` is contained within the interval of this time series.
        """
        return self.time_index[0] <= ts <= self.time_index[-1]

    def map(self,
            fn: Union[Callable[[np.number], np.number],
                      Callable[[Union[pd.Timestamp, int], np.number], np.number]]) -> 'TimeSeries':  # noqa: E501
        """
        Applies the function `fn` elementwise to all values in this TimeSeries.
        Returns a new TimeSeries instance. If `fn` takes 1 argument it is simply applied elementwise.
        If it takes 2 arguments, it is applied elementwise on the (timestamp, value) tuples.

        At the moment this function works only on deterministic time series (i.e., made of 1 sample).

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`
            The type of `timestamp` is either `pd.Timestamp` (if the series is indexed with a DatetimeIndex),
            or an integer otherwise (if the series is indexed with a RangeIndex).

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
            df = self.pd_dataframe().applymap(fn)
        elif num_args == 2:  # map function uses timestamp f(timestamp, x)
            def apply_fn_wrapper(row):
                timestamp = row.name
                return row.map(lambda x: fn(timestamp, x))
            df = self.pd_dataframe().apply(apply_fn_wrapper, axis=1)
        else:
            raise_log(ValueError("fn must have either one or two arguments"), logger)

        return TimeSeries.from_dataframe(df)

    def to_json(self) -> str:
        """
        Converts the `TimeSeries` object to a JSON String

        At the moment this function works only on deterministic time series (i.e., made of 1 sample).

        Returns
        -------
        str
            A JSON String representing the time series
        """
        return self.pd_dataframe().to_json(orient='split', date_format='iso')

    def plot(self,
             new_plot: bool = False,
             central_quantile: Union[float, str] = 0.5,
             confidence_low_quantile: Optional[float] = 0.05,
             confidence_high_quantile: Optional[float] = 0.95,
             *args,
             **kwargs):
        """
        A wrapper method around `xarray.DataArray.plot()`.

        Parameters
        ----------
        new_plot
            whether to spawn a new Figure
        central_quantile
            The quantile (between 0 and 1) to plot as a "central" value, if the series is stochastic (i.e., if
            it has multiple samples). This will be applied on each component separately (i.e., to display quantiles
            of the components' marginal distributions). For instance, setting `central_quantile=0.5` will plot the
            median of each component. `central_quantile` can also be set to 'mean'.
        confidence_low_quantile
            The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `confidence_low_quantile` is None (default 0.05).
        confidence_high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `confidence_high_quantile` is None (default 0.95).
        args
            some positional arguments for the `plot()` method
        kwargs
            some keyword arguments for the `plot()` method
        """
        colors = ['black', 'blue', 'magenta', 'mediumturquoise', 'green', 'darkorange', 'red']
        alpha_confidence_intvls = 0.25

        if central_quantile != 'mean':
            raise_if_not(isinstance(central_quantile, float) and 0. <= central_quantile <= 1.,
                         'central_quantile must be either "mean", or a float between 0 and 1.')

        if confidence_high_quantile is not None and confidence_low_quantile is not None:
            raise_if_not(0. <= confidence_low_quantile <= 1. and 0. <= confidence_high_quantile <= 1.,
                         'confidence interval low and high quantiles must be between 0 and 1.')

        fig = (plt.figure() if new_plot else (kwargs['figure'] if 'figure' in kwargs else plt.gcf()))
        kwargs['figure'] = fig
        label = kwargs['label'] if 'label' in kwargs else None

        if 'lw' not in kwargs:
            kwargs['lw'] = 2

        if self.n_components > 7:
            logger.warn('Number of components is larger than 7 ({}). Plotting only the first 15 components.'.format(
                self.n_components
            ))

        for i, c in enumerate(self._xa.component[:7]):
            comp_name = str(c.values)

            if i > 0:
                kwargs['figure'] = plt.gcf()
            if 'label' in kwargs:
                kwargs['label'] = label + '_' + str(comp_name)
            else:
                kwargs['label'] = str(comp_name)

            comp = self._xa.sel(component=c)

            if comp.sample.size > 1:
                if confidence_low_quantile is not None and confidence_high_quantile is not None:
                    low_series = comp.quantile(q=confidence_low_quantile, dim=DIMS[2])
                    high_series = comp.quantile(q=confidence_high_quantile, dim=DIMS[2])
                    color = kwargs['color'] if 'color' in kwargs else colors[i % len(colors)]
                    plt.fill_between(self.time_index, low_series, high_series, color=color,
                                     alpha=(alpha_confidence_intvls if 'alpha' not in kwargs else kwargs['alpha']))

                if central_quantile == 'mean':
                    central_series = comp.mean(dim=DIMS[2])
                else:
                    central_series = comp.quantile(q=central_quantile, dim=DIMS[2])

            else:
                central_series = comp.mean(dim=DIMS[2])

            if 'color' not in kwargs:
                kwargs['color'] = colors[i % len(colors)]

            # temporarily set alpha to 1 to plot the central value (this way alpha impacts only the confidence intvls)
            alpha = kwargs['alpha'] if 'alpha' in kwargs else None
            kwargs['alpha'] = 1
            central_series.plot(*args, **kwargs)
            kwargs['alpha'] = alpha if alpha is not None else alpha_confidence_intvls

        plt.legend()
        plt.title(self._xa.name);

