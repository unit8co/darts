"""
Timeseries
----------

`TimeSeries` is the main class in `darts`. It represents a univariate or multivariate time series.
It can represent a stochastic time series by storing several samples (trajectories).
"""

import pickle
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, Any, List, Union, TextIO, Sequence
from inspect import signature
from collections import defaultdict
from pandas.tseries.frequencies import to_offset

from .logging import raise_log, raise_if_not, raise_if, get_logger

logger = get_logger(__name__)

# dimension names in the DataArray
# the "time" one can be different, if it has a name in the underlying Series/DataFrame.
DIMS = ('time', 'component', 'sample')

VALID_INDEX_TYPES = (pd.DatetimeIndex, pd.RangeIndex, pd.Int64Index)


class TimeSeries:
    def __init__(self, xa: xr.DataArray):
        """
        Wrapper around a (well formed) DataArray. Use the static factory methods to build instances unless
        you know what you are doing.
        """
        raise_if_not(isinstance(xa, xr.DataArray), 'Data must be provided as an xarray DataArray instance. '
                                                   'If you need to create a TimeSeries from another type '
                                                   '(e.g. a DataFrame), look at TimeSeries factory methods '
                                                   '(e.g. TimeSeries.from_dataframe(), '
                                                   'TimeSeries.from_xarray(), TimeSeries.from_values()'
                                                   'TimeSeries.from_times_and_values(), etc...).', logger)
        raise_if_not(xa.size > 0, 'The time series array must not be empty.', logger)
        raise_if_not(len(xa.shape) == 3, 'TimeSeries require DataArray of dimensionality 3 ({}).'.format(DIMS), logger)

        # Ideally values should be np.float, otherwise certain functionalities like diff()
        # relying on np.nan (which is a float) won't work very properly.
        raise_if_not(np.issubdtype(xa.values.dtype, np.number), 'The time series must contain numeric values only.',
                     logger)

        val_dtype = xa.values.dtype
        if not (np.issubdtype(val_dtype, np.float64) or np.issubdtype(val_dtype, np.float32)):
            logger.warn('TimeSeries is using a numeric type different from np.float32 or np.float64. '
                        'Not all functionalities may work properly. It is recommended casting your data to floating '
                        'point numbers before using TimeSeries.')

        if xa.dims[-2:] != DIMS[-2:]:
            # The first dimension represents the time and may be named differently.
            raise_log(ValueError('The last two dimensions of the DataArray must be named {}'.format(DIMS[-2:])), logger)

        # check that columns/component names are unique
        components = xa.get_index(DIMS[1])
        raise_if_not(len(set(components)) == len(components),
                     'The components (columns) names must be unique. Provided: {}'.format(components),
                     logger)

        self._time_dim = str(xa.dims[0])  # how the time dimension is named; we convert hashable to string

        # The following sorting returns a copy, which we are relying on.
        # As of xarray 0.18.2, this sorting discards the freq of the index for some reason
        # https://github.com/pydata/xarray/issues/5466
        # We sort only if the time axis is not already sorted (monotically increasing).
        self._xa = xa.copy() if xa.get_index(self._time_dim).is_monotonic_increasing else xa.sortby(self._time_dim)

        self._time_index = self._xa.get_index(self._time_dim)

        if not isinstance(self._time_index, VALID_INDEX_TYPES):
            raise_log(ValueError('The time dimension of the DataArray must be indexed either with a DatetimeIndex,'
                                 'or with an Int64Index (this can include a RangeIndex).'), logger)

        self._has_datetime_index = isinstance(self._time_index, pd.DatetimeIndex)

        if self._has_datetime_index:
            freq_tmp = xa.get_index(self._time_dim).freq  # store original freq (see bug of sortby() above).
            self._freq: pd.DateOffset = (freq_tmp if freq_tmp is not None else
                                         to_offset(self._xa.get_index(self._time_dim).inferred_freq))
            raise_if(self._freq is None,
                     'The time index of the provided DataArray is missing the freq attribute, and the frequency could '
                     'not be directly inferred. '
                     'This probably comes from inconsistent date frequencies with missing dates. '
                     'If you know the actual frequency, try setting `fill_missing_dates=True, freq=actual_frequency`. '
                     'If not, try setting `fill_missing_dates=True, freq=None` to see if a frequency can be inferred.',
                     logger)

            self._freq_str: str = self._freq.freqstr

            # reset freq inside the xarray index (see bug of sortby() above).
            self._xa.get_index(self._time_dim).freq = self._freq

            # We have to check manually if the index is complete. Another way could be to rely
            # on `inferred_freq` being present, but this fails for series of length < 3.

            is_index_complete = len(pd.date_range(self._time_index.min(),
                                                  self._time_index.max(),
                                                  freq=self._freq).difference(self._time_index)) == 0

            raise_if_not(is_index_complete, 'Not all timestamps seem to be present in the time index. Does '
                                            'the series contain holes? If you are using a factory method, '
                                            'try specifying `fill_missing_dates=True` '
                                            'or specify the `freq` parameter.', logger)
        else:
            self._freq = 1
            self._freq_str = None

    """ 
    Factory Methods
    ===============
    """
    @classmethod
    def from_xarray(cls,
                    xa: xr.DataArray,
                    fill_missing_dates: Optional[bool] = False,
                    freq: Optional[str] = None,
                    fillna_value: Optional[float] = None) -> 'TimeSeries':
        """
        Returns a TimeSeries instance built from an xarray DataArray.
        The dimensions of the DataArray have to be (time, component, sample), in this order. The time
        dimension can have an arbitrary name, but component and sample must be named "component" and "sample",
        respectively.

        The first dimension (time), and second dimension (component) must be indexed (i.e., have coordinates).
        The time must be indexed either with a pandas DatetimeIndex or a pandas Int64Index. If a DatetimeIndex is
        used, it is better if it has no holes; alternatively setting `fill_missing_dates` can in some cases solve
        these issues (filling holes with NaN, or with the provided `fillna_value` numeric value, if any).

        If two components have the same name or are not strings, this method will disambiguate the components
        names by appending a suffix of the form "<name>_N" to the N-th column with name "name".

        Parameters
        ----------
        xa
            The xarray DataArray
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values. This requires
            either a provided `freq` or the possibility to infer the frequency from the provided timestamps. See
            :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex. This is useful in order to
            fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.
        """
        xa_index = xa.get_index(xa.dims[0])
        has_datetime_index = isinstance(xa_index, pd.DatetimeIndex)
        has_frequency = has_datetime_index and xa_index.freq is not None
        # optionally fill missing dates; do it only when there is a DatetimeIndex (and not a Int64Index)
        if fill_missing_dates and has_datetime_index:
            xa_ = cls._fill_missing_dates(xa, freq=freq)
        # The provided index does not have a freq; using the provided freq
        elif has_datetime_index and freq is not None and not has_frequency:
            xa_ = cls._restore_xarray_from_frequency(xa, freq=freq)
        else:
            xa_ = xa
        if fillna_value is not None:
            xa_ = xa_.fillna(fillna_value)

        # clean components (columns) names if needed (if names are not unique, or not strings)
        components = xa_.get_index(DIMS[1])
        if len(set(components)) != len(components) or any([not isinstance(s, str) for s in components]):

            def _clean_component_list(columns) -> List[str]:
                # return a list of string containing column names
                # make each column name unique in case some columns have the same names
                clist = columns.to_list()

                # convert everything to string if needed
                for i, column in enumerate(clist):
                    if not isinstance(column, str):
                        clist[i] = str(column)

                has_duplicate = len(set(clist)) != len(clist)
                while has_duplicate:
                    # we may have to loop several times (e.g. we could have columns ["0", "0_1", "0"] and not
                    # noticing when renaming the last "0" into "0_1" that "0_1" already exists...)
                    name_to_occurence = defaultdict(int)
                    for i, column in enumerate(clist):
                        name_to_occurence[clist[i]] += 1

                        if name_to_occurence[clist[i]] > 1:
                            clist[i] = clist[i] + '_{}'.format(name_to_occurence[clist[i]]-1)

                    has_duplicate = len(set(clist)) != len(clist)

                return clist

            time_index_name = xa_.dims[0]
            columns_list = _clean_component_list(components)

            # TODO: is there a way to just update the component index without re-creating a new DataArray?
            xa_ = xr.DataArray(xa_.values,
                               dims=xa_.dims,
                               coords={time_index_name: xa_.get_index(time_index_name), DIMS[1]: columns_list})

        # We cast the array to float
        if np.issubdtype(xa_.values.dtype, np.float32) or np.issubdtype(xa_.values.dtype, np.float64):
            return cls(xa_)
        else:
            return cls(xa_.astype(np.float64))

    @classmethod
    def from_csv(cls,
                 filepath_or_buffer: pd._typing.FilePathOrBuffer,
                 time_col: Optional[str] = None,
                 value_cols: Optional[Union[List[str], str]] = None,
                 fill_missing_dates: Optional[bool] = False,
                 freq: Optional[str] = None,
                 fillna_value: Optional[float] = None,
                 **kwargs, ) -> 'TimeSeries':
        """
        Returns a deterministic TimeSeries instance built from a single CSV file.
        One column can be used to represent the time (if not present, the time index will be an Int64Index)
        and a list of columns `value_cols` can be used to indicate the values for this time series.

        Parameters
        ----------
        filepath_or_buffer
            The path to the CSV file, or the file object; consistent with the argument of `pandas.read_csv` function 
        time_col
            The time column name. If set, the column will be cast to a pandas DatetimeIndex.
            If not set, the pandas Int64Index will be used. 
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the CSV file. If set to
            `None`, all columns from the CSV file will be used (except for the time_col, if specified) 
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values. This requires
            either a provided `freq` or the possibility to infer the frequency from the provided timestamps. See
            :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex. This is useful in order to
            fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.
        **kwargs
            Optional arguments to be passed to `pandas.read_csv` function

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.
        """

        df = pd.read_csv(filepath_or_buffer=filepath_or_buffer, **kwargs)
        return cls.from_dataframe(df=df,
                                  time_col=time_col,
                                  value_cols=value_cols,
                                  fill_missing_dates=fill_missing_dates,
                                  freq=freq,
                                  fillna_value=fillna_value)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       time_col: Optional[str] = None,
                       value_cols: Optional[Union[List[str], str]] = None,
                       fill_missing_dates: Optional[bool] = False,
                       freq: Optional[str] = None,
                       fillna_value: Optional[float] = None) -> 'TimeSeries':
        """
        Returns a deterministic TimeSeries instance built from a selection of columns of a DataFrame.
        One column (or the DataFrame index) has to represent the time,
        and a list of columns `value_cols` has to represent the values for this time series.

        Parameters
        ----------
        df
            The DataFrame
        time_col
            The time column name. If set, the column will be cast to a pandas DatetimeIndex.
            If not set, the DataFrame index will be used. In this case the DataFrame must contain an index that is
            either a pandas DatetimeIndex or a pandas Int64Index (incl. RangeIndex). If a DatetimeIndex is
            used, it is better if it has no holes; alternatively setting `fill_missing_dates` can in some casees solve
            these issues (filling holes with NaN, or with the provided `fillna_value` numeric value, if any).
        value_cols
            A string or list of strings representing the value column(s) to be extracted from the DataFrame. If set to
            `None`, the whole DataFrame will be used.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values. This requires
            either a provided `freq` or the possibility to infer the frequency from the provided timestamps. See
            :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex. This is useful in order to
            fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.
        """

        # get values
        if value_cols is None:
            series_df = df.loc[:, df.columns != time_col]
        else:
            if isinstance(value_cols, str):
                value_cols = [value_cols]
            series_df = df[value_cols]

        # get time index
        if time_col:
            if time_col in df.columns:
                if np.issubdtype(df[time_col].dtype, object):
                    try:
                        time_index = pd.Int64Index(df[time_col].astype(int))
                    except ValueError:
                        try:
                            time_index = pd.DatetimeIndex(df[time_col])
                        except ValueError:
                            raise_log(
                                AttributeError("'time_col' is of 'object' dtype but doesn't contain valid timestamps"))
                elif np.issubdtype(df[time_col].dtype, np.integer):
                    time_index = pd.Int64Index(df[time_col])
                elif np.issubdtype(df[time_col].dtype, np.datetime64):
                    time_index = pd.DatetimeIndex(df[time_col])
                else:
                    raise_log(AttributeError(
                        "Invalid type of `time_col`: it needs to be of either 'str', 'datetime' or 'int' dtype."))
            else:
                raise_log(AttributeError('time_col=\'{}\' is not present.'.format(time_col)))
        else:
            raise_if_not(isinstance(df.index, VALID_INDEX_TYPES),
                         'If time_col is not specified, the DataFrame must be indexed either with'
                         'a DatetimeIndex, or with a Int64Index (incl. RangeIndex).', logger)
            time_index = df.index

        if not time_index.name:
            time_index.name = DIMS[0]

        xa = xr.DataArray(series_df.values[:, :, np.newaxis],
                          dims=(time_index.name,) + DIMS[-2:],
                          coords={time_index.name: time_index, DIMS[1]: series_df.columns})

        return cls.from_xarray(xa=xa, fill_missing_dates=fill_missing_dates, freq=freq, fillna_value=fillna_value)

    @classmethod
    def from_series(cls,
                    pd_series: pd.Series,
                    fill_missing_dates: Optional[bool] = False,
                    freq: Optional[str] = None,
                    fillna_value: Optional[float] = None) -> 'TimeSeries':
        """
        Returns a univariate and deterministic TimeSeries built from a pandas Series.

        The series must contain an index that is
        either a pandas DatetimeIndex or a pandas Int64Index (incl. RangeIndex). If a DatetimeIndex is
        used, it is better if it has no holes; alternatively setting `fill_missing_dates` can in some cases solve
        these issues (filling holes with NaN, or with the provided `fillna_value` numeric value, if any).

        Parameters
        ----------
        pd_series
            The pandas Series instance.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values. This requires
            either a provided `freq` or the possibility to infer the frequency from the provided timestamps. See
            :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex. This is useful in order to
            fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A univariate and deterministic TimeSeries constructed from the inputs.
        """

        df = pd.DataFrame(pd_series)
        return cls.from_dataframe(df,
                                  time_col=None,
                                  value_cols=None,
                                  fill_missing_dates=fill_missing_dates,
                                  freq=freq,
                                  fillna_value=fillna_value)

    @classmethod
    def from_times_and_values(cls,
                              times: Union[pd.DatetimeIndex, pd.Int64Index],
                              values: np.ndarray,
                              fill_missing_dates: Optional[bool] = False,
                              freq: Optional[str] = None,
                              columns: Optional[pd._typing.Axes] = None,
                              fillna_value: Optional[float] = None) -> 'TimeSeries':
        """
        Returns a TimeSeries built from an index and value array.

        Parameters
        ----------
        times
            A `pandas.DateTimeIndex` or `pandas.Int64Index` (or `pandas.RangeIndex`) representing the time axis
            for the time series. If a DatetimeIndex is used, it is better if it has no holes; alternatively setting
            `fill_missing_dates` can in some cases solve these issues (filling holes with NaN, or with the
            provided `fillna_value` numeric value, if any).
        values
            A Numpy array of values for the TimeSeries. Both 2-dimensional arrays, for deterministic series,
            and 3-dimensional arrays, for probabilistic series, are accepted. In the former case the dimensions
            should be (time, component), and in the latter case (time, component, sample).
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values. This requires
            either a provided `freq` or the possibility to infer the frequency from the provided timestamps. See
            :meth:`_fill_missing_dates() <TimeSeries._fill_missing_dates>` for more info.
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex. This is useful in order to
            fill in missing values if some dates are missing and `fill_missing_dates` is set to `True`.
        columns
            Columns to be used by the underlying pandas DataFrame.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.
        """
        raise_if_not(isinstance(times, VALID_INDEX_TYPES),
                     'the `times` argument must be a Int64Index (or RangeIndex), or a DateTimeIndex. Use '
                     'TimeSeries.from_values() if you want to use an automatic RangeIndex.')

        times_name = DIMS[0] if not times.name else times.name

        # avoid copying if data is already np.ndarray:
        values = np.array(values) if not isinstance(values, np.ndarray) else values

        if len(values.shape) == 1:
            values = np.expand_dims(values, 1)
        if len(values.shape) == 2:
            values = np.expand_dims(values, 2)

        coords = {times_name: times}
        if columns is not None:
            coords[DIMS[1]] = columns

        xa = xr.DataArray(values,
                          dims=(times_name,) + DIMS[-2:],
                          coords=coords)

        return cls.from_xarray(xa=xa, fill_missing_dates=fill_missing_dates, freq=freq, fillna_value=fillna_value)

    @classmethod
    def from_values(cls,
                    values: np.ndarray,
                    columns: Optional[pd._typing.Axes] = None,
                    fillna_value: Optional[float] = None) -> 'TimeSeries':
        """
        Returns a TimeSeries built from an array of values.
        The series will have an integer index (Int64Index).

        Parameters
        ----------
        values
            A Numpy array of values for the TimeSeries. Both 2-dimensional arrays, for deterministic series,
            and 3-dimensional arrays, for probabilistic series, are accepted. In the former case the dimensions
            should be (time, component), and in the latter case (time, component, sample).
        columns
            Columns to be used by the underlying pandas DataFrame.
        fillna_value
            Optionally, a numeric value to fill missing values (NaNs) with.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.
        """
        time_index = pd.RangeIndex(0, len(values), 1)
        values_ = np.reshape(values, (len(values), 1)) if len(values.shape) == 1 else values

        return cls.from_times_and_values(times=time_index,
                                         values=values_,
                                         fill_missing_dates=False,
                                         freq=None,
                                         columns=columns,
                                         fillna_value=fillna_value)

    @classmethod
    def from_json(cls, json_str: str) -> 'TimeSeries':
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
        return cls.from_dataframe(df)

    @classmethod
    def from_pickle(cls, path: str) -> 'TimeSeries':
        """
        Reads Timeseries object that was pickled.

        **Note**: Xarray docs[1] suggest not using pickle as a long-term data storage.

        ..[1] http://xarray.pydata.org/en/stable/user-guide/io.html#pickle

        Parameters
        ----------
        path : string
            path pointing to a pickle file that will be loaded

        Returns
        -------
        TimeSeries
            timeseries object loaded from file
        """
        with open(path, 'rb') as fh:
            return pickle.load(fh)

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
        return len(self._time_index)

    @property
    def is_deterministic(self):
        return self.n_samples == 1

    @property
    def is_stochastic(self):
        return not self.is_deterministic

    @property
    def is_probabilistic(self):
        return self.is_stochastic

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
    def dtype(self):
        return self._xa.values.dtype

    @property
    def components(self):
        """
        The names of the components (equivalent to DataFrame columns) as a Pandas Index
        """
        return self._xa.get_index(DIMS[1]).copy()

    @property
    def columns(self):
        """
        Same as `components` property
        """
        return self.components

    @property
    def time_index(self) -> Union[pd.DatetimeIndex, pd.Int64Index]:
        """
        Returns
        -------
        Union[pd.DatetimeIndex, pd.Int64Index]
            The time index of this time series.
        """
        return self._time_index.copy()

    @property
    def time_dim(self) -> str:
        """
        The name of the time dimension for this time series
        """
        return self._time_dim

    @property
    def has_datetime_index(self) -> bool:
        """
        Whether this series is indexed with a DatetimeIndex (otherwise it is indexed with an Int64Index)
        """
        return self._has_datetime_index

    @property
    def has_range_index(self) -> bool:
        """
        Whether this series is indexed with an Int64Index (otherwise it is indexed with a DatetimeIndex)
        """
        return not self._has_datetime_index

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
            # Not that the converse doesn't apply (a time-indexed series can be called with an integer)
            raise_if_not(self._has_datetime_index,
                         'Function called with a timestamp, but series not time-indexed.',
                         logger)
            is_inside = self.start_time() <= ts <= self.end_time()
        else:
            if self._has_datetime_index:
                is_inside = 0 <= ts <= len(self)
            else:
                is_inside = self.start_time() <= ts <= self.end_time()

        raise_if_not(is_inside, 'Timestamp must be between {} and {}'.format(self.start_time(),
                                                                             self.end_time()),
                     logger)

    def _get_first_timestamp_after(self, ts: pd.Timestamp) -> pd.Timestamp:
        return next(filter(lambda t: t >= ts, self._time_index))

    def _get_last_timestamp_before(self, ts: pd.Timestamp) -> pd.Timestamp:
        return next(filter(lambda t: t <= ts, self._time_index[::-1]))

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
        xarray.DataArray
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
            return pd.Series(self._xa[:, 0, 0].values.copy(), index=self._time_index.copy())
        else:
            return pd.Series(self._xa[:, 0, 0].values, index=self._time_index)

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
        if not self.is_deterministic:
            raise_log(AssertionError('The pd_dataframe() method can only return DataFrames of deterministic '
                                     'time series, and this series is not deterministic (it contains several samples). '
                                     'Consider calling quantile_df() instead.'))
        if copy:
            return pd.DataFrame(self._xa[:, :, 0].values.copy(),
                                index=self._time_index.copy(),
                                columns=self._xa.get_index(DIMS[1]).copy())
        else:
            return pd.DataFrame(self._xa[:, :, 0].values,
                                index=self._time_index,
                                columns=self._xa.get_index(DIMS[1]))

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
                     'The quantile values must be expressed as fraction (between 0 and 1 inclusive).', logger)

        # column names
        cnames = [s + '_{}'.format(quantile) for s in self.columns]

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
        return self.__class__.from_dataframe(self.quantile_df(quantile))

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

    def astype(self, dtype: Union[str, np.dtype]) -> 'TimeSeries':
        """
        Converts this TimeSeries to a new TimeSeries with desired dtype

        Parameters
        ----------
        dtype
            A NumPy dtype (np.float32 or np.float64)

        Returns
        -------
        TimeSeries
            A TimeSeries having the desired dtype.
        """
        return self.__class__(self._xa.astype(dtype))

    def start_time(self) -> Union[pd.Timestamp, int]:
        """
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the first time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by Int64Index/RangeIndex)
        """
        return self._time_index[0]

    def end_time(self) -> Union[pd.Timestamp, int]:
        """
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the last time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by Int64Index/RangeIndex)
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
        sample
            For stochastic series, the sample for which to return values. Default: 0 (first sample).

        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        raise_if(self.is_deterministic and sample != 0, 'This series contains one sample only (deterministic),'
                                                        'so only sample=0 is accepted.', logger)
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

    def head(self,
             size: Optional[int] = 5,
             axis: Optional[Union[int, str]] = 0) -> 'TimeSeries':
        """
        Return a TimeSeries made of the first n samples of this TimeSeries

        Parameters
        ----------
        size : int, default 5
               number of samples to retain
        axis : str or int, optional, default: 0
               axis along which to slice the series

        Returns
        -------
        TimeSeries
            Three dimensional array of (time, component, samples) where "axis" dimension has been cut off after
            # of ``samples`` samples.
        """

        axis_str = self._get_dim_name(axis)
        display_n = range(min(size, self._xa.sizes[axis_str]))
        return self.__class__(self._xa[{axis_str: display_n}])

    def tail(self,
             size: Optional[int] = 5,
             axis: Optional[Union[int, str]] = 0) -> 'TimeSeries':
        """
        Return last n samples from TimeSeries.

        Parameters
        ----------
        size : int, default: 5
            number of samples to retain
        axis : str or int, optional, default: 0 (time dimension)
            axis along which we intend to display records

        Returns
        -------
        TimeSeries
            Three dimensional array of (time, component, samples) where "axis" dimension has been cut off except
            # of ``samples`` from the bottom. [Default: 5]
        """

        axis_str = self._get_dim_name(axis)
        display_n = range(-min(size, self._xa.sizes[axis_str]), 0)
        return self.__class__(self._xa[{axis_str: display_n}])

    def concatenate(self,
                    other: 'TimeSeries',
                    axis: Optional[Union[str, int]] = 0,
                    ignore_time_axes: Optional[bool] = False) -> 'TimeSeries':
        """Concatenates another timeseries to the current one along given axis.

            Note: when concatenating along the ``time`` dimension, timeseries `self` marks the start date of
            the resulting series, and the remaining series will have their date indices overwritten.

            Parameters
            ----------
            other : TimeSeries
                another timeseries to concatenate to this one
            axis : str or int
                axis along which timeseries will be concatenated. ['time', 'component' or 'sample'; Default: 0 (time)]
            ignore_time_axes : bool, default False
                Ignore errors when time axis varies for some timeseries. Note that this may yield unexpected results

            Returns
            -------
            TimeSeries
                concatenated timeseries
        """
        return concatenate(series=[self, other], axis=axis, ignore_time_axis=ignore_time_axes)

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
            of the gap and the integer length of the gap (in `self.freq` units if the series is indexed
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
            lambda row: intvl(start=row.gap_start, end=row.gap_end), axis=1
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

        # the xarray will be copied in the TimeSeries constructor.
        return self.__class__(self._xa)

    def get_index_at_point(self, point: Union[pd.Timestamp, float, int], after=True) -> int:
        """
        Converts a point into an integer index

        Parameters
        ----------
        point
            This parameter supports 3 different data types: `pd.Timestamp`, `float` and `int`.

            `pd.Timestamp` work only on series that are indexed with a `pd.DatetimeIndex`. In such cases, the returned
            point will be the index of this timestamp if it is present in the series time index. It it's not present
            in the time index, the index of the next timestamp is returned if `after=True` (if it exists in the series),
            otherwise the index of the previous timestamp is returned (if it exists in the series).

            In case of a `float`, the parameter will be treated as the proportion of the time series
            that should lie before the point.

            In the case of `int`, the parameter will returned as such, provided that it is in the series. Otherwise
            it will raise a ValueError.
        after
            If the provided pandas Timestamp is not in the time series index, whether to return the index of the
            next timestamp or the index of the previous one.

        """
        point_index = -1
        if isinstance(point, float):
            raise_if_not(0. <= point <= 1., 'point (float) should be between 0.0 and 1.0.', logger)
            point_index = int((len(self) - 1) * point)
        elif isinstance(point, (int, np.int64)):
            raise_if(point not in range(len(self)), "point (int) should be a valid index in series", logger)
            point_index = point
        elif isinstance(point, pd.Timestamp):
            raise_if_not(self._has_datetime_index,
                         'A Timestamp has been provided, but this series is not time-indexed.', logger)
            self._raise_if_not_within(point)
            if point in self:
                point_index = self._time_index.get_loc(point)
            else:
                point_index = self._time_index.get_loc(self._get_first_timestamp_after(point) if after else
                                                       self._get_last_timestamp_before(point))
        else:
            raise_log(TypeError("`point` needs to be either `float`, `int` or `pd.Timestamp`"), logger)
        return point_index

    def get_timestamp_at_point(self, point: Union[pd.Timestamp, float, int]) -> pd.Timestamp:
        """
        Converts a point into a pandas.Timestamp (if Datetime-indexed) or into an integer (if Int64-indexed).

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
        """
        idx = self.get_index_at_point(point)
        return self._time_index[idx]

    def _split_at(self,
                  split_point: Union[pd.Timestamp, float, int],
                  after: bool = True) -> Tuple['TimeSeries', 'TimeSeries']:

        point_index = self.get_index_at_point(split_point, after)
        return self[:point_index+(1 if after else 0)], self[point_index+(1 if after else 0):]

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

    def drop_after(self, split_point: Union[pd.Timestamp, float, int]):
        """
        Drops everything after the provided timestamp `ts`, included.
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, after `ts`.
        """
        return self.split_before(split_point)[0]

    def drop_before(self, split_point: Union[pd.Timestamp, float, int]):
        """
        Drops everything before the provided timestamp `ts`, included.
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        Parameters
        ----------
        split_point
            The timestamp that indicates cut-off time.

        Returns
        -------
        TimeSeries
            A new TimeSeries, after `ts`.
        """
        return self.split_after(split_point)[1]

    def slice(self, start_ts: Union[pd.Timestamp, int], end_ts: Union[pd.Timestamp, int]):
        """
        Returns a new TimeSeries, starting later than `start_ts` and ending before `end_ts`, inclusive on both ends.
        The timestamps don't have to be in the series.

        Parameters
        ----------
        start_ts
            The timestamp that indicates the left cut-off.
        end_ts
            The timestamp that indicates the right cut-off.

        Returns
        -------
        TimeSeries
            A new series, with indices greater or equal than `start_ts` and smaller or equal than `end_ts`.
        """
        raise_if_not(type(start_ts) == type(end_ts), 'The two timestamps provided to slice() have to be of the '
                                                     'same type.', logger)
        if isinstance(start_ts, pd.Timestamp):
            raise_if_not(self._has_datetime_index, 'Timestamps have been provided to slice(), but the series is '
                                                   'indexed using an integer-based Int64Index.', logger)
            idx = pd.DatetimeIndex(filter(lambda t: start_ts <= t <= end_ts, self._time_index))
        else:
            raise_if(self._has_datetime_index, 'start and end times have been provided as integers to slice(), but '
                                               'the series is indexed with a DatetimeIndex.', logger)
            idx = pd.RangeIndex(start_ts, end_ts, step=1)
        return self[idx]

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

        if isinstance(start_ts, (int, np.int64)):
            return self[start_ts:start_ts+n]
        elif isinstance(start_ts, pd.Timestamp):
            # get first timestamp greater or equal to start_ts
            tss = self._get_first_timestamp_after(start_ts)
            point_index = self.get_index_at_point(tss)
            return self[point_index:point_index + n]
        else:
            raise_log(ValueError('start_ts must be an int or a pandas Timestamp.'), logger)

    def slice_n_points_before(self, end_ts: Union[pd.Timestamp, int], n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending at `start_ts` and having at most `n` points.

        The provided timestamps will be included in the series.

        Parameters
        ----------
        end_ts
            The timestamp or index that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n`, ending at `start_ts`
        """

        raise_if_not(n > 0, 'n should be a positive integer.', logger)
        self._raise_if_not_within(end_ts)

        if isinstance(end_ts, (int, np.int64)):
            return self[end_ts-n+1:end_ts+1]
        elif isinstance(end_ts, pd.Timestamp):
            # get last timestamp smaller or equal to start_ts
            tss = self._get_last_timestamp_before(end_ts)
            point_index = self.get_index_at_point(tss)
            return self[max(0, point_index-n+1):point_index+1]
        else:
            raise_log(ValueError('start_ts must be an int or a pandas Timestamp.'), logger)

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
        return self[time_index]

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
        return self.__class__.from_dataframe(new_series)

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
        if not (np.isnan(self._xa)).any():
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
        coef = value_at_first_step / self._xa.isel({self._time_dim: [0]})
        coef = coef.values.reshape((self.n_components, self.n_samples))  # TODO: test
        new_series = coef * self._xa
        return self.__class__(new_series)

    def shift(self, n: int) -> 'TimeSeries':
        """
        Shifts the time axis of this TimeSeries by `n` time steps.

        If :math:`n > 0`, shifts in the future. If :math:`n < 0`, shifts in the past.

        For example, with :math:`n=2` and `freq='M'`, March 2013 becomes May 2013.
        With :math:`n=-2`, March 2013 becomes Jan 2013.

        Parameters
        ----------
        n
            The number of time steps (in self.freq unit) to shift by. Can be negative.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with a shifted index.
        """
        if not isinstance(n, (int, np.int64)):
            logger.warning(f"TimeSeries.shift(): converting n to int from {n} to {int(n)}")
            n = int(n)

        try:
            self._time_index[-1] + n * self.freq
        except pd.errors.OutOfBoundsDatetime:
            raise_log(OverflowError("the add operation between {} and {} will "
                                    "overflow".format(n * self.freq, self.time_index[-1])), logger)

        if self.has_range_index:
            new_time_index = self._time_index + n*self.freq
        else:
            new_time_index = self._time_index.map(lambda ts: ts + n * self.freq)
        new_xa = self._xa.assign_coords({self._xa.dims[0]: new_time_index})
        return self.__class__(new_xa)

    def diff(self,
             n: Optional[int] = 1,
             periods: Optional[int] = 1,
             dropna: Optional[bool] = True) -> 'TimeSeries':
        """
        Returns a differenced time series. This is often used to make a time series stationary.

        Parameters
        ----------
        n
            Optionally, a positive integer indicating the number of differencing steps (default = 1).
            For instance, n=2 computes the second order differences.
        periods
            Optionally, periods to shift for calculating difference. For instance, periods=12 computes the
            difference between values at time `t` and times `t-12`.
        dropna
            Whether to drop the missing values after each differencing steps. If set to False, the corresponding
            first `periods` time steps will be filled with NaNs.

        Returns
        -------
        TimeSeries
            A TimeSeries constructed after differencing.
        """
        if not isinstance(n, int) or n < 1:
             raise_log(ValueError("'n' must be a positive integer >= 1."), logger)
        if not isinstance(periods, int) or periods < 1:
             raise_log(ValueError("'periods' must be an integer >= 1."), logger)

        def _compute_diff(xa: xr.DataArray):
            # xarray doesn't support Pandas "period" so compute diff() ourselves
            if not dropna:
                # In this case the new DataArray will have the same size and filled with NaNs
                new_xa_ = xa.copy()
                new_xa_.values[:periods, :, :] = np.nan
                new_xa_.values[periods:, :, :] = xa.values[periods:, :, :] - xa.values[:-periods, :, :]
            else:
                # In this case the new DataArray will be shorter
                new_xa_ = xa[periods:, :, :].copy()
                new_xa_.values = xa.values[periods:, :, :] - xa.values[:-periods, :, :]
            return new_xa_

        new_xa = _compute_diff(self._xa)
        for _ in range(n-1):
            new_xa = _compute_diff(new_xa)
        return self.__class__(new_xa)

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
        if len(other) != len(self):
            return False
        return (other.time_index == self.time_index).all()

    def append(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Appends another TimeSeries to this TimeSeries, along the time axis.

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
                     'Both series must have the same type of time index (either DatetimeIndex or Int64Index).', logger)
        raise_if_not(other.freq == self.freq,
                     'Appended TimeSeries must have the same frequency as the current one', logger)
        raise_if_not(other.n_components == self.n_components,
                     'Both series must have the same number of components.', logger)
        raise_if_not(other.n_samples == self.n_samples,
                     'Both series must have the same number of components.', logger)
        if self._has_datetime_index:
            raise_if_not(other.start_time() == self.end_time() + self.freq,
                         'Appended TimeSeries must start one time step after current one.', logger)

        other_xa = other.data_array()

        new_xa = xr.DataArray(np.concatenate((self._xa.values, other_xa.values), axis=0),
                              dims=self._xa.dims,
                              coords={self._time_dim: self._time_index.append(other.time_index),
                                      DIMS[1]: self.components})

        # new_xa = xr.concat(objs=[self._xa, other_xa], dim=str(self._time_dim))
        if not self._has_datetime_index:
            new_xa = new_xa.reset_index(dims_or_levels=new_xa.dims[0])

        return self.__class__.from_xarray(new_xa, fill_missing_dates=True, freq=self._freq_str)

    def append_values(self, values: np.ndarray) -> 'TimeSeries':
        """
        Appends values to current TimeSeries, to the given indices.

        Parameters
        ----------
        values
            An array with the values to append.

        Returns
        -------
        TimeSeries
            A new TimeSeries with the new values appended
        """

        # TODO test
        if self._has_datetime_index:
            idx = pd.DatetimeIndex([self.end_time() + i * self._freq for i in range(1, len(values)+1)], freq=self._freq)
        else:
            idx = pd.RangeIndex(len(self), len(self)+len(values), 1)

        return self.append(self.__class__.from_times_and_values(values=values,
                                                                times=idx,
                                                                fill_missing_dates=False))

    def stack(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Stacks another univariate or multivariate TimeSeries with the same time index on top of
        the current one (along the component axis), and returns the newly formed multivariate TimeSeries that includes
        all the components of `self` and of `other`.

        The resulting TimeSeries will have the same name for its time dimension as this TimeSeries, and the
        same number of samples.

        Parameters
        ----------
        other
            A TimeSeries instance with the same index and the same number of samples as the current one.

        Returns
        -------
        TimeSeries
            A new multivariate TimeSeries instance.
        """
        raise_if_not(self.has_same_time_as(other), 'The indices of the two TimeSeries instances '
                     'must be equal', logger)
        raise_if_not(self.n_samples == other.n_samples, 'Two series can be stacked only if they '
                                                        'have the same number of samples.', logger)

        other_xa = other.data_array(copy=False)
        if other_xa.dims[0] != self._time_dim:
            new_other_xa = xr.DataArray(other_xa.values,
                                        dims=self._xa.dims,
                                        coords={self._time_dim: self._time_index, DIMS[1]: other.components})
        else:
            new_other_xa = other_xa

        new_xa = xr.concat((self._xa, new_other_xa), dim=DIMS[1])

        # we call the factory method here to disambiguate column names if needed.
        return self.__class__.from_xarray(new_xa, fill_missing_dates=False)

    def univariate_component(self, index: Union[str, int]) -> 'TimeSeries':
        """
        Retrieves one of the components of the current TimeSeries instance
        and returns it as new univariate TimeSeries instance.

        Parameters
        ----------
        index
            An zero-indexed integer indicating which component to retrieve. If components have names,
            this can be a string with the component's name.

        Returns
        -------
        TimeSeries
            A new univariate TimeSeries instance.
        """
        if isinstance(index, int):
            new_xa = self._xa.isel(component=index).expand_dims(DIMS[1], axis=1)
        else:
            new_xa = self._xa.sel(component=index).expand_dims(DIMS[1], axis=1)
        return self.__class__(new_xa)

    def add_datetime_attribute(self, attribute, one_hot: bool = False, cyclic: bool = False) -> 'TimeSeries':
        """
        Returns a new TimeSeries instance with one (or more) additional component(s) that contain an attribute
        of the time index of the current series specified with `attribute`, such as 'weekday', 'day' or 'month'.

        This works only for deterministic time series (i.e., made of 1 sample).

        Parameters
        ----------
        attribute
            A pd.DatatimeIndex attribute which will serve as the basis of the new column(s).
        one_hot
            Boolean value indicating whether to add the specified attribute as a one hot encoding
            (results in more columns).
        cyclic
            Boolean value indicating whether to add the specified attribute as a cyclic encoding.
            Alternative to one_hot encoding, enable only one of the two.
            (adds 2 columns, corresponding to sin and cos transformation).

        Returns
        -------
        TimeSeries
            New TimeSeries instance enhanced by `attribute`.
        """
        self._assert_deterministic()
        from .utils import timeseries_generation as tg
        return self.stack(tg.datetime_attribute_timeseries(self.time_index, attribute, one_hot, cyclic))

    def add_holidays(self,
                     country_code: str,
                     prov: str = None,
                     state: str = None) -> 'TimeSeries':
        """
        Adds a binary univariate component to the current series that equals 1 at every index that
        corresponds to selected country's holiday, and 0 otherwise. The frequency of the TimeSeries is daily.

        Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`_.

        This works only for deterministic time series (i.e., made of 1 sample).

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
        self._assert_deterministic()
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

        resample = self._xa.resample({self._time_dim: freq})

        # TODO: check
        if method == 'pad':
            new_xa = resample.pad()
        elif method == 'bfill':
            new_xa = resample.backfill()
        else:
            raise_log(ValueError('Unknown method: {}'.format(method)), logger)
        return self.__class__(new_xa)

    def is_within_range(self, ts: Union[pd.Timestamp, int]) -> bool:
        """
        Check whether a given timestamp or integer is withing the time interval of this time series.
        If a timestamp is provided, it does not need to be an element of the time index of the series.

        Parameters
        ----------
        ts
            The `pandas.Timestamp` (if indexed with DatetimeIndex) or integer (if indexed with Int64Index) to check.

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
            or an integer otherwise (if the series is indexed with an Int64Index).

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
            df = None
            raise_log(ValueError("fn must have either one or two arguments"), logger)

        return self.__class__.from_dataframe(df)

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

    def to_csv(self, *args, **kwargs):

        """
        Writes deterministic time series to CSV file. For a list of parameters, refer to the documentation of
        pandas.DataFrame.to_csv().[1]

        ..[1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv
        """
        if not self.is_deterministic:
            raise_log(AssertionError('The pd_dataframe() method can only return DataFrames of deterministic '
                                     'time series, and this series is not deterministic (it contains several samples). '
                                     'Consider calling quantile_df() instead.'))

        self.pd_dataframe().to_csv(*args, **kwargs)

    def to_pickle(self, path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        """
        Saves Timeseries object in pickle format.

        **Note**: Xarray docs[1] suggest not using pickle as a long-term data storage.

        ..[1] http://xarray.pydata.org/en/stable/user-guide/io.html#pickle

        Parameters
        ----------
        path : string
            path to a file where current object will be pickled
        protocol : integer, default highest
            pickling protocol. The default is best in most cases, use it only if having backward compatibility issues
        """

        with open(path, 'wb') as fh:
            pickle.dump(self, fh, protocol=protocol)

    def plot(self,
             new_plot: bool = False,
             central_quantile: Union[float, str] = 0.5,
             low_quantile: Optional[float] = 0.05,
             high_quantile: Optional[float] = 0.95,
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
        low_quantile
            The quantile to use for the lower bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `confidence_low_quantile` is None (default 0.05).
        high_quantile
            The quantile to use for the upper bound of the plotted confidence interval. Similar to `central_quantile`,
            this is applied to each component separately (i.e., displaying marginal distributions). No confidence
            interval is shown if `high_quantile` is None (default 0.95).
        args
            some positional arguments for the `plot()` method
        kwargs
            some keyword arguments for the `plot()` method
        """
        alpha_confidence_intvls = 0.25

        if central_quantile != 'mean':
            raise_if_not(isinstance(central_quantile, float) and 0. <= central_quantile <= 1.,
                         'central_quantile must be either "mean", or a float between 0 and 1.',
                         logger)

        if high_quantile is not None and low_quantile is not None:
            raise_if_not(0. <= low_quantile <= 1. and 0. <= high_quantile <= 1.,
                         'confidence interval low and high quantiles must be between 0 and 1.',
                         logger)

        fig = (plt.figure() if new_plot else (kwargs['figure'] if 'figure' in kwargs else plt.gcf()))
        kwargs['figure'] = fig
        label = kwargs['label'] if 'label' in kwargs else ''

        if not any(lw in kwargs for lw in ['lw', 'linewidth']):
            kwargs['lw'] = 2

        if self.n_components > 10:
            logger.warn('Number of components is larger than 10 ({}). Plotting only the first 10 components.'.format(
                self.n_components
            ))

        for i, c in enumerate(self._xa.component[:10]):
            comp_name = str(c.values)

            if i > 0:
                kwargs['figure'] = plt.gcf()

            comp = self._xa.sel(component=c)

            if comp.sample.size > 1:
                if central_quantile == 'mean':
                    central_series = comp.mean(dim=DIMS[2])
                else:
                    central_series = comp.quantile(q=central_quantile, dim=DIMS[2])
            else:
                central_series = comp.mean(dim=DIMS[2])

            # temporarily set alpha to 1 to plot the central value (this way alpha impacts only the confidence intvls)
            alpha = kwargs['alpha'] if 'alpha' in kwargs else None
            kwargs['alpha'] = 1

            label_to_use = (label + ('_' + str(i) if len(self.components) > 1 else '')) if label != '' \
                           else '' + str(comp_name)
            kwargs['label'] = label_to_use

            p = central_series.plot(*args, **kwargs)
            color_used = p[0].get_color()
            kwargs['alpha'] = alpha if alpha is not None else alpha_confidence_intvls

            # Optionally show confidence intervals
            if comp.sample.size > 1 and low_quantile is not None and high_quantile is not None:
                    low_series = comp.quantile(q=low_quantile, dim=DIMS[2])
                    high_series = comp.quantile(q=high_quantile, dim=DIMS[2])
                    plt.fill_between(self.time_index, low_series, high_series, color=color_used,
                                     alpha=(alpha_confidence_intvls if 'alpha' not in kwargs else kwargs['alpha']))

        plt.legend()
        plt.title(self._xa.name);

    def with_columns_renamed(self, col_names: Union[List[str], str], col_names_new: Union[List[str], str]) -> 'TimeSeries':
        """
        Changes ts column names and returns a new TimeSeries instance.

        Parameters
        -------
        col_names
            String or list of strings corresponding the the column names to be changed.
        col_names_new
            String or list of strings corresponding to the new column names. Must be the same length as col_names.

        Returns
        -------
        TimeSeries
            A new TimeSeries instance.
        """

        if isinstance(col_names, str):
            col_names = [col_names]
        if isinstance(col_names_new, str):
            col_names_new = [col_names_new]

        raise_if_not(all([(x in self.columns.to_list()) for x in col_names]), 
                                                    "Some column names in col_names don't exist in the time series.", logger)
        
        raise_if_not(len(col_names) == len(col_names_new), 'Length of col_names_new list should be'
                                                    ' equal to the length of col_names list.', logger)


        cols = self.components

        for (o, n) in zip(col_names, col_names_new):
            cols = [n if (c==o) else c for c in cols]

        new_xa = xr.DataArray(
            self._xa.values,
            dims=self._xa.dims,
            coords={
                self._xa.dims[0]: self.time_index, 
                DIMS[1]: pd.Index(cols)
                }
        )
        
        return self.__class__(new_xa)


    """
    Simple statistics. At the moment these work only on deterministic series, and are wrapped around Pandas.
    """

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).mean(axis, skipna, level, numeric_only, **kwargs)

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).skew(axis, skipna, level, numeric_only, **kwargs)

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).kurtosis(axis, skipna, level, numeric_only, **kwargs)

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).min(axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).max(axis, skipna, level, numeric_only, **kwargs)

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs) -> float:
        return self.pd_dataframe(copy=False).sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self.pd_dataframe(copy=False).median(axis, skipna, level, numeric_only, **kwargs)

    def autocorr(self, lag=1) -> float:
        return self.pd_dataframe(copy=False).autocorr(lag)

    def describe(self, percentiles=None, include=None, exclude=None) -> pd.DataFrame:
        return self.pd_dataframe(copy=False).describe(percentiles, include, exclude)

    """
    Dunder methods
    """

    def _combine_arrays(self,
                        other: Union['TimeSeries', xr.DataArray, np.ndarray],
                        combine_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> 'TimeSeries':
        """
        This is a helper function that allows us to combine this series with another one,
        directly applying an operation on their underlying numpy arrays.
        """

        if isinstance(other, TimeSeries):
            other_vals = other.data_array(copy=False).values
        elif isinstance(other, xr.DataArray):
            other_vals = other.values
        else:
            other_vals = other

        raise_if_not(self._xa.values.shape == other_vals.shape, 'Attempted to perform operation on two TimeSeries '
                                                                'of unequal shapes.', logger)
        new_xa = self._xa.copy()
        new_xa.values = combine_fn(new_xa.values, other_vals)
        return self.__class__(new_xa)

    @classmethod
    def _fill_missing_dates(cls,
                            xa: xr.DataArray,
                            freq: Optional[str] = None) -> xr.DataArray:
        """ Returns an xarray DataArray instance with missing dates inserted from an input xarray DataArray.
        The first dimension of the input DataArray `xa` has to be the time dimension.

        This requires either a provided `freq` or the possibility to infer a unique frequency (see
        `offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
        for more info on supported frequencies) from the provided timestamps.

        Parameters
        ----------
        xa
            The xarray DataArray
        freq
            Optionally, a string representing the frequency of the Pandas DateTimeIndex to fill in the missing dates.

        Raises
        -------
        ValueError
            If `xa`'s DateTimeIndex contains less than 3 elements;
            if no unique frequency can be inferred from `xa`'s DateTimeIndex;
            if the resampled DateTimeIndex does not contain all dates from `xa` (see
                :meth:`_restore_xarray_from_frequency() <TimeSeries._restore_xarray_from_frequency>`)

        Returns
        -------
        xarray DataArray
            xarray DataArray with filled missing dates from `xa`
        """

        if freq is not None:
            return cls._restore_xarray_from_frequency(xa, freq)

        raise_if(len(xa) <= 2, f"Input time series must be of (length>=3) when fill_missing_dates=True and freq=None.",
                 logger)

        time_dim = xa.dims[0]
        sorted_xa = xa.copy() if xa.get_index(time_dim).is_monotonic_increasing else xa.sortby(time_dim)
        time_index = sorted_xa.get_index(time_dim)

        offset_alias_info = "For more information about frequency aliases, read " \
                            "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"

        step_size = 3
        n_dates = len(time_index)
        # this creates n steps containing 3 timestamps each; used to infer frequency of time_index
        steps = np.column_stack([time_index[i: (n_dates - step_size + (i + 1))] for i in range(step_size)])
        observed_freqs = set(pd.infer_freq(step) for step in steps)
        observed_freqs.discard(None)

        raise_if_not(
            len(observed_freqs) == 1,
            f"Could not observe an inferred frequency. An explicit frequency must be evident over a span of at least "
            f"3 consecutive time stamps in the input data. {offset_alias_info}" if not len(observed_freqs) else
            f"Could not find a unique inferred frequency (not constant). Observed frequencies: {observed_freqs}. "
            f"If any of those is the actual frequency, try passing it with fill_missing_dates=True "
            f"and freq=your_frequency. {offset_alias_info}",
            logger)

        freq = observed_freqs.pop()

        return cls._restore_xarray_from_frequency(sorted_xa, freq)

    @staticmethod
    def _restore_xarray_from_frequency(xa: xr.DataArray,
                                       freq: str) -> xr.DataArray:
        """ Returns an xarray DataArray instance that is resampled from an input xarray DataArray `xa` with frequency
        `freq`. `freq` should be the inferred or actual frequency of `xa`. All data from `xa` is maintained in the
        output DataArray at the corresponding dates. Any missing dates from `xa` will be inserted into the returned
        DataArray with np.nan values.

        The first dimension of the input DataArray `xa` has to be the time dimension.

        This requires a provided `freq` (see `offset aliases
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more info on
        supported frequencies).

        Parameters
        ----------
        xa
            The xarray DataArray
        freq
            A string representing the actual or inferred frequency of the Pandas DateTimeIndex from `xa`.

        Raises
        -------
        ValueError
            If the resampled DateTimeIndex does not contain all dates from `xa`

        Returns
        -------
        xarray DataArray
            xarray DataArray resampled from `xa` with `freq` including all data from `xa` and inserted missing dates
        """

        time_dim = xa.dims[0]
        sorted_xa = xa.copy() if xa.get_index(time_dim).is_monotonic_increasing else xa.sortby(time_dim)

        time_index = sorted_xa.get_index(time_dim)
        resampled_time_index = pd.Series(index=time_index, dtype='object').asfreq(freq)

        # check if new time index with inferred frequency contains all input data
        contains_all_data = time_index.isin(resampled_time_index.index).all()

        offset_alias_info = "For more information about frequency aliases, read " \
                            "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases"
        raise_if_not(
            contains_all_data,
            f"Could not correctly fill missing dates with the observed/passed frequency freq='{freq}'. "
            f"Not all input time stamps contained in the newly created TimeSeries. {offset_alias_info}",
            logger)

        coords = {
            xa.dims[0]: pd.DatetimeIndex(resampled_time_index.index),
            xa.dims[1]: xa.coords[DIMS[1]]
        }

        resampled_xa = xr.DataArray(data=np.empty(shape=((len(resampled_time_index),) + xa.shape[1:])),
                                    dims=xa.dims,
                                    coords=coords)
        resampled_xa[:] = np.nan
        resampled_xa[resampled_time_index.index.isin(time_index)] = sorted_xa.data
        return resampled_xa

    def _get_dim_name(self, axis: Union[int, str]) -> str:
        if isinstance(axis, int):
            if axis == 0:
                return self._time_dim
            elif axis == 1 or axis == 2:
                return DIMS[axis]
            else:
                raise_if(True, 'If `axis` is an integer it must be between 0 and 2.')
        else:
            known_dims = (self._time_dim,) + DIMS[1:]
            raise_if_not(axis in known_dims,
                         '`axis` must be a known dimension of this series: {}'.format(known_dims))
            return axis

    def _get_dim(self, axis: Union[int, str]) -> int:
        if isinstance(axis, int):
            raise_if_not(0 <= axis <= 2, 'If `axis` is an integer it must be between 0 and 2.')
            return axis
        else:
            known_dims = (self._time_dim,) + DIMS[1:]
            raise_if_not(axis in known_dims,
                         '`axis` must be a known dimension of this series: {}'.format(known_dims))
            return known_dims.index(axis)

    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            return self._xa.equals(other.data_array(copy=False))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._xa)

    def __add__(self, other):
        if isinstance(other, (int, float, np.integer)):
            return self.__class__(self._xa + other)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 + s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for + or add(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float, np.integer)):
            return self.__class__(self._xa - other)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 - s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for - or sub(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.integer)):
            return self.__class__(self._xa * other)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(other, lambda s1, s2: s1 * s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for * or mul(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float, np.integer)):
            raise_if(n < 0, 'Attempted to raise a series to a negative power.', logger)
            return self.__class__(self._xa ** float(n))
        if isinstance(n, (TimeSeries, xr.DataArray, np.ndarray)):
            return self._combine_arrays(n, lambda s1, s2: s1 ** s2)  # elementwise power
        else:
            raise_log(TypeError('unsupported operand type(s) for ** or pow(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(n).__name__)), logger)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer)):
            if other == 0:
                raise_log(ZeroDivisionError('Cannot divide by 0.'), logger)
            return self.__class__(self._xa / other)
        elif isinstance(other, (TimeSeries, xr.DataArray, np.ndarray)):
            if not (other.all_values() != 0).all():
                raise_log(ZeroDivisionError('Cannot divide by a TimeSeries with a value 0.'), logger)
            return self._combine_arrays(other, lambda s1, s2: s1 / s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for / or truediv(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        return self.__class__(abs(self._xa))

    def __neg__(self):
        return self.__class__(-self._xa)

    def __contains__(self, ts: Union[int, pd.Timestamp]) -> bool:
        return ts in self.time_index

    def __round__(self, n=None):
        return self.__class__(self._xa.round(n))

    def __lt__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            series = self._xa < other
        elif isinstance(other, TimeSeries):
            series = self._xa < other.data_array(copy=False)
        else:
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series  # Note: we return a DataArray

    def __gt__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            series = self._xa > other
        elif isinstance(other, TimeSeries):
            series = self._xa > other.data_array(copy=False)
        else:
            series = None
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series  # Note: we return a DataArray

    def __le__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            series = self._xa <= other
        elif isinstance(other, TimeSeries):
            series = self._xa <= other.data_array(copy=False)
        else:
            series = None
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series  # Note: we return a DataArray

    def __ge__(self, other) -> xr.DataArray:
        if isinstance(other, (int, float, np.integer, np.ndarray, xr.DataArray)):
            series = self._xa >= other
        elif isinstance(other, TimeSeries):
            series = self._xa >= other.data_array(copy=False)
        else:
            series = None
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(other).__name__)), logger)
        return series  # Note: we return a DataArray

    def __str__(self):
        return str(self._xa).replace('xarray.DataArray', 'TimeSeries (DataArray)')

    def __repr__(self):
        return self._xa.__repr__().replace('xarray.DataArray', 'TimeSeries (DataArray)')

    def _repr_html_(self):
        return self._xa._repr_html_().replace('xarray.DataArray', 'TimeSeries (DataArray)')

    def __copy__(self, deep: bool = True):
        return self.copy()

    def __deepcopy__(self):
        return self.__class__(self._xa.copy())

    def __getitem__(self,
                    key: Union[pd.DatetimeIndex,
                               pd.Int64Index,
                               List[str],
                               List[int],
                               List[pd.Timestamp],
                               str,
                               int,
                               pd.Timestamp,
                               Any]) -> 'TimeSeries':
        """Allow indexing on darts TimeSeries.

        The supported index types are the following base types as a single value, a list or a slice:
        - pd.Timestamp -> return a TimeSeries corresponding to the value(s) at the given timestamp(s).
        - str -> return a TimeSeries including the column(s) (components) specified as str.
        - int -> return a TimeSeries with the value(s) at the given row (time) index.

        `pd.DatetimeIndex` and `pd.Int64Index` are also supported and will return the corresponding value(s)
        at the provided time indices.

        .. warning::
            slices use pandas convention of including both ends of the slice.
        """
        def _check_dt():
            raise_if_not(self._has_datetime_index, 'Attempted indexing a series with a DatetimeIndex or a timestamp, '
                                                   'but the series uses an Int64Index.', logger)

        def _check_range():
            raise_if(self._has_datetime_index, 'Attempted indexing a series with an Int64Index, '
                                               'but the series uses a DatetimeIndex.', logger)

        def _set_freq_in_xa(xa_: xr.DataArray):
            # mutates the DataArray to make sure it contains the freq
            if isinstance(xa_.get_index(self._time_dim), pd.DatetimeIndex):
                inferred_freq = xa_.get_index(self._time_dim).inferred_freq
                if inferred_freq is not None:
                    xa_.get_index(self._time_dim).freq = to_offset(inferred_freq)
                else:
                    xa_.get_index(self._time_dim).freq = self._freq

        # handle DatetimeIndex and Int64Index:
        if isinstance(key, pd.DatetimeIndex):
            _check_dt()
            xa_ = self._xa.sel({self._time_dim: key})

            # indexing may discard the freq so we restore it...
            # TODO: unit-test this
            _set_freq_in_xa(xa_)

            return self.__class__(xa_)
        elif isinstance(key, (pd.Int64Index, pd.RangeIndex)):
            _check_range()

            return self.__class__(self._xa.sel({self._time_dim: key}))

        # handle slices:
        elif isinstance(key, slice):
            if isinstance(key.start, str) or isinstance(key.stop, str):
                return self.__class__(self._xa.sel({DIMS[1]: key}))
            elif isinstance(key.start, (int, np.int64)) or isinstance(key.stop, (int, np.int64)):
                xa_ = self._xa.isel({self._time_dim: key})
                _set_freq_in_xa(xa_)  # indexing may discard the freq so we restore it...
                return self.__class__(xa_)
            elif isinstance(key.start, pd.Timestamp) or isinstance(key.stop, pd.Timestamp):
                _check_dt()

                # indexing may discard the freq so we restore it...
                xa_ = self._xa.sel({self._time_dim: key})
                _set_freq_in_xa(xa_)
                return self.__class__(xa_)

        # handle simple types:
        elif isinstance(key, str):
            return self.__class__(self._xa.sel({DIMS[1]: [key]}))  # have to put key in a list not to drop the dimension
        elif isinstance(key, (int, np.int64)):
            xa_ = self._xa.isel({self._time_dim: [key]})
            _set_freq_in_xa(xa_)  # indexing may discard the freq so we restore it...
            return self.__class__(xa_)
        elif isinstance(key, pd.Timestamp):
            _check_dt()

            # indexing may discard the freq so we restore it...
            xa_ = self._xa.sel({self._time_dim: [key]})
            _set_freq_in_xa(xa_)
            return self.__class__(xa_)

        # handle lists:
        if isinstance(key, list):
            if all(isinstance(s, str) for s in key):
                # when string(s) are provided, we consider it as (a list of) component(s)
                return self.__class__(self._xa.sel({DIMS[1]: key}))
            elif all(isinstance(i, (int, np.int64)) for i in key):
                xa_ = self._xa.isel({self._time_dim: key})
                _set_freq_in_xa(xa_)  # indexing may discard the freq so we restore it...
                return self.__class__(xa_)
            elif all(isinstance(t, pd.Timestamp) for t in key):
                _check_dt()

                # indexing may discard the freq so we restore it...
                xa_ = self._xa.sel({self._time_dim: key})
                _set_freq_in_xa(xa_)
                return self.__class__(xa_)

        raise_log(IndexError("The type of your index was not matched."), logger)


def concatenate(series: Sequence['TimeSeries'],
                axis: Union[str, int] = 0,
                ignore_time_axis: bool = False):
    """Concatenates multiple ``TimeSeries`` along a given axis.

    ``axis`` can be an integer in (0, 1, 2) to denote (time, component, sample) or, alternatively,
    a string denoting the corresponding dimension of the underlying ``DataArray``.

    Parameters
    ----------
    series : Sequence[TimeSeries]
        sequence of ``TimeSeries`` to concatenate
    axis : Union[str, int]
        axis along which the series will be concatenated.
    ignore_time_axis : bool
        Allow concatenation even when some series do not have matching time axes.
        When done along component or sample dimensions, concatenation will work as long as the series
        have the same lengths (in this case the resulting series will have the time axis of the first
        provided series). When done along time dimension, concatenation will work even if the time axes
        are not contiguous (in this case, the resulting series will have a start time matching the start time
        of the first provided series). Default: False.

    Returns
    -------
    TimeSeries
        concatenated series
    """

    time_dims = [ts.time_dim for ts in series]
    if isinstance(axis, str):
        if axis == DIMS[1]:
            axis = 1
        elif axis == DIMS[2]:
            axis = 2
        else:
            raise_if_not(len(set(time_dims)) == 1 and axis == time_dims[0],
                         'Unrecognised `axis` name. If `axis` denotes the time axis, all provided '
                         'series must have the same time axis name (if that is not the case, try providing '
                         '`axis=0` to concatenate along time dimension).')
            axis = 0
    time_dim_name = time_dims[0]  # At this point all series are supposed to have same time dim name

    da_sequence = [ts.data_array(copy=False) for ts in series]

    component_axis_equal = len(set([ts.width for ts in series])) == 1
    sample_axis_equal = len(set([ts.n_samples for ts in series])) == 1

    if axis == 0:
        # time
        raise_if((not (component_axis_equal and sample_axis_equal)),
                 'when concatenating along time dimension, the component and sample dimensions of all '
                 'provided series must match.')

        da_concat = xr.concat(da_sequence, dim=time_dim_name)

        # check, if timeseries are consecutive
        consecutive_time_axes = True
        for i in range(1, len(series)):
            if series[i - 1].end_time() + series[0].freq != series[i].start_time():
                consecutive_time_axes = False
                break

        if not consecutive_time_axes:
            raise_if_not(ignore_time_axis, "When concatenating over time axis, all series need to be contiguous"
                                           "in the time dimension. Use `ignore_time_axis=True` to override "
                                           "this behavior and concatenate the series by extending the time axis "
                                           "of the first series.")

            from darts.utils.timeseries_generation import _generate_index
            tindex = _generate_index(start=series[0].start_time(), freq=series[0].freq_str, length=da_concat.shape[0])

            da_concat = da_concat.assign_coords({time_dim_name: tindex})

    else:
        time_axes_equal = all(list(map(lambda t: t[0].has_same_time_as(t[1]), zip(series[0:-1], series[1:]))))
        time_axes_ok = (time_axes_equal if not ignore_time_axis else len(set([len(ts) for ts in series])) == 1)

        raise_if_not((time_axes_ok and ((axis == 1 and sample_axis_equal) or (axis == 2 and component_axis_equal))),
                     'When concatenating along component or sample dimensions, all the series must have the same time '
                     'axes (unless `ignore_time_axis` is True), or time axes of same lengths (if `ignore_time_axis` is '
                     'True), and all series must have the same number of samples (if concatenating along component '
                     'dimension), or the same number of components (if concatenating along sample dimension).')

        # we concatenate raw values using Numpy because not all series might have the same time axes
        # and joining using xarray.concatenate() won't work in some cases
        concat_vals = np.concatenate([da.values for da in da_sequence], axis=axis)

        if axis == 1:
            # when concatenating along component dimension, we have to re-create a component index
            component_coords = []
            existing_components = set()
            for i, ts in enumerate(series):
                for comp in ts.components:
                    if comp not in existing_components:
                        component_coords.append(comp)
                        existing_components.add(comp)
                    else:
                        new_comp_name = '{}_{}'.format(i, comp)
                        component_coords.append(new_comp_name)
                        existing_components.add(new_comp_name)
            component_index = pd.Index(component_coords)
        else:
            component_index = da_sequence[0].get_index(DIMS[1])

        da_concat = xr.DataArray(concat_vals,
                                 dims=(time_dim_name,) + DIMS[-2:],
                                 coords={time_dim_name: series[0].time_index, DIMS[1]: component_index})

    return TimeSeries(da_concat)
