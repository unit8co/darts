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
            self._freq = None
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

    """ 
    Other methods
    =============
    """
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

    def all_values(self, copy=True):
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



