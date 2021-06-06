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

        self._time_dim: str = xa.dims[0]  # how the time dimension is named
        self._time_index = xa.get_index(self._time_dim)

        if not isinstance(self._time_index, pd.DatetimeIndex) and not isinstance(self._time_index, pd.RangeIndex):
            raise_log(ValueError('The time dimension of the DataArray must be indexed either with a DatetimeIndex,'
                                 'or with a RangeIndex.'))

        self._xa: xr.DataArray = xa.sortby(self._time_dim)  # returns a copy

        self._has_datetime_index = isinstance(self._time_index, pd.DatetimeIndex)

        if self._has_datetime_index:
            self._freq: pd.DateOffset = self._time_index.freq
        else:
            self._freq = None

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
    def is_univariate(self):
        return self.n_components == 1

    @property
    def freq(self):
        return self._freq

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

