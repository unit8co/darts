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

        time_index = xa.get_index(self._time_dim)
        if not isinstance(time_index, pd.DatetimeIndex) and not isinstance(time_index, pd.RangeIndex):
            raise_log(ValueError('The time dimension of the DataArray must be indexed either with a DatetimeIndex,'
                                 'or with a RangeIndex.'))

        self._xa: xr.DataArray = xa.sortby(self._time_dim)  # returns a copy
        if isinstance(self._xa.get_index(self._time_dim), pd.DatetimeIndex):
            self._freq: pd.DateOffset = self._xa.get_index(self._time_dim).freq
        else:
            self._freq = 1

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
    def freq(self):
        return self._freq



    """ Factory Methods
    """

    @staticmethod
    def from_xarray(xa, fill_missing_dates = None) -> 'TimeSeries':
        # most of the logic to interpolate, infer, etc. will go there
        # optionally fill missing dates
        # set name

        return TimeSeries(xa)

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       time_col: Optional[str] = None,
                       value_cols: Optional[Union[List[str], str]] = None,
                       fill_missing_dates: Optional[bool] = True) -> 'TimeSeries':
        # keep & clean column names, if any, then create xarray

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

        return TimeSeries.from_xarray(xa)

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
class DeterministicTimeSeries(TimeSeries):
    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)
"""
