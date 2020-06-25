"""
Timeseries
----------

`TimeSeries` is the main class in `darts`. It represents a univariate time series,
possibly with lower and upper confidence bounds.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any, List, Union

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
            Optionally, a string representing the frequency of the Pandas DataFrame. When creating a TimeSeries
            instance with a length smaller than 3, this argument must be passed.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.
        """

        raise_if_not(isinstance(df, pd.DataFrame), "Data must be provided in form of a pandas.DataFrame instance",
                     logger)

        # consistent column names
        df.columns = range(df.shape[1])

        raise_if_not(len(df) > 0 and df.shape[1] > 0, 'Time series must not be empty.', logger)
        raise_if_not(isinstance(df.index, pd.DatetimeIndex), 'Time series must be indexed with a DatetimeIndex.',
                     logger)
        raise_if_not(df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), 'Time series must'
                     ' contain only numerical values.', logger)
        raise_if_not(len(df) >= 3 or freq is not None, 'Time series must have at least 3 values if the "freq" argument'
                     'is not passed', logger)

        self._df = df.sort_index()  # Sort by time

        if (len(df) < 3):
            self._freq: str = freq
            logger.info('A TimeSeries with length below 3 is being created. Please note that this can lead to'
                        ' unexpected behavior.')
        else:
            if not df.index.inferred_freq:
                if fill_missing_dates:
                    self._df = self._fill_missing_dates(self._df)
                else:
                    raise_if_not(False, 'Could not infer frequency. Are some dates missing? '
                                        'Try specifying `fill_missing_dates=True`.', logger)
            self._freq: str = self._df.index.inferred_freq  # Infer frequency
            raise_if_not(freq is None or self._freq == freq, 'The inferred frequency does not match the'
                         'value of the "freq" argument.', logger)

        self._df.index.freq = self._freq  # Set the inferred frequency in the Pandas dataframe

        # The actual values
        self._values: np.ndarray = self._df.values

    def _assert_univariate(self):
        """
        Raises an error if the current TimeSeries instance is not univariate.
        """
        if (self._df.shape[1] != 1):
            raise_log(AssertionError('Only univariate TimeSeries instances support this method'), logger)

    def pd_series(self) -> pd.Series:
        """
        Returns
        -------
        pandas.Series
            A Pandas Series representation of this univariate time series.
        """
        self._assert_univariate()
        return self._df.iloc[:, 0].copy()

    def pd_dataframe(self) -> pd.DataFrame:
        """
        Returns
        -------
        pandas.DataFrame
            A copy of the Pandas Dataframe underlying this time series
        """
        return self._df.copy()

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

    def values(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            A copy of the values composing the time series
        """
        return np.copy(self._values)

    def univariate_values(self) -> np.ndarray:
        """
        Returns
        -------
        numpy.ndarray
            A copy of the values composing the time series guaranteed to be univariate.
        """
        self._assert_univariate()
        return np.copy(self._df.iloc[:, 0].values)

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
            return TimeSeries(self.pd_dataframe(), self.freq())
        else:
            return TimeSeries(self._df, self.freq())

    def _raise_if_not_within(self, ts: pd.Timestamp):
        if (ts < self.start_time()) or (ts > self.end_time()):
            raise_log(ValueError('Timestamp must be between {} and {}'.format(self.start_time(),
                                                                              self.end_time())), logger)

    def split_after(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, around a provided timestamp `ts`.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        first of the two TimeSeries, and not in the second.

        Parameters
        ----------
        ts
            The timestamp that indicates the splitting time.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series is before `ts`, and the second one is after `ts`.
        """
        self._raise_if_not_within(ts)
        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)
        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def split_before(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits a TimeSeries in two, around a provided timestamp `ts`.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        second of the two TimeSeries, and not in the first.

        Parameters
        ----------
        ts
            The timestamp that indicates the splitting time.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries]
            A tuple of two time series. The first time series is before `ts`, and the second one is after `ts`.
        """
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

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> 'TimeSeries':
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
        return TimeSeries(_slice_not_none(self._df), self.freq())

    def slice_n_points_after(self, start_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than `start_ts` (included) and having at most `n` points.

        The timestamp may not be in the time series. If it is, it will be included in the new TimeSeries.

        Parameters
        ----------
        start_ts
            The timestamp that indicates the splitting time.
        n
            The maximal length of the new TimeSeries.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n` and indices greater or equal than `start_ts`.
        """
        raise_if_not(n >= 0, 'n should be a positive integer.', logger)  # TODO: logically raise if n<3, cf. init
        self._raise_if_not_within(start_ts)
        start_ts = self.time_index()[self.time_index() >= start_ts][0]  # closest index after start_ts (new start_ts)
        end_ts: pd.Timestamp = start_ts + (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def slice_n_points_before(self, end_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending before `end_ts` (included) and having at most `n` points.

        The timestamp may not be in the TimeSeries. If it is, it will be included in the new TimeSeries.

        Parameters
        ----------
        end_ts
            The timestamp that indicates the splitting time.
        n
            The maximal length of the new time series.

        Returns
        -------
        TimeSeries
            A new TimeSeries, with length at most `n` and indices smaller or equal than `end_ts`.
        """
        raise_if_not(n >= 0, 'n should be a positive integer.', logger)
        self._raise_if_not_within(end_ts)
        end_ts = self.time_index()[self.time_index() <= end_ts][-1]
        start_ts: pd.Timestamp = end_ts - (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

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
        return TimeSeries(new_series, self.freq())

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
        try:
            self.time_index()[-1] + n * self.freq()
        except pd.errors.OutOfBoundsDatetime:
            raise_log(OverflowError("the add operation between {} and {} will "
                                    "overflow".format(n * self.freq(), self.time_index()[-1])), logger)
        new_time_index = self._df.index.map(lambda ts: ts + n * self.freq())
        new_series = self._df.copy()
        new_series.index = new_time_index
        return TimeSeries(new_series, self.freq())

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
                       time_col: Optional[str],
                       value_cols: Union[List[str], str],
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
            A string or list of strings representing the value column(s) to be extracted from the DataFrame.
        freq
            Optionally, a string representing the frequency of the Pandas DataFrame.
        fill_missing_dates
            Optionally, a boolean value indicating whether to fill missing dates with NaN values
            in case the frequency of `series` cannot be inferred.

        Returns
        -------
        TimeSeries
            A univariate TimeSeries constructed from the inputs.
        """

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
                              values: np.ndarray,
                              freq: Optional[str] = None,
                              fill_missing_dates: Optional[bool] = True) -> 'TimeSeries':
        """
        Returns a TimeSeries built from an index and values.

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

        Returns
        -------
        TimeSeries
            A TimeSeries constructed from the inputs.
        """
        df = pd.DataFrame(values, index=times)

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
        return TimeSeries(series, self.freq())

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
        series = self._df.append(new_series)

        return TimeSeries(series, self.freq())

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
        series = values if values is None else pd.DataFrame(np.delete(values, ignored_indices, axis=0), index=index)
        raise_if_not(len(index) > 0, "Must give at least one correct index.", logger)

        new_series = self.pd_dataframe()
        if series is not None:
            new_series.update(series)
            new_series = new_series.astype(self._df.dtypes)
        return TimeSeries(new_series, self.freq())

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
        return TimeSeries(new_dataframe, self.freq())

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

        return TimeSeries.from_series(self.pd_dataframe().iloc[:, index], self.freq())

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

    @staticmethod
    def _combine_or_none(df_a: Optional[pd.DataFrame],
                         df_b: Optional[pd.DataFrame],
                         combine_fn: Callable[[pd.DataFrame, pd.DataFrame], Any]) -> Optional[pd.DataFrame]:
        """
        Combines two Pandas DataFrames `df_a and `df_b` using `combine_fn` if neither is `None`.

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

        series = combine_fn(self._df, other.pd_dataframe())
        return TimeSeries(series, self.freq())

    @staticmethod
    def _fill_missing_dates(series: pd.DataFrame) -> pd.DataFrame:
        """
        Tries to fill missing dates in series with NaN.
        Method is successful only when explicit frequency can be determined from all consecutive triple timestamps.

        Parameters
        ----------
        series
            The actual time series, as a pandas DataFrame with a proper time index.

        Returns
        -------
        pandas.Series
            A new Pandas DataFrame without missing dates.
        """
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

        inferred_frequency = observed_frequencies.pop()
        return series.resample(inferred_frequency).asfreq(fill_value=None)

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
            return TimeSeries(new_series, self.freq())
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
            return TimeSeries(new_series, self.freq())
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
            return TimeSeries(new_series, self.freq())
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
            return TimeSeries(new_series, self.freq())
        else:
            raise_log(TypeError('unsupported operand type(s) for ** or pow(): \'{}\' and \'{}\'.'
                                .format(type(self).__name__, type(n).__name__)), logger)

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.integer, np.float)):
            if (other == 0):
                raise_log(ZeroDivisionError('Cannot divide by 0.'), logger)

            new_series = self._df / other
            return TimeSeries(new_series, self.freq())

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
        return TimeSeries(series, self.freq())

    def __neg__(self):
        series = -self._df
        return TimeSeries(series, self.freq())

    def __contains__(self, ts: pd.Timestamp) -> bool:
        return ts in self._df.index

    def __round__(self, n=None):
        series = self._df.round(n)
        return TimeSeries(series, self.freq())

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

    # TODO: also support integer 0-D and 1-D indexing
    def __getitem__(self, item):
        # return only main series if nb of values < 3
        if isinstance(item, (int, pd.Timestamp)):
            return self._df.loc[[item]]
        elif isinstance(item, (pd.DatetimeIndex, slice, list, np.ndarray)):
            if isinstance(item, slice):
                # if create a slice with timestamp, convert to indices
                if item.start.__class__ == pd.Timestamp or item.stop.__class__ == pd.Timestamp:
                    istart = None if item.start is None else self.time_index().get_loc(item.start)
                    istop = None if item.stop is None else self.time_index().get_loc(item.stop)
                    item = slice(istart, istop, item.step)
                elif item.start.__class__ == str or item.stop.__class__ == str:
                    istart = None if item.start is None else self.time_index().get_loc(pd.Timestamp(item.start))
                    istop = None if item.stop is None else self.time_index().get_loc(pd.Timestamp(item.stop))
                    item = slice(istart, istop, item.step)
                # cannot reverse order
                if item.indices(len(self))[-1] == -1:
                    raise_log(IndexError("Cannot have a backward TimeSeries"), logger)

                if (isinstance(item.start, int) or isinstance(item.stop, int)):
                    item = self._df.index[item]

            # Verify that values in item are really in index to avoid the creation of NaN values
            if isinstance(item, (np.ndarray, pd.DatetimeIndex)):
                check = np.array([elem in self.time_index() for elem in item])
                if not np.all(check):
                    raise_log(IndexError("None of {} in the index".format(item[~check])), logger)

            return TimeSeries(self._df.loc[item, :], self.freq())
        elif isinstance(item, str):
            return self._df[[pd.Timestamp(item)]]
        else:
            raise_log(IndexError("Input {} of class {} is not a possible key.\n Please use integers, "
                                 "pd.DateTimeIndex, arrays or slice".format(item, item.__class__)), logger)
