"""
Timeseries
----------
"""
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any

from .custom_logging import raise_log, raise_if_not, get_logger

logger = get_logger(__name__)

class TimeSeries:
    def __init__(self, series: pd.Series, confidence_lo: pd.Series = None, confidence_hi: pd.Series = None):
        """
        A TimeSeries is an immutable object, representing a univariate time series, and optional confidence intervals.

        It is defined by the following three components:

        :param series: The actual time series, as a pandas Series with a proper time index.
        :param confidence_lo: Optionally, a Pandas Series representing lower confidence interval.
        :param confidence_hi: Optionally, a Pandas Series representing upper confidence interval.

        Within this class, TimeSeries type annotations are 'TimeSeries'; see:
        https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
        """
    def __init__(self, series: pd.Series, confidence_lo: pd.Series = None, confidence_hi: pd.Series = None):

        raise_if_not(len(series) >= 3, 'Series must have at least three values.', logger)  # cannot create a timeseries with n<3 -> can add less than 3 elements with add function
        raise_if_not(isinstance(series.index, pd.DatetimeIndex), 'Series must be indexed with a DatetimeIndex.', logger)
        raise_if_not(np.issubdtype(series.dtype, np.number), 'Series must contain numerical values.', logger)


        self._series: pd.Series = series.sort_index()  # Sort by time
        self._freq: str = self._series.index.inferred_freq  # Infer frequency

        # TODO: optionally fill holes (including missing dates) - for now we assume no missing dates
        raise_if_not(self._freq is not None, 'Could not infer frequency. Are some dates missing? Is Series too short (n=2)?', logger)

        # TODO: are there some pandas Series where the line below causes issues?
        self._series.index.freq = self._freq  # Set the inferred frequency in the Pandas series

        # Handle confidence intervals:
        self._confidence_lo = None
        self._confidence_hi = None
        if confidence_lo is not None:
            self._confidence_lo = confidence_lo.sort_index()
            raise_if_not(len(self._confidence_lo) == len(self._series), 'Lower confidence interval must have same size as ' \
                                                                  'the main time series.', logger)
            raise_if_not((self._confidence_lo.index == self._series.index).all(), 'Lower confidence interval and main ' \
                                                                            'series must have the same time index.', logger)
        if confidence_hi is not None:
            self._confidence_hi = confidence_hi.sort_index()
            raise_if_not(len(self._confidence_hi) == len(self._series), 'Upper confidence interval must have same size as ' \
                                                                  'the main time series.', logger)
            raise_if_not((self._confidence_hi.index == self._series.index).all(), 'Upper confidence interval and main ' \
                                                                            'series must have the same time index.', logger)

    def pd_series(self) -> pd.Series:
        """
        Returns the underlying Pandas Series of this TimeSeries.

        :return: A Pandas Series.
        """
        return self._series.copy()

    def conf_lo_pd_series(self) -> Optional[pd.Series]:
        """
        Returns the underlying Pandas Series of the lower confidence interval if it exists.

        :return: A Pandas Series for the lower confidence interval.
        """
        return self._confidence_lo.copy() if self._confidence_lo is not None else None

    def conf_hi_pd_series(self) -> Optional[pd.Series]:
        """
        Returns the underlying Pandas Series of the upper confidence interval if it exists.

        :return: A Pandas Series for the upper confidence interval.
        """
        return self._confidence_hi.copy() if self._confidence_hi is not None else None

    def start_time(self) -> pd.Timestamp:
        """
        Returns the start time of the time index.

        :return: A timestamp containing the first time of the TimeSeries.
        """
        return self._series.index[0]

    def end_time(self) -> pd.Timestamp:
        """
        Returns the end time of the time index.

        :return: A timestamp containing the last time of the TimeSeries.
        """
        return self._series.index[-1]

    def values(self) -> np.ndarray:
        """
        Returns the values of the TimeSeries.

        :return: A numpy array containing the values of the TimeSeries.
        """
        return np.copy(self._series.values)

    def time_index(self) -> pd.DatetimeIndex:
        """
        Returns the index of the TimeSeries.

        :return: A DatetimeIndex containing the index of the TimeSeries.
        """
        return deepcopy(self._series.index)

    def freq(self) -> pd.DateOffset:
        """
        Returns the frequency of the TimeSeries.

        :return: A DateOffset with the frequency.
        """
        return to_offset(self._freq)

    def freq_str(self) -> str:
        """
        Returns the frequency of the TimeSeries.

        :return: A string with the frequency.
        """
        return self._freq

    def duration(self) -> pd.Timedelta:
        """
        Returns the duration of the TimeSeries.

        :return: A Timedelta of the duration of the TimeSeries.
        """
        return self._series.index[-1] - self._series.index[0]

    def copy(self, deep: bool = True) -> 'TimeSeries':
        """
        Make a copy of this object time series
        :param deep: Make a deep copy. If False, the Series will be the same
        :return: A copy of the TimeSeries
        """
        if deep:
            return TimeSeries(self.pd_series(), self.conf_lo_pd_series(), self.conf_hi_pd_series())
        else:
            return TimeSeries(self._series, self._confidence_lo, self._confidence_hi)

    def _raise_if_not_within(self, ts: pd.Timestamp):
        if (ts < self.start_time()) or (ts > self.end_time()):
            raise_log(ValueError('Timestamp must be between {} and {}'.format(self.start_time(), self.end_time())), logger)

    def split_after(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, around a provided timestamp [ts].
        
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        first of the two TimeSeries, and not in the second.
        
        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller or equal to [ts]
                 and greater than [ts] respectively.
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def split_before(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits a TimeSeries in two, around a provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        second of the two TimeSeries, and not in the first.

        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller than [ts]
                 and greater or equal to [ts] respectively.
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_first_series: pd.Timestamp = ts - self.freq()  # second series does not include ts
        return self.slice(self.start_time(), end_first_series), self.slice(ts, self.end_time())

    def drop_after(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything after the provided timestamp [ts], included.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices smaller than [ts].
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_series: pd.Timestamp = ts - self.freq()  # new series does not include ts
        return self.slice(self.start_time(), end_series)

    def drop_before(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything before the provided timestamp [ts], included.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices greater than [ts].
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_series: pd.Timestamp = ts + self.freq()  # new series does not include ts
        return self.slice(start_series, self.end_time())

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than [start_ts] and ending before [end_ts], inclusive on both ends.

        The timestamps may not be in the time series. TODO: should new timestamps be added? Think not

        :param start_ts: The timestamp that indicates the left cut-off.
        :param end_ts: The timestamp that indicates the right cut-off.
        :return: A new TimeSeries, which indices greater or equal than [start_ts] and smaller or equal than [end_ts].
        """
        raise_if_not(end_ts > start_ts, 'End timestamp must be strictly after start timestamp when slicing.', logger)
        raise_if_not(end_ts >= self.start_time(), 'End timestamp must be after the start of the time series when slicing.', logger)
        raise_if_not(start_ts <= self.end_time(), 'Start timestamp must be after the end of the time series when slicing.', logger)

        def _slice_not_none(s: Optional[pd.Series]) -> Optional[pd.Series]:
            if s is not None:
                s_a = s[s.index >= start_ts]
                return s_a[s_a.index <= end_ts]
            return None

        return TimeSeries(_slice_not_none(self._series),
                          _slice_not_none(self._confidence_lo),
                          _slice_not_none(self._confidence_hi))

    def slice_n_points_after(self, start_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than [start_ts] (included) and having (at most) [n] points.

        The timestamp may not be in the time series. If it is, it will be included in the new TimeSeries.

        :param start_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new TimeSeries.
        :return: A new TimeSeries, with length at most [n] and indices greater or equal than [start_ts].
        """

        raise_if_not(n >= 0, 'n should be a positive integer.', logger)  # TODO: logically raise if n<3, cf. init

        self._raise_if_not_within(start_ts)

        start_ts = self.time_index()[self.time_index() >= start_ts][0]  # closest index after start_ts (new start_ts)

        end_ts: pd.Timestamp = start_ts + (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def slice_n_points_before(self, end_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending before [end_ts] (included) and having (at most) [n] points.

        The timestamp may not be in the TimeSeries. If it is, it will be included in the new TimeSeries.

        :param end_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new time series.
        :return: A new TimeSeries, with length at most [n] and indices smaller or equal than [end_ts].
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
        :param other: the other TimeSeries
        :return: a new TimeSeries, containing the values of this TimeSeries,
                 over the time-span common to both time series.
        """
        time_index = self.time_index().intersection(other.time_index())
        return self.__getitem__(time_index)

    # TODO: other rescale? such as giving a ratio, or a specific position? Can be the same function
    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Returns a new TimeSeries, which is a multiple of this TimeSeries such that
        the first value is [value_at_first_step].
        Numerical imprecisions appear with [value_at_first_step] > 1e+24

        :param value_at_first_step: The new value for the first entry of the TimeSeries.
        :return: A new TimeSeries, where the first value is [value_at_first_step] and other values
                 have been scaled accordingly.
        """
        raise_if_not(self.values()[0] != 0, 'Cannot rescale with first value 0.', logger)

        coef = value_at_first_step / self.values()[0]  # TODO: should the new TimeSeries have the same dtype?
        new_series = coef * self._series
        new_conf_lo = self._op_or_none(self._confidence_lo, lambda s: s * coef)
        new_conf_hi = self._op_or_none(self._confidence_hi, lambda s: s * coef)
        return TimeSeries(new_series, new_conf_lo, new_conf_hi)

    def shift(self, n: int) -> 'TimeSeries':
        """
        Shifts the time axis of this TimeSeries by [n] time steps.

        If n > 0, shifts in the future. If n < 0, shifts in the past.

        For example, with n=2 and freq='M', March 2013 becomes May 2013. With n=-2, March 2013 becomes Jan 2013.

        :param n: The signed number of time steps to shift by.
        :return: A new TimeSeries, with a shifted index.
        """
        try:
            self.time_index()[-1] + n * self.freq()
        except pd.errors.OutOfBoundsDatetime:
            raise_log(OverflowError("the add operation between {} and {} will overflow".format(n * self.freq(),
                                                                                           self.time_index()[-1])), logger)
        new_time_index = self._series.index.map(lambda ts: ts + n * self.freq())
        new_series = self._series.copy()
        new_series.index = new_time_index
        new_conf_lo = None
        new_conf_hi = None
        if self._confidence_lo is not None:
            new_conf_lo = self._confidence_lo.copy()
            new_conf_lo.index = new_time_index
        if self._confidence_hi is not None:
            new_conf_hi = self._confidence_hi.copy()
            new_conf_hi.index = new_time_index
        return TimeSeries(new_series, new_conf_lo, new_conf_hi)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, time_col: Optional[str], value_col: str,
                       conf_lo_col: str = None, conf_hi_col: str = None) -> 'TimeSeries':
        """
        Returns a TimeSeries built from a DataFrame. One column (or the DataFrame index) has to represent the time,
        and another column has to represent the values for this univariate time series.

        :param df: The DataFrame
        :param time_col: The time column name (mandatory). If set to `None`, the DataFrame index will be used.
        :param value_col: The value column name (mandatory).
        :param conf_lo_col: The lower confidence interval column name (optional).
        :param conf_hi_col: The upper confidence interval column name (optional).
        :return: A TimeSeries constructed from the inputs.
        """
        if time_col is None:
            times: pd.DatetimeIndex = pd.to_datetime(df.index, errors='raise')
        else:
            times: pd.Series = pd.to_datetime(df[time_col], errors='raise')
        series: pd.Series = pd.Series(df[value_col].values, index=times)

        conf_lo = pd.Series(df[conf_lo_col], index=times) if conf_lo_col is not None else None
        conf_hi = pd.Series(df[conf_hi_col], index=times) if conf_hi_col is not None else None

        return TimeSeries(series, conf_lo, conf_hi)

    @staticmethod
    def from_times_and_values(times: pd.DatetimeIndex,
                              values: np.ndarray,
                              confidence_lo: np.ndarray = None,
                              confidence_hi: np.ndarray = None) -> 'TimeSeries':
        """
        Returns a TimeSeries built from an index and values.

        :param times: A DateTimeIndex for the TimeSeries.
        :param values: An array of values for the TimeSeries.
        :param confidence_lo: The lower confidence interval values (optional).
        :param confidence_hi: The higher confidence interval values (optional).
        :return: A TimeSeries constructed from the inputs.
        """
        series = pd.Series(values, index=times)
        series_lo = pd.Series(confidence_lo, index=times) if confidence_lo is not None else None
        series_hi = pd.Series(confidence_hi, index=times) if confidence_hi is not None else None

        return TimeSeries(series, series_lo, series_hi)

    def plot(self, plot_ci=True, new_plot=False, *args, **kwargs):
        """
        Currently this is just a wrapper around pd.Series.plot()
        """
        fig = (plt.figure() if new_plot else (kwargs['figure'] if 'figure' in kwargs else plt.gcf()))
        kwargs['figure'] = fig
        self._series.plot(*args, **kwargs)
        x_label = self.time_index().name
        if x_label is not None and len(x_label) > 0:
            plt.xlabel(x_label)
        # TODO: use pandas plot in the future
        if plot_ci and self._confidence_lo is not None and self._confidence_hi is not None:
            plt.fill_between(self.time_index(), self._confidence_lo.values, self._confidence_hi.values, alpha=0.5)

    """
    Some useful methods for TimeSeries combination:
    """
    def has_same_time_as(self, other: 'TimeSeries') -> bool:
        """
        Checks whether this TimeSeries and another one have the same index.

        :param other: A second TimeSeries.
        :return: A boolean. True if both TimeSeries have the same index, False otherwise.
        """

        if self.__len__() != len(other):
            return False
        return (other.time_index() == self.time_index()).all()

    def append(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Appends another TimeSeries to this TimeSeries.

        :param other: A second TimeSeries.
        :return: A new TimeSeries, obtained by appending the second TimeSeries to the first.
        """
        raise_if_not(other.start_time() == self.end_time() + self.freq(), 'Appended TimeSeries must start one time step ' \
                                                                    'after current one.', logger)
        # TODO additional check?
        raise_if_not(other.freq() == self.freq(), 'Appended TimeSeries must have the same frequency as the current one', logger)

        series = self._series.append(other.pd_series())
        conf_lo = None
        conf_hi = None
        if self._confidence_lo is not None and other.conf_lo_pd_series() is not None:
            conf_lo = self._confidence_lo.append(other.conf_lo_pd_series())
        if self._confidence_hi is not None and other.conf_hi_pd_series() is not None:
            conf_hi = self._confidence_hi.append(other.conf_hi_pd_series())
        return TimeSeries(series, conf_lo, conf_hi)

    def append_values(self, values: np.ndarray, index: pd.DatetimeIndex = None,
            conf_lo: np.ndarray = None, conf_hi: np.ndarray = None) -> 'TimeSeries':
        """
        Appends values to current TimeSeries, to the given indices.

        If no index is provided, assumes that it follows the original data.
        Does not add new confidence values if there were none first.
        Does not update value if already existing indices are provided.

        :param values: An array with the values to append.
        :param index: A DateTimeIndex for each value (optional).
        :param conf_lo: The lower confidence interval values (optional).
        :param conf_hi: The higher confidence interval values (optional).
        :return: A new TimeSeries with the new values appended
        """
        if len(values) < 1:
            return self
        if isinstance(values, list):
            values = np.array(values)
        if index is None:
            index = pd.DatetimeIndex([self.end_time() + i * self.freq() for i in range(1, 1+len(values))])
        raise_if_not(isinstance(index, pd.DatetimeIndex), 'Values must be indexed with a DatetimeIndex.', logger)
        raise_if_not(len(index) == len(values), 'Values and index must have same length.', logger)
        raise_if_not(self.time_index().intersection(index).empty, "Cannot add already present time index.", logger)
        new_indices = index.argsort()
        index = index[new_indices]
        # TODO do we really want that?
        raise_if_not(index[0] == self.end_time() + self.freq(), 'Appended index must start one time step ' \
                                                          'after current one.', logger)
        if len(index) > 2:
            raise_if_not(index.inferred_freq == self.freq_str(), 'Appended index must have ' \
                                                           'the same frequency as the current one.', logger)
        elif len(index) == 2:
            raise_if_not(index[-1] == index[0] + self.freq(), 'Appended index must have ' \
                                                        'the same frequency as the current one.', logger)
        values = values[new_indices]
        new_series = pd.Series(values, index=index)
        series = self._series.append(new_series)
        if conf_lo is not None and self._confidence_lo is not None:
            raise_if_not(len(index) == len(conf_lo), 'Confidence intervals must have same length as index.', logger)
            conf_lo = conf_lo[new_indices]
            conf_lo = self._confidence_lo.append(pd.Series(conf_lo, index=index))
        if conf_hi is not None and self._confidence_hi is not None:
            raise_if_not(len(index) == len(conf_hi), 'Confidence intervals must have same length as index.', logger)
            conf_hi = conf_hi[new_indices]
            conf_hi = self._confidence_hi.append(pd.Series(conf_hi, index=index))

        return TimeSeries(series, conf_lo, conf_hi)

    def update(self, index: pd.DatetimeIndex, values: np.ndarray = None,
               conf_lo: np.ndarray = None, conf_hi: np.ndarray = None, inplace: bool = True) -> 'TimeSeries':
        """
        Updates the Series with the new values provided.
        If indices are not in original TimeSeries, they will be discarded.
        At least one parameter other than index must be filled.
        Use np.nan to ignore a specific index in a series.

        It will raise an error if try to update a missing CI series

        :param index: A DateTimeIndex containing the indices to replace.
        :param values: An array containing the values to replace (optional).
        :param conf_lo: The lower confidence interval values to change (optional).
        :param conf_hi: The higher confidence interval values (optional).
        :param inplace: If True, do operation inplace and return None, defaults to True.
        :return: A TimeSeries with values updated

        TODO: Do we need this method? Where/how is it used? We should avoid mutating values at all cost.
        """
        raise_if_not(not (values is None and conf_lo is None and conf_hi is None), "At least one parameter must be filled " \
                                                                             "other than index", logger)
        raise_if_not(not index is None, "Index must be filled.")                                                            
        if (not values is None):
            raise_if_not(len(values) == len(index), \
                "The number of values must correspond to the number of indices: {} != {}".format(len(values), len(index)), logger)
        if (not conf_lo is None):
            raise_if_not(len(conf_lo) == len(index), \
                "The number of values must correspond to the number of indices: {} != {}".format(len(conf_lo), len(index)), logger)
        if (not conf_hi is None):
            raise_if_not(len(conf_hi) == len(index), \
                "The number of values must correspond to the number of indices: {} != {}".format(len(conf_hi), len(index)), logger)
        ignored_indices = [index.get_loc(ind) for ind in (set(index)-set(self.time_index()))]
        index = index.delete(ignored_indices)
        series = values if values is None else pd.Series(np.delete(values, ignored_indices), index=index)
        conf_lo = conf_lo if conf_lo is None else pd.Series(np.delete(conf_lo, ignored_indices), index=index)
        conf_hi = conf_hi if conf_hi is None else pd.Series(np.delete(conf_hi, ignored_indices), index=index)
        raise_if_not(len(index) > 0, "Must give at least one correct index.", logger)
        if inplace:
            if series is not None:
                self._series.update(series)
            if conf_lo is not None:
                self._confidence_lo.update(conf_lo)
            if conf_hi is not None:
                self._confidence_hi.update(conf_hi)
            return None
        else:
            new_series = self.pd_series()
            new_lo = self.conf_lo_pd_series()
            new_hi = self.conf_hi_pd_series()
            if series is not None:
                new_series.update(series)
            if conf_lo is not None:
                new_lo.update(conf_lo)
            if conf_hi is not None:
                new_hi.update(conf_hi)
            return TimeSeries(new_series, new_lo, new_hi)

    def drop_values(self, index: pd.DatetimeIndex, inplace: bool = True, **kwargs):
        """
        Remove elements of all series with specified indices.

        :param index: The indices to be dropped
        :param kwargs: Option to pass to pd.Series drop method
        :param inplace: If True, do operation inplace and return None, defaults to True.
        :return: A TimeSeries with values dropped

        TODO: Do we need this method? Where/how is it used? We should avoid mutating values at all cost.
        """
        series = self._series.drop(index=index, inplace=inplace, **kwargs)
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: s.drop(index, inplace=inplace, **kwargs))
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: s.drop(index, inplace=inplace, **kwargs))
        if inplace:
            return None
        return TimeSeries(series, conf_lo, conf_hi)

    def is_within_range(self, ts: pd.Timestamp) -> bool:
        index = self.time_index()
        return index[0] <= ts <= index[-1]

    @staticmethod
    def _combine_or_none(series_a: Optional[pd.Series],
                         series_b: Optional[pd.Series],
                         combine_fn: Callable[[pd.Series, pd.Series], Any]):
        """
        Combines two Pandas Series [series_a] and [series_b] using [combine_fn] if neither is None.

        :param series_a: A Pandas Series.
        :param series_b: A Pandas Series.
        :param combine_fn: An operation with input two Pandas Series and output one Pandas Series.
        :return: A new Pandas Series, the result of [combine_fn], or None.
        """

        if series_a is not None and series_b is not None:
            return combine_fn(series_a, series_b)
        return None

    @staticmethod
    def _op_or_none(series: Optional[pd.Series], op: Callable[[pd.Series], Any]):
        return op(series) if series is not None else None

    def _combine_from_pd_ops(self, other: 'TimeSeries',
                             combine_fn: Callable[[pd.Series, pd.Series], pd.Series]) -> 'TimeSeries':
        """
        Combines this TimeSeries with another one, using the [combine_fn] on the underlying Pandas Series.

        :param other: A second TimeSeries.
        :param combine_fn: An operation with input two Pandas Series and output one Pandas Series.
        :return: A new TimeSeries, with underlying Pandas Series the series obtained with [combine_fn].
        """

        raise_if_not(self.has_same_time_as(other), 'The two TimeSeries must have the same time index.', logger)

        series = combine_fn(self._series, other.pd_series())
        conf_lo = self._combine_or_none(self._confidence_lo, other.conf_lo_pd_series(), combine_fn)
        conf_hi = self._combine_or_none(self._confidence_hi, other.conf_hi_pd_series(), combine_fn)
        return TimeSeries(series, conf_lo, conf_hi)

    """
    Definition of some useful statistical methods.
    
    These methods rely on the Pandas implementation.
    """

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.mean(axis, skipna, level, numeric_only, **kwargs)

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._series.var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._series.std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.skew(axis, skipna, level, numeric_only, **kwargs)

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.kurtosis(axis, skipna, level, numeric_only, **kwargs)

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.min(axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.max(axis, skipna, level, numeric_only, **kwargs)

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs) -> float:
        return self._series.sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.median(axis, skipna, level, numeric_only, **kwargs)

    def autocorr(self, lag=1) -> float:
        return self._series.autocorr(lag)

    def describe(self, percentiles=None, include=None, exclude=None) -> pd.Series:
        return self._series.describe(percentiles, include, exclude)

    """
    Definition of some dunder methods
    """
    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            if not self._series.equals(other.pd_series()):
                return False
            for other_ci, self_ci in zip([other.conf_lo_pd_series(), other.conf_hi_pd_series()],
                                         [self._confidence_lo, self._confidence_hi]):
                if (other_ci is None) ^ (self_ci is None):
                    # only one is None
                    return False
                if self._combine_or_none(self_ci, other_ci, lambda s1, s2: s1.equals(s2)) is False:
                    # Note: we check for "False" explicitly, because None is OK..
                    return False
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._series)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series + other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s + other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s + other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 + s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for + or add(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series - other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s - other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s - other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 - s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for - or sub(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series * other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s * other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s * other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 * s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for * or mul(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float)):
            if n < 0 and not all(self.values() != 0):
                raise_log(ZeroDivisionError('Cannot divide by a TimeSeries with a value 0.'), logger)

            new_series = self._series ** float(n)
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s ** float(n))
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s ** float(n))
            return TimeSeries(new_series, conf_lo, conf_hi)
        else:
            raise_log(TypeError('unsupported operand type(s) for ** or pow(): \'{}\' and \'{}\'.' \
                            .format(type(self).__name__, type(n).__name__)), logger)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if (other == 0):
                raise_log(ZeroDivisionError('Cannot divide by 0.'), logger)

            new_series = self._series / other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s / other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s / other)
            return TimeSeries(new_series, conf_lo, conf_hi)

        elif isinstance(other, TimeSeries):
            if (not all(other.values() != 0)):
                raise_log(ZeroDivisionError('Cannot divide by a TimeSeries with a value 0.'), logger)

            return self._combine_from_pd_ops(other, lambda s1, s2: s1 / s2)
        else:
            raise_log(TypeError('unsupported operand type(s) for / or truediv(): \'{}\' and \'{}\'.' \
                            .format(type(self).__name__, type(other).__name__)), logger)

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        series = abs(self._series)
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: abs(s))
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: abs(s))
        return TimeSeries(series, conf_lo, conf_hi)

    def __neg__(self):
        series = -self._series
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: -s)
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: -s)
        return TimeSeries(series, conf_lo, conf_hi)

    def __contains__(self, ts: pd.Timestamp) -> bool:
        return ts in self._series.index

    def __round__(self, n=None):
        series = self._series.round(n)
        confidence_lo = self._op_or_none(self._confidence_lo, lambda s: s.round(n))
        confidence_hi = self._op_or_none(self._confidence_hi, lambda s: s.round(n))
        return TimeSeries(series, confidence_lo, confidence_hi)

    # TODO: Ignoring confidence series for now
    def __lt__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series < other
        elif isinstance(other, TimeSeries):
            series = self._series < other.pd_series()
        else:
            raise_log(TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)
        return series  # TODO should we return only the ndarray, the pd series, or our timeseries?

    def __gt__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series > other
        elif isinstance(other, TimeSeries):
            series = self._series > other.pd_series()
        else:
            raise_log(TypeError('unsupported operand type(s) for > : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __le__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series <= other
        elif isinstance(other, TimeSeries):
            series = self._series <= other.pd_series()
        else:
            raise_log(TypeError('unsupported operand type(s) for <= : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __ge__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series >= other
        elif isinstance(other, TimeSeries):
            series = self._series >= other.pd_series()
        else:
            raise_log(TypeError('unsupported operand type(s) for >= : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__)), logger)
        return series

    def __str__(self):
        df = pd.DataFrame({'value': self._series})
        if self._confidence_lo is not None:
            df['conf_low'] = self._confidence_lo
        if self._confidence_hi is not None:
            df['conf_high'] = self._confidence_hi
        return str(df) + '\nFreq: {}'.format(self.freq_str())

    def __repr__(self):
        return self.__str__()

    def __copy__(self, deep: bool = True):
        return self.copy(deep=deep)

    def __deepcopy__(self):
        return self.copy(deep=True)

    def __getitem__(self, item):
        # return only main series if nb of values < 3
        if isinstance(item, (int, pd.Timestamp)):
            return self._series[[item]]
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
            # Verify that values in item are really in index to avoid the creation of NaN values
            if isinstance(item, (np.ndarray, pd.DatetimeIndex)):
                check = np.array([elem in self.time_index() for elem in item])
                if not np.all(check):
                    raise_log(IndexError("None of {} in the index".format(item[~check])), logger)
            try:
                return TimeSeries(self._series[item],
                                  self._op_or_none(self._confidence_lo, lambda s: s[item]),
                                  self._op_or_none(self._confidence_hi, lambda s: s[item]))
            except ValueError:
                # return only main series if nb of values < 3
                return TimeSeries(self._series[item])
        elif isinstance(item, str):
            return self._series[[pd.Timestamp(item)]]
        else:
            raise_log(IndexError("Input {} of class {} is not a possible key.\n"\
                             "Please use integers, pd.DateTimeIndex, arrays or slice".format(item, item.__class__)), logger)
