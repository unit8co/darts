import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose


class TimeSeries:
    def __init__(self, series: pd.Series, confidence_lo: pd.Series = None, confidence_hi: pd.Series = None):
        """
        A TimeSeries is an immutable object defined by the following three components:

        :param series: The actual time series, as a pandas Series with a proper time index.
        :param confidence_lo: Optionally, a Pandas Series representing lower confidence interval.
        :param confidence_hi: Optionally, a Pandas Series representing upper confidence interval.

        Within this class, TimeSeries type annotations are 'TimeSeries'; see:
        https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
        """

        assert len(series) >= 1, 'Series must have at least one value.'
        assert isinstance(series.index, pd.DatetimeIndex), 'Series must be indexed with a DatetimeIndex.'
        assert np.issubdtype(series.dtype, np.number), 'Series must contain numerical values.'

        self._series: pd.Series = series.sort_index()  # Sort by time
        self._freq: str = self._series.index.inferred_freq  # Infer frequency

        # TODO: optionally fill holes (including missing dates) - for now we assume no missing dates
        assert self._freq is not None, 'Could not infer frequency. Are some dates missing? Is Series too short (n=2)?'

        # TODO: are there some pandas Series where the line below causes issues?
        self._series.index.freq = self._freq  # Set the inferred frequency in the Pandas series

        # Handle confidence intervals:
        self._confidence_lo = None
        self._confidence_hi = None
        if confidence_lo is not None:
            self._confidence_lo = confidence_lo.sort_index()
            assert len(self._confidence_lo) == len(self._series), 'Lower confidence interval must have same size as ' \
                                                                  'the main time series.'
            assert (self._confidence_lo.index == self._series.index).all(), 'Lower confidence interval and main ' \
                                                                            'series must have the same time index.'
        if confidence_hi is not None:
            self._confidence_hi = confidence_hi.sort_index()
            assert len(self._confidence_hi) == len(self._series), 'Upper confidence interval must have same size as ' \
                                                                  'the main time series.'
            assert (self._confidence_hi.index == self._series.index).all(), 'Upper confidence interval and main ' \
                                                                            'series must have the same time index.'

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
        return self._series.values

    def time_index(self) -> pd.DatetimeIndex:
        """
        Returns the index of the TimeSeries.

        :return: A DatetimeIndex containing the index of the TimeSeries.
        """
        return self._series.index

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

    def split_after(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, around a provided timestamp [ts].
        
        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        first of the two TimeSeries, and not in the second.
        
        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller or equal to ts and greater than ts respectively.
        """

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def split_before(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits a TimeSeries in two, around a provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        second of the two TimeSeries, and not in the first.

        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller than ts and greater or equal to ts respectively.
        """

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_first_series: pd.Timestamp = ts - self.freq()  # second series does not include ts
        return self.slice(self.start_time(), end_first_series), self.slice(ts, self.end_time())

    def drop_end(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything after the provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices smaller or equal than [ts].
        """

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_series: pd.Timestamp = ts - self.freq()  # new series does not include ts
        return self.slice(self.start_time(), end_series)

    def drop_beginning(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything before the provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices greater than [ts].
        """

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_series: pd.Timestamp = ts + self.freq()  # new series does not include ts
        return self.slice(start_series, self.end_time())

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than [start_ts] and ending before [end_ts].

        The timestamps may not be in the time series. If any is, it will be included in the new time series.

        :param start_ts: The timestamp that indicates the left cut-off.
        :param end_ts: The timestamp that indicates the right cut-off.
        :return: A new TimeSeries, which indices greater or equal than [start_ts] and smaller or equal than [end_ts].
        """

        assert end_ts > start_ts, 'End timestamp must be strictly after start timestamp when slicing.'
        assert end_ts >= self.start_time(), 'End timestamp must be after the start of the time series when slicing.'
        assert start_ts <= self.end_time(), 'Start timestamp must be after the end of the time series when slicing.'

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
        Returns a new TimeSeries, starting later than [start_ts] and having (at most) [n] points.

        The timestamp may not be in the time series. If it is, it will be included in the new TimeSeries.

        :param start_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new TimeSeries.
        :return: A new TimeSeries, with length at most [n] and indices greater or equal than [start_ts].
        """

        assert n >= 0, 'n should be a positive integer.'

        start_ts = self.time_index()[self.time_index() >= start_ts][0]  # closest index after start_ts (new start_ts)

        end_ts: pd.Timestamp = start_ts + (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def slice_n_points_before(self, end_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending before [end_ts] and having (at most) [n] points.

        The timestamp may not be in the TimeSeries. If it is, it will be included in the new TimeSeries.

        :param end_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new time series.
        :return: A new TimeSeries, with length at most [n] and indices smaller or equal than [end_ts].
        """

        assert n >= 0, 'n should be a positive integer.'

        end_ts = self.time_index()[self.time_index() <= end_ts][-1]

        start_ts: pd.Timestamp = end_ts - (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def intersect(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Returns a slice containing the intersection of this TimeSeries and the one provided in argument.

        :param other: A second TimeSeries.
        :return: A new TimeSeries, with values of this TimeSeries and indices the intersection of both
                TimeSeries' indices.
        """

        def _intersect_not_none(s: Optional[pd.Series]) -> Optional[pd.Series]:
            if s is not None:
                new_index = self.time_index().intersection(other.time_index())
                return s[new_index]
            return None

        return TimeSeries(_intersect_not_none(self._series),
                          _intersect_not_none(self._confidence_lo),
                          _intersect_not_none(self._confidence_hi))

    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Returns a new TimeSeries, which is a multiple of this TimeSeries such that
        the first value is [value_at_first_step].

        :param value_at_first_step: The new value for the first entry of the TimeSeries.
        :return: A new TimeSeries, whose first value was changed to [value_at_first_step] and whose others values
                have been scaled accordingly.
        """

        assert self.values()[0] != 0, 'Cannot rescale with first value 0.'

        coef = value_at_first_step / self.values()[0]
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
    def from_dataframe(df: pd.DataFrame, time_col: str, value_col: str,
                       conf_lo_col: str = None, conf_hi_col: str = None) -> 'TimeSeries':
        """
        Returns a TimeSeries built from some DataFrame columns (ignoring DataFrame index)

        :param df: The DataFrame
        :param time_col: The time column name (mandatory).
        :param value_col: The value column name (mandatory).
        :param conf_lo_col: The lower confidence interval column name (optional).
        :param conf_hi_col: The upper confidence interval column name (optional).
        :return: A TimeSeries constructed from the inputs.
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

    def plot(self, *args, plot_ci=True, **kwargs):
        """
        Currently this is just a wrapper around pd.Series.plot()
        """
        # temporary work-around for the pandas.plot issue
        plt.plot(self.time_index(), self.values(), *args, **kwargs)
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

        :param other: As second TimeSeries.
        :return: A new TimeSeries, obtained by appending the second TimeSeries to the first.
        """

        assert other.start_time() == self.end_time() + self.freq(), 'Appended TimeSeries must start one time step ' \
                                                                    'after current one.'

        series = self._series.append(other.pd_series())
        conf_lo = None
        conf_hi = None
        if self._confidence_lo is not None and other.conf_lo_pd_series() is not None:
            conf_lo = self._confidence_lo.append(other.conf_lo_pd_series())
        if self._confidence_hi is not None and other.conf_hi_pd_series() is not None:
            conf_hi = self._confidence_hi.append(other.conf_hi_pd_series())
        return TimeSeries(series, conf_lo, conf_hi)

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

        assert self.has_same_time_as(other), 'The two TimeSeries must have the same time index.'

        series = combine_fn(self._series, other.pd_series())
        conf_lo = self._combine_or_none(self._confidence_lo, other.conf_lo_pd_series(), combine_fn)
        conf_hi = self._combine_or_none(self._confidence_hi, other.conf_hi_pd_series(), combine_fn)
        return TimeSeries(series, conf_lo, conf_hi)

    """
    Definition of some methods related to seasonality.
    """

    def check_seasonality(self, m: int = 0) -> Tuple[bool, int]:
        """
        Checks if the TimeSeries has a seasonality of order [m].

        :param m: An integer, the order of seasonality to check.
        :return: A tuple (a, b), where a indicates whether there is seasonality of order [m] or not and b = m
                unless the TimeSeries is constant.
        """
        n_unique = np.unique(self.values()).shape[0]
        if n_unique > 1:
            r = acf(self.values(), nlags=max([m, 24]))  # In case user-defined period exceeds 24
            if m == 0:  # Handles the case with unknown seasonality period
                grad = np.gradient(r)
                signs_changes = np.diff(np.sign(grad))
                m = np.nonzero((signs_changes == -2))[0][0] + 1  # +1 correction as np.nonzero returns an index.
            r = r[1:]  # Removes the auto-correlation with lag 0.
            stat = self.compute_stats(r, m, len(self))
            return (abs(r[m-1]) / stat) > norm.ppf(0.95), m
        return False, 0

    @staticmethod
    def compute_stats(r: np.ndarray, m: int, N: int) -> float:
        """
        Computes the standard error of [r] with order [m] for a sample of size [N] according to Bartlett's formula.

        :param r: A numpy array.
        :param m: An integer, the upper limit on the number of terms of r to use.
        :param length: An integer, the length of the sample to test.
        :return: A float, the standard error of [r] with respect to [m] and [length].
        """
        if m == 1:
            return math.sqrt(1/float(N))
        else:
            return math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r[:m - 1]))) / float(N))

    def seasonal_adjustment(self, model: str = 'multiplicative') -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Returns the TimeSeries, adjusted for seasonality and the seasonality TimeSeries.

        :param model: A string, the type of seasonality to consider (additive or multiplicative).
        :return: A Tuple (a,b) of two TimeSeries, where a is obtained from this TimeSeries after seasonal adjustment
                and b is the seasonality TimeSeries itself.
        """
        decomp = seasonal_decompose(self.pd_series(), model=model)
        seasonality = decomp.seasonal
        if (seasonality < 1e-6).any():
            print("WARNING seasonal indexes equal to zero, using non-seasonal Theta method.")
            ts = self
        else:
            if model == 'multiplicative':
                ts = TimeSeries.from_times_and_values(self.time_index(), self.values() / seasonality)
            else:
                ts = TimeSeries.from_times_and_values(self.time_index(), self.values() - seasonality)
        return ts, seasonality

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
            for other_ci in [other.conf_lo_pd_series(), other.conf_hi_pd_series()]:
                if (other_ci is None) ^ (self._confidence_lo is None):
                    # only one is None
                    return False
                if self._combine_or_none(self._confidence_lo, other_ci, lambda s1, s2: s1.equals(s2)) is False:
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
        return self._combine_from_pd_ops(other, lambda s1, s2: s1 + s2)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series - other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s - other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s - other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        return self._combine_from_pd_ops(other, lambda s1, s2: s1 - s2)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series * other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s * other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s * other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        return self._combine_from_pd_ops(other, lambda s1, s2: s1 * s2)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if n < 0:
            assert all(self.values() != 0), 'Cannot divide by a TimeSeries with a value 0.'

        new_series = self._series ** n
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: s ** n)
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: s ** n)
        return TimeSeries(new_series, conf_lo, conf_hi)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            assert other != 0, 'Cannot divide by 0.'

            new_series = self._series / other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s / other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s / other)
            return TimeSeries(new_series, conf_lo, conf_hi)

        assert all(other.values() != 0), 'Cannot divide by a TimeSeries with a value 0.'

        return self._combine_from_pd_ops(other, lambda s1, s2: s1 / s2)

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

    def __contains__(self, item):
        if isinstance(item, pd.Timestamp):
            return item in self._series.index
        return False

    def __str__(self):
        df = pd.DataFrame({'value': self._series})
        if self._confidence_lo is not None:
            df['conf_low'] = self._confidence_lo
        if self._confidence_hi is not None:
            df['conf_high'] = self._confidence_hi
        return str(df) + '\nFreq: {}'.format(self.freq_str())
