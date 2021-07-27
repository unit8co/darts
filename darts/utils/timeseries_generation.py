"""
Utils for time series generation
--------------------------------
"""

import math

from typing import Union, Optional

import numpy as np
import pandas as pd
import holidays

from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger, raise_log, raise_if

logger = get_logger(__name__)


def constant_timeseries(value: float = 1,
                        length: int = 10,
                        freq: str = 'D',
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01'),
                        column_name: Optional[str] = 'constant') -> TimeSeries:
    """
    Creates a constant univariate TimeSeries with the given value, length, start date and frequency.

    Parameters
    ----------
    value
        The constant value that the TimeSeries object will assume at every index.
    length
        The length of the returned TimeSeries.
    freq
        The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
        see `docs <https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects>`_.
    start_ts
        The time index of the first entry in the returned TimeSeries.

    Returns
    -------
    TimeSeries
        A constant TimeSeries with value 'value'.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.full(length, value)

    return TimeSeries.from_times_and_values(times, values, freq=freq, columns=pd.Index([column_name]))


def linear_timeseries(start_value: float = 0,
                      end_value: float = 1,
                      length: int = 10,
                      freq: str = 'D',
                      start_ts: pd.Timestamp = pd.Timestamp('2000-01-01'),
                      column_name: Optional[str] = 'linear') -> TimeSeries:
    """
    Creates a univariate TimeSeries with a starting value of `start_value` that increases linearly such that
    it takes on the value `end_value` at the last entry of the TimeSeries. This means that
    the difference between two adjacent entries will be equal to
    (`end_value` - `start_value`) / (`length` - 1).

    Parameters
    ----------
    start_value
        The value of the first entry in the TimeSeries.
    end_value
        The value of the last entry in the TimeSeries.
    length
        The length of the returned TimeSeries.
    freq
        The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    start_ts
        The time index of the first entry in the returned TimeSeries.

    Returns
    -------
    TimeSeries
        A linear TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.linspace(start_value, end_value, length)
    return TimeSeries.from_times_and_values(times, values, freq=freq, columns=pd.Index([column_name]))


def sine_timeseries(value_frequency: float = 0.1,
                    value_amplitude: float = 1.,
                    value_phase: float = 0.,
                    value_y_offset: float = 0.,
                    length: int = 10,
                    freq: str = 'D',
                    start_ts: pd.Timestamp = pd.Timestamp('2000-01-01'),
                    column_name: Optional[str] = 'sine') -> TimeSeries:
    """
    Creates a univariate TimeSeries with a sinusoidal value progression with a given frequency, amplitude,
    phase and y offset.

    Parameters
    ----------
    value_frequency
        The number of periods that take place within one time unit given in `freq`.
    value_amplitude
        The maximum  difference between any value of the returned TimeSeries and `y_offset`.
    value_phase
        The relative position within one period of the first value of the returned TimeSeries (in radians).
    value_y_offset
        The shift of the sine function along the y axis.
    length
        The length of the returned TimeSeries.
    freq
        The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    start_ts
        The time index of the first entry in the returned TimeSeries.

    Returns
    -------
    TimeSeries
        A sinusoidal TimeSeries parametrized as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.array(range(length), dtype=float)
    f = np.vectorize(
        lambda x: value_amplitude * math.sin(2 * math.pi * value_frequency * x + value_phase) + value_y_offset
    )
    values = f(values)

    return TimeSeries.from_times_and_values(times, values, freq=freq, columns=pd.Index([column_name]))


def gaussian_timeseries(length: int = 10,
                        freq: str = 'D',
                        mean: Union[float, np.ndarray] = 0.,
                        std: Union[float, np.ndarray] = 1.,
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01'),
                        column_name: Optional[str] = 'gaussian') -> TimeSeries:
    """
    Creates a gaussian univariate TimeSeries by sampling all the series values independently,
    from a gaussian distribution with mean `mean` and standard deviation `std`.

    Parameters
    ----------
    length
        The length of the returned TimeSeries.
    freq
        The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    mean
        The mean of the gaussian distribution that is sampled at each step.
        If a float value is given, the same mean is used at every step.
        If a numpy.ndarray of floats with the same length as `length` is
        given, a different mean is used at each time step.
    std
        The standard deviation of the gaussian distribution that is sampled at each step.
        If a float value is given, the same standard deviation is used at every step.
        If an array of dimension `(length, length)` is given, it will
        be used as covariance matrix for a multivariate gaussian distribution.
    start_ts
        The time index of the first entry in the returned TimeSeries.

    Returns
    -------
    TimeSeries
        A white noise TimeSeries created as indicated above.
    """

    if (type(mean) == np.ndarray):
        raise_if_not(mean.shape == (length,), 'If a vector of means is provided, '
                                              'it requires the same length as the TimeSeries.', logger)
    if (type(std) == np.ndarray):
        raise_if_not(std.shape == (length, length), 'If a matrix of standard deviations is provided, '
                                                    'its shape has to match the length of the TimeSeries.', logger)

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.random.normal(mean, std, size=length)

    return TimeSeries.from_times_and_values(times, values, freq=freq, columns=pd.Index([column_name]))


def random_walk_timeseries(length: int = 10,
                           freq: str = 'D',
                           mean: float = 0.,
                           std: float = 1.,
                           start_ts: pd.Timestamp = pd.Timestamp('2000-01-01'),
                           column_name: Optional[str] = 'random_walk') -> TimeSeries:
    """
    Creates a random walk univariate TimeSeries, where each step is obtained by sampling a gaussian distribution
    with mean `mean` and standard deviation `std`.

    Parameters
    ----------
    length
        The length of the returned TimeSeries.
    freq
        The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    mean
        The mean of the gaussian distribution that is sampled at each step.
    std
        The standard deviation of the gaussian distribution that is sampled at each step.
    start_ts
        The time index of the first entry in the returned TimeSeries.
    Returns
    -------
    TimeSeries
        A random walk TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.cumsum(np.random.normal(mean, std, size=length))

    return TimeSeries.from_times_and_values(times, values, freq=freq, columns=pd.Index([column_name]))


def _extend_time_index_until(time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
                             until: Optional[Union[int, str, pd.Timestamp]],
                             add_length: int,
                             ) -> pd.DatetimeIndex:

    if not add_length and not until:
        return time_index

    raise_if(bool(add_length) and bool(until), "set only one of add_length and until")

    end = time_index[-1]
    freq = time_index.freq

    if add_length:
        raise_if_not(add_length >= 0, f"Expected add_length, by which to extend the time series by, "
                                      f"to be positive, got {add_length}")

        try:
            end += add_length*freq
        except pd.errors.OutOfBoundsDatetime:
            raise_log(ValueError(f"the add operation between {end} and {add_length * freq} will overflow"), logger)
    else:
        datetime_index = isinstance(time_index, pd.DatetimeIndex)

        if datetime_index:
            raise_if_not(isinstance(until, (str, pd.Timestamp)), "Expected valid timestamp for TimeSeries, "
                                                                 "indexed by DatetimeIndex, "
                                                                 f"for parameter until, got {type(end)}", logger)
        else:
            raise_if_not(isinstance(until, int),  "Expected integer for TimeSeries,"
                                                  "indexed by RangeIndex, ",
                                                  f"for parameter until, got {type(end)}", logger)

        timestamp = pd.Timestamp(until) if datetime_index else until

        raise_if_not(timestamp > end, f"Expected until, {timestamp} to lie past end of time index {end}")

        ahead = timestamp - end
        raise_if_not((ahead % freq) == pd.Timedelta(0), f"End date must correspond with frequency {freq} of the time axis", logger)

        end = timestamp

    new_time_index = pd.date_range(start=time_index[0], end=end, freq=freq)
    return new_time_index


def holidays_timeseries(time_index: pd.DatetimeIndex,
                        country_code: str,
                        prov: str = None,
                        state: str = None,
                        column_name: Optional[str] = 'holidays',
                        until: Optional[Union[int, str, pd.Timestamp]] = None,
                        add_length: int = 0,
                        ) -> TimeSeries:
    """
    Creates a binary univariate TimeSeries with index `time_index` that equals 1 at every index that lies within
    (or equals) a selected country's holiday, and 0 otherwise.

    Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`_.

    Parameters
    ----------
    time_index
        The time index over which to generate the holidays
    country_code
        The country ISO code
    prov
        The province
    state
        The state
    until
        Extend the time_index up until timestamp for datetime indexed series
        and int for range indexed series, should match or exceed forecasting window.
    add_length
        Extend the time_index by add_length, should match or exceed forecasting window.
        Set only one of until and add_length.

    Returns
    -------
    TimeSeries
        A new binary holiday TimeSeries instance.
    """

    time_index = _extend_time_index_until(time_index, until, add_length)
    scope = range(time_index[0].year, (time_index[-1] + pd.Timedelta(days=1)).year)
    country_holidays = holidays.CountryHoliday(country_code, prov=prov, state=state, years=scope)
    index_series = pd.Series(time_index, index=time_index)
    values = index_series.apply(lambda x: x in country_holidays).astype(int)
    return TimeSeries.from_times_and_values(time_index, values, columns=pd.Index([column_name]))


def datetime_attribute_timeseries(time_index: Union[pd.DatetimeIndex, TimeSeries],
                                  attribute: str,
                                  one_hot: bool = False,
                                  cyclic: bool = False,
                                  until: Optional[Union[int, str, pd.Timestamp]] = None,
                                  add_length: int = 0) -> TimeSeries:
    """
    Returns a new TimeSeries with index `time_index` and one or more dimensions containing
    (optionally one-hot encoded or cyclic encoded) pd.DatatimeIndex attribute information derived from the index.


    Parameters
    ----------
    time_index
        Either a `pd.DatetimeIndex` attribute which will serve as the basis of the new column(s), or
        a `TimeSeries` whose time axis will serve this purpose.
    attribute
        An attribute of `pd.DatetimeIndex` - e.g. "month", "weekday", "day", "hour", "minute", "second"
    one_hot
        Boolean value indicating whether to add the specified attribute as a one hot encoding
        (results in more columns).
    cyclic
        Boolean value indicating whether to add the specified attribute as a cyclic encoding.
        Alternative to one_hot encoding, enable only one of the two.
        (adds 2 columns, corresponding to sin and cos transformation)
    until
        Extend the time_index up until timestamp for datetime indexed series
        and int for range indexed series, should match or exceed forecasting window.
    add_length
        Extend the time_index by add_length, should match or exceed forecasting window.
        Set only one of until and add_length.

    Returns
    -------
    TimeSeries
        New datetime attribute TimeSeries instance.
    """

    if isinstance(time_index, TimeSeries):
        time_index = time_index.time_index

    time_index = _extend_time_index_until(time_index, until, add_length)

    raise_if_not(hasattr(pd.DatetimeIndex, attribute), '"attribute" needs to be an attribute '
                 'of pd.DatetimeIndex', logger)

    raise_if(one_hot and cyclic, "set only one of one_hot or cyclic to true", logger)

    num_values_dict = {
        'month': 12,
        'day': 31,
        'weekday': 7,
        'hour': 24,
        'quarter': 4
    }

    values = getattr(time_index, attribute)

    if one_hot or cyclic:
        raise_if_not(attribute in num_values_dict, "Given datetime attribute not supported"
                                                   " with one-hot or cyclical encoding.", logger)

    if one_hot:
        values_df = pd.get_dummies(values)
        # fill missing columns (in case not all values appear in time_index)
        for i in range(1, num_values_dict[attribute] + 1):
            if not (i in values_df.columns):
                values_df[i] = 0
        values_df = values_df[range(1, num_values_dict[attribute] + 1)]
    elif cyclic:
        if attribute == "day":
            periods = [time_index[i].days_in_month for i in time_index.month]
            freq = 2*np.pi * np.reciprocal(periods)
        else:
            period = num_values_dict[attribute]
            freq = 2*np.pi/period

        values_df = pd.DataFrame({
            attribute+"_sin": np.sin(freq * values),
            attribute+"_cos": np.cos(freq * values)
        })
    else:
        values_df = pd.DataFrame(values)
    values_df.index = time_index

    if one_hot:
        values_df.columns = [attribute + '_' + str(column_name) for column_name in values_df.columns]

    return TimeSeries.from_dataframe(values_df)

