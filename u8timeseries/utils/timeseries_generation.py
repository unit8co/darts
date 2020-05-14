"""
Utils for time series generation
--------------------------------
"""

import math
from typing import Union

import numpy as np
import pandas as pd
import holidays

from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger

logger = get_logger(__name__)


def constant_timeseries(value: float = 1,
                        length: int = 10,
                        freq: str = 'D',
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a constant TimeSeries with the given value, length, start date and frequency.

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

    return TimeSeries.from_times_and_values(times, values)


def linear_timeseries(start_value: float = 0,
                      end_value: float = 1,
                      length: int = 10,
                      freq: str = 'D',
                      start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a TimeSeries with a starting value of `start_value` that increases linearly such that
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
    return TimeSeries.from_times_and_values(times, values)


def sine_timeseries(value_frequency: float = 0.1,
                    value_amplitude: float = 1.,
                    value_phase: float = 0.,
                    value_y_offset: float = 0.,
                    length: int = 10,
                    freq: str = 'D',
                    start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a TimeSeries with a sinusoidal value progression with a given frequency, amplitude, phase and y offset.

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

    return TimeSeries.from_times_and_values(times, values)


def gaussian_timeseries(length: int = 10,
                        freq: str = 'D',
                        mean: Union[float, np.ndarray] = 0.,
                        std: Union[float, np.ndarray] = 1.,
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a gaussian TimeSeries by sampling all the series values independently,
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

    return TimeSeries.from_times_and_values(times, values)


def random_walk_timeseries(length: int = 10,
                           freq: str = 'D',
                           mean: float = 0.,
                           std: float = 1.,
                           start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a random walk time series, where each step is obtained by sampling a gaussian distribution
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

    return TimeSeries.from_times_and_values(times, values)


def holiday_timeseries(country_code: str,
                       prov: str = None,
                       state: str = None,
                       length: int = 10,
                       start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> TimeSeries:
    """
    Creates a binary TimeSeries that equals 1 at every index that corresponds to selected country's holiday,
    and 0 otherwise. The frequency of the TimeSeries is daily.

    Available countries can be found `here <https://github.com/dr-prodigy/python-holidays#available-countries>`_.

    Parameters
    ----------
    country_code
        The country ISO code
    prov
        The province
    state
        The state
    length
        The length of the returned TimeSeries.
    start_ts
        The timestamp of the first entry in the returned TimeSeries.

    Returns
    -------
    TimeSeries
        Binary daily TimeSeries for country's holidays.
    """

    times = pd.date_range(periods=length, start=start_ts)
    country_holidays = holidays.CountryHoliday(country_code, prov=prov, state=state)
    scoped_country_holidays = country_holidays[times[0]:times[-1] + pd.Timedelta(days=1)]
    values = times.isin(scoped_country_holidays).astype(int)

    return TimeSeries.from_times_and_values(times, values)
