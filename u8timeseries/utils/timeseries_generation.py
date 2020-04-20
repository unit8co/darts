import numpy as np
import pandas as pd
import math
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import Union

from u8timeseries.timeseries import TimeSeries
from ..custom_logging import assert_log, get_logger

logger = get_logger(__name__)


def constant_timeseries(value: float = 1, length: int = 10, freq: str = 'D',
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a constant TimeSeries with the given value, length, start date and frequency.

    :param value: The constant value that the TimeSeries object will assume at every index.
    :param length: The length of the returned TimeSeries.
    :param freq: The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.full(length, value)

    return TimeSeries.from_times_and_values(times, values)


def linear_timeseries(start_value: float = 0, end_value: float = 1, length: int = 10, freq: str = 'D',
                      start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a TimeSeries with a starting value of 'start_value' that increases linearly such that
    it takes on the value 'end_value' at the last entry of the TimeSeries. This means that
    the difference between two adjacent entries will be equal to 
    ('end_value' - 'start_value') / ('length' - 1).

    :param start_value: The value of the first entry in the TimeSeries.
    :param end_value: The value of the last entry in the TimeSeries.
    :param length: The length of the returned TimeSeries.
    :param freq: The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: A linear TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.linspace(start_value, end_value, length)

    return TimeSeries.from_times_and_values(times, values)


def sine_timeseries(value_frequency: float = 0.1, value_amplitude: float = 1, value_phase: float = 0, 
                        value_y_offset: float = 0, length: int = 10, freq: str = 'D',
                        start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a TimeSeries with a sinusoidal value progression with a given frequency, amplitude, phase and y offset.

    :param value_frequency: The number of periods that take place within one time unit given in 'freq'.
    :param value_amplitude: The maximum  difference between any value of the returned TimeSeries and 'y_offset'.
    :param value_phase: The relative position within one period of the first value of the returned TimeSeries (in radians).
    :param value_y_offset: The shift of the sine function along the y axis.
    :param length: The length of the returned TimeSeries.
    :param freq: The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: A sinusoidal TimeSeries parametrized as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.array(range(length), dtype=float)
    f = np.vectorize(lambda x: value_amplitude * math.sin(2 * math.pi * value_frequency * x + value_phase) + value_y_offset)
    values = f(values)

    return TimeSeries.from_times_and_values(times, values)


def gaussian_timeseries(length: int = 10, freq: str = 'D', mean: Union[float, np.ndarray] = 0, 
                        std: Union[float, np.ndarray] = 1, start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a gaussian noise TimeSeries by sampling a gaussian distribution with mean 'mean' and 
    standard deviation 'std'. Each value represents a sample of the distribution.
    When the mean is set to 0, it can be considered a white noise TimeSeries.

    :param mean: The mean of the gaussian distribution that is sampled at each step.
                 If a float value is given, the same mean is used at every step.
                 If a numpy.ndarray of floats with the same length as 'length' is
                 given, a different mean is used at each step.
    :param std: The standard deviation of the gaussian distribution that is sampled at each step.
                If a float value is given, the same standard deviation is used at every step.
                If a 'length' x 'length' numpy.ndarray of floats  is given, it will
                be used as covariance matrix for a multivariate gaussian distribution.
    :param length: The length of the returned TimeSeries.
    :param freq: The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: A white noise TimeSeries created as indicated above.
    """

    if (type(mean) == np.ndarray):
        assert_log(mean.shape == (length,), 'If a vector of means is provided, it requires the same length as the TimeSeries.', logger)
    if (type(std) == np.ndarray):
        assert_log(std.shape == (length, length), 'If a matrix of standard deviations is provided,' \
                                              ' its shape has to match the length of the TimeSeries.', logger)

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.random.normal(mean, std, size=length)

    return TimeSeries.from_times_and_values(times, values)


def random_walk_timeseries(length: int = 10, freq: str = 'D', mean: float = 0, std: float = 1,
                           start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a random walk TimeSeries by sampling a gaussian distribution with mean 'mean' and 
    standard deviation 'std'. The first value is one such random sample. Every subsequent value
    is equal to the previous value plus a random sample.

    :param mean: The mean of the gaussian distribution that is sampled at each step.
    :param std: The standard deviation of the gaussian distribution that is sampled at each step.
    :param length: The length of the returned TimeSeries.
    :param freq: The time difference between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: A random walk TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=freq, start=start_ts)
    values = np.cumsum(np.random.normal(mean, std, size=length))

    return TimeSeries.from_times_and_values(times, values)


def us_holiday_timeseries(length: int = 10, start_ts: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a binary TimeSeries that equals 1 at every index that corresponds to a US holiday, 
    and 0 otherwise. The frequency of the TimeSeries is daily.

    :param length: The length of the returned TimeSeries.
    :param start_ts: The time index of the first entry in the returned TimeSeries.
    :return: Binary TimeSeries for US holidays.
    """

    times = pd.date_range(periods=length, freq='D', start=start_ts)
    us_holidays = USFederalHolidayCalendar().holidays()
    values = times.isin(us_holidays).astype(int)
    
    return TimeSeries.from_times_and_values(times, values)




    
