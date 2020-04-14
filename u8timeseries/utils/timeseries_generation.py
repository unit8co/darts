import numpy as np
import pandas as pd
import math
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import USFederalHolidayCalendar

from u8timeseries.timeseries import TimeSeries



def constant_timeseries(value: float = 0, length: int = 10, offset: str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a constant TimeSeries with the given value, length, start date and offset.

    :param value: The constant value that the TimeSeries object will assume at every index.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/TimeSeries.html#dateoffset-objects.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.full(length, value)

    return TimeSeries.from_times_and_values(times, values)


def linear_timeseries(start_value: float = 0, value_delta: float = 1, length: int = 10, offset: str = 'D',
                      start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a TimeSeries with a starting value of 'start_value' that increases by 'value_delta' at each step.
    The last entry of the time series will be equal to 'start_value' + ('length' - 1) * 'value_delta'.

    :param start_value: The value of the first entry in the TimeSeries.
    :param value_delta: The difference in value between to adjacent entries in the TimeSeries.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A linear TimeSeries with gradient 'value_delta'.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.linspace(start_value, start_value + (length - 1) * value_delta, length)

    return TimeSeries.from_times_and_values(times, values)


def periodic_timeseries(frequency: float = 0.1, amplitude: float = 1, phase: float = 0, y_offset: float = 0,
                        length: int = 10, offset: str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a TimeSeries with a sinusoidal value progression with a given frequency, amplitude, phase and y offset.

    :param frequency: The number of periods that take place within one time unit given in 'offset'.
    :param amplitude: The maximum  difference between any value of the returned TimeSeries and 'y_offset'.
    :param phase: The relative position within one period of the first value of the returned TimeSeries (in radians).
    :param y_offset: The shift of the sine function along the y axis.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A sinusoidal TimeSeries parametrized as indicated above.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.array(range(length), dtype=float)
    f = np.vectorize(lambda x: amplitude * math.sin(2 * math.pi * frequency * x + phase) + y_offset)
    values = f(values)

    return TimeSeries.from_times_and_values(times, values)


def white_noise_timeseries(length: int = 10, offset: str = 'D', mean: float = 0, std: float = 1,
                           start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a white noise TimeSeries by sampling a gaussian distribution with mean 'mean' and 
    standard deviation 'std'. Each value represents an indipendent sample of the distribution.

    :param mean: The mean of the gaussian distribution that is sampled at each step.
    :param std: The standard deviation of the gaussian distribution that is sampled at each step.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A white noise TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.random.normal(mean, std, size=length)

    return TimeSeries.from_times_and_values(times, values)


def random_walk_timeseries(length: int = 10, offset: str = 'D', mean: float = 0, std: float = 1,
                           start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a random walk TimeSeries by sampling a gaussian distribution with mean 'mean' and 
    standard deviation 'std'. The first value is one such random sample. Every subsequent value
    is equal to the previous value plus a random sample.

    :param mean: The mean of the gaussian distribution that is sampled at each step.
    :param std: The standard deviation of the gaussian distribution that is sampled at each step.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A random walk TimeSeries created as indicated above.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = [np.random.normal(mean, std)]
    while (len(values) < length):
        values.append(values[-1] + np.random.normal(mean, std))

    return TimeSeries.from_times_and_values(times, values)


def us_holiday_timeseries(length: int = 10, start_date: pd.Timestamp = pd.Timestamp('2000-01-01')) -> 'TimeSeries':
    """
    Creates a binary TimeSeries that equals 1 at every index that corresponds to a US holiday, 
    and 0 otherwise. The frequency of the TimeSeries is daily.

    :param length: The length of the returned TimeSeries.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: Binary TimeSeries for US holidays.
    """

    times = pd.date_range(periods=length, freq='D', start=start_date)
    us_holidays = USFederalHolidayCalendar().holidays()
    values = times.isin(us_holidays).astype(int)
    
    return TimeSeries.from_times_and_values(times, values)




    
