import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset

from u8timeseries.timeseries import TimeSeries



def constant_timeseries(value: float = 0, length: int = 10, offset : str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('20000101')) -> 'TimeSeries':
    """
    Creates a timeseries with a constant given value, length, start date and offset.

    :param value: The constant value that the TimeSeries object will assume at every index.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.full(length, value)

    return TimeSeries.from_times_and_values(times, values)


def linear_timeseries(start_value: float = 0, value_delta: float = 1, length: int = 10, offset : str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('20000101')) -> 'TimeSeries':
    """
    Creates a timeseries with a starting value of 'start_value that increases by 'value_delta' at each step.
    The last entry of the time series will equal start_value + (length - 1) * value_delta.

    :param start_value: The value of the first entry in the TimeSeries.
    :param value_delta: The difference in value between to adjacent entries in the TimeSeries.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    times = pd.date_range(periods=length, freq=offset, start=start_date)
    values = np.linspace(start_value, start_value + (length - 1) * value_delta, length)

    return TimeSeries.from_times_and_values(times, values)


def holiday_timeseries(length: int = 10, offset : str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('20000101')) -> 'TimeSeries':
    """
    Creates a binary timeseries that equals 1 at every index that corresponds to a holiday, 
    and 0 otherwise.

    :param value: The constant value that the TimeSeries object will assume at every index.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    # TODO

    return None




    
