import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset

from u8timeseries.timeseries import TimeSeries


def generate_datetime_index(length: int, offset: DateOffset, start_date) -> pd.DatetimeIndex:
    """
    Creates a pandas datetime index with the given start date 'start_date', 
    the given length 'length' and the given time delta 'offset' between two adjacent entries.

    :param length: The length of the returned DatetimeIndex.
    :param offset: The time differene between two adjacent entries in the DatetimeIndex. A DateOffset alias is expected;
                   see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects.
    :param start_date: The time index of the first entry in the DatetimeIndex.
    """

    

    return None


def constant_timeseries(value: float = 0, length: int = 10, offset : str = 'D',
                        start_date: pd.Timestamp = pd.Timestamp('20000101')) -> 'TimeSeries':
    """
    Creates a timeseries with a constant given value value, length, start date and offset.

    :param value: The constant value that the TimeSeries object will assume at every index.
    :param length: The length of the returned TimeSeries.
    :param offset: The time differene between two adjacent entries in the returned TimeSeries.
    :param start_date: The time index of the first entry in the returned TimeSeries.
    :return: A constant TimeSeries with value 'value'.
    """

    index = generate_datetime_index(length, offset, start_date)

    
