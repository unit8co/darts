from u8timeseries.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def na_ratio(ts: 'TimeSeries') -> float:
    """
    Computes the ratio of missing values in percent.

    :param ts: The TimeSeries to check for missing values.
    :return: A float, the percentage of missing values in `ts`.
    """

    return 100 * ts.pd_series().isnull().sum() / len(ts)


def fillna_0(ts: 'TimeSeries') -> 'TimeSeries':
    """
    Fills the missing values of `ts` with zeroes only.

    :param ts: The TimeSeries to check for missing values.
    :return: A TimeSeries, `ts` with al missing values set to 0.
    """
    return TimeSeries.from_times_and_values(ts.time_index(), ts.pd_series().fillna(value=0))


def nan_structure_visual(ts: 'TimeSeries'):
    """
    Plots the indicator function of missing values of `ts`.

    Missing values have value 1 and non-missing values are 0.

    :param ts: The TimeSeries to check.
    """

    nans = np.isnan(ts.values())

    plt.scatter(np.arange(len(nans)), nans)
    plt.show()


def change_of_state(ts: 'TimeSeries') -> Tuple[int, List]:
    """
    Determines the indices where `ts` changes from missing values to non-missing and vice-versa.

    :param ts: The TimeSeries to analyze.
    :return: A tuple (a, l), where a is the number of change of states and l is the list of indices where \
    changes of states occur.
    """

    a = np.where(np.diff(np.isnan(ts.values())) is True)

    return len(a[0]), list(a[0])


def auto_fillna(ts: 'TimeSeries') -> 'TimeSeries':
    """
    This function automatically fills the missing value in the TimeSeries `ts`.

    The rules for completion are given below.

    Missing values at the beginning are set to 0.
    Missing values at the end are filled with a forward-fill.
    Missing values between to numeric values are set by linear interpolation.

    .. todo: be more flexible on the filling methods.

    :param ts: A TimeSeries `ts`.
    :return: A new TimeSeries with all missing values filled according to the rules above.
    """

    # We compute the number of times entries of the TimeSeries go from mssing to numeric and vice-versa
    arr = np.where(np.diff(np.isnan(ts.values())) == True)[0]

    ts_temp = ts.pd_series()

    if len(arr) == 0:
        raise ValueError('Your TimeSeries has no missing value.')

    # Checks if first value is missing
    entry_1 = np.isnan(ts.values()[0])

    # Stocks the indices for going from nan to numeric and numeric to nan in two separate lists
    # The lists depend on entry_1's value
    if entry_1:

        nan_num = [arr[i] for i in range(len(arr)) if i % 2 == 0]
        num_nan = [arr[i] for i in range(len(arr)) if i % 2 == 1]

        # If the TimeSeries starts with missing value, insert 0 everywhere
        ts_temp[:nan_num[0] + 1] = 0

        nan_num.pop(0)

    else:

        nan_num = [arr[i] for i in range(len(arr)) if i % 2 == 1]
        num_nan = [arr[i] for i in range(len(arr)) if i % 2 == 0]

    # One has that len(nan_num) = len(num_nan) or len(num_nan) - 1

    # As long as nan_num is not empty, the missing values are both preceded and followed by numeric values
    # Thus we can use linear interpolation between the closest known values to fill in the gaps.
    while len(nan_num) > 0:
        h = nan_num[0] - num_nan[0] + 1

        k = (ts_temp[nan_num[0] + 1] - ts_temp[num_nan[0]]) / h

        fill_values = [ts_temp[num_nan[0]] + (i + 1) * k for i in range(h - 1)]

        ts_temp[num_nan[0] + 1: nan_num[0] + 1] = fill_values

        nan_num.pop(0)
        num_nan.pop(0)

    # Treats the case with additional missing values at the end with a forward fill.
    if len(num_nan) > 0:
        ts_temp[num_nan[0] + 1:] = ts_temp[num_nan[0]]

    return TimeSeries.from_times_and_values(ts.time_index(), ts_temp.values())