from u8timeseries.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from scipy.interpolate import interp1d


def na_ratio(ts: 'TimeSeries') -> float:
    """
    Computes the ratio of missing values in percent.

    :param ts: The TimeSeries to check for missing values.
    :return: A float, the percentage of missing values in `ts`.
    """

    return 100 * ts.pd_series().isnull().sum() / len(ts)


def fillna_val(ts: 'TimeSeries', fill: float = 0) -> 'TimeSeries':
    """
    Fills the missing values of `ts` with only the value provided (default zeroes).

    :param ts: The TimeSeries to check for missing values.
    :param fill: the value used to replace the missing values.
    :return: A TimeSeries, `ts` with al missing values set to 0.
    """
    return TimeSeries.from_times_and_values(ts.time_index(), ts.pd_series().fillna(value=fill))


def nan_structure_visual(ts: 'TimeSeries', plot: bool = True) -> np.ndarray:
    """
    Plots the indicator function of missing values of `ts`.

    Missing values have value 1 and non-missing values are 0.

    :param ts: The TimeSeries to check.
    :param plot: Boolean to choose to plot the result or not.
    :return: An numpy array containing the indices where the missing values are.
    """

    nans = np.isnan(ts.values())

    if plot: # TODO: find a better way to visualize NaN data
        plt.scatter(np.arange(len(nans)), nans)
        plt.yticks([0, 1], ["Non-missing", "NaN"])
        plt.show()

    return np.where(nans)[0]


def change_of_state(ts: 'TimeSeries') -> Tuple[int, List]:
    """
    Determines the indices where `ts` changes from missing values to non-missing and vice-versa.

    :param ts: The TimeSeries to analyze.
    :return: A tuple (a, l), where a is the number of change of states and l is the list of indices where \
    changes of states occur.
    """

    a = np.where(np.diff(ts.pd_series().isna()))

    return len(a[0]), list(a[0])


def auto_fillna(ts: 'TimeSeries', first: float = None, last: float = None,
                interpolate: str = 'linear', **kwargs) -> 'TimeSeries':
    """
    This function automatically fills the missing value in the TimeSeries `ts`, assuming they are represented by np.nan.

    The rules for completion are given below.

    Missing values at the beginning are filled with constant value `first`. Defaults to backwards-fill.
    Missing values at the end are filled with constant value `last`. Defaults to forward-fill.
    Missing values between to numeric values are set using the interpolation wrapper of pandas with `method`.
    Defaults to linear interpolation.

    Add the option `fill_value` to 'extrapolate' to fill the missing values at the beginning and the end with
    the regression function computed. Must set `first` and `last` to None

    .. todo: be more flexible on the filling methods.

    :param ts: A TimeSeries `ts`.
    :param first: The value to use for filling the beginning of the TimeSeries. Defaults to first known value in `ts`.
    :param last: The value to use for filling the ending of the TimeSeries. Defaults to last known value in `ts`.
    :param interpolate: The function used for filling the middle of the TimeSeries. Defaults to linear interpolation.
    :return: A new TimeSeries with all missing values filled according to the rules above.
    """

    # We compute the number of times entries of the TimeSeries go from missing to numeric and vice-versa
    arr = change_of_state(ts)[1]

    if len(arr) == 0:
        return ts

    ts_temp = ts.pd_series()

    # if first value is missing and `first` is specified, fill values
    if np.isnan(ts.values()[0]) and first is not None:
        ts_temp[:arr[0] + 1] = first

    # if last value is missing and `last` is specified, fill values
    if np.isnan(ts.values()[-1]) and last is not None:
        ts_temp[arr[-1]+1:] = last

    # pandas interpolate wrapper, with chosen `method`
    ts_temp.interpolate(method=interpolate, inplace=True, limit_direction='both', limit=len(ts_temp), **kwargs)

    return TimeSeries.from_times_and_values(ts.time_index(), ts_temp.values)
