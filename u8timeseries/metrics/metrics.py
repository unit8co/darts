import numpy as np
from typing import Tuple
from u8timeseries.timeseries import TimeSeries

""" Helper function
"""
def get_values_or_raise(series_a: TimeSeries, series_b: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the numpy values of two time series, launching an Exception if time series cannot be compared
    """
    assert series_a.has_same_time_as(series_b), 'The two time series must have same time index.' \
                                                '\nFirst series: {}\nSecond series: {}'.format(
                                                series_a.time_index(), series_b.time_index())
    return series_a.values(), series_b.values()


def mape(true_series: TimeSeries, pred_series: TimeSeries):
    """
    Computes the Mean Absolute Percentage Error of `true_series` and `pred_series`

    :param true_series: The series to match
    :param pred_series: The predicred values

    :return: The MAPE.
    """
    y_true, y_hat = get_values_or_raise(true_series, pred_series)
    return 100. * np.mean(np.abs((y_true - y_hat) / y_true))


def mase(true_series: TimeSeries, pred_series: TimeSeries):
    y_true, y_pred = get_values_or_raise(true_series, pred_series)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-1) * np.sum(np.abs(np.diff(y_true)))
    return errors / scale


def overall_percentage_error(true_series: TimeSeries, pred_series: TimeSeries):
    y_true, y_pred = get_values_or_raise(true_series, pred_series)
    y_true_sum, y_pred_sum = np.sum(np.array(y_true)), np.sum(np.array(y_pred))
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.
