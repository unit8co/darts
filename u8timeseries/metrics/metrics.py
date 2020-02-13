import numpy as np
from typing import Tuple
from u8timeseries.timeseries import TimeSeries


def _get_values_or_raise(series_a: TimeSeries, series_b: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the numpy values of two time series, launching an Exception if time series cannot be compared
    """
    assert series_a.has_same_time_as(series_b), 'The two time series must have same time index.' \
                                                '\nFirst series: {}\nSecond series: {}'.format(
                                                series_a.time_index(), series_b.time_index())
    return series_a.values(), series_b.values()


def mape(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    This function computes the MAPE of `pred_series` with respect to `true_series`.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the MAPE of `pred_series` with respect to `true_series`.
    """
    # Mean absolute percentage error
    y_true, y_hat = _get_values_or_raise(true_series, pred_series)
    return 100. * np.mean(np.abs((y_true - y_hat) / y_true))


def mase(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the Mean Absolute Scaled Error (MASE).

    This function computes the MASE of `pred_series` with respect to `true_series`.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the MASE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-1) * np.sum(np.abs(np.diff(y_true)))
    return errors / scale


def mase_seasonal(true_series: TimeSeries, pred_series: TimeSeries, m: int = 1) -> float:
    """
    Computes the Mean Absolute Scaled Error (MASE).

    This function computes the MASE of `pred_series` with respect to `true_series` and the seasonal period m.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param m: A int, the seasonality period to take into account. If None, infer one from ACF?
    :return: A float, the MASE of `pred_series` with respect to `true_series`.
    """
    if m is None:
        m = 1  # todo change to find and check seasonality
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-m) * np.sum(np.abs(np.diff(y_true, m)))
    return errors / scale


def overall_percentage_error(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the Overall Percentage Erroe (OPE):

    This function computes the OPE of `pred_series` with respect to `true_series`.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the OPE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    y_true_sum, y_pred_sum = np.sum(np.array(y_true)), np.sum(np.array(y_pred))
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.


def marre(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the Mean Absolute Ranged Relative Error (MARRE).

    This function computes the MARRE of `pred_series` with respect to `true_series`.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the MARRE of `pred_series` with respect to `true_series`.
    """
    y_true, y_hat = _get_values_or_raise(true_series, pred_series)
    true_range = y_true.max() - y_true.min()
    return 100. * np.mean(np.abs((y_true - y_hat) / true_range))


def r2_score(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the coefficient of determination R2

    This function computes the R2 score of `pred_series` with respect to `true_series`.
    :param true_series: A TimeSeries
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the coefficient R2
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    ss_errors = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true-y_true.mean())**2)
    return 1 - ss_errors/ss_tot

