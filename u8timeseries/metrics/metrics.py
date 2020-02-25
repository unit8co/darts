import numpy as np
from typing import Tuple
from u8timeseries.timeseries import TimeSeries
from warnings import warn


def _import_check_seasonality():
    try:
        from u8timeseries.models.statistics import check_seasonality as cs
    except ImportError as e:
        raise ImportError('Cannot import check_seasonality. Choose a fixed period')
    return cs


def _get_values_or_raise(series_a: TimeSeries, series_b: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the numpy values of two time series, launching an Exception if time series cannot be compared
    """
    assert series_a.has_same_time_as(series_b), 'The two time series must have same time index.' \
                                                '\nFirst series: {}\nSecond series: {}'.format(
                                                series_a.time_index(), series_b.time_index())
    return series_a.values(), series_b.values()


def mae(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Compute the Mean Absolute Error (MAE).

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the MAE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_pred = np.diff(y_true), np.diff(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mse(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Compute the Mean Squared Error (MSE).

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the MSE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_pred = np.diff(y_true), np.diff(y_pred)
    return np.mean((y_true - y_pred)**2)


def rmse(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the RMSE of `pred_series` with respect to `true_series`.
    """
    return np.sqrt(mse(true_series, pred_series, time_diff))


def rmsle(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Compute the Root Mean Squared Log Error (RMSLE).

    Penalize more the under-estimate than the over-estimate.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the RMSLE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    y_true, y_pred = np.log(y_true + 1), np.log(y_pred + 1)
    return np.sqrt(np.mean((y_true - y_pred)**2))


def coefficient_variation(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the RMSE of `pred_series` with respect to `true_series`.
    """
    return 100 * rmse(true_series, pred_series, time_diff)/true_series.mean()


def mape(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE).

    This function computes the MAPE of `pred_series` with respect to `true_series`.
    Use `time_diff=True` when the time series has a strong auto-correlation to have a more accurate analysis.
    Use `time_diff` mainly when the time series are monotonic.
    Otherwise, it will be difficult to analyze the results.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the MAPE of `pred_series` with respect to `true_series`.
    """
    # Mean absolute percentage error
    y_true, y_hat = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_hat = np.diff(y_true), np.diff(y_hat)
    return 100. * np.mean(np.abs((y_true - y_hat) / y_true))


def mase_old(true_series: TimeSeries, pred_series: TimeSeries) -> float:
    """
    Computes the Mean Absolute Scaled Error (MASE).

    This function computes the MASE of `pred_series` with respect to `true_series`.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :return: A float, the MASE of `pred_series` with respect to `true_series`.
    """
    warn("This function does not take into account the seasonality of the time series. "
         "Please use mase_seasonal to be accurate", FutureWarning)
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-1) * np.sum(np.abs(np.diff(y_true)))
    return errors / scale


def mase(true_series: TimeSeries, pred_series: TimeSeries, m: int = 1, time_diff: bool = False) -> float:
    """
    Computes the Mean Absolute Scaled Error (MASE).

    This function computes the MASE of `pred_series` with respect to `true_series` and the seasonal period m.
    Use `time_diff=True` when the time series has a strong auto-correlation to have a more accurate analysis.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param m: A int, the seasonality period to take into account. If None, try to infer one from ACF
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the MASE of `pred_series` with respect to `true_series`.
    """
    if m is None:
        check_seasonality = _import_check_seasonality()
        test_season, m = check_seasonality()
        if not test_season:
            warn("No seasonality found. The period is fixed to 1.", UserWarning)
            m = 1
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_pred = np.diff(y_true), np.diff(y_pred)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-m) * np.sum(np.abs(y_true[m:] - y_true[:-m]))
    assert not np.isclose(scale, 0), "cannot use MASE with periodical signals"
    return errors / scale


def overall_percentage_error(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Computes the Overall Percentage Erroe (OPE):

    This function computes the OPE of `pred_series` with respect to `true_series`.
    Use `time_diff=True` when the time series has a strong auto-correlation to have a more accurate analysis.
    Use `time_diff` mainly when the time series are monotonic.
    Otherwise, it will be difficult to analyze the results.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the OPE of `pred_series` with respect to `true_series`.
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_pred = np.diff(y_true), np.diff(y_pred)
    y_true_sum, y_pred_sum = np.sum(np.array(y_true)), np.sum(np.array(y_pred))
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.


def marre(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Computes the Mean Absolute Ranged Relative Error (MARRE).

    This function computes the MARRE of `pred_series` with respect to `true_series`.
    Use `time_diff=True` when the time series has a strong auto-correlation to have a more accurate analysis.

    :param true_series: A TimeSeries.
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the MARRE of `pred_series` with respect to `true_series`.
    """
    y_true, y_hat = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_hat= np.diff(y_true), np.diff(y_hat)
    true_range = y_true.max() - y_true.min()
    return 100. * np.mean(np.abs((y_true - y_hat) / true_range))


def r2_score(true_series: TimeSeries, pred_series: TimeSeries, time_diff: bool = False) -> float:
    """
    Computes the coefficient of determination R2.
    Use `time_diff=True` when the time series has a strong auto-correlation to have a more accurate analysis.

    This function computes the R2 score of `pred_series` with respect to `true_series`.
    :param true_series: A TimeSeries
    :param pred_series: A TimeSeries to be compared with `true_series`.
    :param time_diff: If True, analyze the time differentiated series, instead of the index one.
    :return: A float, the coefficient R2
    """
    y_true, y_pred = _get_values_or_raise(true_series, pred_series)
    if time_diff:
        y_true, y_pred = np.diff(y_true), np.diff(y_pred)
    ss_errors = np.sum((y_true - y_pred)**2)
    y_hat = y_true.mean()
    ss_tot = np.sum((y_true-y_hat)**2)
    return 1 - ss_errors/ss_tot

