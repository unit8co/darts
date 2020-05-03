"""
Metrics
-------

Some metrics to compare time series.
"""

import numpy as np
from typing import Tuple
from ..timeseries import TimeSeries
from ..utils.statistics import check_seasonality
from ..logging import raise_if_not, get_logger
from warnings import warn
from typing import Optional

logger = get_logger(__name__)


def _get_values_or_raise(series_a: TimeSeries, series_b: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the numpy values of two time series, launching an Exception if time series cannot be compared
    """
    raise_if_not(series_a.has_same_time_as(series_b), 'The two time series must have same time index.' \
                                                '\nFirst series: {}\nSecond series: {}'.format(
                                                series_a.time_index(), series_b.time_index()), logger)
    return series_a.values(), series_b.values()


def mae(series1: TimeSeries, series2: TimeSeries) -> float:
    """ Mean Absolute Error (MAE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(|y^1_t - y^2_t|)}.

    Parameters
    ----------
    series1
        The first time series
    series2
        The second time series

    Returns
    -------
    float
        The Mean Absolute Error (MAE)
    """

    y1, y2 = _get_values_or_raise(series1, series2)
    return np.mean(np.abs(y1 - y2))


def mse(series1: TimeSeries, series2: TimeSeries) -> float:
    """ Mean Squared Error (MSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}.

    Parameters
    ----------
    series1
        The first time series
    series2
        The second time series

    Returns
    -------
    float
        The Mean Squared Error (MSE)
    """

    y_true, y_pred = _get_values_or_raise(series1, series2)
    return np.mean((y_true - y_pred)**2)


def rmse(series1: TimeSeries, series2: TimeSeries) -> float:
    """ Root Mean Squared Error (RMSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}}.

    Parameters
    ----------
    series1
        The first time series
    series2
        The second time series

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE)
    """
    return np.sqrt(mse(series1, series2))


def rmsle(series1: TimeSeries, series2: TimeSeries) -> float:
    """ Root Mean Squared Log Error (RMSLE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{\\left(\\log{(y^1_t + 1)} - \\log{(y^2_t + 1)}\\right)^2}},

    using the natural logarithm.

    Parameters
    ----------
    series1
        The first time series
    series2
        The second time series

    Returns
    -------
    float
        The Root Mean Squared Log Error (RMSLE)
    """

    y1, y2 = _get_values_or_raise(series1, series2)
    y1, y2 = np.log(y1 + 1), np.log(y2 + 1)
    return np.sqrt(np.mean((y1 - y2)**2))


def coefficient_of_variation(actual_series: TimeSeries, pred_series: TimeSeries) -> float:
    """ Coefficient of Variation (percentage).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`,
    it is a percentage value, computed as

    .. math:: 100 \\cdot \\text{RMSE}(y_t, \\hat{y}_t) / \\bar{y_t},

    where :math:`\\text{RMSE}()` denotes the root mean squred error, and :math:`\\bar{y_t}` is the average of :math:`y_t`.

    Parameters
    ----------
    actual_series
        The series of actual values
    pred_series
        The series of predicted values

    Returns
    -------
    float
        The Coefficient of Variation
    """

    return 100 * rmse(actual_series, pred_series) / actual_series.mean()


def mape(actual_series: TimeSeries, pred_series: TimeSeries) -> float:
    """ Mean Absolute Percentage Error (MAPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T}{\\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|}.

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Mean Absolute Scaled Error (MASE) in these cases.

    Parameters
    ----------
    actual_series
        The series of actual values
    pred_series
        The series of predicted values

    Raises
    ------
    ValueError
        If the actual series contains some zeros.

    Returns
    -------
    float
        The Mean Absolute Percentage Error (MAPE)
    """

    y_true, y_hat = _get_values_or_raise(actual_series, pred_series)
    raise_if_not(all(y_true > 0), 'The actual series must be strictly positive to compute the MAPE.')
    return 100. * np.mean(np.abs((y_true - y_hat) / y_true))


def mase(actual_series: TimeSeries, pred_series: TimeSeries, m: Optional[int] = 1) -> float:
    """ Mean Absolute Scaled Error (MASE).

    See `the Wikipedia page <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_
    for details about the MASE and how it is computed.

    Parameters
    ----------
    actual_series
        The series of actual values
    pred_series
        The series of predicted values
    m
        Optionally, the seasonality to use for differencing.
        `m=1` corresponds to the non-seasonal MASE, whereas `m>1` corresponds to seasonal MASE.
        If `m=None`, it will be tentatively inferred
        from the auto-correlation function (ACF). It will fall back to a value of 1 if this fails.

    Returns
    -------
    float
        The Mean Absolute Scaled Error (MASE)
    """

    if m is None:
        test_season, m = check_seasonality(actual_series)
        if not test_season:
            warn("No seasonality found when computing MASE. Fixing the period to 1.", UserWarning)
            m = 1
    y_true, y_hat = _get_values_or_raise(actual_series, pred_series)
    errors = np.sum(np.abs(y_true - y_hat))
    t = y_true.size
    scale = t/(t-m) * np.sum(np.abs(y_true[m:] - y_true[:-m]))
    raise_if_not(not np.isclose(scale, 0), "cannot use MASE with periodical signals", logger)
    return errors / scale


def ope(actual_series: TimeSeries, pred_series: TimeSeries) -> float:
    """ Overall Percentage Error (OPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\left| \\frac{\\sum_{t=1}^{T}{y_t} - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}} \\right|.

    Parameters
    ----------
    actual_series
        The series of actual values
    pred_series
        The series of predicted values

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        The Overall Percentage Error (OPE)
    """

    y_true, y_pred = _get_values_or_raise(actual_series, pred_series)
    y_true_sum, y_pred_sum = np.sum(y_true), np.sum(y_pred)
    raise_if_not(y_true_sum > 0, 'The series of actual value cannot sum to zero when computing OPE.')
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.


def marre(actual_series: TimeSeries, pred_series: TimeSeries) -> float:
    """ Mean Absolute Ranged Relative Error (MARRE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} - \\min_t{y_t}} \\right|}

    Parameters
    ----------
    actual_series
        The series of actual values
    pred_series
        The series of predicted values

    Raises
    ------
    ValueError
        If :math:`\\max_t{y_t} = \\min_t{y_t}`.

    Returns
    -------
    float
        The Mean Absolute Ranged Relative Error (MARRE)
    """

    y_true, y_hat = _get_values_or_raise(actual_series, pred_series)
    raise_if_not(max(y_true) > min(y_true), 'The difference between the max and min values must be strictly'
                                             'positive to compute the MARRE.')
    true_range = max(y_true) - min(y_true)
    return 100. * np.mean(np.abs((y_true - y_hat) / true_range))


def r2_score(series1: TimeSeries, series2: TimeSeries) -> float:
    """ Coefficient of Determination :math:`R^2`.

    See `the Wikipedia page <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    for details about the :math:`R^2` score and how it is computed.

    Parameters
    ----------
    series1
        The first time series
    series2
        The second time series

    Returns
    -------
    float
        The Coefficient of Determination :math:`R^2`
    """

    y1, y2 = _get_values_or_raise(series1, series2)
    ss_errors = np.sum((y1 - y2)**2)
    y_hat = y1.mean()
    ss_tot = np.sum((y1-y_hat)**2)
    return 1 - ss_errors/ss_tot
