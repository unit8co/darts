"""
Metrics
-------

Some metrics to compare time series.
"""

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing import dtw
from darts.logging import get_logger, raise_log
from darts.metrics.utils import (
    METRIC_OUTPUT_TYPE,
    SMPL_AX,
    TIME_AX,
    _compute_score,
    _confusion_matrix,
    _get_error_scale,
    _get_quantile_intervals,
    _get_values_or_raise,
    _get_wrapped_metric,
    _LabelReduction,
    classification_support,
    interval_support,
    multi_ts_support,
    multivariate_support,
)

logger = get_logger(__name__)


@multi_ts_support
@multivariate_support
def err(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Error (ERR).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: y_t - \\hat{y}_t

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    return y_true - y_pred


@multi_ts_support
@multivariate_support
def merr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Error (MERR).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)}

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(err)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def ae(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Error (AE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: |y_t - \\hat{y}_t|

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    return np.abs(y_true - y_pred)


@multi_ts_support
@multivariate_support
def mae(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Error (MAE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{|y_t - \\hat{y}_t|}

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(ae)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def ase(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: int = 1,
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Scaled Error (ASE) (see [1]_ for more information on scaled forecasting errors).

    It is the Absolute Error (AE) scaled by the Mean AE (MAE) of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: \\frac{AE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`AE` is the Absolute
    Error (:func:`~darts.metrics.metrics.ae`), and :math:`E_m` is the Mean AE (MAE) of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MAE(y_{m:t_p}, y_{0:t_p - m}).

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` . This series serves to compute the scale of the error
        obtained by a naive forecaster on the training data.
    m
        The seasonality to use for differencing to compute the error scale :math:`E_m` (as described in the metric
        description). :math:`m=1` corresponds to a non-seasonal :math:`E_m` (e.g. naive repetition of the last observed
        value), whereas :math:`m>1` corresponds to a seasonal :math:`E_m`.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/
    """
    error_scale = _get_error_scale(insample, pred_series, m=m, metric="mae")
    errors = _get_wrapped_metric(ae)(
        actual_series,
        pred_series,
        intersect,
        q=q,
    )
    return errors / error_scale


@multi_ts_support
@multivariate_support
def mase(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: int = 1,
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Scaled Error (MASE) (see [1]_ for more information on scaled forecasting errors).

    It is the Mean Absolute Error (MAE) scaled by the MAE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{MAE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`MAE` is the Mean
    Absolute Error (:func:`~darts.metrics.metrics.mae`), and :math:`E_m` is the MAE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MAE(y_{m:t_p}, y_{0:t_p - m}).

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` . This series serves to compute the scale of the error
        obtained by a naive forecaster on the training data.
    m
        The seasonality to use for differencing to compute the error scale :math:`E_m` (as described in the metric
        description). :math:`m=1` corresponds to a non-seasonal :math:`E_m` (e.g. naive repetition of the last observed
        value), whereas :math:`m>1` corresponds to a seasonal :math:`E_m`.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/
    """
    return np.nanmean(
        _get_wrapped_metric(ase)(
            actual_series,
            pred_series,
            insample,
            m=m,
            intersect=intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def se(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Error (SE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: (y_t - \\hat{y}_t)^2.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    return (y_true - y_pred) ** 2


@multi_ts_support
@multivariate_support
def mse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Squared Error (MSE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(se)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def sse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: int = 1,
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Scaled Error (SSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Squared Error (SE) scaled by the Mean SE (MSE) of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: \\frac{SE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`SE` is the Squared
    Error (:func:`~darts.metrics.metrics.se`), and :math:`E_m` is the Mean SE (MSE) of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MSE(y_{m:t_p}, y_{0:t_p - m}).

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` . This series serves to compute the scale of the error
        obtained by a naive forecaster on the training data.
    m
        The seasonality to use for differencing to compute the error scale :math:`E_m` (as described in the metric
        description). :math:`m=1` corresponds to a non-seasonal :math:`E_m` (e.g. naive repetition of the last observed
        value), whereas :math:`m>1` corresponds to a seasonal :math:`E_m`.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/
    """
    error_scale = _get_error_scale(insample, pred_series, m=m, metric="mse")
    errors = _get_wrapped_metric(se)(
        actual_series,
        pred_series,
        intersect,
        q=q,
    )
    return errors / error_scale


@multi_ts_support
@multivariate_support
def msse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: int = 1,
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Squared Scaled Error (MSSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Mean Squared Error (MSE) scaled by the MSE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{MSE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`MSE` is the Mean
    Squared Error (:func:`~darts.metrics.metrics.mse`), and :math:`E_m` is the MSE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MSE(y_{m:t_p}, y_{0:t_p - m}).

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` . This series serves to compute the scale of the error
        obtained by a naive forecaster on the training data.
    m
        The seasonality to use for differencing to compute the error scale :math:`E_m` (as described in the metric
        description). :math:`m=1` corresponds to a non-seasonal :math:`E_m` (e.g. naive repetition of the last observed
        value), whereas :math:`m>1` corresponds to a seasonal :math:`E_m`.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/
    """
    return np.nanmean(
        _get_wrapped_metric(sse)(
            actual_series,
            pred_series,
            insample,
            m=m,
            intersect=intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def rmse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Error (RMSE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.sqrt(
        _get_wrapped_metric(mse)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        )
    )


@multi_ts_support
@multivariate_support
def rmsse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: int = 1,
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Scaled Error (RMSSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Root Mean Squared Error (RMSE) scaled by the RMSE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\frac{RMSE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`RMSE` is the Root
    Mean Squared Error (:func:`~darts.metrics.metrics.rmse`), and :math:`E_m` is the RMSE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = RMSE(y_{m:t_p}, y_{0:t_p - m}).

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` . This series serves to compute the scale of the error
        obtained by a naive forecaster on the training data.
    m
        The seasonality to use for differencing to compute the error scale :math:`E_m` (as described in the metric
        description). :math:`m=1` corresponds to a non-seasonal :math:`E_m` (e.g. naive repetition of the last observed
        value), whereas :math:`m>1` corresponds to a seasonal :math:`E_m`.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/
    """
    error_scale = _get_error_scale(insample, pred_series, m=m, metric="rmse")
    errors = _get_wrapped_metric(rmse)(
        actual_series,
        pred_series,
        intersect,
        q=q,
    )
    return errors / error_scale


@multi_ts_support
@multivariate_support
def sle(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Log Error (SLE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column, (optional) quantile, and time step :math:`t` as:

    .. math:: \\left(\\log{(y_t + 1)} - \\log{(\\hat{y} + 1)}\\right)^2

    using the natural logarithm.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    y_true, y_pred = np.log(y_true + 1), np.log(y_pred + 1)
    return (y_true - y_pred) ** 2


@multi_ts_support
@multivariate_support
def rmsle(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Log Error (RMSLE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{\\left(\\log{(y_t + 1)} - \\log{(\\hat{y}_t + 1)}\\right)^2}}

    using the natural logarithm.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.sqrt(
        np.nanmean(
            _get_wrapped_metric(sle)(
                actual_series,
                pred_series,
                intersect,
                q=q,
            ),
            axis=TIME_AX,
        )
    )


@multi_ts_support
@multivariate_support
def ape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Percentage Error (APE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and time step :math:`t` with:

    .. math:: 100 \\cdot \\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Absolute Scaled Error (:func:`~darts.metrics.metrics.ase`) in these cases.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If `actual_series` contains some zeros.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    if not (y_true != 0).all():
        raise_log(
            ValueError(
                "`actual_series` must be strictly positive to compute the MAPE."
            ),
            logger=logger,
        )
    return 100.0 * np.abs((y_true - y_pred) / y_true)


@multi_ts_support
@multivariate_support
def mape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Percentage Error (MAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and (optional) quantile with:

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T}{\\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|}

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Mean Absolute Scaled Error (:func:`~darts.metrics.metrics.mase`) in these cases.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If `actual_series` contains some zeros.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    return np.nanmean(
        _get_wrapped_metric(ape)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def wmape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Weighted Mean Absolute Percentage Error (WMAPE). (see [1]_ for more information).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and (optional) quantile with:

    .. math:: 100 \\cdot \\frac{\\sum_{t=1}^T |y_t - \\hat{y}_t|}{\\sum_{t=1}^T |y_t|}

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If `actual_series` contains some zeros.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mean_absolute_percentage_error#WMAPE
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )

    return (
        100.0
        * np.nansum(np.abs(y_true - y_pred), axis=TIME_AX)
        / np.nansum(np.abs(y_true), axis=TIME_AX)
    )


@multi_ts_support
@multivariate_support
def sape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """symmetric Absolute Percentage Error (sAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column, (optional) quantile and time step :math:`t` with:

    .. math::
        200 \\cdot \\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|}

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0` for some
    :math:`t`. Consider using the Absolute Scaled Error (:func:`~darts.metrics.metrics.ase`)  in these cases.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If `actual_series` and `pred_series` contain some zeros at the same time index.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    numerator = 200*np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)

@multi_ts_support
@multivariate_support
def smape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """symmetric Mean Absolute Percentage Error (sMAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and (optional) quantile with:

    .. math::
        200 \\cdot \\frac{1}{T}
        \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|} }

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0`
    for some :math:`t`. Consider using the Mean Absolute Scaled Error (:func:`~darts.metrics.metrics.mase`) in these
    cases.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If the `actual_series` and the `pred_series` contain some zeros at the same time index.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    return np.nanmean(
        _get_wrapped_metric(sape)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def ope(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Overall Percentage Error (OPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and (optional) quantile with:

    .. math:: 100 \\cdot \\left| \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}} \\right|.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_true_sum, y_pred_sum = (
        np.nansum(y_true, axis=TIME_AX),
        np.nansum(y_pred, axis=TIME_AX),
    )
    if not (y_true_sum > 0).all():
        raise_log(
            ValueError(
                "The series of actual value cannot sum to zero when computing OPE."
            ),
            logger=logger,
        )
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100.0


@multi_ts_support
@multivariate_support
def arre(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Ranged Relative Error (ARRE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column, (optional) quantile and time step :math:`t` with:

    .. math:: 100 \\cdot \\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} - \\min_t{y_t}} \\right|

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If :math:`\\max_t{y_t} = \\min_t{y_t}`.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_max, y_min = np.nanmax(y_true, axis=TIME_AX), np.nanmin(y_true, axis=TIME_AX)
    if not (y_max > y_min).all():
        raise_log(
            ValueError(
                "The difference between the max and min values must "
                "be strictly positive to compute the MARRE."
            ),
            logger=logger,
        )
    true_range = y_max - y_min
    return 100.0 * np.abs((y_true - y_pred) / true_range)


@multi_ts_support
@multivariate_support
def marre(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Ranged Relative Error (MARRE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and (optional) quantile with:

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} -
              \\min_t{y_t}} \\right|}

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Raises
    ------
    ValueError
        If :math:`\\max_t{y_t} = \\min_t{y_t}`.

    float
        A single metric score for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - a single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(arre)(
            actual_series,
            pred_series,
            intersect,
            q=q,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def r2_score(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Determination :math:`R^2` (see [1]_ for more details).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as:

    .. math:: 1 - \\frac{\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}{\\sum_{t=1}^T{(y_t - \\bar{y})^2}},

    where :math:`\\bar{y}` is the mean of :math:`y` over all time steps.

    This metric is not symmetric.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    ss_errors = np.nansum((y_true - y_pred) ** 2, axis=TIME_AX)
    y_hat = np.nanmean(y_true, axis=TIME_AX)
    ss_tot = np.nansum((y_true - y_hat) ** 2, axis=TIME_AX)
    return 1 - ss_errors / ss_tot


@multi_ts_support
@multivariate_support
def coefficient_of_variation(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Variation (percentage).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and (optional) quantile as a percentage value with:

    .. math:: 100 \\cdot \\text{RMSE}(y_t, \\hat{y}_t) / \\bar{y},

    where :math:`RMSE` is the Root Mean Squared Error (:func:`~darts.metrics.metrics.rmse`), and :math:`\\bar{y}` is
    the average of :math:`y` over all time steps.

    If :math:`\\hat{y}_t` are stochastic (contains several samples) or quantile predictions, use parameter `q` to
    specify on which quantile(s) to compute the metric on. By default, it uses the median 0.5 quantile
    (over all samples, or, if given, the quantile prediction itself).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        Optionally, the quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    # not calling rmse as y_true and y_pred are np.ndarray
    return (
        100
        * np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=TIME_AX))
        / np.nanmean(y_true, axis=TIME_AX)
    )


# Dynamic Time Warping
@multi_ts_support
@multivariate_support
def dtw_metric(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    metric: Callable[
        [
            Union[TimeSeries, Sequence[TimeSeries]],
            Union[TimeSeries, Sequence[TimeSeries]],
        ],
        METRIC_OUTPUT_TYPE,
    ] = mae,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    **kwargs,
) -> METRIC_OUTPUT_TYPE:
    """
    Applies Dynamic Time Warping to `actual_series` and `pred_series` before passing it into the metric.
    Enables comparison between series of different lengths, phases and time indices.

    Defaults to using :func:`~darts.metrics.metrics.mae` as a metric.

    See :func:`~darts.dataprocessing.dtw.dtw.dtw` for more supported parameters.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    metric
        The selected metric with signature '[[TimeSeries, TimeSeries], float]' to use. Default: `mae`.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    alignment = dtw.dtw(actual_series, pred_series, **kwargs)
    warped_actual_series, warped_pred_series = alignment.warped()
    return _get_wrapped_metric(metric)(
        warped_actual_series,
        warped_pred_series,
    )


@multi_ts_support
@multivariate_support
def qr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Union[float, list[float], tuple[np.ndarray, pd.Index]] = 0.5,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Quantile Risk (QR)

    QR is a metric that quantifies the accuracy of a specific quantile :math:`q` from the predicted value
    distribution of a stochastic/probabilistic `pred_series` containing N samples.

    The main difference to the Quantile Loss (QL) is that QR computes the quantile and loss on the aggregate of all
    sample values summed up along the time axis (QL computes the quantile and loss per time step).

    For the true series :math:`y` and predicted stochastic/probabilistic series (containing N samples) :math:`\\hat{y}`
    of of shape :math:`T \\times N`, it is computed per column/component and quantile as:

    .. math:: 2 \\frac{QL(Z, \\hat{Z}_q)}{Z},

    where :math:`QL` is the Quantile Loss (:func:`~darts.metrics.metrics.ql`), :math:`Z = \\sum_{t=1}^{T} y_t` is
    the sum of all target/actual values, :math:`\\hat{Z} = \\sum_{t=1}^{T} \\hat{y}_t` is the sum of all predicted
    samples along the time axis, and :math:`\\hat{Z}_q` is the quantile :math:`q` of that sum.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        The quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    if not pred_series.is_stochastic:
        raise_log(
            ValueError(
                "quantile risk (qr) should only be computed for stochastic predicted TimeSeries."
            ),
            logger=logger,
        )

    z_true, z_hat = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        q=None,
        remove_nan_union=True,
    )
    z_true = np.nansum(z_true, axis=TIME_AX)
    z_hat = np.nansum(
        z_hat, axis=TIME_AX
    )  # aggregate all individual sample realizations
    # quantile loss
    q, _ = q
    z_hat_rho = np.quantile(
        z_hat, q=q, axis=1
    ).T  # get the quantile from aggregated samples

    errors = z_true - z_hat_rho
    losses = 2 * np.maximum((q - 1) * errors, q * errors)
    return losses / z_true


@multi_ts_support
@multivariate_support
def ql(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Union[float, list[float], tuple[np.ndarray, pd.Index]] = 0.5,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Quantile Loss (QL).

    Also known as Pinball Loss. QL is a metric that quantifies the accuracy of a specific quantile :math:`q` from the
    predicted deterministic quantiles or value distribution of a stochastic/probabilistic `pred_series` containing N
    samples.

    QL computes the quantile of all sample values and the loss per time step.

    For the true series :math:`y` and predicted stochastic/probabilistic series (containing N samples) :math:`\\hat{y}`
    of of shape :math:`T \\times N`, it is computed per column/component, quantile and time step :math:`t` as:

    .. math:: 2 \\max((q - 1) (y_t - \\hat{y}_{t,q}), q (y_t - \\hat{y}_{t,q})),

    where :math:`\\hat{y}_{t,q}` is quantile value :math:`q` (of all predicted quantiles or samples) at time :math:`t`.
    The factor `2` makes the loss more interpretable, as for `q=0.5` the loss is identical to the Absolute Error
    (:func:`~darts.metrics.metrics.ae`).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        The quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time-
        and component reductions, and shape (n time steps, n quantiles) without time- but with component reduction and
        `len(q) > 1`. For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        q=q,
        remove_nan_union=True,
    )
    q, _ = q
    errors = y_true - y_pred
    losses = 2.0 * np.maximum((q - 1) * errors, q * errors)
    return losses


@multi_ts_support
@multivariate_support
def mql(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Union[float, list[float], tuple[np.ndarray, pd.Index]] = 0.5,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Quantile Loss (MQL).

    Also known as Pinball Loss. QL is a metric that quantifies the accuracy of a specific quantile :math:`q` from the
    predicted deterministic quantiles or value distribution of a stochastic/probabilistic `pred_series` containing N
    samples.

    MQL first computes the quantile of all sample values and the loss per time step, and then takes the mean over the
    time axis.

    For the true series :math:`y` and predicted stochastic/probabilistic series (containing N samples) :math:`\\hat{y}`
    of of shape :math:`T \\times N`, it is computed per column/component and quantile as:

    .. math:: 2 \\frac{1}{T}\\sum_{t=1}^T{\\max((q - 1) (y_t - \\hat{y}_{t,q}), q (y_t - \\hat{y}_{t,q}))},

    where :math:`\\hat{y}_{t,q}` is quantile value :math:`q` (of all predicted quantiles or samples) at time :math:`t`.
    The factor `2` makes the loss more interpretable, as for `q=0.5` the loss is identical to the Mean Absolute Error
    (:func:`~darts.metrics.metrics.mae`).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q
        The quantile (float [0, 1]) or list of quantiles of interest to compute the metric on.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n quantiles,) without component reduction,
        and shape (n quantiles,) with component reduction and `len(q) > 1`.
        For:

        - the same input arguments that result in the `float` return case from above but with `len(q) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(ql)(
            actual_series,
            pred_series,
            q=q,
            intersect=intersect,
        ),
        axis=TIME_AX,
    )


@interval_support
@multi_ts_support
@multivariate_support
def iw(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Interval Width (IW).

    IL gives the width / length of predicted quantile intervals.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step
    :math:`t` as:

    .. math:: U_t - L_t,

    where :math:`U_t` are the predicted upper bound quantile values :math:`\\hat{y}_{q_h,t}` (of all predicted
    quantiles or samples) at time :math:`t`, and :math:`L_t` are the predicted lower bound quantile values
    :math:`\\hat{y}_{q_l,t}`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n q intervals) without time-
        and component reductions, and shape (n time steps, n q intervals) without time- but with component reduction and
        `len(q_interval) > 1`. For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_pred_lo, y_pred_hi = _get_quantile_intervals(y_pred, q=q, q_interval=q_interval)
    return y_pred_hi - y_pred_lo


@interval_support
@multi_ts_support
@multivariate_support
def miw(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Interval Width (MIW).

    MIW gives the time-aggregated width / length of predicted quantile intervals.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step
    :math:`t` as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{U_t - L_t},

    where :math:`U_t` are the predicted upper bound quantile values :math:`\\hat{y}_{q_h,t}` (of all predicted
    quantiles or samples) at time :math:`t`, and :math:`L_t` are the predicted lower bound quantile values
    :math:`\\hat{y}_{q_l,t}`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n q intervals,) without component reduction,
        and shape (n q intervals,) with component reduction and `len(q_interval) > 1`.
        For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(iw, n_wrappers=3)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            q_interval=q_interval,
        ),
        axis=TIME_AX,
    )


@interval_support
@multi_ts_support
@multivariate_support
def iws(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Interval Winkler Score (IWS) [1]_.

    IWS gives the length / width of the quantile intervals plus a penalty if the observation is outside the interval.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math::
        \\begin{equation}
            \\begin{cases}
                (U_t - L_t) + \\frac{1}{q_l} (L_t - y_t) & \\text{if } y_t < L_t \\\\
                (U_t - L_t) & \\text{if } L_t \\leq y_t \\leq U_t \\\\
                (U_t - L_t) + \\frac{1}{1 - q_h} (y_t - U_t) & \\text{if } y_t > U_t
            \\end{cases}
        \\end{equation}

    where :math:`U_t` are the predicted upper bound quantile values :math:`\\hat{y}_{q_h,t}` (of all predicted
    quantiles or samples) at time :math:`t`, and :math:`L_t` are the predicted lower bound quantile values
    :math:`\\hat{y}_{q_l,t}`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n q intervals) without time-
        and component reductions, and shape (n time steps, n q intervals) without time- but with component reduction and
        `len(q_interval) > 1`. For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://otexts.com/fpp3/distaccuracy.html
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_pred_lo, y_pred_hi = _get_quantile_intervals(y_pred, q=q, q_interval=q_interval)
    interval_width = y_pred_hi - y_pred_lo

    # `c_alpha = 2 / alpha` corresponds to:
    #   - `1 / (1 - q_hi)` for the high quantile
    #   - `1 / q_lo` for the low quantile
    c_alpha_hi = 1 / (1 - q_interval[:, 1])
    c_alpha_lo = 1 / q_interval[:, 0]

    score = np.where(
        y_true < y_pred_lo,
        interval_width + c_alpha_lo * (y_pred_lo - y_true),
        np.where(
            y_true > y_pred_hi,
            interval_width + c_alpha_hi * (y_true - y_pred_hi),
            interval_width,
        ),
    )
    return score


@interval_support
@multi_ts_support
@multivariate_support
def miws(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Interval Winkler Score (IWS) [1]_.

    MIWS gives the time-aggregated length / width of the quantile intervals plus a penalty if the observation is
    outside the interval.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{W_t(y_t, \\hat{y}_{t}, q_h, q_l)},

    where :math:`W` is the Winkler Score :func:`~darts.metrics.metrics.iws`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n q intervals,) without component reduction,
        and shape (n q intervals,) with component reduction and `len(q_interval) > 1`.
        For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://otexts.com/fpp3/distaccuracy.html
    """
    return np.nanmean(
        _get_wrapped_metric(iws, n_wrappers=3)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            q_interval=q_interval,
        ),
        axis=TIME_AX,
    )


@interval_support
@multi_ts_support
@multivariate_support
def ic(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Interval Coverage (IC).

    IC gives a binary outcome with `1` if the observation is within the interval, and `0` otherwise.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math::
        \\begin{equation}
            \\begin{cases}
                1 & \\text{if } L_t < y_t < U_t \\\\
                0 & \\text{otherwise}
            \\end{cases}
        \\end{equation}

    where :math:`U_t` are the predicted upper bound quantile values :math:`\\hat{y}_{q_h,t}` (of all predicted
    quantiles or samples) at time :math:`t`, and :math:`L_t` are the predicted lower bound quantile values
    :math:`\\hat{y}_{q_l,t}`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n q intervals) without time-
        and component reductions, and shape (n time steps, n q intervals) without time- but with component reduction and
        `len(q_interval) > 1`. For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_pred_lo, y_pred_hi = _get_quantile_intervals(y_pred, q=q, q_interval=q_interval)
    return np.where((y_pred_lo <= y_true) & (y_true <= y_pred_hi), 1.0, 0.0)


@interval_support
@multi_ts_support
@multivariate_support
def mic(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Interval Coverage (MIC).

    MIC gives the time-aggregated Interval Coverage :func:`~darts.metrics.metrics.ic` - the ratio of observations
    being within the interval.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{C(y_t, \\hat{y}_{t}, q_h, q_l)},

    where :math:`C` is the Interval Coverage :func:`~darts.metrics.metrics.ic`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n q intervals,) without component reduction,
        and shape (n q intervals,) with component reduction and `len(q_interval) > 1`.
        For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(ic, n_wrappers=3)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            q_interval=q_interval,
        ),
        axis=TIME_AX,
    )


@interval_support
@multi_ts_support
@multivariate_support
def incs_qr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    symmetric: bool = True,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Interval Non-Conformity Score for Quantile Regression (INCS_QR).

    INCS_QR gives the absolute error to the closest predicted quantile interval bound when the observation is outside
    the interval. Otherwise, it gives the negative absolute error to the closer bound.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math:: \\max(L_t - y_t, y_t - U_t)

    where :math:`U_t` are the predicted upper bound quantile values :math:`\\hat{y}_{q_h,t}` (of all predicted
    quantiles or samples) at time :math:`t`, and :math:`L_t` are the predicted lower bound quantile values
    :math:`\\hat{y}_{q_l,t}`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    symmetric
        Whether to return symmetric non-conformity scores. If `False`, returns asymmetric scores (individual scores
        for lower- and upper quantile interval bounds; returned in the component axis).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    time_reduction
        Optionally, a function to aggregate the metrics over the time axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(c,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        time axis. If `None`, will return a metric per time step.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n q intervals) without time-
        and component reductions, and shape (n time steps, n q intervals) without time- but with component reduction and
        `len(q_interval) > 1`. For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=True,
        q=q,
    )
    y_pred_lo, y_pred_hi = _get_quantile_intervals(y_pred, q=q, q_interval=q_interval)
    if symmetric:
        return np.maximum(y_pred_lo - y_true, y_true - y_pred_hi)
    else:
        return np.concatenate([y_pred_lo - y_true, y_true - y_pred_hi], axis=SMPL_AX)


@interval_support
@multi_ts_support
@multivariate_support
def mincs_qr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q_interval: Union[tuple[float, float], Sequence[tuple[float, float]]] = None,
    symmetric: bool = True,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Interval Non-Conformity Score for Quantile Regression (MINCS_QR).

    MINCS_QR gives the time-aggregated INCS_QR :func:`~darts.metrics.metrics.incs_qr`.

    For the true series :math:`y` and predicted stochastic or quantile series :math:`\\hat{y}` of length :math:`T`,
    it is computed per component/column, quantile interval :math:`(q_l,q_h)`, and time step :math:`t` as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{INCS_QR(y_t, \\hat{y}_{t}, q_h, q_l)},

    where :math:`INCS_QR` is the Interval Non-Conformity Score for Quantile Regression
    :func:`~darts.metrics.metrics.incs_qr`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    q_interval
        The quantile interval(s) to compute the metric on. Must be a tuple (single interval) or sequence of tuples
        (multiple intervals) with elements (low quantile, high quantile).
    symmetric
        Whether to return symmetric non-conformity scores. If `False`, returns asymmetric scores (individual scores
        for lower- and upper quantile interval bounds; returned in the component axis).
    q
        Quantiles `q` not supported by this metric; use `q_interval` instead.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for (with `len(q_interval) <= 1`):

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n q intervals,) without component reduction,
        and shape (n q intervals,) with component reduction and `len(q_interval) > 1`.
        For:

        - the input from the `float` return case above but with `len(q_interval) > 1`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(incs_qr, n_wrappers=3)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            q_interval=q_interval,
            symmetric=symmetric,
        ),
        axis=TIME_AX,
    )


@classification_support
@multi_ts_support
@multivariate_support
def accuracy(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Accuracy Score [1]_.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T\\mathbb{I}(y_t = \\hat{y}_t)

    Where :math:`\\mathbb{I}` is the indicator function.

    If :math:`\\hat{y}_t` are stochastic (contains several samples), it takes the label with the highest count from
    each time step and component.
    If :math:`\\hat{y}_t` represent the predict class label probabilities, it takes the label with the highest
    probability from each time step and component.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction.
        For:

        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Precision_and_recall
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        is_classification=True,
    )
    return np.nanmean(np.array(y_true == y_pred, dtype=y_true.dtype), axis=TIME_AX)


@classification_support
@multi_ts_support
@multivariate_support
def precision(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    labels: Optional[Union[int, list[int], np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    label_reduction: Union[Optional[str], _LabelReduction] = _LabelReduction.MACRO,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Precision Score [1]_.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{TP}{TP + FP}

    Where :math:`TP` are the true positives, and :math:`FP` are the false positives.

    If :math:`\\hat{y}_t` are stochastic (contains several samples), it takes the label with the highest count from
    each time step and component.
    If :math:`\\hat{y}_t` represent the predict class label probabilities, it takes the label with the highest
    probability from each time step and component.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    labels
        Optionally, the labels and their order to compute the metric on. If `None`, will use all unique values from the
        actual and predicted series.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    label_reduction
        The method to reduce the label-specific metrics. Can be one of:

        - ``None``: no reduction, returns a metric per label.
        - ``'micro'``: computes the metric globally by counting the total true positives, false negatives and false
          positives.
        - ``'macro'``: computes the metric for each label, and returns the unweighted mean.
        - ``'weighted'``: computes the metric for each label, and returns the weighted mean by support (the number of
          true instances for each label).
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `label_reduction != None`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n labels,) without component reduction,
        and shape (n labels,) with component reduction. `n labels` is the number of labels when `label_reduction=None`,
        and `1` otherwise.
        For:

        - the same input arguments that result in the `float` return case from above but with `label_reduction=None`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Precision_and_recall
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        is_classification=True,
    )

    def _score_func(tn, fp, fn, tp) -> np.ndarray:
        """Compute score from confusion matrix components."""
        return tp / (tp + fp)

    return _compute_score(
        y_true,
        y_pred,
        score_func=_score_func,
        label_reduction=label_reduction,
        labels=labels,
    )


@classification_support
@multi_ts_support
@multivariate_support
def recall(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    labels: Optional[Union[int, list[int], np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    label_reduction: Union[Optional[str], _LabelReduction] = _LabelReduction.MACRO,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Recall Score [1]_.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{TP}{TP + FN}

    Where :math:`TP` are the true positives, and :math:`FN` are the false negatives.

    If :math:`\\hat{y}_t` are stochastic (contains several samples), it takes the label with the highest count from
    each time step and component.
    If :math:`\\hat{y}_t` represent the predict class label probabilities, it takes the label with the highest
    probability from each time step and component.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    labels
        Optionally, the labels and their order to compute the metric on. If `None`, will use all unique values from the
        actual and predicted series.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    label_reduction
        The method to reduce the label-specific metrics. Can be one of:

        - ``None``: no reduction, returns a metric per label.
        - ``'micro'``: computes the metric globally by counting the total true positives, false negatives and false
          positives.
        - ``'macro'``: computes the metric for each label, and returns the unweighted mean.
        - ``'weighted'``: computes the metric for each label, and returns the weighted mean by support (the number of
          true instances for each label).
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `label_reduction != None`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n labels,) without component reduction,
        and shape (n labels,) with component reduction. `n labels` is the number of labels when `label_reduction=None`,
        and `1` otherwise.
        For:

        - the same input arguments that result in the `float` return case from above but with `label_reduction=None`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Precision_and_recall
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        is_classification=True,
    )

    def _score_func(tn, fp, fn, tp) -> np.ndarray:
        """Compute score from confusion matrix components."""
        return tp / (tp + fn)

    return _compute_score(
        y_true,
        y_pred,
        score_func=_score_func,
        label_reduction=label_reduction,
        labels=labels,
    )


@classification_support
@multi_ts_support
@multivariate_support
def f1(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    labels: Optional[Union[int, list[int], np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    label_reduction: Union[Optional[str], _LabelReduction] = _LabelReduction.MACRO,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """F1 Score [1]_.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{2TP}{2TP + FP + FN}

    Where :math:`TP` are the true positives, :math:`FP` are the false positives, and :math:`FN` are the false
    negatives.

    If :math:`\\hat{y}_t` are stochastic (contains several samples), it takes the label with the highest count from
    each time step and component.
    If :math:`\\hat{y}_t` represent the predict class label probabilities, it takes the label with the highest
    probability from each time step and component.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    labels
        Optionally, the labels and their order to compute the metric on. If `None`, will use all unique values from the
        actual and predicted series.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    label_reduction
        The method to reduce the label-specific metrics. Can be one of:

        - ``None``: no reduction, returns a metric per label.
        - ``'micro'``: computes the metric globally by counting the total true positives, false negatives and false
          positives.
        - ``'macro'``: computes the metric for each label, and returns the unweighted mean.
        - ``'weighted'``: computes the metric for each label, and returns the weighted mean by support (the number of
          true instances for each label).
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    float
        A single metric score (when `label_reduction != None`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components * n labels,) without component reduction,
        and shape (n labels,) with component reduction. `n labels` is the number of labels when `label_reduction=None`,
        and `1` otherwise.
        For:

        - the same input arguments that result in the `float` return case from above but with `label_reduction=None`.
        - a single multivariate series and at least `component_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    list[float]
        Same as for type `float` but for a sequence of series.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Precision_and_recall
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        is_classification=True,
    )

    def _score_func(tn, fp, fn, tp) -> np.ndarray:
        """Compute score from confusion matrix components."""
        return 2 * tp / (2 * tp + fp + fn)

    scores = _compute_score(
        y_true,
        y_pred,
        score_func=_score_func,
        label_reduction=label_reduction,
        labels=labels,
    )
    return scores


@classification_support
@multi_ts_support
@multivariate_support
def confusion_matrix(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    labels: Optional[Union[int, list[int], np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nansum,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Confusion Matrix (CM) [1]_.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and label as:

    The confusion matrix :math:`C` is such that :math:`C_{i,j}` is equal to the number of observations
    :math:`y` known to be in group :math:`i` and predicted :math:`\\hat{y}` to be in group :math:`j`.

    If :math:`\\hat{y}_t` are stochastic (contains several samples), it takes the label with the highest count from
    each time step and component.
    If :math:`\\hat{y}_t` represent the predict class label probabilities, it takes the label with the highest
    probability from each time step and component.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    labels
        Optionally, the labels and their order to compute the metric on. If `None`, will use all unique values from the
        actual and predicted series.
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over multiple series. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. For example with `np.nanmean`, will return the average over all series metrics. If `None`, will
        return a metric per component.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress.

    Returns
    -------
    np.ndarray
        The Confusion Matrix as a numpy array for a single series. The array has shape (n components, n labels,
        n labels) without component reduction, and shape (n labels, n labels) with component reduction.
    list[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Confusion_matrix
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        is_classification=True,
    )
    return _confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        compute_multilabel=False,
    )[0]
