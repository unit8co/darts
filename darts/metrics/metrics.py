"""
Metrics
-------

Some metrics to compare time series.
"""

from functools import wraps
from inspect import signature
from typing import Callable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np

from darts import TimeSeries
from darts.dataprocessing import dtw
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.utils.statistics import check_seasonality

logger = get_logger(__name__)
TIME_AX = 0

# Note: for new metrics added to this module to be able to leverage the two decorators, it is required both having
# the `actual_series` and `pred_series` parameters, and not having other ``Sequence`` as args (since these decorators
# don't "unpack" parameters different from `actual_series` and `pred_series`). In those cases, the new metric must take
# care of dealing with Sequence[TimeSeries] and multivariate TimeSeries on its own (See mase() implementation).
METRIC_OUTPUT_TYPE = Union[float, List[float], np.ndarray, List[np.ndarray]]
METRIC_TYPE = Callable[
    [Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]],
    METRIC_OUTPUT_TYPE,
]


def multi_ts_support(func) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """
    This decorator further adapts the metrics that took as input two univariate/multivariate ``TimeSeries`` instances,
    adding support for equally-sized sequences of ``TimeSeries`` instances. The decorator computes the pairwise metric
    for ``TimeSeries`` with the same indices, and returns a float value that is computed as a function of all the
    pairwise metrics using a `series_reduction` subroutine passed as argument to the metric function.

    If a 'Sequence[TimeSeries]' is passed as input, this decorator provides also parallelisation of the metric
    evaluation regarding different ``TimeSeries`` (if the `n_jobs` parameter is not set 1).
    """

    @wraps(func)
    def wrapper_multi_ts_support(*args, **kwargs):
        actual_series = (
            kwargs["actual_series"] if "actual_series" in kwargs else args[0]
        )
        pred_series = (
            kwargs["pred_series"]
            if "pred_series" in kwargs
            else args[0] if "actual_series" in kwargs else args[1]
        )

        n_jobs = kwargs.pop("n_jobs", signature(func).parameters["n_jobs"].default)
        verbose = kwargs.pop("verbose", signature(func).parameters["verbose"].default)

        if not isinstance(n_jobs, int):
            raise_log(ValueError("n_jobs must be an integer"), logger=logger)
        if not isinstance(verbose, bool):
            raise_log(ValueError("verbose must be a bool"), logger=logger)

        actual_series = (
            [actual_series]
            if not isinstance(actual_series, Sequence)
            else actual_series
        )
        pred_series = (
            [pred_series] if not isinstance(pred_series, Sequence) else pred_series
        )

        if not len(actual_series) == len(pred_series):
            raise_log(
                ValueError("The two TimeSeries sequences must have the same length."),
                logger=logger,
            )

        num_series_in_args = int("actual_series" not in kwargs) + int(
            "pred_series" not in kwargs
        )
        kwargs.pop("actual_series", 0)
        kwargs.pop("pred_series", 0)

        iterator = _build_tqdm_iterator(
            iterable=zip(actual_series, pred_series),
            verbose=verbose,
            total=len(actual_series),
        )

        vals = _parallel_apply(
            iterator=iterator,
            fn=func,
            n_jobs=n_jobs,
            fn_args=args[num_series_in_args:],
            fn_kwargs=kwargs,
        )
        # in case the reduction is not reducing the metrics sequence to a single value, e.g., if returning the
        # np.ndarray of values with the identity function, we must handle the single TS case, where we should
        # return a single value instead of a np.array of len 1
        vals = vals[0] if len(vals) == 1 else vals

        if kwargs.get("series_reduction") is not None:
            return kwargs["series_reduction"](vals)
        else:
            return vals

    return wrapper_multi_ts_support


def multivariate_support(func) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """
    This decorator transforms a metric function that takes as input two univariate TimeSeries instances
    into a function that takes two equally-sized multivariate TimeSeries instances, computes the pairwise univariate
    metrics for components with the same indices, and returns a float value that is computed as a function of all the
    univariate metrics using a `component_reduction` subroutine passed as argument to the metric function.
    """

    @wraps(func)
    def wrapper_multivariate_support(*args, **kwargs) -> METRIC_OUTPUT_TYPE:
        # we can avoid checks about args and kwargs since the input is adjusted by the previous decorator
        actual_series = args[0]
        pred_series = args[1]

        if not actual_series.width == pred_series.width:
            raise_log(
                ValueError("The two TimeSeries instances must have the same width."),
                logger=logger,
            )

        vals = func(actual_series, pred_series, *args[2:], **kwargs)
        if "component_reduction" in kwargs:
            return (
                kwargs["component_reduction"](vals)
                if kwargs["component_reduction"] is not None
                else vals
            )
        else:
            return signature(func).parameters["component_reduction"].default(vals)

    return wrapper_multivariate_support


def _get_values(
    vals: np.ndarray, stochastic_quantile: Optional[float] = 0.5
) -> np.ndarray:
    """
    Returns a deterministic or probabilistic numpy array from the values of a time series.
    For stochastic input values, return either all sample values with (stochastic_quantile=None) or the quantile sample
    value with (stochastic_quantile {>=0,<=1})
    """
    if vals.shape[2] == 1:  # deterministic
        out = vals[:, :, 0]
    else:  # stochastic
        if stochastic_quantile is None:
            out = vals
        else:
            out = np.quantile(vals, stochastic_quantile, axis=2)
    return out


def _get_values_or_raise(
    series_a: TimeSeries,
    series_b: TimeSeries,
    intersect: bool,
    stochastic_quantile: Optional[float] = 0.5,
    remove_nan_union: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.

    Raises a ValueError if the two time series (or their intersection) do not have the same time index.

    Parameters
    ----------
    series_a
        A univariate deterministic ``TimeSeries`` instance (the actual series).
    series_b
        A univariate (deterministic or stochastic) ``TimeSeries`` instance (the predicted series).
    intersect
        A boolean for whether to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, remove all indices from `series_a` and `series_b` which have a NaN value
        in either of the two input series.
    """

    if not series_a.width == series_b.width:
        raise_log(
            ValueError("The two time series must have the same number of components"),
            logger=logger,
        )

    if not isinstance(intersect, bool):
        raise_log(ValueError("The intersect parameter must be a bool"), logger=logger)

    make_copy = True
    if series_a.has_same_time_as(series_b) or not intersect:
        vals_a_common = series_a.all_values(copy=make_copy)
        vals_b_common = series_b.all_values(copy=make_copy)
    else:
        vals_a_common = series_a.slice_intersect_values(series_b, copy=make_copy)
        vals_b_common = series_b.slice_intersect_values(series_a, copy=make_copy)

    if not len(vals_a_common) == len(vals_b_common):
        raise_log(
            ValueError(
                "The two time series (or their intersection) "
                "must have the same time index."
            ),
            logger=logger,
        )

    vals_a_det = _get_values(vals_a_common, stochastic_quantile=stochastic_quantile)
    vals_b_det = _get_values(vals_b_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union:
        return vals_a_det, vals_b_det

    b_is_deterministic = bool(len(vals_b_det.shape) == 2)
    if b_is_deterministic:
        isnan_mask = np.logical_or(np.isnan(vals_a_det), np.isnan(vals_b_det))
        isnan_mask_pred = isnan_mask
    else:
        isnan_mask = np.logical_or(
            np.isnan(vals_a_det), np.isnan(vals_b_det).any(axis=2)
        )
        isnan_mask_pred = np.repeat(
            np.expand_dims(isnan_mask, axis=-1), vals_b_det.shape[2], axis=2
        )
    return np.where(isnan_mask, np.nan, vals_a_det), np.where(
        isnan_mask_pred, np.nan, vals_b_det
    )


@multi_ts_support
@multivariate_support
def mae(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Error (MAE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(|y^1_t - y^2_t|)}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    return np.nanmean(np.abs(y_true - y_pred), axis=TIME_AX)


@multi_ts_support
@multivariate_support
def mse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Mean Squared Error (MSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    return np.nanmean((y_true - y_pred) ** 2, axis=TIME_AX)


@multi_ts_support
@multivariate_support
def rmse(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Error (RMSE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y^1_t - y^2_t)^2}}.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """
    return np.sqrt(
        mse(
            actual_series,
            pred_series,
            intersect,
            component_reduction=None,
            series_reduction=None,
        )
    )


@multi_ts_support
@multivariate_support
def rmsle(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Log Error (RMSLE).

    For two time series :math:`y^1` and :math:`y^2` of length :math:`T`, it is computed as

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{\\left(\\log{(y^1_t + 1)} - \\log{(y^2_t + 1)}\\right)^2}},

    using the natural logarithm.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    y_true, y_pred = np.log(y_true + 1), np.log(y_pred + 1)
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=TIME_AX))


@multi_ts_support
@multivariate_support
def coefficient_of_variation(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Variation (percentage).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`,
    it is a percentage value, computed as

    .. math:: 100 \\cdot \\text{RMSE}(y_t, \\hat{y}_t) / \\bar{y_t},

    where :math:`\\text{RMSE}()` denotes the root mean squared error, and
    :math:`\\bar{y_t}` is the average of :math:`y_t`.

    Currently, this only supports deterministic series (made of one sample).

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    # not calling rmse as y_true and y_pred are np.ndarray
    return (
        100
        * np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=TIME_AX))
        / np.nanmean(y_true, axis=TIME_AX)
    )


@multi_ts_support
@multivariate_support
def mape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Percentage Error (MAPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T}{\\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|}.

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Mean Absolute Scaled Error (MASE) in these cases.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the actual series contains some zeros.

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    if not (y_true != 0).all():
        raise_log(
            ValueError(
                "The actual series must be strictly positive to compute the MAPE."
            ),
            logger=logger,
        )
    return 100.0 * np.nanmean(np.abs((y_true - y_pred) / y_true), axis=TIME_AX)


@multi_ts_support
@multivariate_support
def smape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """symmetric Mean Absolute Percentage Error (sMAPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math::
        200 \\cdot \\frac{1}{T}
        \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|} }.

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0`
     for some :math:`t`. Consider using the Mean Absolute Scaled Error (MASE) in these cases.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the actual series and the pred series contains some zeros at the same time index.

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    if not np.logical_or(y_true != 0, y_pred != 0).all():
        raise_log(
            ValueError(
                "The actual series must be strictly positive to compute the sMAPE."
            ),
            logger=logger,
        )
    return 200.0 * np.nanmean(
        np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)), axis=TIME_AX
    )


# mase cannot leverage multivariate and multi_ts with the decorator since also the `insample` is a Sequence[TimeSeries]
def mase(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    insample: Union[TimeSeries, Sequence[TimeSeries]],
    m: Optional[int] = 1,
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Scaled Error (MASE).

    See `Mean absolute scaled error wikipedia page <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_
    for details about the MASE and how it is computed.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    insample
        The training series used to forecast `pred_series` .
        This series serves to compute the scale of the error obtained by a naive forecaster on the training data.
    m
        Optionally, the seasonality to use for differencing.
        `m=1` corresponds to the non-seasonal MASE, whereas `m>1` corresponds to seasonal MASE.
        If `m=None`, it will be tentatively inferred
        from the auto-correlation function (ACF). It will fall back to a value of 1 if this fails.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`X_t = X_{t-m}` )

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    def _multivariate_mase(
        actual_series: TimeSeries,
        pred_series: TimeSeries,
        insample: TimeSeries,
        m: int,
        intersect: bool,
        component_reduction: Callable[[np.ndarray], float],
    ):

        if not actual_series.width == pred_series.width:
            raise_log(
                ValueError("The two TimeSeries instances must have the same width."),
                logger=logger,
            )
        if not actual_series.width == insample.width:
            raise_log(
                ValueError(
                    "The insample TimeSeries must have the same width as the other series."
                ),
                logger=logger,
            )
        if not insample.end_time() + insample.freq == pred_series.start_time():
            raise_log(
                ValueError(
                    "The pred_series must be the forecast of the insample series"
                ),
                logger=logger,
            )

        insample_ = (
            insample.quantile_timeseries(quantile=0.5)
            if insample.is_stochastic
            else insample
        )

        value_list = []
        for i in range(actual_series.width):
            # old implementation of mase on univariate TimeSeries
            if m is None:
                test_season, m = check_seasonality(insample)
                if not test_season:
                    warn(
                        "No seasonality found when computing MASE. Fixing the period to 1.",
                        UserWarning,
                    )
                    m = 1

            y_true, y_pred = _get_values_or_raise(
                actual_series.univariate_component(i),
                pred_series.univariate_component(i),
                intersect,
                remove_nan_union=False,
            )

            x_t = insample_.univariate_component(i).values()
            errors = np.abs(y_true - y_pred)
            scale = np.mean(np.abs(x_t[m:] - x_t[:-m]))
            if np.isclose(scale, 0):
                raise_log(
                    ValueError("cannot use MASE with periodical signals"), logger=logger
                )
            value_list.append(np.mean(errors / scale))

        return component_reduction(value_list)

    if isinstance(actual_series, TimeSeries):
        if not isinstance(pred_series, TimeSeries):
            raise_log(
                ValueError("Expecting pred_series to be TimeSeries"), logger=logger
            )
        if not isinstance(insample, TimeSeries):
            raise_log(ValueError("Expecting insample to be TimeSeries"), logger=logger)
        return _multivariate_mase(
            actual_series=actual_series,
            pred_series=pred_series,
            insample=insample,
            m=m,
            intersect=intersect,
            component_reduction=component_reduction,
        )

    elif isinstance(actual_series, Sequence) and isinstance(
        actual_series[0], TimeSeries
    ):
        if not (
            isinstance(pred_series, Sequence) and isinstance(pred_series[0], TimeSeries)
        ):
            raise_log(
                ValueError("Expecting pred_series to be a Sequence[TimeSeries]"),
                logger=logger,
            )
        if not (isinstance(insample, Sequence) and isinstance(insample[0], TimeSeries)):
            raise_log(
                ValueError("Expecting insample to be a Sequence[TimeSeries]"),
                logger=logger,
            )
        if not (
            len(pred_series) == len(actual_series) and len(pred_series) == len(insample)
        ):
            raise_log(
                ValueError("The TimeSeries sequences must have the same length."),
                logger=logger,
            )
        if not isinstance(n_jobs, int):
            raise_log(ValueError("n_jobs must be an integer"), logger=logger)
        if not isinstance(verbose, bool):
            raise_log(ValueError("verbose must be a bool"), logger=logger)

        iterator = _build_tqdm_iterator(
            iterable=zip(actual_series, pred_series, insample),
            verbose=verbose,
            total=len(actual_series),
        )

        value_list = _parallel_apply(
            iterator=iterator,
            fn=_multivariate_mase,
            n_jobs=n_jobs,
            fn_args=dict(),
            fn_kwargs={
                "m": m,
                "intersect": intersect,
                "component_reduction": component_reduction,
            },
        )
        return series_reduction(value_list)
    else:
        raise_log(
            ValueError(
                "Input type not supported, only TimeSeries and Sequence[TimeSeries] are accepted."
            )
        )


@multi_ts_support
@multivariate_support
def ope(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Overall Percentage Error (OPE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\left| \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}} \\right|.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\sum_{t=1}^{T}{y_t} = 0`.

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    y_true_sum, y_pred_sum = np.nansum(y_true, axis=TIME_AX), np.nansum(
        y_pred, axis=TIME_AX
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
def marre(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Ranged Relative Error (MARRE).

    Given a time series of actual values :math:`y_t` and a time series of predicted values :math:`\\hat{y}_t`
    both of length :math:`T`, it is a percentage value computed as

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} -
              \\min_t{y_t}} \\right|}

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If :math:`\\max_t{y_t} = \\min_t{y_t}`.

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
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
    return 100.0 * np.nanmean(np.abs((y_true - y_pred) / true_range), axis=TIME_AX)


@multi_ts_support
@multivariate_support
def r2_score(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Determination :math:`R^2`.

    See `Coefficient of determination wikipedia page <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    for details about the :math:`R^2` score and how it is computed.
    Please note that this metric is not symmetric, `actual_series` should correspond to the ground truth series,
    whereas `pred_series` should correspond to the predicted series.

    If any of the series is stochastic (containing several samples), the median sample value is considered.

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
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    ss_errors = np.nansum((y_true - y_pred) ** 2, axis=TIME_AX)
    y_hat = np.nanmean(y_true, axis=TIME_AX)
    ss_tot = np.nansum((y_true - y_hat) ** 2, axis=TIME_AX)
    return 1 - ss_errors / ss_tot


# Dynamic Time Warping
@multi_ts_support
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
    **kwargs
) -> METRIC_OUTPUT_TYPE:
    """
    Applies Dynamic Time Warping to actual_series and pred_series before passing it into the metric.
    Enables comparison between series of different lengths, phases and time indices.

    Defaults to using mae as a metric.

    See `darts.dataprocessing.dtw.dtw` for more supported parameters.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    metric
        The selected metric with signature '[[TimeSeries, TimeSeries], float]' to use. Default: `mae`.
    component_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component. By default, returns the `nanmean` over all component metrics.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """

    alignment = dtw.dtw(actual_series, pred_series, **kwargs)
    if metric == mae and "distance" not in kwargs:
        return alignment.mean_distance()

    warped_actual_series, warped_pred_series = alignment.warped()

    return metric(warped_actual_series, warped_pred_series)


# rho-risk (quantile risk)
@multi_ts_support
@multivariate_support
def rho_risk(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    rho: float = 0.5,
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """:math:`\\rho`-risk (rho-risk or quantile risk).

    Given a time series of actual values :math:`y_t` of length :math:`T` and a time series of stochastic predictions
    (containing N samples) :math:`\\hat{y}_t` of shape :math:`T \\times N`, rho-risk is a metric that quantifies the
    accuracy of a specific quantile :math:`\\rho` from the predicted value distribution.

    For a univariate stochastic predicted TimeSeries the :math:`\\rho`-risk is given by:

    .. math:: \\frac{ L_{\\rho} \\left( Z, \\hat{Z}_{\\rho} \\right) } {Z},

    where :math:`L_{\\rho} \\left( Z, \\hat{Z}_{\\rho} \\right)` is the :math:`\\rho`-loss function:

    .. math:: L_{\\rho} \\left( Z, \\hat{Z}_{\\rho} \\right) = 2 \\left( Z - \\hat{Z}_{\\rho} \\right)
        \\left( \\rho I_{\\hat{Z}_{\\rho} < Z} - \\left( 1 - \\rho \\right) I_{\\hat{Z}_{\\rho} \\geq Z} \\right),

    where :math:`Z = \\sum_{t=1}^{T} y_t` (1) is the aggregated target value and :math:`\\hat{Z}_{\\rho}` is the
    :math:`\\rho`-quantile of the predicted values. For this, each sample realization :math:`i \\in N` is first
    aggregated over the time span similar to (1) with :math:`\\hat{Z}_{i} = \\sum_{t=1}^{T} \\hat{y}_{i,t}`.

    :math:`I_{cond} = 1` if cond is True else :math:`0``

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    rho
        The quantile (float [0, 1]) of interest for the risk evaluation.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """
    if not pred_series.is_stochastic:
        raise_log(
            ValueError(
                "rho (quantile) loss should only be computed for stochastic predicted TimeSeries."
            ),
            logger=logger,
        )

    z_true, z_hat = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        stochastic_quantile=None,
        remove_nan_union=True,
    )
    z_true = np.nansum(z_true, axis=TIME_AX)
    z_hat = np.nansum(
        z_hat, axis=TIME_AX
    )  # aggregate all individual sample realizations
    z_hat_rho = np.quantile(
        z_hat, q=rho, axis=1
    )  # get the quantile from aggregated samples

    # quantile loss
    errors = z_true - z_hat_rho
    losses = 2 * np.maximum((rho - 1) * errors, rho * errors)
    return losses / z_true


# Quantile Loss (Pinball Loss)
@multi_ts_support
@multivariate_support
def quantile_loss(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    tau: float = 0.5,
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> METRIC_OUTPUT_TYPE:
    """
    Also known as Pinball Loss, given a time series of actual values :math:`y` of length :math:`T`
    and a time series of stochastic predictions (containing N samples) :math:`y'` of shape :math:`T  x N`
    quantile loss is a metric that quantifies the accuracy of a specific quantile :math:`tau`
    from the predicted value distribution.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    tau
        The quantile (float [0, 1]) of interest for the loss.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning a scalar value. This function is used to
        aggregate the metrics of different components in case of multivariate ``TimeSeries`` instances. If `None`,
        will return a metric per component.
    series_reduction
        Optionally, a function taking as input a ``np.ndarray`` and returning either a scalar value or a ``np.ndarray``.
        This function is used to aggregate the metrics in case the metric is evaluated on multiple series
        (e.g., on a ``Sequence[TimeSeries]``). By default, returns the metric for each series.
        Example: ``series_reduction=np.nanmean``, will return the average over all series metrics.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for single time series, either univariate or multivariate with `component_reduction`.
    List[float]
        A list of metric scores for multiple series, all either univariate or multivariate with `component_reduction`.
    np.ndarray
        A single array of metric scores for single multivariate time series without `component_reduction`
    List[np.ndarray]
        A list of arrays of metric scores for multiple multivariate series without `component_reduction`.
    """
    if not pred_series.is_stochastic:
        raise_log(
            ValueError(
                "quantile (pinball) loss should only be computed for "
                "stochastic predicted TimeSeries."
            ),
            logger=logger,
        )

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        stochastic_quantile=tau,
        remove_nan_union=True,
    )
    errors = y_true - y_pred
    losses = np.maximum((tau - 1) * errors, tau * errors)
    return np.nanmean(losses, axis=TIME_AX)
