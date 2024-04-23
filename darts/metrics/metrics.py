"""
Metrics
-------

Some metrics to compare time series.
"""

import inspect
from functools import wraps
from inspect import signature
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing import dtw
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply, n_steps_between
from darts.utils.ts_utils import SeriesType, get_series_seq_type, series2seq

logger = get_logger(__name__)
TIME_AX = 0
COMP_AX = 1

# Note: for new metrics added to this module to be able to leverage the two decorators, it is required both having
# the `actual_series` and `pred_series` parameters, and not having other ``Sequence`` as args (since these decorators
# don't "unpack" parameters different from `actual_series` and `pred_series`). In those cases, the new metric must take
# care of dealing with Sequence[TimeSeries] and multivariate TimeSeries on its own (See mase() implementation).
METRIC_OUTPUT_TYPE = Union[float, List[float], np.ndarray, List[np.ndarray]]
METRIC_TYPE = Callable[
    ...,
    METRIC_OUTPUT_TYPE,
]


def multi_ts_support(func) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """
    This decorator further adapts the metrics that took as input two (or three for scaled metrics with `insample`)
    univariate/multivariate ``TimeSeries`` instances, adding support for equally-sized sequences of ``TimeSeries``
    instances. The decorator computes the pairwise metric for ``TimeSeries`` with the same indices, and returns a float
    value that is computed as a function of all the pairwise metrics using a `series_reduction` subroutine passed as
    argument to the metric function.

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
            else args[0]
            if "actual_series" in kwargs
            else args[1]
        )

        params = signature(func).parameters
        n_jobs = kwargs.pop("n_jobs", params["n_jobs"].default)
        if not isinstance(n_jobs, int):
            raise_log(ValueError("n_jobs must be an integer"), logger=logger)

        verbose = kwargs.pop("verbose", params["verbose"].default)
        if not isinstance(verbose, bool):
            raise_log(ValueError("verbose must be a bool"), logger=logger)

        # sanity check reduction functions
        _ = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="time_reduction",
            axis=TIME_AX,
            sanity_check=True,
        )
        _ = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="component_reduction",
            axis=COMP_AX,
            sanity_check=True,
        )
        series_reduction = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="series_reduction",
            axis=0,
            sanity_check=True,
        )

        series_seq_type = get_series_seq_type(actual_series)
        actual_series = series2seq(actual_series)
        pred_series = series2seq(pred_series)

        if len(actual_series) != len(pred_series):
            raise_log(
                ValueError(
                    f"Mismatch between number of series in `actual_series` (n={len(actual_series)}) and "
                    f"`pred_series` (n={len(pred_series)})."
                ),
                logger=logger,
            )
        num_series_in_args = int("actual_series" not in kwargs) + int(
            "pred_series" not in kwargs
        )
        input_series = (actual_series, pred_series)

        kwargs.pop("actual_series", 0)
        kwargs.pop("pred_series", 0)

        # handle `insample` parameter for scaled metrics
        if "insample" in params:
            insample = kwargs.get("insample")
            if insample is None:
                insample = args[
                    2 - ("actual_series" in kwargs) - ("pred_series" in kwargs)
                ]

            insample = [insample] if not isinstance(insample, Sequence) else insample
            if len(actual_series) != len(insample):
                raise_log(
                    ValueError(
                        f"Mismatch between number of series in `actual_series` (n={len(actual_series)}) and "
                        f"`insample` series (n={len(insample)})."
                    ),
                    logger=logger,
                )
            input_series += (insample,)
            num_series_in_args += int("insample" not in kwargs)
            kwargs.pop("insample", 0)

        iterator = _build_tqdm_iterator(
            iterable=zip(*input_series),
            verbose=verbose,
            total=len(actual_series),
        )

        # `vals` is a list of series metrics of length `len(actual_series)`. Each metric has shape
        # `(n time steps, n components)`;
        # - n times step is `1` if `time_reduction` is other than `None`
        # - n components: is 1 if `component_reduction` is other than `None`
        vals = _parallel_apply(
            iterator=iterator,
            fn=func,
            n_jobs=n_jobs,
            fn_args=args[num_series_in_args:],
            fn_kwargs=kwargs,
        )

        # we flatten metrics along the time axis if n time steps == 1,
        # and/or along component axis if n components == 1
        vals = [
            val[
                slice(None) if val.shape[TIME_AX] != 1 else 0,
                slice(None) if val.shape[COMP_AX] != 1 else 0,
            ]
            for val in vals
        ]

        # reduce metrics along series axis
        if series_reduction is not None:
            vals = kwargs["series_reduction"](vals, axis=0)
        elif series_seq_type == SeriesType.SINGLE:
            vals = vals[0]

        # flatten along series axis if n series == 1
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
        params = signature(func).parameters
        # we can avoid checks about args and kwargs since the input is adjusted by the previous decorator
        actual_series = args[0]
        pred_series = args[1]
        num_series_in_args = 2

        if actual_series.width != pred_series.width:
            raise_log(
                ValueError(
                    f"Mismatch between number of components in `actual_series` "
                    f"(n={actual_series.width}) and `pred_series` (n={pred_series.width}."
                ),
                logger=logger,
            )

        # handle `insample` parameters for scaled metrics
        input_series = (actual_series, pred_series)
        if "insample" in params:
            insample = args[2]
            if actual_series.width != insample.width:
                raise_log(
                    ValueError(
                        f"Mismatch between number of components in `actual_series` "
                        f"(n={actual_series.width}) and `insample` (n={insample.width}."
                    ),
                    logger=logger,
                )
            input_series += (insample,)
            num_series_in_args += 1

        vals = func(*input_series, *args[num_series_in_args:], **kwargs)
        if not 1 <= len(vals.shape) <= 2:
            raise_log(
                ValueError(
                    "Metric output must have 1 dimension for aggregated metrics (e.g. `mae()`, ...), "
                    "or 2 dimension for time dependent metrics (e.g. `ae()`, ...)"
                ),
                logger=logger,
            )
        elif len(vals.shape) == 1:
            vals = np.expand_dims(vals, TIME_AX)

        time_reduction = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="time_reduction",
            axis=TIME_AX,
            sanity_check=False,
        )
        if time_reduction is not None:
            vals = np.expand_dims(time_reduction(vals, axis=TIME_AX), axis=TIME_AX)

        component_reduction = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="component_reduction",
            axis=COMP_AX,
            sanity_check=False,
        )
        if component_reduction is not None:
            vals = np.expand_dims(component_reduction(vals, axis=COMP_AX), axis=COMP_AX)
        return vals

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
    is_insample: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.

    Parameters
    ----------
    series_a
        A deterministic ``TimeSeries`` instance. If `is_insample=False`, it is the `actual_series`.
        Otherwise, it is the `insample` series.
    series_b
        A deterministic or stochastic ``TimeSeries`` instance (the predictions `pred_series`).
    intersect
        A boolean for whether to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, sets all values from `series_a` and `series_b` to `np.nan` at indices
        where any of the two series contain a NaN value. Only effective when `is_insample=False`.
    is_insample
        Whether `series_a` corresponds to the `insample` series for scaled metrics.

    Raises
    ------
    ValueError
        If `is_insample=False` and the two time series do not have at least a partially overlapping time index.
    """

    if not series_a.width == series_b.width:
        raise_log(
            ValueError("The two time series must have the same number of components"),
            logger=logger,
        )

    if not isinstance(intersect, bool):
        raise_log(ValueError("The intersect parameter must be a bool"), logger=logger)

    make_copy = False
    if not is_insample:
        # get the time intersection and values of the two series (corresponds to `actual_series` and `pred_series`
        if series_a.has_same_time_as(series_b) or not intersect:
            vals_a_common = series_a.all_values(copy=make_copy)
            vals_b_common = series_b.all_values(copy=make_copy)
        else:
            vals_a_common = series_a.slice_intersect_values(series_b, copy=make_copy)
            vals_b_common = series_b.slice_intersect_values(series_a, copy=make_copy)

        if not len(vals_a_common) == len(vals_b_common):
            raise_log(
                ValueError(
                    "The two time series must have at least a partially overlapping time index."
                ),
                logger=logger,
            )

        vals_b_det = _get_values(vals_b_common, stochastic_quantile=stochastic_quantile)
    else:
        # for `insample` series we extract only values up until before start of `pred_series`
        # find how many steps `insample` overlaps into `series_b`
        end = (
            n_steps_between(
                end=series_b.start_time(), start=series_a.end_time(), freq=series_a.freq
            )
            - 1
        )
        if end > 0 or abs(end) >= len(series_a):
            raise_log(
                ValueError(
                    "The `insample` series must start before the `pred_series` and "
                    "extend at least until one time step before the start of `pred_series`."
                ),
                logger=logger,
            )
        end = end or None
        vals_a_common = series_a.all_values(copy=make_copy)[:end]
        vals_b_det = None
    vals_a_det = _get_values(vals_a_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union or is_insample:
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


def _get_wrapped_metric(
    func: Callable[..., METRIC_OUTPUT_TYPE],
) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """Returns the inner metric function `func` which bypasses the decorators `multi_ts_support` and
    `multivariate_support`. It significantly decreases process time compared to calling `func` directly.
    Only use this to compute a pre-defined metric within the scope of another metric.
    """
    return func.__wrapped__.__wrapped__


def _get_reduction(
    kwargs, params, red_name, axis, sanity_check: bool = True
) -> Optional[Callable[..., np.ndarray]]:
    """Returns the reduction function either from user kwargs or metric default.
    Optionally performs sanity checks for presence of `axis` parameter, and correct output type and
    reduced shape."""
    if red_name not in params:
        return None

    red_fn = kwargs[red_name] if red_name in kwargs else params[red_name].default
    if not sanity_check:
        return red_fn

    if red_fn is not None:
        red_params = inspect.signature(red_fn).parameters
        if "axis" not in red_params:
            raise_log(
                ValueError(
                    f"Invalid `{red_name}` function: Must have a parameter called `axis`."
                ),
                logger=logger,
            )
        # verify `red_fn` reduces to array with correct shape
        shape_in = (2, 1) if axis == 0 else (1, 2)
        out = red_fn(np.zeros(shape_in), axis=axis)

        if not isinstance(out, np.ndarray):
            raise_log(
                ValueError(
                    f"Invalid `{red_name}` function output type: Expected type "
                    f"`np.ndarray`, received type=`{type(out)}`."
                ),
                logger=logger,
            )
        shape_invalid = out.shape != (1,)
        if shape_invalid:
            raise_log(
                ValueError(
                    f"Invalid `{red_name}` function output shape: The function must reduce an input "
                    f"`np.ndarray` of shape (t, c) to a `np.ndarray` of shape `(c,)`. "
                    f"However, the function reduced a test array of shape `{shape_in}` to "
                    f"`{out.shape}`."
                ),
                logger=logger,
            )
    return red_fn


def _get_error_scale(
    insample: TimeSeries,
    pred_series: TimeSeries,
    m: int,
    metric: str,
):
    """Computes the error scale based on a naive seasonal forecasts on `insample` values with seasonality `m`."""
    if not isinstance(m, int):
        raise_log(
            ValueError(f"Seasonality `m` must be of type `int`, recevied `m={m}`"),
            logger=logger,
        )

    # `x_t` are the true `y` values before the start of `y_pred`
    x_t, _ = _get_values_or_raise(
        insample, pred_series, intersect=False, remove_nan_union=False, is_insample=True
    )
    diff = x_t[m:] - x_t[:-m]
    if metric == "mae":
        scale = np.nanmean(np.abs(diff), axis=TIME_AX)
    elif metric == "mse":
        scale = np.nanmean(np.power(diff, 2), axis=TIME_AX)
    elif metric == "rmse":
        scale = np.sqrt(np.nanmean(np.power(diff, 2), axis=TIME_AX))
    else:
        raise_log(
            ValueError(
                f"unknown `metric={metric}`. Must be one of ('mae', 'mse', 'rmse')."
            ),
            logger=logger,
        )

    if np.isclose(scale, 0.0).any():
        raise_log(ValueError("cannot use MASE with periodical signals"), logger=logger)
    return scale


@multi_ts_support
@multivariate_support
def err(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Error (ERR).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: y_t - \\hat{y}_t

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    return y_true - y_pred


@multi_ts_support
@multivariate_support
def merr(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Error (MERR).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)}

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(err)(
            actual_series,
            pred_series,
            intersect,
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
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Error (AE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: |y_t - \\hat{y}_t|

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
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
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    return np.abs(y_true - y_pred)


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
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Error (MAE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{|y_t - \\hat{y}_t|}

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
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
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(ae)(
            actual_series,
            pred_series,
            intersect,
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
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Scaled Error (ASE) (see [1]_ for more information on scaled forecasting errors).

    It is the Absolute Error (AE) scaled by the Mean AE (MAE) of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: \\frac{AE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`AE` is the Absolute
    Error (:func:`~darts.metrics.metrics.ae`), and :math:`E_m` is the Mean AE (MAE) of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MAE(y_{m:t_p}, y_{0:t_p - m}).

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Scaled Error (MASE) (see [1]_ for more information on scaled forecasting errors).

    It is the Mean Absolute Error (MAE) scaled by the MAE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{MAE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`MAE` is the Mean
    Absolute Error (:func:`~darts.metrics.metrics.mae`), and :math:`E_m` is the MAE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MAE(y_{m:t_p}, y_{0:t_p - m}).

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Error (SE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: (y_t - \\hat{y}_t)^2.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
    )
    return (y_true - y_pred) ** 2


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
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Squared Error (MSE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(se)(
            actual_series,
            pred_series,
            intersect,
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
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Scaled Error (SSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Squared Error (SE) scaled by the Mean SE (MSE) of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: \\frac{SE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`SE` is the Squared
    Error (:func:`~darts.metrics.metrics.se`), and :math:`E_m` is the Mean SE (MSE) of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MSE(y_{m:t_p}, y_{0:t_p - m}).

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Squared Scaled Error (MSSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Mean Squared Error (MSE) scaled by the MSE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{MSE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`MSE` is the Mean
    Squared Error (:func:`~darts.metrics.metrics.mse`), and :math:`E_m` is the MSE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = MSE(y_{m:t_p}, y_{0:t_p - m}).

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Error (RMSE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.sqrt(
        _get_wrapped_metric(mse)(
            actual_series,
            pred_series,
            intersect,
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Scaled Error (RMSSE) (see [1]_ for more information on scaled forecasting errors).

    It is the Root Mean Squared Error (RMSE) scaled by the RMSE of the naive m-seasonal forecast.

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\frac{RMSE(y_{t_p+1:t_p+T}, \\hat{y}_{t_p+1:t_p+T})}{E_m},

    where :math:`t_p` is the prediction time (one step before the first forecasted point), :math:`RMSE` is the Root
    Mean Squared Error (:func:`~darts.metrics.metrics.rmse`), and :math:`E_m` is the RMSE of the naive m-seasonal
    forecast on the `insample` series :math:`y_{0:t_p}` (the true series ending at :math:`t_p`):

    .. math:: E_m = RMSE(y_{m:t_p}, y_{0:t_p - m}).

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `insample` series is periodic ( :math:`y_t = y_{t-m}` ) or any series in `insample` does not end one
        time step before the start of the corresponding forecast in `pred_series`.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    )
    return errors / error_scale


@multi_ts_support
@multivariate_support
def sle(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Squared Log Error (SLE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column and time step :math:`t` as:

    .. math:: \\left(\\log{(y_t + 1)} - \\log{(\\hat{y} + 1)}\\right)^2

    using the natural logarithm.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Log Error (RMSLE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: \\sqrt{\\frac{1}{T}\\sum_{t=1}^T{\\left(\\log{(y_t + 1)} - \\log{(\\hat{y}_t + 1)}\\right)^2}}

    using the natural logarithm.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.sqrt(
        np.nanmean(
            _get_wrapped_metric(sle)(
                actual_series,
                pred_series,
                intersect,
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

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If `actual_series` contains some zeros.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=False
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Percentage Error (MAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column with:

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T}{\\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right|}

    Note that it will raise a `ValueError` if :math:`y_t = 0` for some :math:`t`. Consider using
    the Mean Absolute Scaled Error (:func:`~darts.metrics.metrics.mase`) in these cases.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If `actual_series` contains some zeros.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    return np.nanmean(
        _get_wrapped_metric(ape)(
            actual_series,
            pred_series,
            intersect,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def sape(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """symmetric Absolute Percentage Error (sAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and time step :math:`t` with:

    .. math::
        200 \\cdot \\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|}

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0` for some
    :math:`t`. Consider using the Absolute Scaled Error (:func:`~darts.metrics.metrics.ase`)  in these cases.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If `actual_series` and `pred_series` contain some zeros at the same time index.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    if not np.logical_or(y_true != 0, y_pred != 0).all():
        raise_log(
            ValueError(
                "`actual_series` must be strictly positive to compute the sMAPE."
            ),
            logger=logger,
        )
    return 200.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))


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
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """symmetric Mean Absolute Percentage Error (sMAPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column with:

    .. math::
        200 \\cdot \\frac{1}{T}
        \\sum_{t=1}^{T}{\\frac{\\left| y_t - \\hat{y}_t \\right|}{\\left| y_t \\right| + \\left| \\hat{y}_t \\right|} }

    Note that it will raise a `ValueError` if :math:`\\left| y_t \\right| + \\left| \\hat{y}_t \\right| = 0`
    for some :math:`t`. Consider using the Mean Absolute Scaled Error (:func:`~darts.metrics.metrics.mase`) in these
    cases.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Raises
    ------
    ValueError
        If the `actual_series` and the `pred_series` contain some zeros at the same time index.

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    return np.nanmean(
        _get_wrapped_metric(sape)(
            actual_series,
            pred_series,
            intersect,
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Overall Percentage Error (OPE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column with:

    .. math:: 100 \\cdot \\left| \\frac{\\sum_{t=1}^{T}{y_t}
              - \\sum_{t=1}^{T}{\\hat{y}_t}}{\\sum_{t=1}^{T}{y_t}} \\right|.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
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
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
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
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Ranged Relative Error (ARRE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column and time step :math:`t` with:

    .. math:: 100 \\cdot \\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} - \\min_t{y_t}} \\right|

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
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
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
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
    return 100.0 * np.abs((y_true - y_pred) / true_range)


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
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Ranged Relative Error (MARRE).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed as a
    percentage value per component/column with:

    .. math:: 100 \\cdot \\frac{1}{T} \\sum_{t=1}^{T} {\\left| \\frac{y_t - \\hat{y}_t} {\\max_t{y_t} -
              \\min_t{y_t}} \\right|}

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
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

    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    return np.nanmean(
        _get_wrapped_metric(arre)(
            actual_series,
            pred_series,
            intersect,
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Determination :math:`R^2` (see [1]_ for more details).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

    .. math:: 1 - \\frac{\\sum_{t=1}^T{(y_t - \\hat{y}_t)^2}}{\\sum_{t=1}^T{(y_t - \\bar{y})^2}},

    where :math:`\\bar{y}` is the mean of :math:`y` over all time steps.

    This metric is not symmetric.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    y_true, y_pred = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
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
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Coefficient of Variation (percentage).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as a percentage value with:

    .. math:: 100 \\cdot \\text{RMSE}(y_t, \\hat{y}_t) / \\bar{y},

    where :math:`RMSE` is the Root Mean Squared Error (:func:`~darts.metrics.metrics.rmse`), and :math:`\\bar{y}` is
    the average of :math:`y` over all time steps.

    If any of the series is stochastic (containing several samples), :math:`\\hat{y}_t` is the median over all samples
    for time step :math:`t`.

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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
    q: float = 0.5,
    intersect: bool = True,
    *,
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
    of of shape :math:`T \\times N`, it is computed per column/component as:

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
    q
        The quantile (float [0, 1]) of interest for the risk evaluation.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
        stochastic_quantile=None,
        remove_nan_union=True,
    )
    z_true = np.nansum(z_true, axis=TIME_AX)
    z_hat = np.nansum(
        z_hat, axis=TIME_AX
    )  # aggregate all individual sample realizations
    z_hat_rho = np.quantile(
        z_hat, q=q, axis=1
    )  # get the quantile from aggregated samples

    # quantile loss
    errors = z_true - z_hat_rho
    losses = 2 * np.maximum((q - 1) * errors, q * errors)
    return losses / z_true


@multi_ts_support
@multivariate_support
def ql(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    q: float = 0.5,
    intersect: bool = True,
    *,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Quantile Loss (QL).

    Also known as Pinball Loss. QL is a metric that quantifies the accuracy of a specific quantile :math:`q` from the
    predicted value distribution of a stochastic/probabilistic `pred_series` containing N samples.

    QL computes the quantile of all sample values and the loss per time step.

    For the true series :math:`y` and predicted stochastic/probabilistic series (containing N samples) :math:`\\hat{y}`
    of of shape :math:`T \\times N`, it is computed per column/component and time step :math:`t` as:

    .. math:: 2 \\max((q - 1) (y_t - \\hat{y}_{t,q}), q (y_t - \\hat{y}_{t,q})),

    where :math:`\\hat{y}_{t,q}` is the quantile :math:`q` of all predicted sample values at time :math:`t`.
    The factor `2` makes the loss more interpretable, as for `q=0.5` the loss is identical to the Absolute Error
    (:func:`~darts.metrics.metrics.ae`).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    q
        The quantile (float [0, 1]) of interest for the loss.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
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
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """
    if not pred_series.is_stochastic:
        raise_log(
            ValueError(
                "quantile/pinball loss (ql) should only be computed for "
                "stochastic predicted TimeSeries."
            ),
            logger=logger,
        )

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        stochastic_quantile=q,
        remove_nan_union=True,
    )
    errors = y_true - y_pred
    losses = 2.0 * np.maximum((q - 1) * errors, q * errors)
    return losses


@multi_ts_support
@multivariate_support
def mql(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    q: float = 0.5,
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Mean Quantile Loss (MQL).

    Also known as Pinball Loss. QL is a metric that quantifies the accuracy of a specific quantile :math:`q` from the
    predicted value distribution of a stochastic/probabilistic `pred_series` containing N samples.

    MQL first computes the quantile of all sample values and the loss per time step, and then takes the mean over the
    time axis.

    For the true series :math:`y` and predicted stochastic/probabilistic series (containing N samples) :math:`\\hat{y}`
    of of shape :math:`T \\times N`, it is computed per column/component as:

    .. math:: 2 \\frac{1}{T}\\sum_{t=1}^T{\\max((q - 1) (y_t - \\hat{y}_{t,q}), q (y_t - \\hat{y}_{t,q}))},

    where :math:`\\hat{y}_{t,q}` is the quantile :math:`q` of all predicted sample values at time :math:`t`.
    The factor `2` makes the loss more interpretable, as for `q=0.5` the loss is identical to the Mean Absolute Error
    (:func:`~darts.metrics.metrics.mae`).

    Parameters
    ----------
    actual_series
        The (sequence of) actual series.
    pred_series
        The (sequence of) predicted series.
    q
        The quantile (float [0, 1]) of interest for the loss.
    intersect
        For time series that are overlapping in time without having the same time index, setting `True`
        will consider the values only over their common time interval (intersection in time).
    component_reduction
        Optionally, a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray`
        of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the
        component axis. If `None`, will return a metric per component.
    series_reduction
        Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray`
        of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a
        parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the
        series axis. If `None`, will return a metric per series.
    n_jobs
        The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
        passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
        (sequential). Setting the parameter to `-1` means using all the available processors.
    verbose
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,) without component reduction. For:

        - single multivariate series and at least `component_reduction=None`.
        - sequence of uni/multivariate series including `series_reduction` and `component_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
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
