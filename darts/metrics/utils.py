import inspect
from collections.abc import Sequence
from enum import Enum
from functools import wraps
from inspect import signature
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.likelihood_models.base import (
    likelihood_component_names,
    quantile_names,
)
from darts.utils.ts_utils import SeriesType, get_series_seq_type, series2seq
from darts.utils.utils import (
    _build_tqdm_iterator,
    _parallel_apply,
    n_steps_between,
)

logger = get_logger(__name__)
TIME_AX = 0
COMP_AX = 1
SMPL_AX = 2

# (True / False Positive / Negative) indices in the confusion matrix
_TN_IDX = (slice(None), slice(None), 0, 0)
_FP_IDX = (slice(None), slice(None), 0, 1)
_FN_IDX = (slice(None), slice(None), 1, 0)
_TP_IDX = (slice(None), slice(None), 1, 1)


# class probabilities suffix
PROBA_SUFFIX = "_p"


# class label reduction methods
class _LabelReduction(Enum):
    NONE = None
    MACRO = "macro"
    MICRO = "micro"
    WEIGHTED = "weighted"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


# Note: for new metrics added to this module to be able to leverage the two decorators, it is required both having
# the `actual_series` and `pred_series` parameters, and not having other ``Sequence`` as args (since these decorators
# don't "unpack" parameters different from `actual_series` and `pred_series`). In those cases, the new metric must take
# care of dealing with Sequence[TimeSeries] and multivariate TimeSeries on its own (See mase() implementation).
METRIC_OUTPUT_TYPE = Union[float, list[float], np.ndarray, list[np.ndarray]]
METRIC_TYPE = Callable[
    ...,
    METRIC_OUTPUT_TYPE,
]


def interval_support(func) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """
    This decorator adds support for quantile interval metrics with sanity checks, processing, and extraction of
    quantiles from the intervals.
    """

    @wraps(func)
    def wrapper_interval_support(*args, **kwargs):
        q = kwargs.get("q")
        if q is not None:
            raise_log(
                ValueError(
                    "`q` is not supported for quantile interval metrics; use `q_interval` instead."
                )
            )
        q_interval = kwargs.get("q_interval")
        if q_interval is None:
            raise_log(
                ValueError("Quantile interval metrics require setting `q_interval`.")
            )
        if isinstance(q_interval, tuple):
            q_interval = [q_interval]
        q_interval = np.array(q_interval)
        if not q_interval.ndim == 2 or q_interval.shape[1] != 2:
            raise_log(
                ValueError(
                    "`q_interval` must be a tuple (float, float) or a sequence of tuples (float, float)."
                ),
                logger=logger,
            )
        if not np.all(q_interval[:, 1] - q_interval[:, 0] > 0):
            raise_log(
                ValueError(
                    "all intervals in `q_interval` must be tuples of (lower q, upper q) with `lower q > upper q`. "
                    f"Received `q_interval={q_interval}`"
                ),
                logger=logger,
            )
        kwargs["q_interval"] = q_interval
        kwargs["q"] = np.sort(np.unique(q_interval))
        return func(*args, **kwargs)

    return wrapper_interval_support


def classification_support(func) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """
    This decorator adds support for classification metrics including sanity checks and handling of class
    probabilities and categorical samples.
    """

    @wraps(func)
    def wrapper_classification_support(*args, **kwargs):
        labels = kwargs.get("labels")
        if labels is not None:
            if isinstance(labels, int):
                labels = np.array([labels])
            else:
                labels = np.array(labels)
            kwargs["labels"] = labels

        params = signature(func).parameters
        if "label_reduction" in params:
            label_reduction = kwargs.get(
                "label_reduction", params["label_reduction"].default
            )
            if not isinstance(label_reduction, _LabelReduction):
                if not _LabelReduction.has_value(label_reduction):
                    raise_log(
                        ValueError(
                            f"Invalid `label_reduction` value: {label_reduction}. "
                            f"Must be one of {_LabelReduction._value2member_map_.keys()}."
                        ),
                        logger=logger,
                    )
                kwargs["label_reduction"] = _LabelReduction(label_reduction)

        kwargs["is_classification"] = True
        return func(*args, **kwargs)

    return wrapper_classification_support


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
        verbose = kwargs.pop("verbose", params["verbose"].default)

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

        # handle `q` (quantile) parameter for probabilistic (or quantile) forecasts
        if "q" in params:
            # convert `q` to tuple of (quantile values, optional quantile component names)
            q = kwargs.get("q", params["q"].default)
            q_comp_names = None
            if q is None:
                kwargs["q"] = None
            else:
                if isinstance(q, tuple):
                    q, q_comp_names = q
                if isinstance(q, float):
                    q = np.array([q])
                else:
                    q = np.array(q)

                if not np.all(q[1:] - q[:-1] > 0.0):
                    raise_log(
                        ValueError(
                            "`q` must be of type `float`, or a sequence of increasing order with unique values only. "
                            f"Received `q={q}`."
                        ),
                        logger=logger,
                    )
                if not np.all(q >= 0.0) & np.all(q <= 1.0):
                    raise_log(
                        ValueError(
                            f"All `q` values must be in the range `(>=0,<=1)`. Received `q={q}`."
                        ),
                        logger=logger,
                    )
                kwargs["q"] = (q, q_comp_names)

        iterator = _build_tqdm_iterator(
            iterable=zip(*input_series),
            verbose=verbose,
            total=len(actual_series),
            desc=f"metric `{func.__name__}()`",
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

        is_classification = kwargs.pop("is_classification", False)
        if is_classification:
            _classification_handling(actual_series, pred_series)
            # confusion matrix is special; returns 4 # dimensions: (1, n components, n labels, n labels)
            is_confusion_matrix = func.__name__ == "confusion_matrix"
        else:
            kwargs = _regression_handling(actual_series, pred_series, params, kwargs)
            is_confusion_matrix = False

        # handle `insample` parameters for scaled metrics
        input_series = (actual_series, pred_series)
        if "insample" in params:
            insample = args[2]
            if actual_series.n_components != insample.n_components:
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
        # bring vals to shape (n time, n components, *QL); *QL stands for quantile or class label dimensions
        n_dims = len(vals.shape)
        if not 2 <= n_dims <= 3:
            if n_dims == 4 and is_confusion_matrix:
                # confusion matrix returns 4 # dimensions: (1, n components, n labels, n labels)
                pass
            else:
                raise_log(
                    ValueError(
                        "Metric output must have 2 dimensions (n components, n quantiles or n labels) "
                        "for aggregated metrics (e.g. `mae()`, ...), "
                        "or 3 dimensions (n times, n components, n quantiles or n labels)  "
                        "for time dependent metrics (e.g. `ae()`, ...)"
                    ),
                    logger=logger,
                )

        if n_dims == 2:
            # quantile metrics aggregated over time -> (1, n components, *QL)
            vals = np.expand_dims(vals, TIME_AX)

        time_reduction = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="time_reduction",
            axis=TIME_AX,
            sanity_check=False,
        )
        if time_reduction is not None:
            # -> (1, n components, *QL)
            vals = np.expand_dims(time_reduction(vals, axis=TIME_AX), axis=TIME_AX)

        component_reduction = _get_reduction(
            kwargs=kwargs,
            params=params,
            red_name="component_reduction",
            axis=COMP_AX,
            sanity_check=False,
        )
        if component_reduction is not None:
            # -> (n time, *QL)
            vals = component_reduction(vals, axis=COMP_AX)
            if is_confusion_matrix:
                vals = np.expand_dims(vals, axis=COMP_AX)
        elif not is_confusion_matrix:
            # -> (n time, n components * n QL), with order [c0_q0, c0_q1, ... c1_q0, c1_q1, ...]
            vals = vals.reshape(vals.shape[0], -1)
        else:  # confusion matrix
            # -> (1, n components, n labels, n labels)
            pass
        return vals

    return wrapper_multivariate_support


def _regression_handling(actual_series, pred_series, params, kwargs):
    """Handles the regression metrics input parameters and checks."""
    q, q_comp_names = kwargs.get("q"), None
    if q is None:
        # without quantiles, the number of components must match
        if actual_series.n_components != pred_series.n_components:
            raise_log(
                ValueError(
                    f"Mismatch between number of components in `actual_series` "
                    f"(n={actual_series.width}) and `pred_series` (n={pred_series.width})."
                ),
                logger=logger,
            )
        # compute median for stochastic predictions
        if pred_series.is_stochastic:
            q = np.array([0.5])
    else:
        # `q` is required to be a tuple (handled by `multi_ts_support` wrapper)
        if not isinstance(q, tuple) or not len(q) == 2:
            raise_log(
                ValueError(
                    "`q` must be of tuple of `(np.ndarray, Optional[pd.Index])` "
                    "where the (quantile values, optional quantile component names). "
                    f"Received `q={q}`."
                ),
                logger=logger,
            )
        q, q_comp_names = q
        if not pred_series.is_stochastic:
            # quantile component names are required if the predictions are not stochastic (as for stochastic
            # predictions, the quantiles can be retrieved from the sample dimension for each component)
            if q_comp_names is None:
                q_comp_names = pd.Index(
                    likelihood_component_names(
                        components=actual_series.components,
                        parameter_names=quantile_names(q=q),
                    )
                )
            if not q_comp_names.isin(pred_series.components).all():
                raise_log(
                    ValueError(
                        f"Computing a metric with quantile(s) `q={q}` is only supported for probabilistic "
                        f"`pred_series` (num samples > 1) or `pred_series` containing the predicted "
                        f"quantiles as columns / components. Either pass a probabilistic `pred_series` or "
                        f"a series containing the expected quantile components: {q_comp_names.tolist()} "
                    ),
                    logger=logger,
                )

    if "q" in params:
        kwargs["q"] = (q, q_comp_names)
    return kwargs


def _classification_handling(actual_series, pred_series):
    """Handles the classification metrics input parameters and checks."""
    # This assumes class probabilities components follow the "<component_name>_p<label>" convention
    # Note: the correct number of labels per component is not checked here
    if not pred_series.components.equals(actual_series.components):
        # "<component_name>_p<label>" -> "<component_name>"
        predicted_components = (
            pred_series.components.str.split(PROBA_SUFFIX)
            .str[:-1]
            .str.join(PROBA_SUFFIX)
            .unique()
        )
        if not predicted_components.equals(actual_series.components):
            raise_log(
                ValueError(
                    f"Could not resolve the predicted components for the classification metric. "
                    f"Mismatch between number of components in `actual_series` "
                    f"(n={actual_series.width}) and `pred_series` (n={pred_series.width})."
                    "If `pred_series` represents class probabilities, it must contain a dedicated component "
                    "per original component and label following the naming convention `<component_name>_p<label>`. "
                    f"Original components: {actual_series.components}, predicted components: "
                    f"{pred_series.components}."
                ),
                logger=logger,
            )


def _get_values(
    vals: np.ndarray,
    vals_components: pd.Index,
    actual_components: pd.Index,
    q: Optional[tuple[Sequence[float], Union[Optional[pd.Index]]]] = None,
    is_classification: bool = False,
) -> np.ndarray:
    """
    Returns a deterministic or probabilistic numpy array from the values of a time series of shape
    (times, components, samples / quantiles).
    To extract quantile (sample) values from quantile or stochastic `vals`, use `q`.

    Parameters
    ----------
    vals
        A numpy array with the values of a TimeSeries (actual values or predictions).
    vals_components
        The components of the `vals` TimeSeries.
    actual_components
        The components of the actual TimeSeries.
    q
        Optionally, for stochastic or quantile series/values, return deterministic quantile values.
        If not `None`, must a tuple with (quantile values,
        `None` if `pred_series` is stochastic else the quantile component names).
    """
    # classification metrics
    if is_classification:
        if vals.shape[SMPL_AX] != 1:  # sampled class labels
            vals = _get_highest_count_label(vals)
        elif vals.shape[COMP_AX] != len(actual_components):  # class label probabilities
            vals = _get_highest_probability_label(
                vals=vals,
                vals_components=vals_components,
                actual_components=actual_components,
            )
        return vals

    # return values as is (times, components, samples)
    if q is None:
        return vals

    q, q_names = q
    if vals.shape[SMPL_AX] == 1:  # deterministic (or quantile components) input
        if q_names is not None:
            # `q_names` are the component names of the predicted quantile parameters
            # we extract the relevant quantile components with shape (times, components * quantiles)
            vals = vals[:, vals_components.get_indexer(q_names)]
            # rearrange into (times, components, quantiles)
            vals = vals.reshape((len(vals), len(actual_components), -1))
        return vals

    # probabilistic input
    # compute multiple quantiles for all times and components; with shape: (quantiles, times, components)
    out = np.quantile(vals, q, axis=SMPL_AX)
    # rearrange into (times, components, quantiles)
    return out.transpose((1, 2, 0))


def _get_values_or_raise(
    actual_series: TimeSeries,
    pred_series: TimeSeries,
    intersect: bool,
    q: Optional[tuple[Sequence[float], Union[Optional[pd.Index]]]] = None,
    remove_nan_union: bool = False,
    is_insample: bool = False,
    is_classification: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, q, remove_nan_union, is_classification`.

    Parameters
    ----------
    actual_series
        A deterministic ``TimeSeries`` instance. If `is_insample=False`, it is the `actual_series`.
        Otherwise, it is the `insample` series.
    pred_series
        A deterministic or stochastic ``TimeSeries`` instance (the predictions `pred_series`).
    intersect
        A boolean for whether to only consider the time intersection between `actual_series` and `pred_series`
    q
        Optionally, for predicted stochastic or quantile series, return deterministic quantile values.
        If not `None`, must a tuple with (quantile values,
        `None` if `pred_series` is stochastic else the quantile component names).
    remove_nan_union
        By setting `remove_non_union` to True, sets all values from `actual_series` and `pred_series` to `np.nan` at
        indices where any of the two series contain a NaN value. Only effective when `is_insample=False`.
    is_insample
        Whether `actual_series` corresponds to the `insample` series for scaled metrics.
    is_classification
        Whether the metric is a classification metric. If `True`, the values are processed as class labels.
        If `pred_series` contains class probabilities, it will extract the class label with the highest
        probability for each time step and component.
        If `pred_series` contains sampled class labels, it will extract the class label with the highest frequency.
        If `pred_series` is deterministic and has the same number of components as `actual_series`, it will return the
        class labels as is.

    Raises
    ------
    ValueError
        If `is_insample=False` and the two time series do not have at least a partially overlapping time index.
    """
    make_copy = False
    if not is_insample:
        # get the time intersection and values of the two series (corresponds to `actual_series` and `pred_series`
        if actual_series.has_same_time_as(pred_series) or not intersect:
            vals_actual_common = actual_series.all_values(copy=make_copy)
            vals_pred_common = pred_series.all_values(copy=make_copy)
        else:
            vals_actual_common = actual_series.slice_intersect_values(
                pred_series, copy=make_copy
            )
            vals_pred_common = pred_series.slice_intersect_values(
                actual_series, copy=make_copy
            )

        vals_b = _get_values(
            vals=vals_pred_common,
            vals_components=pred_series.components,
            actual_components=actual_series.components,
            q=q,
            is_classification=is_classification,
        )
    else:
        # for `insample` series we extract only values up until before start of `pred_series`
        # find how many steps `insample` overlaps into `pred_series`
        end = (
            n_steps_between(
                end=pred_series.start_time(),
                start=actual_series.end_time(),
                freq=actual_series.freq,
            )
            - 1
        )
        if end > 0 or abs(end) >= len(actual_series):
            raise_log(
                ValueError(
                    "The `insample` series must start before the `pred_series` and "
                    "extend at least until one time step before the start of `pred_series`."
                ),
                logger=logger,
            )
        end = end or None
        vals_actual_common = actual_series.all_values(copy=make_copy)[:end]
        vals_b = None
    vals_a = _get_values(
        vals=vals_actual_common,
        vals_components=actual_series.components,
        actual_components=actual_series.components,
        q=([0.5], None),
    )

    if not remove_nan_union or is_insample:
        return vals_a, vals_b

    isnan_mask = np.expand_dims(
        np.logical_or(np.isnan(vals_a), np.isnan(vals_b)).any(axis=SMPL_AX), axis=-1
    )
    isnan_mask_pred = np.repeat(isnan_mask, vals_b.shape[SMPL_AX], axis=SMPL_AX)
    return np.where(isnan_mask, np.nan, vals_a), np.where(
        isnan_mask_pred, np.nan, vals_b
    )


def _get_quantile_intervals(
    vals: np.ndarray,
    q: tuple[Sequence[float], Any],
    q_interval: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the lower and upper bound values from `vals` for all quantile intervals in `q_interval`.

    Parameters
    ----------
    vals
        A numpy array with predicted quantile values of shape (n times, n components, n quantiles).
    q
        A tuple with (quantile values, any).
    q_interval
        A numpy array with the lower and upper quantile interval bound of shape (n intervals, 2).
    """
    q, _ = q
    # find index of every `q_interval` value in `q`; we have guarantees from support wrappers:
    # - `q` has increasing order
    # - `vals` has same order as `q` in dim 3 (quantile dim)
    # - `q_interval` holds (lower q, upper q) in that order
    q_idx = np.searchsorted(q, q_interval.flatten()).reshape(q_interval.shape)
    return vals[:, :, q_idx[:, 0]], vals[:, :, q_idx[:, 1]]


def _mode(vals: np.ndarray) -> np.ndarray:
    """Computes the mode (value with the highest frequency) of a 1D numpy array.

    Parameters
    ----------
    arr
        A numpy array representing the predicted samples of a specific time step and component.
    """
    vals, cnts = np.unique(vals, return_counts=True)
    return vals[cnts.argmax()]


def _get_highest_count_label(vals: np.ndarray) -> np.ndarray:
    """Computes the mode (value with the highest frequency) for all time steps and components.

    Parameters
    ----------
    vals
        A numpy array with predicted class label samples of shape (n times, n components, n samples).
    """
    return np.apply_along_axis(func1d=_mode, axis=SMPL_AX, arr=vals).reshape(
        vals.shape[:SMPL_AX] + (1,)
    )


def _get_highest_probability_label(
    vals: np.ndarray,
    vals_components: Sequence[str],
    actual_components: Sequence[str],
) -> np.ndarray:
    """Computes the class label with highest probability for all time steps and components.

    Parameters
    ----------
    vals
        A numpy array with predicted class label probabilities of shape (n times, n components * n class labels, 1).
    vals_components
        The class probability component names, which must follow the naming convention
        "<component_name: str>_p<label: int>".
    actual_components
        The components of the actual TimeSeries.
    """
    names_indices_label = []
    for idx, name in enumerate(vals_components):
        # format {component_name: unique str}_p{class label: int}
        name_split = name.split(PROBA_SUFFIX)
        c_name = PROBA_SUFFIX.join(name_split[:-1])
        label = name_split[-1]
        try:
            label = int(label)
        except Exception:
            raise_log(
                ValueError(
                    f"Could not parse class label from name: {name}. "
                    f"The component names must follow the naming convention: '<component_name: str>_p<label: int>'."
                ),
            )
        names_indices_label.append((c_name, idx, label))

    # get label with highest probability for each component and time step
    comp_labels = np.zeros((len(vals), len(actual_components), 1), dtype=vals.dtype)
    for comp_idx, component_name in enumerate(actual_components):
        labels, indices = [], []
        for name, param_idx, label in names_indices_label:
            if name == component_name:
                labels.append(label)
                indices.append(param_idx)

        comp_labels[:, comp_idx, :] = np.take(
            labels, np.apply_along_axis(np.argmax, COMP_AX, vals[:, indices])
        )
    return comp_labels


def _get_wrapped_metric(
    func: Callable[..., METRIC_OUTPUT_TYPE], n_wrappers: int = 2
) -> Callable[..., METRIC_OUTPUT_TYPE]:
    """Returns the inner metric function `func` which bypasses the decorators `multi_ts_support` and
    `multivariate_support`. It significantly decreases process time compared to calling `func` directly.
    Only use this to compute a pre-defined metric within the scope of another metric.
    """
    if not 2 <= n_wrappers <= 3:
        raise_log(
            NotImplementedError("Only 2-3 wrappers are currently supported"),
            logger=logger,
        )
    if n_wrappers == 2:
        return func.__wrapped__.__wrapped__
    else:
        return func.__wrapped__.__wrapped__.__wrapped__


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
            ValueError(f"Seasonality `m` must be of type `int`, received `m={m}`"),
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


def _unique_labels(y_true: np.ndarray, y_pred: np.ndarray) -> list[np.ndarray]:
    """Returns unique labels for each component in the true and predicted labels."""
    labels = []
    for comp_idx in range(y_true.shape[1]):
        labels_true = np.unique(y_true[:, comp_idx])
        labels_pred = np.unique(y_pred[:, comp_idx])
        labels.append(np.unique(np.concatenate([labels_true, labels_pred])))
    return labels


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
    compute_multilabel: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Computes a confusion matrix using numpy for two np.arrays `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    labels : Optional[np.ndarray]
        The labels to consider for the confusion matrix. If `None`, will use unique labels from `y_true` and `y_pred`.
    compute_multilabel : bool
        Whether to compute a multilabel confusion matrix. If `True`, will return a component- and label-specific
        confusion matrix.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray]]
        The confusion matrix and optionally the multilabel confusion matrix.
    """
    n_comps = y_true.shape[COMP_AX]

    # global unique labels
    labels_all = np.unique(np.concatenate([y_true, y_pred]))

    # optionally, add user labels (might not be present in global labels)
    if labels is not None:
        labels_all = np.unique(np.concatenate([labels, labels_all]))

    # component-specific labels
    labels_comp = _unique_labels(y_true, y_pred)

    label_to_idx = {label: idx for idx, label in enumerate(labels_all)}
    n_labels = len(labels_all)
    conf_matrix = np.zeros(
        shape=(
            1,  # for time dimension
            n_comps,
            n_labels,
            n_labels,
        ),
        dtype=y_true.dtype,
    )
    # fill the confusion matrix
    for comp_idx in range(n_comps):
        for label in labels_all:
            if label not in labels_comp[comp_idx]:
                conf_matrix[0, comp_idx, label_to_idx[label], :] = np.nan
                conf_matrix[0, comp_idx, :, label_to_idx[label]] = np.nan

        for t_i, p_i in zip(y_true[:, comp_idx, 0], y_pred[:, comp_idx, 0]):
            conf_matrix[0, comp_idx, label_to_idx[t_i], label_to_idx[p_i]] += 1

    multilabel_conf_matrix = None
    if compute_multilabel:
        # create a component- and label-specific confusion matrix with elements:
        # [
        #     ["TN", "FP"],
        #     ["FN", "TP"],
        # ]
        multilabel_conf_matrix = np.zeros(
            shape=(
                n_comps,
                n_labels,
                2,
                2,
            ),
            dtype=y_true.dtype,
        )
        for comp_idx in range(n_comps):
            cm_total = np.nansum(conf_matrix[0, comp_idx])
            for t_idx in range(n_labels):
                # TP
                tp = conf_matrix[0, comp_idx, t_idx, t_idx]
                # FN
                fn = np.nansum(conf_matrix[0, comp_idx, t_idx, :]) - tp
                # FP
                fp = np.nansum(conf_matrix[0, comp_idx, :, t_idx]) - tp
                # TN
                tn = cm_total - tp - fn - fp

                multilabel_conf_matrix[comp_idx, t_idx] = [[tn, fp], [fn, tp]]

    # optionally, extract specific labels
    if labels is not None:
        labels_sel_idx = [label_to_idx.get(label) for label in labels]
        conf_matrix = conf_matrix[:, :, labels_sel_idx][:, :, :, labels_sel_idx]
        if compute_multilabel:
            multilabel_conf_matrix = multilabel_conf_matrix[:, labels_sel_idx]
    return conf_matrix, multilabel_conf_matrix


def _compute_score(
    y_true,
    y_pred,
    score_func: Callable,
    label_reduction: _LabelReduction,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes a score on the confusion matrix of two np.arrays `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    score_func : Callable
        The function to compute the score from the confusion matrix.
    label_reduction : Optional[str]
        The label reduction method to apply. Can be one of `None`, `"micro"`, `"macro"`, or `"weighted"`.
    labels : Optional[np.ndarray]
        The labels to consider for the confusion matrix. If `None`, will use unique labels from `y_true` and `y_pred`.

    Returns
    -------
    np.ndarray
        The computed scores based on the confusion matrix.
    """
    _, cm = _confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        compute_multilabel=True,
        labels=labels,
    )

    tn, tp, fp, fn = cm[_TN_IDX], cm[_TP_IDX], cm[_FP_IDX], cm[_FN_IDX]
    if label_reduction == _LabelReduction.MICRO:
        # micro f1 score: 2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn))
        tn = np.nansum(tn, axis=COMP_AX)
        tp = np.nansum(tp, axis=COMP_AX)
        fp = np.nansum(fp, axis=COMP_AX)
        fn = np.nansum(fn, axis=COMP_AX)

    scores = score_func(tn, fp, fn, tp)
    if label_reduction == _LabelReduction.NONE:
        # label-specific score: score_func(x)
        scores = scores.reshape((1, y_true.shape[COMP_AX], -1))
    elif label_reduction == _LabelReduction.MACRO:
        # macro score: mean(score_fun(x))
        scores = np.nanmean(scores, axis=COMP_AX)
        scores = scores.reshape((-1, 1))
    elif label_reduction == _LabelReduction.WEIGHTED:
        # weighted f1 score: sum(score_func(x) * weights) / sum(weights)
        # weights are the number of positives (where y_true == label; TP + FN)
        weights = tp + fn
        scores = np.nansum(scores * weights, axis=1, keepdims=True) / np.nansum(
            weights, axis=1, keepdims=True
        )
    else:  # "micro"
        # micro f1 score: score_func(sum(x))
        scores = scores.reshape((-1, 1))
    return scores
