from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np

from darts.logging import get_logger, raise_log
from darts.metrics.metrics import (
    COMP_AX,
    METRIC_OUTPUT_TYPE,
    TIME_AX,
    _get_values_or_raise,
    multi_ts_support,
    multivariate_support,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


@multi_ts_support
@multivariate_support
def acc(
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
    """Accuarcy (ACC).



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
        Optionally, whether to print operations progress

    Returns
    -------
    float
        A single metric score for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,1) without component reduction.
        For:

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
        remove_nan_union=False,
    )
    return np.mean(
        y_true == y_pred,
        axis=TIME_AX,
    )


def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    if y_true.shape[0] != y_pred.shape[0]:
        raise_log(
            ValueError(
                "y_true and y_pred have different number of samples. "
                f"y_true: {y_true.shape}, y_pred: {y_pred.shape}"
            )
        )

    if normalize not in {None, "true", "pred", "all"}:
        raise_log(
            ValueError(
                "normalize must be one of [None, 'true', 'pred', 'all']. "
                f"Got {normalize} instead."
            )
        )
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.array(labels)
        n_labels = labels.size
        if n_labels == 0:
            raise ValueError("'labels' should contains at least one label.")
        elif y_true.size == 0:
            return np.zeros((n_labels, n_labels), dtype=int)
        elif len(np.intersect1d(y_true, labels)) == 0:
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    if sample_weight.shape[0] != y_true.shape[0]:
        raise_log(
            ValueError(
                "sample_weight and y_true have different number of samples. "
                f"y_true: {y_true.shape}, sample_weight: {sample_weight.shape}"
            )
        )

    n_labels = labels.size
    # If labels are not consecutive integers starting from zero, then
    # y_true and y_pred must be converted into index form
    need_index_conversion = not (
        labels.dtype.kind in {"i", "u", "b"}
        and np.all(labels == np.arange(n_labels))
        and y_true.min() >= 0
        and y_pred.min() >= 0
    )

    if need_index_conversion:
        label_to_ind = {y: x for x, y in enumerate(labels)}
        y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
        y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    if not np.all(ind):
        y_pred = y_pred[ind]
        y_true = y_true[ind]
        # also eliminate weights of eliminated items
        sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {"i", "u", "b"}:
        dtype = np.int64
    else:
        dtype = np.float64

    result = np.zeros((n_labels, n_labels), dtype=dtype)
    for i in range(y_true.shape[0]):
        result[y_true[i]][y_pred[i]] += sample_weight[i]

    if normalize is not None:
        result = result.astype(np.float64, copy=False)
        if normalize == "true":
            result /= np.sum(result, axis=1, keepdims=True)
        elif normalize == "pred":
            result /= np.sum(result, axis=0, keepdims=True)
        elif normalize == "all":
            result /= np.sum(result)
    return result


def precision_recall_fscore_support(y_true, y_pred, labels=None, sample_weight=None):
    C = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    precision = np.diag(C) / np.sum(C, axis=0)
    recall = np.diag(C) / np.sum(C, axis=1)
    fscore = 2 * precision * recall / (precision + recall)
    support = np.sum(C, axis=1)

    return precision, recall, fscore, support


@multi_ts_support
@multivariate_support
def bacc(
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
    """Balanced Accuarcy (BACC).

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.

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
        Optionally, whether to print operations progress

     Returns
    -------
    float
        A single metric score for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,1) without component reduction.
        For:

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
        remove_nan_union=False,
    )
    scores = np.zeros((y_true.shape[COMP_AX], 1))
    for comp in range(y_true.shape[COMP_AX]):
        C = confusion_matrix(
            y_true[:, comp],
            y_pred[:, comp],
            labels=None,
            sample_weight=None,
            normalize=None,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class = np.diag(C) / C.sum(axis=1)
        if np.any(np.isnan(per_class)):
            logger.warning("y_pred contains classes not in y_true")
            per_class = per_class[~np.isnan(per_class)]

        scores[comp] = np.mean(per_class)

    return scores


@multi_ts_support
@multivariate_support
def p(
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
    """Precision (P).



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
        Optionally, whether to print operations progress

     Returns
    -------
    float
        A single metric score for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction` and `component_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n components,1) without component reduction.
        For:

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
        remove_nan_union=False,
    )
    scores = np.zeros((y_true.shape[COMP_AX], 1))
    for comp in range(y_true.shape[COMP_AX]):
        precision, _, _, _ = precision_recall_fscore_support(
            y_true[:, comp],
            y_pred[:, comp],
            labels=None,
            sample_weight=None,
        )

        scores[comp] = np.nanmean(precision)

    return scores
