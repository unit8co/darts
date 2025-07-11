from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np

from darts.logging import get_logger
from darts.metrics.metrics import (
    COMP_AX,
    METRIC_OUTPUT_TYPE,
    SMPL_AX,
    TIME_AX,
    _get_values_or_raise,
    classification_support,
    multi_ts_support,
    multivariate_support,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


def _transform_samples_to_highest_count_sample(y_pred):
    # Get the most frequent item in 1D array, smaller item is picked in case of equality
    get_most_frequent_value_in_1d = lambda array_1d: np.bincount(array_1d).argmax()
    most_frequent_samples = np.apply_along_axis(
        get_most_frequent_value_in_1d, SMPL_AX, y_pred
    )
    return np.expand_dims(most_frequent_samples, SMPL_AX)  # Add back sample idx


def _transform_probabilities_to_most_likely(
    y_pred, component_names, probabilities_names
):
    # format {component_name: unique str}_p_{class label: int}
    names_indices_label = [
        ("_p_".join(name.split("_p_")[:-1]), idx, int(name.split("_p_")[-1]))
        for idx, name in enumerate(probabilities_names)
    ]

    # Shape is reduced to the number of component is the original series
    new_shape = list(y_pred.shape)
    new_shape[COMP_AX] = len(component_names)
    sampled_class = np.zeros(new_shape)

    for i, component_name in enumerate(component_names):
        labels, indices = zip(*[
            (label, idx)
            for name, idx, label in names_indices_label
            if name == component_name
        ])
        sampled_class[:, i] = np.take(
            labels, np.apply_along_axis(np.argmax, COMP_AX, y_pred[:, indices])
        )

    return sampled_class

    # most_likely_class = np.apply_along_axis(np.argmax, SMPL_AX, y_pred)
    # return np.expand_dims(most_likely_class, SMPL_AX)  # Add back sample idx


@classification_support
@multi_ts_support
@multivariate_support
def macc(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """Accuracy (ACC).

    For the true series :math:`y` and predicted series :math:`\\hat{y}` of length :math:`T`, it is computed per
    component/column as:

     .. math:: \\frac{1}{T}\\sum_{t=1}^T\\mathbb{I}(y_t = \\hat{y}_t)

    Where :math::`mathbb{I}` is the indicator function.
    If :math:`\\hat{y}_t` are stochastic (contains several samples) takes the label with the highest count,
    if :math:`\\hat{y}_t` are probabilistic (contains ClassProbability likelihood parameters)
    takes the label with the highest score.

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

    print(y_pred.shape)

    # check for probabilistic first as it would satisfy stochastic condition
    # Prediction are probabilistic
    if y_pred.shape[COMP_AX] != y_true.shape[COMP_AX]:
        y_pred = _transform_probabilities_to_most_likely(
            y_pred,
            component_names=actual_series.components,
            probabilities_names=pred_series.components,
        )

    print(y_pred.shape)

    # Prediction are stochastic
    if y_pred.shape[SMPL_AX] != 1:
        y_pred = _transform_samples_to_highest_count_sample(y_pred)

    return np.mean(
        y_true == y_pred,
        axis=TIME_AX,
    )
