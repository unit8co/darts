from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from darts.metrics.metrics import (
    METRIC_OUTPUT_TYPE,
    TIME_AX,
    _get_values_or_raise,
    multi_ts_support,
    multivariate_support,
)
from darts.timeseries import TimeSeries


@multi_ts_support
@multivariate_support
def acc(
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
    """Accuarcy (acc).



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
        A single metric score (when `len(q) <= 1`) for:

        - a single univariate series.
        - a single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components * n quantiles) without time
        and component reductions, and shape (n time steps, n quantiles) without time but component reduction and
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
    )
    return np.mean(
        y_true == y_pred,
        axis=TIME_AX,
    )
