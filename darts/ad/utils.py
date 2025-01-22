"""
Utils for Anomaly Detection
---------------------------

Common functions used throughout the Anomaly Detection module.
"""

# TODO:
#     - migrate metrics function to darts.metric
#     - check error message
#     - create a zoom option on anomalies for a show function
#     - add an option to visualize: "by window", "unique", "together"
#     - create a normalize option in plot function (norm every anomaly score btw 1 and 0) -> to be seen on the same plot

from collections.abc import Sequence
from typing import Callable, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


def eval_metric_from_scores(
    anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    pred_scores: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Computes a score/metric between anomaly scores against true anomalies.

    `anomalies` and `pred_scores` must have the same shape.
    `anomalies` must be binary and have values belonging to the two classes (0 and 1).

    If one series is given for `anomalies` and `pred_scores` contains more than
    one series, the function will consider `anomalies` as the ground truth anomalies for
    all scores in `pred_scores`.

    Parameters
    ----------
    anomalies
        The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
    pred_scores
        The (sequence of) of estimated anomaly score series indicating how anomalous each window of size w is.
    window
        Integer value indicating the number of past samples each point represents in the `pred_scores`.
        The parameter will be used to transform `anomalies`.
        If a list of integers, the length must match the number of series in `pred_scores`.
        If an integer, the value will be used for every series in `pred_scores` and `anomalies`.
    metric
        The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
        Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
        Default: "AUC_ROC".

    Returns
    -------
    float
        A single score/metric for univariate `pred_scores` series (with only one component/column).
    Sequence[float]
        A sequence (list) of scores for:

        - multivariate `pred_scores` series (multiple components). Gives a score for each component.
        - a sequence (list) of univariate `pred_scores` series. Gives a score for each series.
    Sequence[Sequence[float]]
        A sequence of sequences of scores for a sequence of multivariate `pred_scores` series.
        Gives a score for each series (outer sequence) and component (inner sequence).
    """
    return _eval_metric(
        anomalies=anomalies,
        pred_series=pred_scores,
        window=window,
        metric=metric,
        pred_is_binary=False,
    )


def eval_metric_from_binary_prediction(
    anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    pred_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: Literal["recall", "precision", "f1", "accuracy"] = "recall",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Computes a score/metric between predicted anomalies against true anomalies.

    `pred_anomalies` and `anomalies` must have:

        - identical dimensions (number of time steps and number of components/columns),
        - binary values belonging to the two classes (`1` if it is an anomaly and `0` if not)

    If one series is given for `anomalies` and `pred_anomalies` contains more than
    one series, the function will consider `anomalies` as the true anomalies for
    all scores in `pred_scores`.

    Parameters
    ----------
    anomalies
        The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
    pred_anomalies
        The (sequence of) predicted binary anomaly series.
    window
        Integer value indicating the number of past samples each point represents in the `pred_scores`.
        The parameter will be used to transform `anomalies`.
        If a list of integers, the length must match the number of series in `pred_scores`.
        If an integer, the value will be used for every series in `pred_scores` and `anomalies`.
    metric
        The name of the metric function to use. Must be one of "recall", "precision", "f1", and "accuracy".
        Default: "recall".

    Returns
    -------
    float
        A single score for univariate `pred_anomalies` series (with only one component/column).
    Sequence[float]
        A sequence (list) of scores for:

        - multivariate `pred_anomalies` series (multiple components). Gives a score for each component.
        - a sequence (list) of univariate `pred_anomalies` series. Gives a score for each series.
    Sequence[Sequence[float]]
        A sequence of sequences of scores for a sequence of multivariate `pred_anomalies` series.
        Gives a score for each series (outer sequence) and component (inner sequence).
    """
    return _eval_metric(
        anomalies=anomalies,
        pred_series=pred_anomalies,
        window=window,
        metric=metric,
        pred_is_binary=True,
    )


def _eval_metric(
    anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]],
    metric: Literal["AUC_ROC", "AUC_PR", "recall", "precision", "f1", "accuracy"],
    pred_is_binary: bool,
):
    """Computes a score/metric between anomaly scores or binary predicted anomalies against true
    anomalies.

    Parameters
    ----------
    anomalies
        The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
    pred_series
        The (sequence of) anomaly scores or predicted binary anomaly series.
    window
        Integer value indicating the number of past samples each point represents in the `pred_scores`.
        The parameter will be used to transform `anomalies`.
        If a list of integers, the length must match the number of series in `pred_scores`.
        If an integer, the value will be used for every series in `pred_scores` and `anomalies`.
    metric
        The name of the scoring function to use. Must be one of "recall", "precision",
        "f1", and "accuracy" if `pred_is_binary` is `True`. Otherwise, must be one of "AUC_ROC", "AUC_PR".
    pred_is_binary
        Whether `pred_series` refers predicted binary anomalies or anomaly scores.

    Returns
    -------
    float
        A single score for univariate `pred_series` series (with only one component/column).
    Sequence[float]
        A sequence (list) of scores for:

        - multivariate `pred_series` series (multiple components). Gives a score for each component.
        - a sequence (list) of univariate `pred_series` series. Gives a score for each series.
    Sequence[Sequence[float]]
        A sequence of sequences of scores for a sequence of multivariate `pred_series` series.
        Gives a score for each series (outer sequence) and component (inner sequence).
    """
    metrics_exp = (
        {"recall", "precision", "f1", "accuracy"}
        if pred_is_binary
        else {"AUC_ROC", "AUC_PR"}
    )
    if metric not in metrics_exp:
        raise_log(
            ValueError(f"Argument `metric` must be one of {metrics_exp}"),
            logger=logger,
        )

    if metric == "AUC_ROC":
        metric_fn = roc_auc_score
    elif metric == "AUC_PR":
        metric_fn = average_precision_score
    elif metric == "recall":
        metric_fn = recall_score
    elif metric == "precision":
        metric_fn = precision_score
    elif metric == "f1":
        metric_fn = f1_score
    else:
        metric_fn = accuracy_score

    called_with_single_series = isinstance(pred_series, TimeSeries)
    anomalies = series2seq(anomalies)
    pred_series = series2seq(pred_series)
    window = [window] if not isinstance(window, Sequence) else window

    if len(anomalies) == 1 and len(pred_series) > 1:
        anomalies = anomalies * len(pred_series)

    name = "anomalies"
    pred_name = "pred_anomalies" if pred_is_binary else "pred_scores"
    _assert_same_length(
        anomalies,
        pred_series,
        name,
        pred_name,
    )

    if len(window) == 1:
        window = window * len(anomalies)
    else:
        if len(window) != len(anomalies):
            raise_log(
                ValueError(
                    f"The list of windows must be the same length as the list of `{pred_name}` and "
                    f"`{name}`. There must be one window value for each series. "
                    f"Found length {len(window)}, expected {len(anomalies)}."
                ),
                logger=logger,
            )

    sol = []
    for s_anomalies, s_pred, s_window in zip(anomalies, pred_series, window):
        _assert_timeseries(s_pred, name=pred_name)
        _assert_timeseries(s_anomalies, name=name)
        _assert_binary(s_anomalies, name)
        if pred_is_binary:
            _assert_binary(s_pred, pred_name)

        # if s_window > 1, the anomalies will be adjusted so that it can be compared timewise with s_pred
        s_anomalies = _max_pooling(s_anomalies, s_window)

        _sanity_check_two_series(s_pred, s_anomalies, pred_name, name)

        s_pred_vals = s_pred.slice_intersect_values(s_anomalies, copy=False)
        s_anomalies_vals = s_anomalies.slice_intersect_values(s_pred, copy=False)

        if not len(s_pred_vals) == len(s_anomalies_vals):
            raise_log(
                ValueError(
                    f"The two time series `{pred_name}` and `{name}` "
                    f"must have at least a partially overlapping time index."
                ),
                logger=logger,
            )

        if not pred_is_binary:  # `pred_series` is an anomaly score
            nr_anomalies_per_component = s_anomalies_vals.sum(axis=0).flatten()

            if nr_anomalies_per_component.min() == 0:
                raise_log(
                    ValueError(
                        f"`{name}` does not contain anomalies. {metric} cannot be computed."
                    ),
                    logger=logger,
                )
            if nr_anomalies_per_component.max() == len(s_anomalies_vals):
                add_txt = (
                    ""
                    if s_window <= 1
                    else f" Consider decreasing the window size (window={s_window})"
                )
                raise_log(
                    ValueError(
                        f"`{name}` only contains anomalies. {metric} cannot be computed."
                        + add_txt
                    ),
                    logger=logger,
                )

        # TODO: could we vectorize this?
        metrics = []
        for component_idx in range(s_pred.width):
            metrics.append(
                metric_fn(
                    s_anomalies_vals[:, component_idx],
                    s_pred_vals[:, component_idx],
                )
            )
        sol.append(metrics if len(metrics) > 1 else metrics[0])

    return sol[0] if called_with_single_series else sol


def show_anomalies_from_scores(
    series: TimeSeries,
    anomalies: TimeSeries = None,
    pred_series: TimeSeries = None,
    pred_scores: Union[TimeSeries, Sequence[TimeSeries]] = None,
    window: Union[int, Sequence[int]] = 1,
    names_of_scorers: Union[str, Sequence[str]] = None,
    title: str = None,
    metric: Optional[Literal["AUC_ROC", "AUC_PR"]] = None,
    component_wise: bool = False,
):
    """Plot the results generated by an anomaly model.

    The plot will be composed of the following:
        - the actual series itself with the output of the model (if given)
        - the anomaly score of each scorer. The scorer with different windows will be separated.
        - the actual anomalies, if given.

    If `pred_series` is stochastic (i.e., if it has multiple samples), the function will plot:
        - the mean per timestamp
        - the quantile 0.95 for an upper bound
        - the quantile 0.05 for a lower bound

    Possible to:
        - add a title to the figure with the parameter `title`
        - give personalized names for the scorers with `names_of_scorers`
        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies is given

    Parameters
    ----------
    series
        The actual series to visualize anomalies from.
    anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
    pred_series
        Output of the model given as input the `series` (can be stochastic).
    pred_scores
        Output of the scorers given the output of the model and `series`.
    window
        Window parameter for each anomaly scores.
        Default: 1. If a list of anomaly scores is given, the same default window will be used for every score.
    names_of_scorers
        Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        Only effective when `pred_scores` is not `None`.
    title
        Title of the figure
    metric
        Optionally, the name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
        Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
        Only effective when `pred_scores` is not `None`.
        Default: "AUC_ROC".
    component_wise
        If True, will separately plot each component in case of multivariate anomaly detection.
    """
    series = _check_input(
        series,
        name="series",
        num_series_expected=1,
        check_multivariate=component_wise,
    )[0]

    if title is None and pred_scores is not None:
        title = "Anomaly results"

    nbr_plots = 1
    if anomalies is not None:
        nbr_plots = nbr_plots + 1
    elif metric is not None:
        raise_log(
            ValueError("`anomalies` must be given in order to calculate a metric."),
            logger=logger,
        )

    pred_scores = series2seq(pred_scores)
    if pred_scores is not None:
        if names_of_scorers is not None:
            names_of_scorers = (
                [names_of_scorers]
                if isinstance(names_of_scorers, str)
                else names_of_scorers
            )
            if len(names_of_scorers) != len(pred_scores):
                raise_log(
                    ValueError(
                        f"The number of names in `names_of_scorers` must match the "
                        f"number of anomaly score given as input, found "
                        f"{len(names_of_scorers)} and expected {len(pred_scores)}."
                    ),
                    logger=logger,
                )

        window = [window] if isinstance(window, int) else window
        if not all([w > 0 for w in window]):
            raise_log(
                ValueError(
                    "Parameter `window` must be a positive integer, "
                    "or a sequence of positive integers."
                ),
                logger=logger,
            )
        window = window if len(window) > 1 else window * len(pred_scores)
        if len(window) != len(pred_scores):
            raise_log(
                ValueError(
                    f"The number of window in `window` must match the "
                    f"number of anomaly score given as input. One window "
                    f"value for each series. Found length {len(window)}, "
                    f"and expected {len(pred_scores)}."
                ),
                logger=logger,
            )

        if not all([w < len(s) for (w, s) in zip(window, pred_scores)]):
            raise_log(
                ValueError(
                    "Parameter `window` must be an integer or sequence of integers "
                    "with value(s) smaller than the length of the corresponding series "
                    "in `pred_scores`."
                ),
                logger=logger,
            )

        nbr_plots += len(set(window))

    series_width = series.n_components
    if pred_series is not None:
        pred_series = _check_input(
            pred_series,
            name="pred_series",
            width_expected=series_width,
            num_series_expected=1,
            check_multivariate=component_wise,
        )[0]

    if anomalies is not None and component_wise:
        anomalies = _check_input(
            anomalies,
            name="anomalies",
            width_expected=series_width,
            num_series_expected=1,
            check_binary=True,
            check_multivariate=component_wise,
        )[0]

    if pred_scores is not None and component_wise:
        for pred_score in pred_scores:
            _ = _check_input(
                pred_score,
                name="pred_score",
                width_expected=series_width,
                num_series_expected=1,
                check_multivariate=component_wise,
            )[0]

    plots_per_ts = nbr_plots * series_width if component_wise else nbr_plots
    height_ratios = ([2] + [1] * (nbr_plots - 1)) * (plots_per_ts // nbr_plots)
    height_total = 2 * sum(height_ratios)
    fig, axs = plt.subplots(
        nrows=plots_per_ts,
        figsize=(8, height_total),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    for i in range(series_width if component_wise else 1):
        if component_wise:
            series_ = series[series.components[i]]
            anomalies_ = (
                anomalies[anomalies.components[i]] if anomalies is not None else None
            )
            pred_series_ = (
                pred_series[pred_series.components[i]]
                if pred_series is not None
                else None
            )
            pred_scores_ = (
                [pc[pc.components[i]] for pc in pred_scores]
                if pred_scores is not None
                else None
            )
        else:
            series_ = series
            anomalies_ = anomalies
            pred_series_ = pred_series
            pred_scores_ = pred_scores

        _plot_series_and_anomalies(
            series=series_,
            anomalies=anomalies_,
            pred_series=pred_series_,
            pred_scores=pred_scores_,
            window=window,
            names_of_scorers=names_of_scorers,
            metric=metric,
            axs=axs,
            index_ax=i * nbr_plots,
        )
    # make title fit nicely on plot
    title_height = 0.1
    title_y = 1 - title_height / height_total

    fig.suptitle(title, y=title_y)
    fig.tight_layout()


def _assert_binary(series: TimeSeries, name: str):
    """Checks if series is a binary timeseries (1 and 0)"

    Parameters
    ----------
    series
        series to check for.
    name
        name of the series.
    """

    vals = series.values(copy=False)
    if not np.array_equal(vals, vals.astype(bool)):
        raise_log(
            ValueError(f"Input series `{name}` must have binary values only."),
            logger=logger,
        )


def _assert_timeseries(series: TimeSeries, name: str = "series"):
    """Checks if given input is of type Darts TimeSeries"""
    if not isinstance(series, TimeSeries):
        raise_log(
            ValueError(
                f"all series in `{name}` must be `TimeSeries`. Received {type(series)}."
            ),
            logger=logger,
        )


def _sanity_check_two_series(
    series_1: TimeSeries,
    series_2: TimeSeries,
    name_series_1: str,
    name_series_2: str,
):
    """Performs sanity check on the two given inputs

    Checks if the two inputs:
        - type is Darts Timeseries
        - have the same number of components
        - if their intersection in time is not null

    Parameters
    ----------
    series_1
        1st time series
    series_2:
        2nd time series
    """

    _assert_timeseries(series_1, name=name_series_1)
    _assert_timeseries(series_2, name=name_series_2)

    # check if the two inputs time series have the same number of components
    if series_1.width != series_2.width:
        raise_log(
            ValueError(
                f"The series from `{name_series_1}` and `{name_series_2}` must have the "
                f"same number of components, found {series_1.width} and {series_2.width}."
            ),
            logger=logger,
        )


def _max_pooling(series: TimeSeries, window: int) -> TimeSeries:
    """Slides a window of size `window` along the input series, and replaces the value of the
    input time series by the maximum of the values contained in the window.

    The binary time series output represents if there is an anomaly (=1) or not (=0) in the past
    window points. The new series will equal the length of the input series - window. Its first
    point will start at the first time index of the input time series + window points.

    Parameters
    ----------
    series:
        Binary time series.
    window:
        Integer value indicating the number of past samples each point represents.

    Returns
    -------
    Binary TimeSeries
    """
    if window <= 0:
        raise_log(
            ValueError(
                f"Parameter `window` must be strictly greater than 0, found size {window}."
            ),
            logger=logger,
        )
    if window >= len(series):
        raise_log(
            ValueError(
                f"Parameter `window` must be smaller than the length of the "
                f"input series, found window size {window}, and max size {len(series)}."
            ),
            logger=logger,
        )

    if window == 1:
        # the process results in replacing every value by itself -> return directly the series
        return series

    return series.window_transform(
        transforms={
            "window": window,
            "function": "max",
            "mode": "rolling",
            "min_periods": window,
        },
        treat_na="dropna",
    )


def _assert_same_length(
    list_series_1: Sequence[TimeSeries],
    list_series_2: Sequence[TimeSeries],
    name_series_1: str,
    name_series_2: str,
):
    """Checks if the two sequences contain the same number of TimeSeries."""

    if len(list_series_1) != len(list_series_2):
        raise_log(
            ValueError(
                f"Number of `{name_series_2}` must match the number of given "
                f"`{name_series_1}`, found length {len(list_series_2)} and "
                f"expected {len(list_series_1)}."
            ),
            logger=logger,
        )


def _plot_series(series, ax_id, linewidth, label_name, **kwargs):
    """Internal function called by `show_anomalies_from_scores()`

    Plot the series on the given axes ax_id.

    Parameters
    ----------
    series
        The series to plot.
    ax_id
        The axis the series will be plotted on.
    linewidth
        Thickness of the line.
    label_name
        Name that will appear in the legend.
    """
    for i, c in enumerate(series._xa.component[:10]):
        comp = series._xa.sel(component=c)

        if comp.sample.size > 1:
            central_series = comp.mean(dim="sample")
            low_series = comp.quantile(q=0.05, dim="sample")
            high_series = comp.quantile(q=0.95, dim="sample")
        else:
            central_series = comp

        label_to_use = (
            (label_name + ("_" + str(i) if len(series.components) > 1 else ""))
            if label_name != ""
            else "" + str(str(c.values))
        )

        central_series.plot(ax=ax_id, linewidth=linewidth, label=label_to_use, **kwargs)

        if comp.sample.size > 1:
            ax_id.fill_between(
                series.time_index, low_series, high_series, alpha=0.25, **kwargs
            )


def _check_input(
    series: Union[TimeSeries, Sequence[TimeSeries]],
    name: str,
    width_expected: Optional[int] = None,
    check_deterministic: bool = False,
    check_binary: bool = False,
    check_multivariate: bool = False,
    num_series_expected: Optional[int] = None,
    extra_checks: Optional[Callable] = None,
):
    """
    Input `series` checks used for Aggregators, Detectors, ...

    - `series` must be (sequence of) series with length (`num_series_expected`) where each series must:
        - have width `width_expected` if it is not `None`
        - be deterministic if `check_deterministic=True`
        - be binary if `check_binary=True`
        - be multivariate if `check_multivariate=True`

    By default, all checks except the `TimeSeries` check are disabled.

    Parameters
    ----------
    series
        A (sequence of) multivariate series.
    name
        The name of the series.
    width_expected
        Optionally, the expected number of components/width of each series.
    check_multivariate
        Whether to check if all series are multivariate.
    """
    series = series2seq(series)
    if num_series_expected is not None and len(series) != num_series_expected:
        if num_series_expected == 1:
            err_txt = f"`{name}` must be single `TimeSeries` or a sequence of `TimeSeries` of length `1`."
        else:
            err_txt = f"`{name}` must be a sequence of `TimeSeries` of length `{num_series_expected}`."
        raise_log(
            ValueError(err_txt),
            logger=logger,
        )
    for s in series:
        if not isinstance(s, TimeSeries):
            raise_log(
                ValueError(f"all series in `{name}` must be of type `TimeSeries`."),
                logger=logger,
            )
        if check_deterministic and not s.is_deterministic:
            raise_log(
                ValueError(
                    f"all series in `{name}` must be deterministic (number of samples=1)."
                ),
                logger=logger,
            )
        if check_binary:
            _assert_binary(s, name=name)
        if check_multivariate and s.width <= 1:
            raise_log(
                ValueError(f"all series in `{name}` must be multivariate (width>1)."),
                logger=logger,
            )
        if width_expected is not None and s.width != width_expected:
            raise_log(
                ValueError(
                    f"all series in `{name}` must have `{width_expected}` component(s) (width={width_expected})."
                ),
                logger=logger,
            )
        if extra_checks is not None:
            extra_checks(s)
    return series


def _assert_fit_called(fit_called: bool, name: str):
    """Checks that `fit_called` is `True`."""
    if not fit_called:
        raise_log(
            ValueError(
                f"The `{name}` has not been fitted yet. Call `{name}.fit()` first."
            ),
            logger=logger,
        )


def _plot_series_and_anomalies(
    series: TimeSeries,
    anomalies: TimeSeries,
    pred_series: TimeSeries,
    pred_scores: Sequence[TimeSeries],
    window: Sequence[int],
    names_of_scorers: Sequence[str],
    metric: str,
    axs: plt.Axes,
    index_ax: int,
):
    """Helper function to plot series and anomalies.

    Parameters
    ----------
    series
        The actual series to visualize anomalies from.
    anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
    pred_series
        Output of the model given as input the `series` (can be stochastic).
    pred_scores
        Output of the scorers given the output of the model and `series`.
    window
        Window parameter for each anomaly scores.
    names_of_scorers
        Name of the scores.
    metric
        The name of the metric function to use.
    axs
        The axes to plot on.
    index_ax
        The index of the current axis.
    """
    _plot_series(series=series, ax_id=axs[index_ax], linewidth=0.5, label_name="")

    if pred_series is not None:
        _plot_series(
            series=pred_series,
            ax_id=axs[index_ax],
            linewidth=0.5,
            label_name="model output",
        )

    axs[index_ax].set_title("")

    if anomalies is not None or pred_scores is not None:
        axs[index_ax].set_xlabel("")

    axs[index_ax].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

    if pred_scores is not None:
        dict_input = {}

        for idx, (score, w) in enumerate(zip(pred_scores, window)):
            dict_input[idx] = {"series_score": score, "window": w, "name_id": idx}

        for index, elem in enumerate(
            sorted(dict_input.items(), key=lambda x: x[1]["window"])
        ):
            if index == 0:
                current_window = elem[1]["window"]
                index_ax = index_ax + 1

            idx = elem[1]["name_id"]
            w = elem[1]["window"]

            if w != current_window:
                current_window = w
                index_ax = index_ax + 1

            if metric is not None:
                value = round(
                    eval_metric_from_scores(
                        anomalies=anomalies,
                        pred_scores=pred_scores[idx],
                        window=w,
                        metric=metric,
                    ),
                    3,
                )
            else:
                value = None

            if names_of_scorers is not None:
                label = names_of_scorers[idx] + [f" ({value})", ""][value is None]
            else:
                label = f"score_{str(idx)}" + [f" ({value})", ""][value is None]

            _plot_series(
                series=elem[1]["series_score"],
                ax_id=axs[index_ax],
                linewidth=0.5,
                label_name=label,
            )

            axs[index_ax].legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2)
            axs[index_ax].set_title(f"Window: {str(w)}", loc="left")
            axs[index_ax].set_title("")
            axs[index_ax].set_xlabel("")

    if anomalies is not None:
        _plot_series(
            series=anomalies,
            ax_id=axs[index_ax + 1],
            linewidth=1,
            label_name="anomalies",
            color="red",
        )

        axs[index_ax + 1].set_title("")
        axs[index_ax + 1].set_ylim([-0.1, 1.1])
        axs[index_ax + 1].set_yticks([0, 1])
        axs[index_ax + 1].set_yticklabels(["no", "yes"])
        axs[index_ax + 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    else:
        axs[index_ax].set_xlabel("timestamp")
