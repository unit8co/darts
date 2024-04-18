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

from typing import Optional, Sequence, Tuple, Union

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
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    anomaly_score: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "AUC_ROC",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Computes a score/metric between anomaly scores against true anomalies.

    `actual_anomalies` and `anomaly_score` must have the same shape.
    `actual_anomalies` must be binary and have values belonging to the two classes (0 and 1).

    If one series is given for `actual_anomalies` and `anomaly_score` contains more than
    one series, the function will consider `actual_anomalies` as the ground truth anomalies for
    all scores in `anomaly_score`.

    Parameters
    ----------
    actual_anomalies
        The (sequence of) ground truth anomaly series (`1` if it is an anomaly and `0` if not).
    anomaly_score
        The (sequence of) of estimated anomaly score series indicating how anomalous each window of size w is.
    window
        Integer value indicating the number of past samples each point represents in the `anomaly_score`.
        The parameter will be used to transform `actual_anomalies`.
        If a list of integers, the length must match the number of series in `anomaly_score`.
        If an integer, the value will be used for every series in `anomaly_score` and `actual_anomalies`.
    metric
        The name of the scoring function to use. Must be one of "AUC_ROC" (Area Under the
        Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
        Default: "AUC_ROC"

    Returns
    -------
    float
        A single score/metric for univariate `anomaly_score` series (with only one component/column).
    Sequence[float]
        A sequence (list) of scores for:

        - multivariate `anomaly_score` series (multiple components). Gives a score for each component.
        - a sequence (list) of univariate `anomaly_score` series. Gives a score for each series.
    Sequence[Sequence[float]]
        A sequence of sequences of scores for a sequence of multivariate `anomaly_score` series.
        Gives a score for each series (outer sequence) and component (inner sequence).
    """
    return _eval_metric(
        actual_series=actual_anomalies,
        pred_series=anomaly_score,
        window=window,
        metric=metric,
        pred_is_binary=False,
    )


def eval_metric_from_binary_prediction(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "recall",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Computes a score/metric between predicted anomalies against true anomalies.

    `pred_anomalies` and `actual_series` must have:

        - identical dimensions (number of time steps and number of components/columns),
        - binary values belonging to the two classes (`1` if it is an anomaly and `0` if not)

    If one series is given for `actual_series` and `pred_series` contains more than
    one series, the function will consider `actual_series` as the true anomalies for
    all scores in `anomaly_score`.

    Parameters
    ----------
    actual_series
        The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
    pred_series
        The (sequence of) predicted binary anomaly series.
    window
        Integer value indicating the number of past samples each point represents in the `anomaly_score`.
        The parameter will be used to transform `actual_series`.
        If a list of integers, the length must match the number of series in `anomaly_score`.
        If an integer, the value will be used for every series in `anomaly_score` and `actual_series`.
    metric
        The name of the scoring function to use. Must be one of "recall", "precision", "f1", and "accuracy".
         Default: "recall"

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
    return _eval_metric(
        actual_series=actual_series,
        pred_series=pred_series,
        window=window,
        metric=metric,
        pred_is_binary=True,
    )


def _eval_metric(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]],
    metric: str,
    pred_is_binary: bool,
):
    """Computes a score/metric between anomaly scores or binary predicted anomalies against true
    anomalies.

    Parameters
    ----------
    actual_series
        The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
    pred_series
        The (sequence of) anomaly scores or predicted binary anomaly series.
    window
        Integer value indicating the number of past samples each point represents in the `anomaly_score`.
        The parameter will be used to transform `actual_series`.
        If a list of integers, the length must match the number of series in `anomaly_score`.
        If an integer, the value will be used for every series in `anomaly_score` and `actual_series`.
    metric
        Optionally, the name of the scoring function to use. Must be one of "recall", "precision",
        "f1", and "accuracy". Default: "recall"
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
    actual_series = series2seq(actual_series)
    pred_series = series2seq(pred_series)
    window = [window] if not isinstance(window, Sequence) else window

    if len(actual_series) == 1 and len(pred_series) > 1:
        actual_series = actual_series * len(pred_series)

    _assert_same_length(actual_series, pred_series)

    actual_name = "actual_series" if pred_is_binary else "actual_anomalies"
    pred_name = "pred_series" if pred_is_binary else "anomaly_score"
    if len(window) == 1:
        window = window * len(actual_series)
    else:
        if len(window) != len(actual_series):
            raise_log(
                ValueError(
                    f"The list of windows must be the same length as the list of `{pred_name}` and "
                    f"`{actual_name}`. There must be one window value for each series. "
                    f"Found length {len(window)}, expected {len(actual_series)}."
                ),
                logger=logger,
            )

    sol = []
    for s_anomalies, s_pred, s_window in zip(actual_series, pred_series, window):
        _assert_timeseries(s_pred, name=pred_name)
        _assert_timeseries(s_anomalies, name=actual_name)
        _assert_binary(s_anomalies, actual_name)
        if pred_is_binary:
            _assert_binary(s_pred, pred_name)

        # if s_window > 1, the anomalies will be adjusted so that it can be compared timewise with s_pred
        s_anomalies = _max_pooling(s_anomalies, s_window)

        _sanity_check_two_series(s_pred, s_anomalies)

        s_pred, s_anomalies = _intersect(s_pred, s_anomalies)

        if not pred_is_binary:  # `pred_series` is an anomaly score
            nr_anomalies_per_component = (
                s_anomalies.values(copy=False).sum(axis=0).flatten()
            )

            if nr_anomalies_per_component.min() == 0:
                raise_log(
                    ValueError(
                        f"`{actual_name}` does not contain anomalies. {metric} cannot be computed."
                    ),
                    logger=logger,
                )
            if nr_anomalies_per_component.max() == len(s_anomalies):
                add_txt = (
                    ""
                    if s_window <= 1
                    else f" Consider decreasing the window size (window={s_window})"
                )
                raise_log(
                    ValueError(
                        f"`{actual_name}` only contains anomalies. {metric} cannot be computed."
                        + add_txt
                    ),
                    logger=logger,
                )

        # TODO: could we vectorize this?
        s_anomalies_vals = s_anomalies.all_values(copy=False)
        s_pred_vals = s_pred.all_values(copy=False)
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
    model_output: TimeSeries = None,
    anomaly_scores: Union[TimeSeries, Sequence[TimeSeries]] = None,
    window: Union[int, Sequence[int]] = 1,
    names_of_scorers: Union[str, Sequence[str]] = None,
    actual_anomalies: TimeSeries = None,
    title: str = None,
    metric: str = None,
):
    """Plot the results generated by an anomaly model.

    The plot will be composed of the following:
        - the series itself with the output of the model (if given)
        - the anomaly score of each scorer. The scorer with different windows will be separated.
        - the actual anomalies, if given.

    If model_output is stochastic (i.e., if it has multiple samples), the function will plot:
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
        The series to visualize anomalies from.
    model_output
        Output of the model given as input the series (can be stochastic).
    anomaly_scores
        Output of the scorers given the output of the model and the series.
    window
        Window parameter for each anomaly scores.
        Default: 1. If a list of anomaly scores is given, the same default window will be used for every score.
    names_of_scorers
        Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        Only effective when `anomaly_scores` is not `None`.
    actual_anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
    title
        Title of the figure
    metric
        Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
        Only effective when `anomaly_scores` is not `None`. Default: "AUC_ROC"
    """
    if not isinstance(series, TimeSeries):
        raise_log(
            ValueError("`series` must be a single `TimeSeries`."),
            logger=logger,
        )

    if title is None and anomaly_scores is not None:
        title = "Anomaly results"

    nbr_plots = 1
    if actual_anomalies is not None:
        nbr_plots = nbr_plots + 1
    elif metric is not None:
        raise_log(
            ValueError(
                "`actual_anomalies` must be given in order to calculate a metric."
            ),
            logger=logger,
        )

    anomaly_scores = series2seq(anomaly_scores)
    if anomaly_scores is not None:
        if names_of_scorers is not None:
            names_of_scorers = (
                [names_of_scorers]
                if isinstance(names_of_scorers, str)
                else names_of_scorers
            )
            if len(names_of_scorers) != len(anomaly_scores):
                raise_log(
                    ValueError(
                        f"The number of names in `names_of_scorers` must match the "
                        f"number of anomaly score given as input, found "
                        f"{len(names_of_scorers)} and expected {len(anomaly_scores)}."
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
        window = window if len(window) > 1 else window * len(anomaly_scores)
        if len(window) != len(anomaly_scores):
            raise_log(
                ValueError(
                    f"The number of window in `window` must match the "
                    f"number of anomaly score given as input. One window "
                    f"value for each series. Found length {len(window)}, "
                    f"and expected {len(anomaly_scores)}."
                ),
                logger=logger,
            )

        if not all([w < len(s) for (w, s) in zip(window, anomaly_scores)]):
            raise_log(
                ValueError(
                    "Parameter `window` must be an integer or sequence of integers "
                    "with value(s) smaller than the length of the corresponding series "
                    "in `anomaly_scores`."
                ),
                logger=logger,
            )

        nbr_plots = nbr_plots + len(set(window))

    fig, axs = plt.subplots(
        nbr_plots,
        figsize=(8, 4 + 2 * (nbr_plots - 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [2] + [1] * (nbr_plots - 1)},
        squeeze=False,
    )

    index_ax = 0

    _plot_series(series=series, ax_id=axs[index_ax][0], linewidth=0.5, label_name="")

    if model_output is not None:
        _plot_series(
            series=model_output,
            ax_id=axs[index_ax][0],
            linewidth=0.5,
            label_name="model output",
        )

    axs[index_ax][0].set_title("")

    if actual_anomalies is not None or anomaly_scores is not None:
        axs[index_ax][0].set_xlabel("")

    axs[index_ax][0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

    if anomaly_scores is not None:

        dict_input = {}

        for idx, (score, w) in enumerate(zip(anomaly_scores, window)):

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
                        anomaly_score=anomaly_scores[idx],
                        actual_anomalies=actual_anomalies,
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
                ax_id=axs[index_ax][0],
                linewidth=0.5,
                label_name=label,
            )

            axs[index_ax][0].legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2
            )
            axs[index_ax][0].set_title(f"Window: {str(w)}", loc="left")
            axs[index_ax][0].set_title("")
            axs[index_ax][0].set_xlabel("")

    if actual_anomalies is not None:

        _plot_series(
            series=actual_anomalies,
            ax_id=axs[index_ax + 1][0],
            linewidth=1,
            label_name="anomalies",
            color="red",
        )

        axs[index_ax + 1][0].set_title("")
        axs[index_ax + 1][0].set_ylim([-0.1, 1.1])
        axs[index_ax + 1][0].set_yticks([0, 1])
        axs[index_ax + 1][0].set_yticklabels(["no", "yes"])
        axs[index_ax + 1][0].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2
        )
    else:
        axs[index_ax][0].set_xlabel("timestamp")

    fig.suptitle(title)


def _intersect(
    series_1: TimeSeries,
    series_2: TimeSeries,
) -> Tuple[TimeSeries, TimeSeries]:
    """Returns the sub-series of series_1 and of series_2 that share the same time index.
    (Intersection in time of the two time series)

    Parameters
    ----------
    series_1
        1st time series
    series_2:
        2nd time series

    Returns
    -------
    Tuple[TimeSeries, TimeSeries]
    """
    new_series_1 = series_1.slice_intersect(series_2)
    if len(new_series_1) == 0:
        raise_log(
            ValueError("Time intersection between the two series must be non empty."),
            logger=logger,
        )
    return new_series_1, series_2.slice_intersect(series_1)


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

    _assert_timeseries(series_1)
    _assert_timeseries(series_2)

    # check if the two inputs time series have the same number of components
    if series_1.width != series_2.width:
        raise_log(
            ValueError(
                f"Series must have the same number of components, "
                f"found {series_1.width} and {series_2.width}."
            ),
            logger=logger,
        )

    # check if the time intersection between the two inputs time series is not empty
    if len(series_1.time_index.intersection(series_2.time_index)) == 0:
        raise_log(
            ValueError("Series must have a non-empty intersection timestamps."),
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
):
    """Checks if the two sequences contain the same number of TimeSeries."""
    if len(list_series_1) != len(list_series_2):
        raise_log(
            ValueError(
                f"Sequences of series must be of the same length, "
                f"found length: {len(list_series_1)} and {len(list_series_2)}."
            ),
            logger=logger,
        )


def _plot_series(series, ax_id, linewidth, label_name, **kwargs):
    """Internal function called by ``show_anomalies_from_scores()``

    Plot the series on the given axes ax_id.

    Parameters
    ----------
    series
        The series to plot.
    ax_id
        The axis the series will be ploted on.
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
    width_expected: Optional[int],
    check_binary: bool,
    check_multivariate: bool,
):
    """
    Input `series` checks used for Aggregators, Detectors, ...

    - `series` must be (sequence of) series where each series must:
        * be deterministic
        * have width `width_expected` if it is not `None`
        * be binary if `check_binary=True`
        * be multivariate if `check_multivariate=True`

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
    for s in series:
        if not isinstance(s, TimeSeries):
            raise_log(
                ValueError(f"all series in `{name}` must be of type TimeSeries."),
                logger=logger,
            )
        if not s.is_deterministic:
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
