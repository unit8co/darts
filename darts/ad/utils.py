"""
Utils for Anomaly Detection
---------------------------

Common functions used by anomaly_model.py, scorers.py, aggregators.py and detectors.py
"""

# TODO:
#     - change structure of eval_accuracy_from_scores and eval_accuracy_from_binary_prediction (a lot of repeated code)
#     - migrate metrics function to darts.metric
#     - check error message
#     - create a zoom option on anomalies for a show function
#     - add an option visualize: "by window", "unique", "together"
#     - create a normalize option in plot function (norm every anomaly score btw 1 and 0) -> to be seen on the same plot

from typing import Sequence, Tuple, Union

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
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


def _assert_binary(series: TimeSeries, name_series: str):
    """Checks if series is a binary timeseries (1 and 0)"

    Parameters
    ----------
    series
        series to check for.
    name_series
        name str of the series.
    """

    raise_if_not(
        np.array_equal(
            series.values(copy=False),
            series.values(copy=False).astype(bool),
        ),
        f"Input series {name_series} must be a binary time series.",
    )


def eval_accuracy_from_scores(
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    anomaly_score: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "AUC_ROC",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Scores the results against true anomalies.

    `actual_anomalies` and `anomaly_score` must have the same shape.
    `actual_anomalies` must be binary and have values belonging to the two classes (0 and 1).

    If one series is given for `actual_anomalies` and `anomaly_score` contains more than
    one series, the function will consider `actual_anomalies` as the ground truth anomalies for
    all scores in `anomaly_score`.

    Parameters
    ----------
    actual_anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
    anomaly_score
        Series indicating how anomoulous each window of size w is.
    window
        Integer value indicating the number of past samples each point represents
        in the anomaly_score. The parameter will be used by the function
        ``_window_adjustment_anomalies()`` to transform actual_anomalies.
        If a list is given. the length must match the number of series in anomaly_score
        and actual_anomalies. If only one window is given, the value will be used for every
        series in anomaly_score and actual_anomalies.
    metric
        Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
        Default: "AUC_ROC"

    Returns
    -------
    Union[float, Sequence[float], Sequence[Sequence[float]]]
        Score of the anomalies score prediction
            * ``float`` if `anomaly_score` is a univariate series (dimension=1).
            * ``Sequence[float]``

                * if `anomaly_score` is a multivariate series (dimension>1),
                  returns one value per dimension.
                * if `anomaly_score` is a sequence of univariate series, returns one
                  value per series
            * ``Sequence[Sequence[float]]`` if `anomaly_score` is a sequence of
              multivariate series. Outer Sequence is over the sequence input, and the inner
              Sequence is over the dimensions of each element in the sequence input.
    """

    raise_if_not(
        metric in {"AUC_ROC", "AUC_PR"},
        "Argument `metric` must be one of 'AUC_ROC', 'AUC_PR'",
    )
    metric_fn = roc_auc_score if metric == "AUC_ROC" else average_precision_score

    list_actual_anomalies, list_anomaly_scores, list_window = (
        _to_list(actual_anomalies),
        _to_list(anomaly_score),
        _to_list(window),
    )

    if len(list_actual_anomalies) == 1 and len(list_anomaly_scores) > 1:
        list_actual_anomalies = list_actual_anomalies * len(list_anomaly_scores)

    _assert_same_length(list_actual_anomalies, list_anomaly_scores)

    if len(list_window) == 1:
        list_window = list_window * len(actual_anomalies)
    else:
        raise_if_not(
            len(list_window) == len(list_actual_anomalies),
            "The list of windows must be the same length as the list of `anomaly_score` and"
            + " `actual_anomalies`. There must be one window value for each series."
            + f" Found length {len(list_window)}, expected {len(list_actual_anomalies)}.",
        )

    sol = []
    for idx, (s_anomalies, s_score) in enumerate(
        zip(list_actual_anomalies, list_anomaly_scores)
    ):

        _assert_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(
                s_anomalies, s_score, list_window[idx], metric_fn, metric
            )
        )

    if len(sol) == 1 and not isinstance(anomaly_score, Sequence):
        return sol[0]
    else:
        return sol


def eval_accuracy_from_binary_prediction(
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    binary_pred_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "recall",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Score the results against true anomalies.

    checks that `pred_anomalies` and `actual_anomalies` are the same:
        - type,
        - length,
        - number of components
        - binary and has values belonging to the two classes (1 and 0)

    If one series is given for `actual_anomalies` and `pred_anomalies` contains more than
    one series, the function will consider `actual_anomalies` as the true anomalies for
    all scores in `anomaly_score`.

    Parameters
    ----------
    actual_anomalies
        The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)
    binary_pred_anomalies
        Anomaly predictions.
    window
        Integer value indicating the number of past samples each point represents
        in the pred_anomalies. The parameter will be used to transform actual_anomalies.
        If a list is given. the length must match the number of series in pred_anomalies
        and actual_anomalies. If only one window is given, the value will be used for every
        series in pred_anomalies and actual_anomalies.
    metric
        Optionally, Scoring function to use. Must be one of "recall", "precision",
        "f1", and "accuracy".
        Default: "recall"

    Returns
    -------
    Union[float, Sequence[float], Sequence[Sequence[float]]]
        Score of the anomalies prediction

            * ``float`` if `binary_pred_anomalies` is a univariate series (dimension=1).
            * ``Sequence[float]``

                * if `binary_pred_anomalies` is a multivariate series (dimension>1),
                  returns one value per dimension.
                * if `binary_pred_anomalies` is a sequence of univariate series, returns one
                  value per series
            * ``Sequence[Sequence[float]]`` if `binary_pred_anomalies` is a sequence of
              multivariate series. Outer Sequence is over the sequence input, and the inner
              Sequence is over the dimensions of each element in the sequence input.
    """

    raise_if_not(
        metric in {"recall", "precision", "f1", "accuracy"},
        "Argument `metric` must be one of 'recall', 'precision', "
        "'f1' and 'accuracy'.",
    )

    if metric == "recall":
        metric_fn = recall_score
    elif metric == "precision":
        metric_fn = precision_score
    elif metric == "f1":
        metric_fn = f1_score
    else:
        metric_fn = accuracy_score

    list_actual_anomalies, list_binary_pred_anomalies, list_window = (
        _to_list(actual_anomalies),
        _to_list(binary_pred_anomalies),
        _to_list(window),
    )

    if len(list_actual_anomalies) == 1 and len(list_binary_pred_anomalies) > 1:
        list_actual_anomalies = list_actual_anomalies * len(list_binary_pred_anomalies)

    _assert_same_length(list_actual_anomalies, list_binary_pred_anomalies)

    if len(list_window) == 1:
        list_window = list_window * len(actual_anomalies)
    else:
        raise_if_not(
            len(list_window) == len(list_actual_anomalies),
            "The list of windows must be the same length as the list of `pred_anomalies` and"
            + " `actual_anomalies`. There must be one window value for each series."
            + f" Found length {len(list_window)}, expected {len(list_actual_anomalies)}.",
        )

    sol = []
    for idx, (s_anomalies, s_pred) in enumerate(
        zip(list_actual_anomalies, list_binary_pred_anomalies)
    ):

        _assert_binary(s_pred, "pred_anomalies")
        _assert_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(
                s_anomalies, s_pred, list_window[idx], metric_fn, metric
            )
        )

    if len(sol) == 1 and not isinstance(binary_pred_anomalies, Sequence):
        return sol[0]
    else:
        return sol


def _eval_accuracy_from_data(
    s_anomalies: TimeSeries,
    s_data: TimeSeries,
    window: int,
    metric_fn,
    metric_name: str,
) -> Union[float, Sequence[float]]:
    """Internal function for:
        - ``eval_accuracy_from_binary_prediction()``
        - ``eval_accuracy_from_scores()``

    Score the results against true anomalies.

    Parameters
    ----------
    actual_anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
    s_data
        series prediction
    window
        Integer value indicating the number of past samples each point represents
        in the anomaly_score. The parameter will be used by the function
        ``_window_adjustment_anomalies()`` to transform s_anomalies.
    metric_fn
        Function to use. Can be "average_precision_score", "roc_auc_score", "accuracy_score",
        "f1_score", "precision_score" and "recall_score".
    metric_name
        Name str of the function to use. Can be "AUC_PR", "AUC_ROC", "accuracy",
        "f1", "precision" and "recall".

    Returns
    -------
    Union[float, Sequence[float]]
        Score of the anomalies prediction
            - float -> if `s_data` is a univariate series (dimension=1).
            - Sequence[float] -> if `s_data` is a multivariate series (dimension>1),
            returns one value per dimension.
    """

    _assert_timeseries(s_data, "Prediction series input")
    _assert_timeseries(s_anomalies, "actual_anomalies input")

    # if window > 1, the anomalies will be adjusted so that it can be compared timewise with s_data
    s_anomalies = _max_pooling(s_anomalies, window)

    _sanity_check_two_series(s_data, s_anomalies)

    s_data, s_anomalies = _intersect(s_data, s_anomalies)

    if metric_name == "AUC_ROC" or metric_name == "AUC_PR":

        nr_anomalies_per_component = (
            s_anomalies.sum(axis=0).values(copy=False).flatten()
        )

        raise_if(
            nr_anomalies_per_component.min() == 0,
            f"`actual_anomalies` does not contain anomalies. {metric_name} cannot be computed.",
        )

        raise_if(
            nr_anomalies_per_component.max() == len(s_anomalies),
            f"`actual_anomalies` only contains anomalies. {metric_name} cannot be computed."
            + ["", f" Consider decreasing the window size (window={window})"][
                window > 1
            ],
        )

    # TODO: could we vectorize this?
    metrics = []
    for component_idx in range(s_data.width):
        metrics.append(
            metric_fn(
                s_anomalies.all_values(copy=False)[:, component_idx],
                s_data.all_values(copy=False)[:, component_idx],
            )
        )

    if len(metrics) == 1:
        return metrics[0]
    else:
        return metrics


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
    raise_if(
        len(new_series_1) == 0,
        "Time intersection between the two series must be non empty.",
    )

    return new_series_1, series_2.slice_intersect(series_1)


def _assert_timeseries(series: TimeSeries, message: str = None):
    """Checks if given input is of type Darts TimeSeries"""

    raise_if_not(
        isinstance(series, TimeSeries),
        "{} must be type darts.timeseries.TimeSeries and not {}.".format(
            message if message is not None else "Series input", type(series)
        ),
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
    raise_if_not(
        series_1.width == series_2.width,
        "Series must have the same number of components,"
        + f" found {series_1.width} and {series_2.width}.",
    )

    # check if the time intersection between the two inputs time series is not empty
    raise_if_not(
        len(series_1.time_index.intersection(series_2.time_index)) > 0,
        "Series must have a non-empty intersection timestamps.",
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

    raise_if_not(
        isinstance(window, int),
        f"Parameter `window` must be of type int, found {type(window)}.",
    )

    raise_if_not(
        window > 0,
        f"Parameter `window` must be stricly greater than 0, found size {window}.",
    )

    raise_if_not(
        window < len(series),
        "Parameter `window` must be smaller than the length of the input series, "
        + f" found window size {(window)}, and max size {len(series)}.",
    )

    if window == 1:
        # the process results in replacing every value by itself -> return directly the series
        return series
    else:
        return series.window_transform(
            transforms={
                "window": window,
                "function": "max",
                "mode": "rolling",
                "min_periods": window,
            },
            treat_na="dropna",
        )


def _to_list(series: Union[TimeSeries, Sequence[TimeSeries]]) -> Sequence[TimeSeries]:
    """If not already, it converts the input into a sequence

    Parameters
    ----------
    series
        single TimeSeries, or a sequence of TimeSeries

    Returns
    -------
    Sequence[TimeSeries]
    """

    return [series] if not isinstance(series, Sequence) else series


def _assert_same_length(
    list_series_1: Sequence[TimeSeries],
    list_series_2: Sequence[TimeSeries],
):
    """Checks if the two sequences contain the same number of TimeSeries."""

    raise_if_not(
        len(list_series_1) == len(list_series_2),
        "Sequences of series must be of the same length, found length:"
        + f" {len(list_series_1)} and {len(list_series_2)}.",
    )


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
    actual_anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
    title
        Title of the figure
    metric
        Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
        Default: "AUC_ROC"
    """

    raise_if_not(
        isinstance(series, TimeSeries),
        f"Input `series` must be of type TimeSeries, found {type(series)}.",
    )

    if title is None:
        if anomaly_scores is not None:
            title = "Anomaly results"
    else:
        raise_if_not(
            isinstance(title, str),
            f"Input `title` must be of type str, found {type(title)}.",
        )

    nbr_plots = 1

    if model_output is not None:
        raise_if_not(
            isinstance(model_output, TimeSeries),
            f"Input `model_output` must be of type TimeSeries, found {type(model_output)}.",
        )

    if actual_anomalies is not None:
        raise_if_not(
            isinstance(actual_anomalies, TimeSeries),
            f"Input `actual_anomalies` must be of type TimeSeries, found {type(actual_anomalies)}.",
        )

        nbr_plots = nbr_plots + 1
    else:
        raise_if_not(
            metric is None,
            "`actual_anomalies` must be given in order to calculate a metric.",
        )

    if anomaly_scores is not None:

        if isinstance(anomaly_scores, Sequence):
            for idx, s in enumerate(anomaly_scores):
                raise_if_not(
                    isinstance(s, TimeSeries),
                    f"Elements of anomaly_scores must be of type TimeSeries, found {type(s)} at index {idx}.",
                )
        else:
            raise_if_not(
                isinstance(anomaly_scores, TimeSeries),
                f"Input `anomaly_scores` must be of type TimeSeries or Sequence, found {type(actual_anomalies)}.",
            )
            anomaly_scores = [anomaly_scores]

        if names_of_scorers is not None:

            if isinstance(names_of_scorers, str):
                names_of_scorers = [names_of_scorers]
            elif isinstance(names_of_scorers, Sequence):
                for idx, name in enumerate(names_of_scorers):
                    raise_if_not(
                        isinstance(name, str),
                        f"Elements of names_of_scorers must be of type str, found {type(name)} at index {idx}.",
                    )
            else:
                raise ValueError(
                    f"Input `names_of_scorers` must be of type str or Sequence, found {type(names_of_scorers)}."
                )

            raise_if_not(
                len(names_of_scorers) == len(anomaly_scores),
                "The number of names in `names_of_scorers` must match the number of anomaly score "
                + f"given as input, found {len(names_of_scorers)} and expected {len(anomaly_scores)}.",
            )

        if isinstance(window, int):
            window = [window]
        elif isinstance(window, Sequence):
            for idx, w in enumerate(window):
                raise_if_not(
                    isinstance(w, int),
                    f"Every window must be of type int, found {type(w)} at index {idx}.",
                )
        else:
            raise ValueError(
                f"Input `window` must be of type int or Sequence, found {type(window)}."
            )

        raise_if_not(
            all([w > 0 for w in window]),
            "All windows must be positive integer.",
        )

        if len(window) == 1:
            window = window * len(anomaly_scores)
        else:
            raise_if_not(
                len(window) == len(anomaly_scores),
                "The number of window in `window` must match the number of anomaly score given as input. One "
                + f"window value for each series. Found length {len(window)}, and expected {len(anomaly_scores)}.",
            )

        raise_if_not(
            all([w < len(s) for (w, s) in zip(window, anomaly_scores)]),
            "All windows must be smaller than the length of their corresponding score.",
        )

        nbr_plots = nbr_plots + len(set(window))
    else:
        if window is not None:
            logger.warning(
                "The parameter `window` is given, but the input `anomaly_scores` is None."
            )

        if names_of_scorers is not None:
            logger.warning(
                "The parameter `names_of_scorers` is given, but the input `anomaly_scores` is None."
            )

        if metric is not None:
            logger.warning(
                "The parameter `metric` is given, but the input `anomaly_scores` is None."
            )

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

        current_window = window[0]
        index_ax = index_ax + 1

        for elem in sorted(dict_input.items(), key=lambda x: x[1]["window"]):

            idx = elem[1]["name_id"]
            w = elem[1]["window"]

            if w != current_window:
                current_window = w
                index_ax = index_ax + 1

            if metric is not None:
                value = round(
                    eval_accuracy_from_scores(
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
