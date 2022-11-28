"""
Utils
---------------

Common functions used by anomaly_model.py, scorers.py, aggregators.py and detectors.py

TODO:
    - change structure of eval_accuracy_from_scores and eval_accuracy_from_binary_prediction (a lot of repeated code)
    - migrate metrics function to darts.metric
    - check error message
    - clean function show_anomalies_from_scores
    - allow plots for probabilistic timeseries (for now we take the mean when plotting)
    - create a zoom option on anomalies for a show function
    - add an option visualize: "by window", "unique", "together"
"""

from typing import Sequence, Union

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
from darts.logging import raise_if, raise_if_not


def check_if_binary(series: TimeSeries, name_series: str):
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
    anomaly_score: Union[TimeSeries, Sequence[TimeSeries]],
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "AUC_ROC",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Scores the results against true anomalies.

    checks:
    - anomaly_score and actual_anomalies are the same type, length, width/dimension
    - actual_anomalies is binary and has values belonging to the two classes (1 and 0)

    Parameters
    ----------
    anomaly_score
        Time series to detect anomalies from.
    actual_anomalies
        The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
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
        Score of the anomaly time series
    """

    if metric == "AUC_ROC":
        metric_fn = roc_auc_score
    elif metric == "AUC_PR":
        metric_fn = average_precision_score
    else:
        raise ValueError("Argument `metric` must be one of 'AUC_ROC', 'AUC_PR'")

    list_anomaly_scores, list_actual_anomalies, list_window = (
        _to_list(anomaly_score),
        _to_list(actual_anomalies),
        _to_list(window),
    )
    _same_length(list_anomaly_scores, list_actual_anomalies)

    if len(list_window) == 1:
        list_window = list_window * len(actual_anomalies)
    else:
        raise_if_not(
            len(list_window) == len(list_actual_anomalies),
            f"List of windows must be of same length as list of anomaly_score and actual_anomalies. One window \
            value for each series. Found length {len(list_window)}, expected {len(list_actual_anomalies)}",
        )

    sol = []
    for idx, (s_score, s_anomalies) in enumerate(
        zip(list_anomaly_scores, list_actual_anomalies)
    ):

        check_if_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(
                s_score, s_anomalies, list_window[idx], metric_fn, metric
            )
        )

    if len(sol) == 1 and not isinstance(anomaly_score, Sequence):
        return sol[0]
    else:
        return sol


def eval_accuracy_from_binary_prediction(
    pred_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    window: Union[int, Sequence[int]] = 1,
    metric: str = "recall",
) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
    """Score the results against true anomalies.

    checks that pred_anomalies and actual_anomalies are the same:
    - type,
    - length,
    -width/dimension
    - binary and has values belonging to the two classes (1 and 0)

    Parameters
    ----------
    pred_anomalies
        Anomalies prediction.
    actual_anomalies
        True anomalies.
    window
        Integer value indicating the number of past samples each point represents
        in the pred_anomalies. The parameter will be used by the function
        ``_window_adjustment_anomalies()`` to transform actual_anomalies.
        If a list is given. the length must match the number of series in pred_anomalies
        and actual_anomalies. If only one window is given, the value will be used for every
        series in pred_anomalies and actual_anomalies.
    metric
        Optionally, Scoring function to use. Must be one of "recall", "precision",
        "f1", and "iou".
        Default: "recall"

    Returns
    -------
    Union[float, Sequence[float], Sequence[Sequence[float]]]
        Score of the anomalies prediction
    """

    if metric == "recall":
        metric_fn = recall_score
    elif metric == "precision":
        metric_fn = precision_score
    elif metric == "f1":
        metric_fn = f1_score
    elif metric == "accuracy":
        metric_fn = accuracy_score
    else:
        raise ValueError(
            "Argument `metric` must be one of 'recall', 'precision', "
            "'f1' and 'accuracy'."
        )

    list_pred_anomalies, list_actual_anomalies, list_window = (
        _to_list(pred_anomalies),
        _to_list(actual_anomalies),
        _to_list(window),
    )
    _same_length(list_pred_anomalies, list_actual_anomalies)

    if len(list_window) == 1:
        list_window = list_window * len(actual_anomalies)
    else:
        raise_if_not(
            len(list_window) == len(list_actual_anomalies),
            f"List of windows must be of same length as list of pred_anomalies and actual_anomalies. One window \
            value for each series. Found length {len(list_window)}, expected {len(list_actual_anomalies)}",
        )

    sol = []
    for idx, (s_pred, s_anomalies) in enumerate(
        zip(list_pred_anomalies, list_actual_anomalies)
    ):

        check_if_binary(s_pred, "pred_anomalies")
        check_if_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(
                s_pred, s_anomalies, list_window[idx], metric_fn, metric
            )
        )

    if len(sol) == 1 and not isinstance(pred_anomalies, Sequence):
        return sol[0]
    else:
        return sol


def _eval_accuracy_from_data(
    s_data: TimeSeries,
    s_anomalies: TimeSeries,
    window: int,
    metric_fn,
    metric_name: str,
) -> Union[float, Sequence[float]]:
    """Internal function for:
    - eval_accuracy_from_binary_prediction()
    - eval_accuracy_from_scores()

    Score the results against true anomalies.

    Parameters
    ----------
    s_data
        series prediction
    actual_anomalies
        True anomalies.
    window
        Integer value indicating the number of past samples each point represents
        in the anomaly_score. The parameter will be used by the function
        ``_window_adjustment_anomalies()`` to transform s_anomalies.
    metric_fn
        Function to use. Can be "average_precision_score", "roc_auc_score", "accuracy_score",
        "f1_score", "precision_score" and "recall_score".
    metric_name
        Function to use. Can be "AUC_PR", "AUC_ROC", "accuracy",
        "f1", "precision" and "recall".

    Returns
    -------
    Union[float, Sequence[float]]
        Score of the anomalies prediction
    """

    _check_timeseries_type(s_data, "Prediction series input")
    _check_timeseries_type(s_anomalies, "actual_anomalies input")

    # if window > 1, the anomalies will be adjusted so that it can be compared timewise with s_data
    s_anomalies = _window_adjustment_anomalies(s_anomalies, window)

    _sanity_check_2series(s_data, s_anomalies)

    s_data, s_anomalies = _intersect(s_data, s_anomalies)

    if metric_name == "AUC_ROC" or metric_name == "AUC_PR":

        raise_if(
            s_anomalies.sum(axis=0).values(copy=False).flatten().min() == 0,
            f"'actual_anomalies' does not contain anomalies. {metric_name} cannot be computed.",
        )

        raise_if(
            s_anomalies.sum(axis=0).values(copy=False).flatten().max()
            == len(s_anomalies),
            f"'actual_anomalies' contains only anomalies. {metric_name} cannot be computed."
            + ["", f" Think about decreasing the window size (window={window})"][
                window > 1
            ],
        )

    metrics = []
    for width in range(s_data.width):
        metrics.append(
            metric_fn(
                s_anomalies.all_values(copy=False)[:, width],
                s_data.all_values(copy=False)[:, width],
            )
        )

    if width == 0:
        return metrics[0]
    else:
        return metrics


def _intersect(
    series_1: TimeSeries,
    series_2: TimeSeries,
) -> tuple[TimeSeries, TimeSeries]:
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

    return series_1.slice_intersect(series_2), series_2.slice_intersect(series_1)


def _check_timeseries_type(series: TimeSeries, message: str = None):
    """Checks if given input is of type Darts TimeSeries"""

    raise_if_not(
        isinstance(series, TimeSeries),
        "{} must be type darts.timeseries.TimeSeries and not {}".format(
            message if message is not None else "Series input", type(series)
        ),
    )


def _sanity_check_2series(
    series_1: TimeSeries,
    series_2: TimeSeries,
):
    """Performs sanity check on the two given inputs

    Checks if the two inputs:
        - type is Darts Timeseries
        - have the same width/dimension
        - if their intersection in time is not null

    Parameters
    ----------
    series_1
        1st time series
    series_2:
        2nd time series
    """

    _check_timeseries_type(series_1)
    _check_timeseries_type(series_2)

    # check if the two inputs time series have the same width
    raise_if_not(
        series_1.width == series_2.width,
        f"Series must have the same width, found {series_1.width} and {series_2.width}",
    )

    # check if the time intersection between the two inputs time series is not empty
    raise_if_not(
        len(series_1._time_index.intersection(series_2._time_index)) > 0,
        "Series must have a non-empty intersection timestamps",
    )


def _window_adjustment_anomalies(series: TimeSeries, window: int) -> TimeSeries:
    """Slides a window of size window along the input series, and replaces the value of the
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
        isinstance(window, int), f"Window must be of type int, found {type(window)}"
    )

    raise_if_not(
        window > 0,
        f"window must be stricly greater than 0, found size {window}",
    )

    if window == 1:
        # the process results in replacing every value by itself -> return directly the series
        return series
    else:
        np_series = series.all_values(copy=False)

        values = [
            np_series[ind : ind + window].max(axis=0)
            for ind in range(len(np_series) - window + 1)
        ]

        return TimeSeries.from_times_and_values(
            series._time_index[window - 1 :], values
        )


def _to_list(series: Union[TimeSeries, Sequence[TimeSeries]]) -> Sequence[TimeSeries]:
    """If not already, it converts the input into a Sequence

    Parameters
    ----------
    series
        single TimeSeries, or a sequence of TimeSeries

    Returns
    -------
    Sequence[TimeSeries]
    """

    return [series] if not isinstance(series, Sequence) else series


def _same_length(
    list_series_1: Sequence[TimeSeries],
    list_series_2: Sequence[TimeSeries],
):
    """Checks if the two sequences contain the same number of TimeSeries."""

    raise_if_not(
        len(list_series_1) == len(list_series_2),
        f"Sequences of series must be of the same length, found length: \
        {len(list_series_1)} and {len(list_series_2)}",
    )


def show_anomalies_from_scores(
    series: TimeSeries,
    model_output: TimeSeries = None,
    anomaly_scores: Union[TimeSeries, Sequence[TimeSeries]] = None,
    window: Union[int, Sequence[int]] = 1,
    names_of_scorers: Union[str, Sequence[str]] = None,
    actual_anomalies: TimeSeries = None,
    title: str = None,
    save_png: str = None,
    metric: str = None,
):
    """Plot the results generated by an anomaly model.

    The plot will be composed of the following:
        - the series itself with the output of the model (if given)
        - the anomaly score of each scorer. The scorer with different windows will be separated.
        - the actual anomalies, if given.

    Possible to:
        - add a title to the figure with the parameter 'title'
        - give personalized names for the scorers with 'names_of_scorers'
        - save the plot as a png at the path 'save_png'
        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies is given

    Parameters
    ----------
    series
        The series to visualize anomalies from.
    model_output
        Output of the model given as input the series.
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
    save_png
        Path to where the plot in format png should be saved
        Default: None (the plot will not be saved)
    metric
        Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
        Default: "AUC_ROC"
    """

    raise_if_not(
        isinstance(series, TimeSeries),
        f"Input `series` must be of type TimeSeries, found {type(series)}.",
    )

    if title is None:
        title = "Anomaly results"
    else:
        raise_if_not(
            isinstance(title, str),
            f"Input `title` must be of type str, found {type(title)}.",
        )

    if save_png is not None:
        raise_if_not(
            isinstance(save_png, str),
            f"Input `save_png` must be of type str, found {type(save_png)}.",
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
            if isinstance(names_of_scorers, Sequence):
                for idx, name in enumerate(names_of_scorers):
                    raise_if_not(
                        isinstance(name, str),
                        f"Elements of names_of_scorers must be of type str, found {type(name)} at index {idx}.",
                    )
            else:
                raise_if_not(
                    isinstance(names_of_scorers, str),
                    f"Input `names_of_scorers` must be of type str or Sequence, found {type(names_of_scorers)}.",
                )

                names_of_scorers = [names_of_scorers]

            raise_if_not(
                len(names_of_scorers) == len(anomaly_scores),
                f"The number of names in `names_of_scorers` must match the number of anomaly score given as input, \
                found {len(names_of_scorers)} and expected {len(anomaly_scores)}.",
            )

        if isinstance(window, Sequence):
            for idx, w in enumerate(window):
                raise_if_not(
                    isinstance(w, int),
                    f"Every window must be of type int, found {type(w)} at index {idx}.",
                )
            list_window = window
        else:
            raise_if_not(
                isinstance(window, int),
                f"Input `window` must be of type int or Sequence, found {type(window)}.",
            )
            list_window = [window]

        if len(list_window) == 1:
            list_window = list_window * len(actual_anomalies)
        else:
            raise_if_not(
                len(list_window) == len(anomaly_scores),
                f"The number of window in `window` must match the number of anomaly score given as input. One window \
                value for each series. Found length {len(list_window)}, and expected {len(anomaly_scores)}.",
            )

        nbr_plots = nbr_plots + len(set(list_window))

    fig, axs = plt.subplots(
        nbr_plots,
        figsize=(8, 4 + 2 * (nbr_plots - 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [2] + [1] * (nbr_plots - 1)},
    )
    fig.suptitle(title, y=0.93)
    fig.subplots_adjust(hspace=0.3)

    index_ax = 0

    for width in range(series.width):
        series._xa[:, width].plot(
            ax=axs[index_ax], color="red", linewidth=0.5, label="true values series"
        )

    if model_output is not None:

        for width in range(model_output.width):
            model_output._xa[:, width].mean(axis=1).plot(
                ax=axs[index_ax], color="blue", linewidth=0.5, label="model output"
            )

    axs[index_ax].set_title("")

    if actual_anomalies is not None and anomaly_scores is not None:
        axs[index_ax].set_xlabel("")

    axs[index_ax].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

    if anomaly_scores is not None:

        dict_input = {}

        for idx, (score, w) in enumerate(zip(anomaly_scores, list_window)):

            dict_input[idx] = {"series_score": score, "window": w, "name_id": idx}

        current_window = list_window[0]
        index_ax = index_ax + 1

        for elem in sorted(dict_input.items(), key=lambda x: x[1]["window"]):

            idx = elem[1]["name_id"]
            window = elem[1]["window"]

            if window != current_window:
                current_window = window
                index_ax = index_ax + 1

            if metric is not None:
                value = round(
                    eval_accuracy_from_scores(
                        anomaly_score=anomaly_scores[idx],
                        actual_anomalies=actual_anomalies,
                        window=window,
                        metric=metric,
                    ),
                    2,
                )
            else:
                value = None

            if names_of_scorers is not None:
                label = names_of_scorers[idx] + [f" ({value})", ""][value is None]
            else:
                label = f"score_{str(idx)}" + [f" ({value})", ""][value is None]

            for width in range(elem[1]["series_score"].width):
                elem[1]["series_score"]._xa[:, width].plot(
                    ax=axs[index_ax], linewidth=0.5, label=label
                )

            axs[index_ax].legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2)
            axs[index_ax].set_title(f"Window: {str(window)}", loc="left")
            axs[index_ax].set_title("")
            axs[index_ax].set_xlabel("")

    if actual_anomalies is not None:

        for width in range(actual_anomalies.width):
            actual_anomalies._xa[:, width].plot(
                ax=axs[index_ax + 1], color="red", linewidth=1, label="true anomalies"
            )

        axs[index_ax + 1].set_title("")
        axs[index_ax + 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)

    if save_png is not None:
        plt.savefig(save_png)

    plt.show()
