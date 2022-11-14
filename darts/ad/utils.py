"""
Utils
---------------

Common functions used by anomaly_model.py, scorers.py, aggregators.py and detectors.py

"""

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
    window: int = 1,
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

    list_anomaly_scores, list_actual_anomalies = _convert_to_list(
        anomaly_score, actual_anomalies
    )

    sol = []
    for s_score, s_anomalies in zip(list_anomaly_scores, list_actual_anomalies):

        check_if_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(s_score, s_anomalies, window, metric_fn, metric)
        )

    if len(sol) == 1 and not isinstance(anomaly_score, Sequence):
        return sol[0]
    else:
        return sol


def eval_accuracy_from_prediction(
    pred_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
    window: int = 1,
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

    list_pred_anomalies, list_actual_anomalies = _convert_to_list(
        pred_anomalies, actual_anomalies
    )

    sol = []
    for s_pred, s_anomalies in zip(list_pred_anomalies, list_actual_anomalies):

        check_if_binary(s_pred, "pred_anomalies")
        check_if_binary(s_anomalies, "actual_anomalies")

        sol.append(
            _eval_accuracy_from_data(s_pred, s_anomalies, window, metric_fn, metric)
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
    - eval_accuracy_from_prediction()
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

    # if window > 1, the anomalies will be adjusted so that it can be compared timewise with s_data
    s_anomalies = _window_adjustment_anomalies(s_anomalies, window)

    _sanity_check(s_data, s_anomalies)
    s_data, s_anomalies = _return_intersect(s_data, s_anomalies)

    if metric_name == "AUC_ROC" or metric_name == "AUC_PR":

        raise_if(
            s_anomalies.sum(axis=0).values(copy=False).flatten().min() == 0,
            f"'actual_anomalies' does not contain anomalies. {metric_name} cannot be computed.",
        )

        raise_if(
            s_anomalies.sum(axis=0).values(copy=False).flatten().max()
            == len(s_anomalies),
            f"'actual_anomalies' contains only anomalies. {metric_name} cannot be computed."
            + ["", f" Think about reducing the window (window={window})"][window > 1],
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


def _return_intersect(
    series_1: TimeSeries,
    series_2: TimeSeries,
) -> tuple[TimeSeries, TimeSeries]:
    """Returns the values of series_1 and the values of series_2 that share the same time index.
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


def _sanity_check(
    series_1: TimeSeries,
    series_2: TimeSeries = None,
):
    """Performs sanity check on the given inputs

    Checks if the two inputs:
    - are 'Darts TimeSeries'
    - have the same width/dimension
    - if their intersection in time is not null

    Parameters
    ----------
    series_1
        1st time series
    series_2:
        Optionally, 2nd time series
    """

    # check if type input is a Darts TimeSeries
    raise_if_not(
        isinstance(series_1, TimeSeries),
        f"Series input must be type darts.timeseries.TimeSeries and not {type(series_1)}",
    )

    if series_2 is not None:

        # check if type input is a Darts TimeSeries
        raise_if_not(
            isinstance(series_2, TimeSeries),
            f"Series input must be type darts.timeseries.TimeSeries and not {type(series_2)}",
        )

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

        _sanity_check(series)

        np_series = series.all_values(copy=False)

        values = [
            np_series[ind : ind + window].max(axis=0)
            for ind in range(len(np_series) - window + 1)
        ]

        return TimeSeries.from_times_and_values(
            series._time_index[window - 1 :], values
        )


def _convert_to_list(
    series_1: Union[TimeSeries, Sequence[TimeSeries]],
    series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
    """If not already, it converts the inputs into a Sequence. Additionaly, it checks if the two sequences
    contain the same number TimeSeries.

    Parameters
    ----------
    series_1
        1st time series
    series_2
        Optionally, 2nd time series

    Returns
    -------
    Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]
    """

    series_1 = [series_1] if not isinstance(series_1, Sequence) else series_1

    if series_2 is not None:

        series_2 = [series_2] if not isinstance(series_2, Sequence) else series_2

        raise_if_not(
            len(series_1) == len(series_2),
            f"Sequences of series must be of the same length, found length: \
            {len(series_1)} and {len(series_2)}",
        )

    return series_1, series_2


def show(
    self,
    series: TimeSeries,
    model_output: TimeSeries = None,
    anomaly_scorers: TimeSeries = None,
    anomalies: TimeSeries = None,
    save_png=None,
    show_ROC_AUC=False,
):

    if anomalies is None:
        nbr_plots = len(self.scorer_dict)
    else:
        nbr_plots = len(self.scorer_dict) + 1

    fig, axs = plt.subplots(
        nbr_plots + 1,
        figsize=(8, 4 + 2 * nbr_plots),
        sharex=True,
        gridspec_kw={"height_ratios": [2] + [1] * nbr_plots},
    )
    fig.suptitle(f"Results for {self.model.__class__.__name__}", y=0.94)
    fig.subplots_adjust(hspace=0.3)

    series.pd_series().plot(
        ax=axs[0], label="true values series", color="red", linewidth=0.5
    )
    if model_output is not None:
        model_output.pd_series().plot(
            ax=axs[0], label="model output", color="blue", linewidth=0.5
        )
    axs[0].set_title("")
    axs[0].set_xlabel("")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)

    for index_ax, key in enumerate(self.scorer_dict):
        for index in self.scorer_dict[key]:
            if show_ROC_AUC is True:
                label = (
                    self.label_scorer[index]
                    + " ("
                    + str(
                        round(
                            self.scorers[index].score(
                                series=anomaly_scorers[index],
                                actual_anomalies=self.scorers[
                                    index
                                ]._window_adjustment_anomalies(anomalies),
                            ),
                            2,
                        )
                    )
                    + ")"
                )
            else:
                label = self.label_scorer[index]
            anomaly_scorers[index].pd_series().plot(
                ax=axs[index_ax + 1], label=label, linewidth=0.5
            )
        axs[index_ax + 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2)
        axs[index_ax + 1].set_title(f"Window: {key}", loc="left")

    if anomalies is not None:
        anomalies.pd_series().plot(
            ax=axs[len(self.scorer_dict) + 1],
            label="true anomalies",
            color="red",
            linewidth=1,
        )
        axs[len(self.scorer_dict) + 1].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2
        )

    if save_png is not None:
        plt.savefig(save_png)

    plt.show()
