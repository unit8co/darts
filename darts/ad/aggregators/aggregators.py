"""
Aggregators
---------------
Module for aggregators. An aggregator combines multiple lists of anomalies into one.

TODO:
- add customize aggregators
- add in trainable aggregators
    - log regression
    - decision tree
- show all combined (info about correlation, and from what path did
the anomaly alarm comes from)

"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np

from darts import TimeSeries
from darts.ad.utils import _intersect, eval_accuracy_from_binary_prediction
from darts.logging import raise_if, raise_if_not


class Aggregator(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __str__(self):
        "returns the name of the aggregator"
        pass

    @abstractmethod
    def _predict_core(self):
        "returns the aggregated results"
        pass

    def _check_input(self, list_series: Sequence[TimeSeries]) -> TimeSeries:
        """
        Checks for input if:
            - it is a Sequence
            - it contains at least two elements
            - each element of input is
                - a Timeseries
                - binary (only values equal to 0 or 1)
                - 1 value per timestamp per dimension (num_samples equal to 1)
            - all elements needs to have the same width/dimension
        """

        raise_if_not(
            isinstance(list_series, Sequence),
            f"Input needs to be a Sequence, found type {type(list_series)}.",
        )

        raise_if(
            len(list_series) <= 1,
            f"Input list needs to contain at least two time series, found {len(list_series)}.",
        )

        for idx, series in enumerate(list_series):

            raise_if_not(
                isinstance(series, TimeSeries),
                f"Element of list needs to be Timeseries, found type {type(series)} for element at index {idx}.",
            )

            raise_if_not(
                np.array_equal(
                    series.values(copy=False),
                    series.values(copy=False).astype(bool),
                ),
                f"Series in list needs to be binary, series at index {idx} is not.",
            )

            raise_if_not(
                series.is_deterministic,
                "Series in list must be deterministic (one value per timestamp per dimension),"
                + f" found {series.n_samples} values for series at index {idx}.",
            )

            if idx == 0:
                series_width = series.width
                series_0 = series

            raise_if_not(
                series.width == series_width,
                "Element of list needs to have the same dimension/width,"
                + f" found width {series.width} and {series_width}.",
            )

            series_0, list_series[idx] = _intersect(series_0, series)

        for idx, series in enumerate(list_series):
            if idx > 0:
                list_series[idx] = series.slice_intersect(series_0)

            raise_if(
                len(list_series[idx]) == 0,
                f"Element {idx} of `list_series` must have a non empty intersection"
                + " with the other series of the sequence.",
            )

        list_series[0] = series_0

        return list_series

    def _predict(self, list_series: Sequence[TimeSeries]) -> TimeSeries:

        np_series = np.concatenate(
            [s.all_values(copy=False) for s in list_series], axis=2
        )

        list_pred = []
        for idx, width in enumerate(range(list_series[0].width)):
            list_pred.append(self._predict_core(np_series[:, width, :], idx))

        return TimeSeries.from_times_and_values(
            list_series[0]._time_index, list(zip(*list_pred))
        )

    def eval_accuracy(
        self,
        actual_anomalies: TimeSeries,
        list_series: Sequence[TimeSeries],
        window: int = 1,
        metric: str = "recall",
    ) -> Union[float, Sequence[float]]:
        """Aggregates the list of series given as input into one series and evaluates
        the results against true anomalies.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        list_series
            The list of binary series to aggregate
        window
            Integer value indicating the number of past samples each point represents
            in the list_series. The parameter will be used by the function
            ``_window_adjustment_anomalies()`` in darts.ad.utils to transform
            actual_anomalies.
        metric
            Metric function to use. Must be one of "recall", "precision",
            "f1", and "accuracy".
            Default: "recall"

        Returns
        -------
        Union[float, Sequence[float]]
            Score for the time series
        """

        raise_if_not(
            isinstance(actual_anomalies, TimeSeries),
            f"`actual_anomalies` must be of type TimeSeries, found type {type(actual_anomalies)}.",
        )

        series = self.predict(list_series)

        raise_if_not(
            actual_anomalies.width == series.width,
            "`actual_anomalies` must have the same width as the series in the sequence "
            + f"`list_series`, found width {actual_anomalies.width} and expected {series.width}.",
        )

        return eval_accuracy_from_binary_prediction(
            actual_anomalies, series, window, metric
        )


class NonFittableAggregator(Aggregator):
    "Base class of Aggregators that do not need training."

    def __init__(self) -> None:
        super().__init__()

        # indicates if the Aggregator is trainable or not
        self.trainable = False

    def predict(self, list_series: Sequence[TimeSeries]) -> TimeSeries:
        """Aggregates the list of series given as input into one series.

        Parameters
        ----------
        list_series
            The list of binary series to aggregate

        Returns
        -------
        TimeSeries
            Aggregated results
        """
        list_series = self._check_input(list_series)
        return self._predict(list_series)


class FittableAggregator(Aggregator):
    "Base class of Aggregators that do need training."

    def __init__(self) -> None:
        super().__init__()

        # indicates if the Aggregator is trainable or not
        self.trainable = True

        # indicates if the Aggregator has been trained yet
        self._fit_called = False

    def check_if_fit_called(self):
        """Checks if the Aggregator has been fitted before calling its `score()` function."""

        raise_if_not(
            self._fit_called,
            f"The Aggregator {self.__str__()} has not been fitted yet. Call `fit()` first.",
        )

    def fit(self, actual_anomalies: TimeSeries, list_series: Sequence[TimeSeries]):
        """Fit the aggregators on the given list of series.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        list_series
            The list of binary series to aggregate

        Returns
        -------
        TimeSeries
            Aggregated results
        """
        self.len_training_set = len(list_series)
        list_series = self._check_input(list_series)
        self.width_trained_on = list_series[0].width
        actual_anomalies = actual_anomalies.slice_intersect(list_series[0])

        raise_if_not(
            isinstance(actual_anomalies, TimeSeries),
            f"`actual_anomalies` must be of type TimeSeries, found type {type(actual_anomalies)}.",
        )

        raise_if_not(
            actual_anomalies.width == self.width_trained_on,
            "`actual_anomalies` must have the same width as the series in the sequence `list_series`,"
            + f" found width {actual_anomalies.width} and width {self.width_trained_on}.",
        )

        for idx, s in enumerate(list_series):
            list_series[idx] = s.slice_intersect(actual_anomalies)

        raise_if(
            len(list_series[0]) == 0,
            "`actual_anomalies` must have a non-empty time intersection with the series in the"
            + " sequence `list_series`.",
        )

        np_training_data = np.concatenate(
            [s.all_values(copy=False) for s in list_series], axis=2
        )
        np_actual_anomalies = actual_anomalies.all_values(copy=False)

        models = []
        for width in range(self.width_trained_on):
            self._fit_core(
                np_training_data[:, width, :], np_actual_anomalies[:, width].flatten()
            )
            models.append(self.model)

        self.models = models
        self._fit_called = True

    def predict(self, list_series: Sequence[TimeSeries]) -> TimeSeries:
        """Aggregates the list of series given as input into one series.

        Parameters
        ----------
        list_series
            The list of binary series to aggregate

        Returns
        -------
        TimeSeries
            Aggregated results
        """
        self.check_if_fit_called()

        list_series = self._check_input(list_series)

        raise_if_not(
            len(list_series) == self.len_training_set,
            f"The model was trained on a list of length {self.len_training_set}, and found for prediciton"
            + f" a list of different length {len(list_series)}.",
        )

        raise_if_not(
            all([s.width == self.width_trained_on for s in list_series]),
            "all series in `series` must have the same width as the data used for training the"
            + f" detector model, training width {self.width_trained_on}.",
        )

        return self._predict(list_series)
