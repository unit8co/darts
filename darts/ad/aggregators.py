"""
Aggregators
---------------
Module for aggregators. An aggregator combines multiple lists of anomalies into one.


TODO:
- add customize aggregators
- add trainable aggregators
    - log regression
    - decision tree
- show all combined

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
                f"Series in list must be deterministic (one value per timestamp per dimension), \
                found {series.n_samples} values for series at index {idx}.",
            )

            if idx == 0:
                series_width = series.width
                series_0 = series

            raise_if_not(
                series.width == series_width,
                f"Element of list needs to have the same dimension/width, \
                found width {series.width} and {series_width}.",
            )

            series_0, list_series[idx] = _intersect(series_0, series)

        for idx, series in enumerate(list_series):
            if idx > 0:
                list_series[idx] = series.slice_intersect(series_0)

        list_series[0] = series_0

        return list_series

    def _predict(self, list_series: Sequence[TimeSeries]) -> TimeSeries:

        list_series = self._check_input(list_series)

        list_pred = []
        for width in range(list_series[0].width):
            list_pred.append(
                self._predict_core(
                    np.concatenate(
                        [s.all_values()[:, width] for s in list_series], axis=1
                    )
                )
            )

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
            "f1", and "iou".
            Default: "recall"

        Returns
        -------
        Union[float, Sequence[float]]
            Score for the time series
        """

        series = self.predict(list_series)

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

        raise_if_not(
            len(list_series) == self.len_training_set,
            "The model was trained on a list of length {}, found for prediciton a list of different \
            length {}.".format(
                self.len_training_set, len(list_series)
            ),
        )

        return self._predict(list_series)

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

        actual_anomalies = actual_anomalies.slice_intersect(list_series[0])

        for idx, s in enumerate(list_series):
            list_series[idx] = s.slice_intersect(actual_anomalies)

        # TODO: for every width -> train different models for each
        width = 0

        training_data = np.concatenate(
            [s.all_values()[:, width] for s in list_series], axis=1
        )
        np_actual_anomalies = actual_anomalies.all_values(copy=False).flatten()

        self._fit_core(training_data, np_actual_anomalies)
        self._fit_called = True


class EnsembleSklearnAggregator(FittableAggregator):
    """Wrapper around Ensemble model of sklearn"""

    def __init__(self, model) -> None:
        self.model = model
        super().__init__()

    def __str__(self):
        return "EnsembleSklearnAggregator: {}".format(
            self.model.__str__().split("(")[0]
        )

    def _fit_core(
        self, np_series: np.ndarray, np_actual_anomalies: np.ndarray
    ) -> np.ndarray:
        self.model.fit(np_series, np_actual_anomalies)

    def _predict_core(self, np_series: np.ndarray) -> np.ndarray:
        return self.model.predict(np_series)


class OrAggregator(NonFittableAggregator):
    """Aggregator that identifies a time point as anomalous as long as it is
    included in one of the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "OrAggregator"

    def _predict_core(self, np_series: np.ndarray) -> np.ndarray:

        return [1 if timestamp.sum() >= 1 else 0 for timestamp in np_series]


class AndAggregator(NonFittableAggregator):
    """Aggregator that identifies a time point as anomalous only if it is
    included in all the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "AndAggregator"

    def _predict_core(self, np_series: np.ndarray) -> np.ndarray:

        return [0 if 0 in timestamp else 1 for timestamp in np_series]
