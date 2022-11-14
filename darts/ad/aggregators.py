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
from darts.ad.utils import _return_intersect, eval_accuracy_from_prediction
from darts.logging import raise_if, raise_if_not


class _Aggregator(ABC):
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

    def predict(self, list_series: Sequence[TimeSeries]) -> TimeSeries:
        """Aggregates the list of series given as input into one series.

        Checks:
        - input is a Sequence
        - input contains at least two elements
        - each element is
            - a Timeseries
            - binary (only values equal to 0 or 1)
            - 1 value per timestamp per dimension (num_samples equal to 1)
        - all elements needs to have the same width/dimension

        Parameters
        ----------
        list_series
            The list of binary series to aggregate

        Returns
        -------
        TimeSeries
            Aggregated results
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
                series.n_samples == 1,
                f"Series in list needs to have one value per timestamp per dimension, \
                found {series.n_samples} for series at index {idx}.",
            )

            if idx == 0:
                series_width = series.width
                series_0 = series

            raise_if_not(
                series.width == series_width,
                f"Element of list needs to have the same dimension/width, \
                found width {series.width} and {series_width}.",
            )

            series_0, list_series[idx] = _return_intersect(series_0, series)

        for idx, series in enumerate(list_series):
            if idx > 0:
                list_series[idx] = series.slice_intersect(series_0)

        list_series[0] = series_0

        list_pred = []
        for width in range(series_width):
            list_pred.append(
                self._predict_core(
                    np.concatenate(
                        [s.all_values()[:, width] for s in list_series], axis=1
                    )
                )
            )

        return TimeSeries.from_times_and_values(
            series_0._time_index, list(zip(*list_pred))
        )

    def eval_accuracy(
        self,
        list_series: Sequence[TimeSeries],
        actual_anomalies: TimeSeries,
        window: int = 1,
        metric: str = "recall",
    ) -> Union[float, Sequence[float]]:
        """Aggregates the list of series given as input into one series and evaluates
        the results against true anomalies.

        Parameters
        ----------
        list_series
            The list of binary series to aggregate
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
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

        return eval_accuracy_from_prediction(series, actual_anomalies, window, metric)


class OrAggregator(_Aggregator):
    """Aggregator that identifies a time point as anomalous as long as it is
    included in one of the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "OrAggregator"

    def _predict_core(self, np_series: np.ndarray) -> np.ndarray:

        return [1 if timestamp.sum() >= 1 else 0 for timestamp in np_series]


class AndAggregator(_Aggregator):
    """Aggregator that identifies a time point as anomalous only if it is
    included in all the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "AndAggregator"

    def _predict_core(self, np_series: np.ndarray) -> np.ndarray:

        return [0 if 0 in timestamp else 1 for timestamp in np_series]
