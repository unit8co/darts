"""
Anomaly aggregators base classes
"""

# TODO:
# - add customize aggregators
# - add in trainable aggregators
#     - log regression
#     - decision tree
# - create show_all_combined (info about correlation, and from what path did
#   the anomaly alarm came from)

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np

from darts import TimeSeries
from darts.ad.utils import _to_list, eval_accuracy_from_binary_prediction
from darts.logging import raise_if_not


class Aggregator(ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __str__(self):
        """returns the name of the aggregator"""
        pass

    @abstractmethod
    def _predict_core(self):
        """returns the aggregated results"""
        pass

    @abstractmethod
    def predict(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Aggregates the (sequence of) multivariate binary series given as
        input into a (sequence of) univariate binary series.

        Parameters
        ----------
        series
            The (sequence of) multivariate binary series to aggregate

        Returns
        -------
        TimeSeries
            (Sequence of) aggregated results
        """
        pass

    def _check_input(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        """
        Checks for input if:
            - it is a (sequence of) multivariate series (width>1)
            - (sequence of) series must be:
                * a deterministic TimeSeries
                * binary (only values equal to 0 or 1)
        """

        list_series = _to_list(series)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all series in `series` must be of type TimeSeries.",
        )

        raise_if_not(
            all([s.width > 1 for s in list_series]),
            "all series in `series` must be multivariate (width>1).",
        )

        raise_if_not(
            all([s.is_deterministic for s in list_series]),
            "all series in `series` must be deterministic (number of samples=1).",
        )

        raise_if_not(
            all(
                [
                    np.array_equal(
                        s.values(copy=False), s.values(copy=False).astype(bool)
                    )
                    for s in list_series
                ]
            ),
            "all series in `series` must be binary (only 0 and 1 values).",
        )

        return list_series

    def eval_accuracy(
        self,
        actual_anomalies: Sequence[TimeSeries],
        series: Sequence[TimeSeries],
        window: int = 1,
        metric: str = "recall",
    ) -> Union[float, Sequence[float]]:
        """Aggregates the (sequence of) multivariate series given as input into one (sequence of)
        series and evaluates the results against true anomalies.

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            The (sequence of) multivariate binary series to aggregate
        window
            (Sequence of) integer value indicating the number of past samples each point
            represents in the (sequence of) series. The parameter will be used by the
            function ``_window_adjustment_anomalies()`` in darts.ad.utils to transform
            actual_anomalies.
        metric
            Metric function to use. Must be one of "recall", "precision",
            "f1", and "accuracy".
            Default: "recall"

        Returns
        -------
        Union[float, Sequence[float]]
            (Sequence of) score for the (sequence of) series
        """

        list_actual_anomalies = _to_list(actual_anomalies)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be of type TimeSeries.",
        )

        raise_if_not(
            all([s.is_deterministic for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be deterministic (number of samples=1).",
        )

        raise_if_not(
            all([s.width == 1 for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be univariate (width=1).",
        )

        raise_if_not(
            len(list_actual_anomalies) == len(_to_list(series)),
            "`actual_anomalies` and `series` must contain the same number of series.",
        )

        preds = self.predict(series)

        return eval_accuracy_from_binary_prediction(
            list_actual_anomalies, preds, window, metric
        )


class NonFittableAggregator(Aggregator):
    "Base class of Aggregators that do not need training."

    def __init__(self) -> None:
        super().__init__()

        # indicates if the Aggregator is trainable or not
        self.trainable = False

    def predict(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Aggregates the (sequence of) multivariate binary series given as
        input into a (sequence of) univariate binary series.

        Parameters
        ----------
        series
            The (sequence of) multivariate binary series to aggregate

        Returns
        -------
        TimeSeries
            (Sequence of) aggregated results
        """
        list_series = self._check_input(series)

        if isinstance(series, TimeSeries):
            return self._predict_core(list_series)[0]
        else:
            return self._predict_core(list_series)


class FittableAggregator(Aggregator):
    "Base class of Aggregators that do need training."

    def __init__(self) -> None:
        super().__init__()

        # indicates if the Aggregator is trainable or not
        self.trainable = True

        # indicates if the Aggregator has been trained yet
        self._fit_called = False

    def _assert_fit_called(self):
        """Checks if the Aggregator has been fitted before calling its `score()` function."""

        raise_if_not(
            self._fit_called,
            f"The Aggregator {self.__str__()} has not been fitted yet. Call `fit()` first.",
        )

    def fit(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        """Fit the aggregators on the (sequence of) multivariate binary series.

        If a list of series is given, they must have the same number of components.

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            The (sequence of) multivariate binary series
        """
        list_series = self._check_input(series)
        self.width_trained_on = list_series[0].width

        raise_if_not(
            all([s.width == self.width_trained_on for s in list_series]),
            "all series in `list_series` must have the same number of components.",
        )

        list_actual_anomalies = _to_list(actual_anomalies)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be of type TimeSeries.",
        )

        raise_if_not(
            all([s.is_deterministic for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be deterministic (width=1).",
        )

        raise_if_not(
            all([s.width == 1 for s in list_actual_anomalies]),
            "all series in `actual_anomalies` must be univariate (width=1).",
        )

        raise_if_not(
            len(list_actual_anomalies) == len(list_series),
            "`actual_anomalies` and `series` must contain the same number of series.",
        )

        same_intersection = list(
            zip(
                *[
                    [anomalies.slice_intersect(series), series.slice_intersect(series)]
                    for (anomalies, series) in zip(list_actual_anomalies, list_series)
                ]
            )
        )
        list_actual_anomalies = list(same_intersection[0])
        list_series = list(same_intersection[1])

        ret = self._fit_core(list_actual_anomalies, list_series)
        self._fit_called = True
        return ret

    def predict(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Aggregates the (sequence of) multivariate binary series given as
        input into a (sequence of) univariate binary series.

        Parameters
        ----------
        series
            The (sequence of) multivariate binary series to aggregate

        Returns
        -------
        TimeSeries
            (Sequence of) aggregated results
        """
        self._assert_fit_called()
        list_series = self._check_input(series)

        raise_if_not(
            all([s.width == self.width_trained_on for s in list_series]),
            "all series in `series` must have the same number of components as the data"
            + " used for training the detector model, number of components in training:"
            + f" {self.width_trained_on}.",
        )

        if isinstance(series, TimeSeries):
            return self._predict_core(list_series)[0]
        else:
            return self._predict_core(list_series)
