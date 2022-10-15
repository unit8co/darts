"""
Scorer
-------

Scorers can be trainable (TrainableScorer) or not trainable (NonTrainableScorer). The main functions are `fit()`
(only for the trainable scorer), `compute()` and `score()`.

`fit()` learns the function `f()`, over the history of one time series. The function `compute()` takes as input
two time series, and applies the function `f()` to obtain an anomaly score time series. The function `score()`
returns the score of an agnostic threshold metric (AUC-ROC or AUC-PR), between an anomaly score time series and a
binary ground truth time series indicating the presence of anomalies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from darts import TimeSeries
from darts.logging import raise_if, raise_if_not


class Scorer(ABC):
    "Base class for all scorers"

    def __init__(self, characteristic_length: Optional[int] = None) -> None:

        if characteristic_length is None:
            characteristic_length = 0

        self.characteristic_length = characteristic_length

    @abstractmethod
    def compute(self, input_1: Any, input_2: Any) -> Any:
        pass

    @abstractmethod
    def _compute_core(self, input_1: Any, input_2: Any) -> Any:
        pass

    def window_adjustment_anomalies(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Slides a window of size self.characteristic_length along the input series, and replaces the value of the
        input time series by the maximum of the values contained in the window (past self.characteristic_length and
        itself points).

        The binary time series output represents if there is an anomaly (=1) or not (=0) in the past
        self.characteristic_length points + itself. The new series will equal the length of the input series
        - self.characteristic_length. Its first point will start at the first time index of the input time series +
        self.characteristic_length points.

        Parameters
        ----------
        series: Binary Darts TimeSeries

        Returns
        -------
        Binary Darts TimeSeries
        """

        if self.characteristic_length == 0:
            return series
        else:
            values = [
                series[ind : ind + self.characteristic_length + 1]
                .max(axis=0)
                .all_values()
                .flatten()[0]
                for (ind, _) in enumerate(
                    series[self.characteristic_length :].pd_series()
                )
            ]
            return TimeSeries.from_times_and_values(
                series._time_index[self.characteristic_length :], values
            )

    def window_adjustment_series(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Slides a window of size self.characteristic_length along the input series, and replaces the value of
        the input time series by the mean of the values contained in the window (past self.characteristic_length
        and itself points).

        Parameters
        ----------
        series: Darts TimeSeries

        Returns
        -------
        Darts TimeSeries
        """

        if self.characteristic_length == 0:
            return series
        else:
            values = [
                series[ind : ind + self.characteristic_length + 1]
                .mean(axis=0)
                .all_values()
                .flatten()[0]
                for (ind, _) in enumerate(
                    series[self.characteristic_length :].pd_series()
                )
            ]
            return TimeSeries.from_times_and_values(
                series._time_index[self.characteristic_length :], values
            )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float]]:
        """Scores the results against true anomalies.

        Parameters
        ----------
        series: Darts TimeSeries
            Time series to detect anomalies from.
        actual_anomalies: Darts TimeSeries
            True anomalies.
        metric: str,
            The selected metric to use. Can be 'AUC_ROC' (default value) or 'AUC_PR'

        Returns
        -------
        float
            Score of the time series
        """

        if metric == "AUC_ROC":
            scoring_fn = roc_auc_score
        elif metric == "AUC_PR":
            scoring_fn = average_precision_score
        else:
            raise ValueError("Argument `metric` must be one of 'AUC_ROC', 'AUC_PR'")

        self.sanity_check(series, actual_anomalies)

        series, actual_anomalies = self.return_intersect(series, actual_anomalies)

        return scoring_fn(
            y_true=actual_anomalies.all_values().flatten(),
            y_score=series.all_values().flatten(),
        )

    def return_intersect(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> tuple:
        """Returns the values of series_1 and the values of series_2 that share the same time index.
        (Intersection in time of the two time series)

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries

        Returns
        -------
        tuple of Darts TimeSeries
        """

        return series_1.slice_intersect(series_2), series_2.slice_intersect(series_1)

    def compute_score(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
        scoring: str = "AUC_ROC",
    ) -> Union[float, Sequence[float]]:
        """Computes the anomaly score between the two given time series, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        actual_anomalies: Binary Darts TimeSeries
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional
        scoring: str, optional
            Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        float
            Score for the time series
        """
        anomaly_score = self.compute(series_1, series_2)

        return self.score(
            anomaly_score, self.window_adjustment_anomalies(actual_anomalies), scoring
        )

    def sanity_check(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
    ):
        """Performs sanity check on the given inputs

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional
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

            # check if the time intersection between the two inputs time series is not empty
            raise_if_not(
                len(series_1._time_index.intersection(series_2._time_index)) > 0,
                "Series must have a non-empty intersection timestamps",
            )

            # check if the two inputs time series have the same width
            raise_if_not(
                series_1.width == series_2.width, "Series must have the same width"
            )


class NonTrainableScorer(Scorer):
    "Base class of scorers that do not need training."

    def __init__(self, characteristic_length) -> None:
        super().__init__(characteristic_length=characteristic_length)
        self.trainable = False

    def compute(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score between the two given time series.

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional

        Returns
        -------
        Darts TimeSeries
            Anomaly score time series
        """
        self.sanity_check(series_1, series_2)

        series_1, series_2 = self.return_intersect(series_1, series_2)

        return self.window_adjustment_series(self._compute_core(series_1, series_2))


class TrainableScorer(Scorer):
    "Base class of scorers that do need training."

    def __init__(self, characteristic_length) -> None:
        super().__init__(characteristic_length=characteristic_length)
        self._fit_called = False
        self.trainable = True

    def compute(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score between the two given time series.

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional

        Returns
        -------
        Darts TimeSeries
            Anomaly score time series
        """
        raise_if_not(
            self._fit_called,
            "The Scorer has not been fitted yet. Call `fit()` first",
        )

        self.sanity_check(series_1, series_2)

        if series_2 is None:
            series = series_1
        else:
            series_1, series_2 = self.return_intersect(series_1, series_2)
            series = self._diff(series_1, series_2)

        return self.window_adjustment_series(self._compute_core(series))

    @abstractmethod
    def _fit_core(self, input: Any) -> Any:
        pass

    def fit(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
    ) -> Union[float, Sequence[float]]:
        """Fits the scorer on the given time series input.

        If 2 time series are given, a distance function, given as input, will be applied
        to transform the 2 time series into 1.

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional

        Returns
        -------
        self
            Fitted Scorer.
        """

        self.sanity_check(series_1, series_2)

        if series_2 is None:
            series = series_1
        else:
            series_1, series_2 = self.return_intersect(series_1, series_2)
            series = self._diff(series_1, series_2)

        self._fit_core(series)

    def _diff(self, series_1, series_2):
        """Applies the distance_function to the two time series

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries

        Returns
        -------
        Darts TimeSeries
            Output of the distance_function given the two time series
        """

        if self.distance_function == "l1":
            return (series_1 - series_2.slice_intersect(series_1)).map(
                lambda x: np.abs(x)
            )
        elif self.distance_function == "l2":
            return (series_1 - series_2.slice_intersect(series_1)) ** 2
        elif self.distance_function == "diff":
            return series_1 - series_2.slice_intersect(series_1)
        else:
            return series_1

    def _check_norm(self):
        """Checks if the given distance_function is known"""

        accepted_norms = ["l1", "l2", "diff"]

        raise_if_not(
            self.distance_function in accepted_norms,
            "Metric should be 'l1', 'l2' or 'diff'",
        )


class GaussianMixtureScorer(TrainableScorer):
    """GaussianMixtureScorer anomaly score"""

    def __init__(
        self,
        characteristic_length: Optional[int] = None,
        n_components: int = 1,
        distance_function="l1",
    ) -> None:
        super().__init__(characteristic_length=characteristic_length)
        self.n_components = n_components
        self.distance_function = distance_function
        super()._check_norm()
        self.model = GaussianMixture(n_components=n_components)

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):

        self._fit_called = True
        self.model.fit(series.all_values().flatten().reshape(-1, 1))

    def _compute_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        np_anomaly_score = np.exp(
            self.model.score_samples(series.all_values().flatten().reshape(-1, 1))
        )
        return TimeSeries.from_times_and_values(series._time_index, np_anomaly_score)


class KmeansScorer(TrainableScorer):
    """Kmean anomaly score"""

    def __init__(
        self,
        characteristic_length: Optional[int] = None,
        k: Union[int, list[int]] = 2,
        distance_function="l1",
    ) -> None:
        super().__init__(characteristic_length=characteristic_length)
        self.k = k
        self.distance_function = distance_function
        self._check_norm()
        self.model = KMeans(n_clusters=k)

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):

        self._fit_called = True
        self.model.fit(series.all_values().flatten().reshape(-1, 1))

    def _compute_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid
        np_anomaly_score = self.model.transform(
            series.all_values().flatten().reshape(-1, 1)
        ).min(axis=1)
        return TimeSeries.from_times_and_values(series._time_index, np_anomaly_score)


class WasserteinScorer(TrainableScorer):
    """WasserteinScorer anomaly score"""

    def __init__(
        self, characteristic_length: Optional[int] = None, distance_function="l1"
    ) -> None:
        if characteristic_length is None:
            characteristic_length = 10
        super().__init__(characteristic_length=characteristic_length)
        self.distance_function = distance_function
        super()._check_norm()

        raise_if(
            self.characteristic_length == 0,
            "characteristic_length must be stricly higher than 0,"
            "(preferably higher than 10 as it is the number of samples of the test distribution)",
        )

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):

        self._fit_called = True
        self.training_data = series.all_values().flatten()

    def _compute_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid
        distance = []

        for i in range(len(series) - self.characteristic_length + 1):
            distance.append(
                wasserstein_distance(
                    self.training_data,
                    series[i : i + self.characteristic_length + 1]
                    .all_values()
                    .flatten(),
                )
            )

        return TimeSeries.from_times_and_values(
            series._time_index[self.characteristic_length - 1 :], distance
        )

    def window_adjustment_series(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        return series


class LocalOutlierFactorScorer(TrainableScorer):
    """LocalOutlierFactor anomaly score"""

    def __init__(
        self,
        characteristic_length: Optional[int] = None,
        n_neighbors: int = 2,
        distance_function="l1",
    ) -> None:
        super().__init__(characteristic_length=characteristic_length)
        self.n_neighbors = n_neighbors
        self.distance_function = distance_function
        super()._check_norm()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        self._fit_called = True
        self.model.fit(series.all_values().flatten().reshape(-1, 1))

    def _compute_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid
        np_anomaly_score = np.abs(
            self.model.score_samples(series.all_values().flatten().reshape(-1, 1))
        )
        return TimeSeries.from_times_and_values(series._time_index, np_anomaly_score)


class L2(NonTrainableScorer):
    """L2 distance metric"""

    def __init__(self, characteristic_length: Optional[int] = None) -> None:
        super().__init__(characteristic_length=characteristic_length)

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2) ** 2


class L1(NonTrainableScorer):
    """L1 distance metric"""

    def __init__(self, characteristic_length: Optional[int] = None) -> None:
        super().__init__(characteristic_length=characteristic_length)

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2).map(lambda x: np.abs(x))


class difference(NonTrainableScorer):
    """difference distance metric"""

    def __init__(self, characteristic_length: Optional[int] = None) -> None:
        super().__init__(characteristic_length=characteristic_length)

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return series_1 - series_2
