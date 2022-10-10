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
from typing import Any, Dict, Optional, Sequence, Union

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def _compute_core(self, input_1: Any, input_2: Any) -> Any:
        pass

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        scoring: str = "AUC_ROC",
    ) -> Union[float, Sequence[float]]:
        """Scores the results against true anomalies.

        Parameters
        ----------
        series: Darts TimeSeries
            Time series to detect anomalies from.
        actual_anomalies: Darts TimeSeries
            True anomalies.
        scoring: str, optional
            Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        float
            Score of the time series
        """

        if scoring == "AUC_ROC":
            scoring_fn = roc_auc_score
        elif scoring == "AUC_PR":
            scoring_fn = average_precision_score
        else:
            raise ValueError("Argument `scoring` must be one of 'AUC_ROC', 'AUC_PR'")

        series = series.slice_intersect(actual_anomalies)
        actual_anomalies = actual_anomalies.slice_intersect(series)

        return scoring_fn(
            y_true=actual_anomalies.all_values().flatten(),
            y_score=series.all_values().flatten(),
        )

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
        # check inputs (type, size, values)
        # check if same timestamp!

        if self.trainable:

            raise_if_not(
                self._fit_called,
                "The Scorer has not been fitted yet. Call `fit()` first",
            )

            if series_2 == None:
                series = series_1
            else:
                series = self._diff(series_1, series_2)

            return self._compute_core(series)

        else:
            raise_if(
                series_2 == None,
                "The Scorer expects two time series input. Only one was given.",
            )
            return self._compute_core(series_1, series_2)

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
        # check inputs (type, size, values) -> of anomaly_score
        # check if same timestamp -> of anomaly_score

        anomaly_score = self.compute(series_1, series_2)

        return self.score(anomaly_score, actual_anomalies, scoring)


class NonTrainableScorer(Scorer):
    "Base class of scorers that do not need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = False


class TrainableScorer(Scorer):
    "Base class of scorers that do need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False
        self.trainable = True

    @abstractmethod
    def _fit_core(self, input: Any) -> Any:
        pass

    def fit(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
        scorer_fit_params: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Sequence[float]]:
        """Fits the scorer on the given time series input.

        If 2 time series are given, a distance function, given as input, will be applied
        to transform the 2 time series into 1.

        Parameters
        ----------
        series_1: Darts TimeSeries
        series_2: Darts TimeSeries, optional
        scorer_fit_params: dict, optional
            parameters of the scorer model

        Returns
        -------
        self
            Fitted Scorer.
        """

        # check inputs (type, size, values)
        # check if same timestamp!

        if scorer_fit_params == None:
            scorer_fit_params = {}

        if series_2 == None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        self._fit_core(series, scorer_fit_params)

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
            "Metric should be l1, l2 or diff",
        )


class GaussianMixtureScorer(TrainableScorer):
    """GaussianMixtureScorer anomaly score"""

    def __init__(self, n_components: int = 1, distance_function="l1") -> None:
        super().__init__()
        self.n_components = n_components
        self.distance_function = distance_function
        super()._check_norm()
        self.model = GaussianMixture(n_components=n_components)

    def _fit_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], scorer_fit_params
    ):

        raise_if_not(
            scorer_fit_params == {},
            ".fit() of GaussianMixtureScorer has no parameters",
        )

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

    def __init__(self, k: Union[int, list[int]] = 2, distance_function="l1") -> None:
        super().__init__()
        self.k = k
        self.distance_function = distance_function
        self._check_norm()
        self.model = KMeans(n_clusters=k)

    def _fit_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], scorer_fit_params
    ):

        self._fit_called = True
        self.model.fit(
            series.all_values().flatten().reshape(-1, 1), **scorer_fit_params
        )

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

    def __init__(self, window: int = 10, distance_function="l1") -> None:
        super().__init__()
        self.distance_function = distance_function
        super()._check_norm()
        self.window = window

    def _fit_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], scorer_fit_params
    ):

        raise_if_not(
            scorer_fit_params == {},
            ".fit() of WasserteinScorer has no parameters",
        )

        self._fit_called = True
        self.training_data = series.all_values().flatten()

    def _compute_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid
        distance = []

        for i in range(len(series) - self.window + 1):
            distance.append(
                wasserstein_distance(
                    self.training_data,
                    series[i : i + self.window].all_values().flatten(),
                )
            )

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], distance
        )


class LocalOutlierFactorScorer(TrainableScorer):
    """LocalOutlierFactor anomaly score"""

    def __init__(self, n_neighbors: int = 2, distance_function="l1") -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.distance_function = distance_function
        super()._check_norm()
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def _fit_core(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], scorer_fit_params
    ):
        self._fit_called = True
        self.model.fit(
            series.all_values().flatten().reshape(-1, 1), **scorer_fit_params
        )

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

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2) ** 2


class L1(NonTrainableScorer):
    """L1 distance metric"""

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2).map(lambda x: np.abs(x))


class difference(NonTrainableScorer):
    """difference distance metric"""

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return series_1 - series_2
