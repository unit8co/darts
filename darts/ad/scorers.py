"""
Anomaly Scorers
---------------

Anomaly scorers can be trainable (FittableAnomalyScorer) or not trainable (NonFittableAnomalyScorer).
The main functions are ``fit()`` (only for the trainable anomaly scorer), ``score()`` and ``eval_accuracy()``.

``fit()`` learns the function ``f()``, over the history of one time series. The function ``score()`` takes as input
two time series, and applies the function ``f()`` to obtain an anomaly score time series.
The function ``eval_accuracy()`` returns the score of an agnostic threshold metric (AUC-ROC or AUC-PR), between an
anomaly score time series and a binary ground truth time series indicating the presence of anomalies.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import gamma, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from darts import TimeSeries
from darts.ad.utils import (
    _convert_to_list,
    _return_intersect,
    _sanity_check,
    eval_accuracy_from_scores,
)
from darts.logging import raise_if_not


class AnomalyScorer(ABC):
    "Base class for all anomaly scorers"

    def __init__(self, window: Optional[int] = None) -> None:

        if window is None:
            window = 1

        raise_if_not(
            window > 0,
            f"window must be stricly greater than 0, found size {window}",
        )

        self.window = window

    def _expects_probabilistic(self) -> bool:
        """Checks if the scorer expects a probabilistic predictions.
        By default, returns False. Needs to be overwritten by scorers that do expects
        probabilistic predictions.
        """
        return False

    def _check_probabilistic_case(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """Checks if the scorer is probabilistic and the corresponding rules regarding the two time series inputs are
        respected.

        For probabilistic scorers, the input must be a probabilistic time series corresponding to the output of a
        probabilistic forecasting model. The second input must be a deterministic time series corresponding to the
        actual value of the forecasted time series.

        If the scorer is not probabilistic, both inputs must be deterministic (num_samples==1).

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

        if self._expects_probabilistic():
            raise_if_not(
                series_1.n_samples > 1,
                f"Scorer {self.__str__()} is expecting a probabilitic timeseries as its first input \
                (number of samples must be higher than 1).",
            )

        else:
            if series_1.n_samples > 1:
                # TODO: output a warning "The scorer expects a non probabilitic input
                # (num of samples needs to be equal to 1)"
                # median along each time stamp is computed to reduce the number of samples to 1
                series_1 = series_1.median(axis=2)

        if series_2 is not None:
            # TODO: create a warning rather than an error, and avg on axis 2
            raise_if_not(
                series_2.n_samples == 1,
                f"Scorer {self.__str__()} is expecting a deterministic timeseries as its second input \
            (number of samples must be equal to 1, found: {series_2.n_samples}).",
            )

        return series_1, series_2

    @abstractmethod
    def __str__(self):
        "returns the name of the scorer"
        pass

    @abstractmethod
    def score(self, input_1: Any, input_2: Any) -> Any:
        pass

    @abstractmethod
    def _score_core(self, input_1: Any, input_2: Any) -> Any:
        pass

    def eval_accuracy(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Computes the anomaly score between the two given time series, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series_1
            1st time series
        series_2
            Optionally, 2nd time series
        metric
            Optionally, metric function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score for the time series
        """
        anomaly_score = self.score(series_1, series_2)

        return eval_accuracy_from_scores(
            anomaly_score, actual_anomalies, self.window, metric
        )


class NonFittableAnomalyScorer(AnomalyScorer):
    "Base class of anomaly scorers that do not need training."

    def __init__(self, window) -> None:
        super().__init__(window=window)

        # indicates if the scorer is trainable or not
        self.trainable = False

    def score(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score between the two given time series.

        Parameters
        ----------
        series_1
            1st time series
        series_2:
            2nd time series

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series
        """
        list_series_1, list_series_2 = _convert_to_list(series_1, series_2)

        anomaly_scores = []

        for s1, s2 in zip(list_series_1, list_series_2):

            _sanity_check(s1, s2)
            s1, s2 = _return_intersect(s1, s2)
            s1, s2 = self._check_probabilistic_case(s1, s2)

            anomaly_scores.append(self._score_core(s1, s2))

        if (
            len(anomaly_scores) == 1
            and not isinstance(series_1, Sequence)
            and not isinstance(series_2, Sequence)
        ):
            return anomaly_scores[0]
        else:
            return anomaly_scores


class FittableAnomalyScorer(AnomalyScorer):
    "Base class of scorers that do need training."

    def __init__(self, window, reduced_function) -> None:
        super().__init__(window=window)

        # indicates if the scorer is trainable or not
        self.trainable = True

        # indicates if the scorer has been trained yet
        self._fit_called = False

        # function used in ._diff() to convert 2 time series into 1
        if reduced_function is None:
            reduced_function = "abs_diff"
        self.reduced_function = reduced_function

    def score(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the given series.

        If 2 time series are given, a reduced function, given as a parameter in the __init__ method
        (reduced_function), will be applied to transform the 2 time series into 1. Default: "abs_diff"

        Parameters
        ----------
        series_1
            1st time series
        series_2:
            Optionally, 2nd time series

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series
        """

        raise_if_not(
            self._fit_called,
            "The Scorer has not been fitted yet. Call `fit()` first",
        )

        list_series_1, list_series_2 = _convert_to_list(series_1, series_2)

        anomaly_scores = []

        if series_2 is None:
            for series in list_series_1:
                _sanity_check(series)
                series, _ = self._check_probabilistic_case(series)
                anomaly_scores.append(self._score_core(series, None))
        else:
            for (s1, s2) in zip(list_series_1, list_series_2):
                _sanity_check(s1, s2)
                s1, s2 = self._check_probabilistic_case(s1, s2)
                anomaly_scores.append(self._score_core(s1, s2))

        if len(anomaly_scores) == 1 and not isinstance(series_1, Sequence):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def _check_window_size(self, series: TimeSeries):
        """Checks if the parameter window is less or equal to the length of the given series"""

        raise_if_not(
            self.window <= len(series),
            f"Window size {self.window} is greater than the targeted series length {len(series)}, \
            must be lower or equal.",
        )

    @abstractmethod
    def _fit_core(self, input: Any) -> Any:
        pass

    def fit(
        self,
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
    ):
        """Fits the scorer on the given time series input.

        If 2 time series are given, a reduced function, given as a parameter in the __init__ method
        (reduced_function), will be applied to transform the 2 time series into 1. Default: "abs_diff"

        Parameters
        ----------
        series_1
            1st time series
        series_2:
            Optionally, 2nd time series

        Returns
        -------
        self
            Fitted Scorer.
        """
        list_series_1, list_series_2 = _convert_to_list(series_1, series_2)

        if series_2 is None:
            for series in list_series_1:
                _sanity_check(series)
        else:
            for (s1, s2) in zip(list_series_1, list_series_2):
                _sanity_check(s1, s2)

        self._fit_core(list_series_1, list_series_2)

    def _diff_sequence(
        self, list_series_1: Sequence[TimeSeries], list_series_2: Sequence[TimeSeries]
    ) -> Sequence[TimeSeries]:
        """Calls the function _diff() on every pair (s1,s2) in the list (list_series_1,list_series_2).

        list_series_1 and list_series_2 must have the same length n.
        Each pair of series in list_series_1 and list_series_2 must be of the same length L and width/dimension W.

        Parameters
        ----------
        series_1
            1st sequence of time series
        series_2:
            2nd sequence of time series

        Returns
        -------
        TimeSeries
            Sequence of series of length n
        """
        list_series = []

        for (series1, series2) in zip(list_series_1, list_series_2):
            list_series.append(self._diff(series1, series2))

        return list_series

    def _diff(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        """Applies the reduced_function to the two time series. Converts two time series into 1.

        series_1 and series_2 must be of the same length L and width/dimension W.

        Parameters
        ----------
        series_1
            1st time series
        series_2:
            2nd time series

        Returns
        -------
        TimeSeries
            series of length L and width/dimension W.
        """
        series_1, series_2 = _return_intersect(series_1, series_2)

        if self.reduced_function == "abs_diff":
            return (series_1 - series_2).map(lambda x: np.abs(x))
        elif self.reduced_function == "diff":
            return series_1 - series_2
        else:
            # found an non-existent reduced_function
            raise ValueError(
                f"Metric should be 'diff' or 'abs_diff', found {self.reduced_function}"
            )


# FittableAnomalyScorer


class GaussianMixtureScorer(FittableAnomalyScorer):
    """GaussianMixtureScorer anomaly score

    Wrapped around the GaussianMixture scikit-learn function.
    Source code: <https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/mixture/_gaussian_mixture.py#L456>
    """

    def __init__(
        self,
        window: Optional[int] = None,
        n_components: int = 1,
        reduced_function=None,
        component_wise: bool = False,
    ) -> None:

        """
        A Gaussian mixture model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will compute the log-likelihood of each sample.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option to change
        these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        component_wise is a boolean parameter in the __init__ method indicating how the model should behave with input
        that is a multivariate series. If set to True, the model will treat each width/dimension of the series
        independently. If the series has a width of d, the model will train and store the d Gaussian mixture models
        and fit them on each dimension. If set to False, the model will concatenate the widths in the considered window
        and compute the score using only one trained Gaussian mixture model.

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned
        into equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter.
        If the series is of length n, width d and the window is set to w, the training phase will generate (n-w+1)/w
        data samples of length d * w. If a list of series is given of length l, each series will be partitioned into
        subsequences, and the results will be concatenated into an array of length l * number of subsequences.

        The Gaussian mixture model will be fitted on the generated subsequences. The model will estimate its parameters
        with the EM algorithm.

        If component_wise is set to True, the algorithm will be applied to each width independently. For each width,
        a Gaussian mixture model will be trained.

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the GaussianMixture model.

        - If the series is multivariate of width w:
            - if component_wise is set to False: it will return a univariate series (width=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if component_wise is set to True: it will return a multivariate series of width w. Each dimension
            represents the anomaly score of the corresponding dimension of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        component_wise.

        A window of size w is rolled on the series with
        a stride equal to 1. It is the same window used during the training phase. At each timestamp, the previous w
        values will be used to form a vector of size w * width of the series. The ``score_samples()`` of the Gaussian
        mixture model will be called, with the vector as parameter. It will return the log-likelihood of the vector,
        and the exponential function is applied. The output will be a series of width 1 and length n-w+1, with n being
        the length of the input series. Each value will represent how anomalous the sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

        If component_wise is set to True, the algorithm will be applied to each width independently. The log-likelihood
        of the window in width w will be computed by the model trained on the corresponding width w during the training.

        Parameters
        ----------
        window
            Size of the window used to create the subsequences of the series.
        n_components
            Optionally, the number of mixture components of the GaussianMixture model.
        reduced_function
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the GaussianMixtureScorer to apply GaussianMixture model on the original series or on its
            residuals (difference between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        super().__init__(window=window, reduced_function=reduced_function)
        self.n_components = n_components

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

    def __str__(self):
        return f"GaussianMixtureScorer (n_components={self.n_components}, window={self.window}, \
        reduced_function={self.reduced_function})"

    def _fit_core(
        self,
        list_series_1: Sequence[TimeSeries],
        list_series_2: Sequence[TimeSeries] = None,
    ):

        if list_series_2 is None:
            list_series = list_series_1
        else:
            list_series = self._diff_sequence(list_series_1, list_series_2)

        self._fit_called = True
        self.width_trained_on = list_series[0].width

        if not self.component_wise:
            self.model = GaussianMixture(n_components=self.n_components)
            self.model.fit(
                np.concatenate(
                    [
                        s.all_values(copy=False).reshape(-1, self.window * s.width)
                        for s in list_series
                    ]
                )
            )
        else:
            models = []
            for width in range(self.width_trained_on):
                model = GaussianMixture(n_components=self.n_components)
                model.fit(
                    np.concatenate(
                        [
                            s.all_values(copy=False)[:, width].reshape(-1, self.window)
                            for s in list_series
                        ]
                    )
                )
                models.append(model)
            self.model = models

    def _score_core(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> TimeSeries:

        if series_2 is None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        raise_if_not(
            self.width_trained_on == series.width,
            f"Input must have the same width of the data used for training the GaussianMixture model, \
            found width: {self.width_trained_on} and {series.width}",
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:

            np_anomaly_score.append(
                np.exp(
                    self.model.score_samples(
                        np.array(
                            [
                                np.array(np_series[i : i + self.window])
                                for i in range(len(series) - self.window + 1)
                            ]
                        ).reshape(-1, self.window * series.width)
                    )
                )
            )
        else:

            for width in range(self.width_trained_on):
                np_anomaly_score_width = self.model[width].score_samples(
                    np.array(
                        [
                            np.array(np_series[i : i + self.window, width])
                            for i in range(len(series) - self.window + 1)
                        ]
                    ).reshape(-1, self.window)
                )

                np_anomaly_score.append(np.exp(np_anomaly_score_width))

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], list(zip(*np_anomaly_score))
        )


class KMeansScorer(FittableAnomalyScorer):
    """KMeans anomaly score

    Wrapped around the KMeans scikit-learn function.
    Source code: <https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/cluster/_kmeans.py#L1126>.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        k: Union[int, list[int]] = 2,
        reduced_function=None,
        component_wise: bool = False,
    ) -> None:
        """
        A KMeans model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the minimal distance between the centroid and the sample.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option to change
        these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        component_wise is a boolean parameter in the __init__ method indicating how the model should behave with input
        that is a multivariate series. If set to True, the model will treat each width/dimension of the series
        independently. If the series has a width of d, the model will train and store the d KMeans models and fit them
        on each dimension. If set to False, the model will concatenate the widths in the considered window and compute
        the score using only one trained KMeans model.

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned into
        equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter. If the series is of length n, width d and the window is set
        to w, the training phase will generate (n-w+1)/w data samples of length d*w. If a list of series is given of
        length l, each series will be partitioned into subsequences, and the results will be concatenated into an array
        of length l * number of subsequences.

        The model KMeans will be fitted on the generated subsequences. The model will find k cluster of dimensions
        equal to the length of the subsequences (d*w).

        If component_wise is set to True, the algorithm will be applied to each width independently. For each width,
        a KMeans model will be trained.

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the KMeans model.

        - If the series is multivariate of width w:
            - if component_wise is set to False: it will return a univariate series (width=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if component_wise is set to True: it will return a multivariate series of width w. Each dimension
            represents the anomaly score of the corresponding dimension of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        component_wise.

        A window of size w is rolled on the series with a stride equal to 1. It is the same window used during
        the training phase. At each timestamp, the previous w values will be used to form a vector of size w * width
        of the series. The KMeans model will then retrieve the closest centroid to this vector, and compute the
        euclidean distance between the centroid and the vector. The output will be a series of width 1 and length
        n-w+1, with n being the length of the input series. Each value will represent how anomalous the sample of
        the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

        If component_wise is set to True, the algorithm will be applied to each width independently. The distance will
        be computed between the vector and the closest centroid found by the model trained on the corresponding width
        during the training.

        Parameters
        ----------
        window
            Size of the window used to create the subsequences of the series.
        k
            The number of clusters to form as well as the number of centroids to generate by the KMeans model
        reduced_function
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the KMeansScorer to apply KMeans on the original series or on its residuals (difference between
            the prediction and the original series). Must be one of "abs_diff" and "diff" (defined in ``_diff()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        super().__init__(window=window, reduced_function=reduced_function)

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

        self.k = k

    def __str__(self):
        return f"KMeansScorer (k={self.k}, window={self.window}, reduced_function={self.reduced_function})"

    def _fit_core(
        self,
        list_series_1: Sequence[TimeSeries],
        list_series_2: Sequence[TimeSeries] = None,
    ):

        if list_series_2 is None:
            list_series = list_series_1
        else:
            list_series = self._diff_sequence(list_series_1, list_series_2)

        self._fit_called = True
        self.width_trained_on = list_series[0].width

        if not self.component_wise:
            self.model = KMeans(n_clusters=self.k)
            self.model.fit(
                np.concatenate(
                    [
                        s.all_values(copy=False).reshape(-1, self.window * s.width)
                        for s in list_series
                    ]
                )
            )
        else:
            models = []
            for width in range(self.width_trained_on):
                model = KMeans(n_clusters=self.k)
                model.fit(
                    np.concatenate(
                        [
                            s.all_values(copy=False)[:, width].reshape(-1, self.window)
                            for s in list_series
                        ]
                    )
                )
                models.append(model)
            self.model = models

    def _score_core(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> TimeSeries:

        if series_2 is None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        raise_if_not(
            self.width_trained_on == series.width,
            f"Input must have the same width of the data used for training the KMeans model, \
            found width: {self.width_trained_on} and {series.width}",
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:

            # return distance to the clostest centroid
            np_anomaly_score.append(
                self.model.transform(
                    np.array(
                        [
                            np.array(np_series[i : i + self.window])
                            for i in range(len(series) - self.window + 1)
                        ]
                    ).reshape(-1, self.window * series.width)
                ).min(axis=1)
            )  # only return the clostest distance out of the k ones (k centroids)
        else:

            for width in range(self.width_trained_on):
                np_anomaly_score_width = (
                    self.model[width]
                    .transform(
                        np.array(
                            [
                                np.array(np_series[i : i + self.window, width])
                                for i in range(len(series) - self.window + 1)
                            ]
                        ).reshape(-1, self.window)
                    )
                    .min(axis=1)
                )

                np_anomaly_score.append(np_anomaly_score_width)

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], list(zip(*np_anomaly_score))
        )


class LocalOutlierFactorScorer(FittableAnomalyScorer):
    """LocalOutlierFactor anomaly score

    Wrapped around the LocalOutlierFactor scikit-learn function.
    Source code: <https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/neighbors/_lof.py#L19>.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        n_neighbors: int = 2,
        reduced_function=None,
        component_wise: bool = False,
    ) -> None:
        """
        A Local outlier factor model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the local deviation of the density of a given sample with respect to its
        neighbors. It is local in that the deviation depends on how isolated the object is with respect to the
        surrounding neighborhood. More precisely, locality is given by k-nearest neighbors, whose distance is used to
        estimate the local density. By comparing the local density of a sample to the local densities of its neighbors,
        one can identify samples that have a substantially lower density than their neighbors.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option
        to change these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        component_wise is a boolean parameter in the __init__ method indicating how the model should behave with input
        that is a multivariate series. If set to True, the model will treat each width/dimension of the series
        independently. If the series has a width of d, the model will train and store the d LocalOutlierFactor models
        and fit them on each dimension. If set to False, the model will concatenate the widths in the considered window
        and compute the score using only one trained LocalOutlierFactor model.

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned into
        equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter.
        If the series is of length n, width d and the window is set to w, the training phase will generate (n-w+1)/w
        data samples of length d*w. If a list of series is given of length l, each series will be partitioned into
        subsequences, and the results will be concatenated into an array of length l * number of subsequences.

        The model LocalOutlierFactor will be fitted on the generated subsequences.

        If component_wise is set to True, the algorithm will be applied to each width independently. For each width,
        a LocalOutlierFactor model will be trained.

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the LocalOutlierFactor model.

        - If the series is multivariate of width w:
            - if component_wise is set to False: it will return a univariate series (width=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if component_wise is set to True: it will return a multivariate series of width w. Each dimension
            represents the anomaly score of the corresponding dimension of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        component_wise.

        A window of size w is rolled on the series with a stride equal to 1. It is the same window used during
        the training phase. At each timestamp, the previous w values will be used to form a vector of size w * width
        of the series. The Local outlier factor model will then return the local deviation of the density of the given
        vector with respect to its neighbors (constituted in the training phase). The output will be a series of width
        1 and length n-w+1, with n being the length of the input series. Each value will represent how anomalous the
        sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

        If component_wise is set to True, the algorithm will be applied to each width independently. For each width,
        the anomaly score will be computed by the model trained on the corresponding width during the training.

        Parameters
        ----------
        window
            Size of the window used to create the subsequences of the series.
        n_neighbors
            Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of
            samples provided, all samples will be used.
        reduced_function
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the LocalOutlierFactorScorer to apply LocalOutlierFactor model on the original series or on
            its residuals (difference between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        super().__init__(window=window, reduced_function=reduced_function)
        self.n_neighbors = n_neighbors

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

    def __str__(self):
        return f"LocalOutlierFactor (n_neighbors= {self.n_neighbors}, window={self.window}, \
            reduced_function={self.reduced_function})"

    def _fit_core(
        self,
        list_series_1: Sequence[TimeSeries],
        list_series_2: Sequence[TimeSeries] = None,
    ):

        if list_series_2 is None:
            list_series = list_series_1
        else:
            list_series = self._diff_sequence(list_series_1, list_series_2)

        self._fit_called = True
        self.width_trained_on = list_series[0].width

        if not self.component_wise:
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
            self.model.fit(
                np.concatenate(
                    [
                        s.all_values(copy=False).reshape(-1, self.window * s.width)
                        for s in list_series
                    ]
                )
            )
        else:
            models = []
            for width in range(self.width_trained_on):
                model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
                model.fit(
                    np.concatenate(
                        [
                            s.all_values(copy=False)[:, width].reshape(-1, self.window)
                            for s in list_series
                        ]
                    )
                )
                models.append(model)
            self.model = models

    def _score_core(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> TimeSeries:

        if series_2 is None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        raise_if_not(
            self.width_trained_on == series.width,
            f"Input must have the same width of the data used for training the Kmeans model, \
                found width: {self.width_trained_on} and {series.width}",
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:

            np_anomaly_score.append(
                self.model.score_samples(
                    np.array(
                        [
                            np.array(np_series[i : i + self.window])
                            for i in range(len(series) - self.window + 1)
                        ]
                    ).reshape(-1, self.window * series.width)
                )
            )
        else:

            for width in range(self.width_trained_on):
                np_anomaly_score_width = self.model[width].score_samples(
                    np.array(
                        [
                            np.array(np_series[i : i + self.window, width])
                            for i in range(len(series) - self.window + 1)
                        ]
                    ).reshape(-1, self.window)
                )

                np_anomaly_score.append(np_anomaly_score_width)

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], list(zip(*np_anomaly_score))
        )


class WassersteinScorer(FittableAnomalyScorer):
    """WassersteinScorer anomaly score

    Wrapped around the Wasserstein scipy.stats functon.
    Source code: <https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L8675-L8749>.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        reduced_function=None,
        component_wise: bool = False,
    ) -> None:
        """
        A Wasserstein model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the Wasserstein distance bewteen the training distribution
        and the window sample distribution. Both distributions are 1D.

        TODO:
        - understand better the math behind the Wasserstein distance (ex: when the test distribution contains
        only one sample)
        - check if there is an equivalent Wasserstein distance for d-D distributions (currently only accepts 1D)

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        component_wise is a boolean parameter in the __init__ method indicating how the model should behave with input
        that is a multivariate series. If set to True, the model will treat each width/dimension of the series
        independently. If set to False, the model will concatenate the widths in the considered window and compute
        the score.

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The element of a list will be
        concatenated to form one continuous array (by definition, each element have the same width/dimensions).

        If the series is of length n and width d, the array will be of length n*d. If component_wise is True, each
        width d is treated independently and the data is stored in a list of size d.
        Each element is an array of length n.

        If a list of series is given of length l, each series will be reduced to an array, and the l arrays will then
        be concatenated to form a continuous array of length l*d*n. If component_wise is True, the data is stored in a
        list of size d. Each element is an array of length l*n.

        The array will be kept in memory, representing the training data distribution.
        In practice, the series or list of series would represent residuals than can be considered independent
        and identically distributed (iid).

        Compute score:

        The input is a series (univariate or multivariate) or a list of series.

        - If the series is multivariate of width w:
            - if component_wise is set to False: it will return a univariate series (width=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if component_wise is set to True: it will return a multivariate series of width w. Each dimension
            represents the anomaly score of the corresponding dimension of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        component_wise.

        A window of size w (given as a parameter named window) is rolled on the series with a stride equal to 1.
        At each timestamp, the previous w values will be used to form a vector of size w * width of the series.
        The Wasserstein distance will be computed between this vector and the train distribution.
        The function will return a float number indicating how different these two distributions are.
        The output will be a series of width 1 and length n-w+1, with n being the length of the input series.
        Each value will represent how anomalous the sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()``
        will be applied independently on each series.

        If component_wise is set to True, the algorithm will be applied to each width independently,
        and be compared to their corresponding training data samples computed in the ``fit()`` method.

        Parameters
        ----------
        window
            Size of the sliding window that represents the number of samples in the testing distribution to compare
            with the training distribution in the Wasserstein function
        reduced_function
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the WassersteinScorer to compute the Wasserstein distance on the original series or on its
            residuals (difference between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        if window is None:
            window = 10
        super().__init__(window=window, reduced_function=reduced_function)

        raise_if_not(
            self.window > 0,
            "window must be stricly higher than 0,"
            "(preferably higher than 10 as it is the number of samples of the test distribution)",
        )

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

    def __str__(self):
        return f"WassersteinScorer (window={self.window}, reduced_function={self.reduced_function})"

    def _fit_core(
        self,
        list_series_1: Sequence[TimeSeries],
        list_series_2: Sequence[TimeSeries] = None,
    ):

        if list_series_2 is None:
            list_series = list_series_1
        else:
            list_series = self._diff_sequence(list_series_1, list_series_2)

        self._fit_called = True
        self.width_trained_on = list_series[0].width

        if self.component_wise and self.width_trained_on > 1:

            concatenated_data = np.concatenate(
                [s.all_values(copy=False) for s in list_series]
            )

            training_data = []
            for width in range(self.width_trained_on):
                training_data.append(concatenated_data[:, width].flatten())

            self.training_data = training_data

        else:
            self.training_data = [
                np.concatenate(
                    [s.all_values(copy=False).flatten() for s in list_series]
                )
            ]

    def _score_core(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> TimeSeries:

        if series_2 is None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        raise_if_not(
            self.width_trained_on == series.width,
            f"Input must have the same width of the data used for training the Wassertein model, \
                found width: {self.width_trained_on} and {series.width}",
        )

        distance = []

        np_series = series.all_values(copy=False)

        for i in range(len(series) - self.window + 1):

            temp_test = np_series[i : i + self.window + 1]

            if self.component_wise:
                width_result = []
                for width in range(self.width_trained_on):
                    width_result.append(
                        wasserstein_distance(
                            self.training_data[width], temp_test[width].flatten()
                        )
                    )

                distance.append(width_result)

            else:
                distance.append(
                    wasserstein_distance(
                        self.training_data[0],
                        temp_test.flatten(),
                    )
                )

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], distance
        )


# NonFittableAnomalyScorer


class Difference(NonFittableAnomalyScorer):
    """Difference distance metric

    Returns the difference between each timestamps of two series.

    If the two series are multivariate, it will return a multivariate series.
    """

    def __init__(self) -> None:
        super().__init__(window=None)

    def __str__(self):
        return "Difference"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        return series_1 - series_2


class Norm(NonFittableAnomalyScorer):
    """Norm anomaly score

    Wrapped around the linalg.norm numpy function.
    Source code: <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>.
    """

    def __init__(self, ord=None, component_wise: bool = False) -> None:
        """
        Returns the norm of a given order for each timestamp of the two input series’ differences.

        If component_wise is set to False, each timestamp of the difference will be considered as a vector
        and its norm will be computed.

        If component_wise is set to True, it will return the absolute value for each element of the difference,
        regardless of the norm's order.

        The ``compute()`` method accepts as input two series:

        - If the two series are multivariate of width w:
            - if component_wise is set to False: it will return a univariate series (width=1).
            - if component_wise is set to True: it will return a multivariate series of width w

        - If the two series are univariate, it will return a univariate series regardless of the parameter
        component_wise.

        Parameters
        ----------
        ord
            Order of the norm. Options are listed under 'Notes' at:
            <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>.
            Default: None
        component_wise
            Boolean value indicating if the norm needs to be computed element-wise (True)
            or component-wise (False) equivalent to axis=1.
            Default: False
        """

        self.ord = ord
        self.component_wise = component_wise
        super().__init__(window=None)

    def __str__(self):
        return f"Norm (ord={self.ord}, component_wise={self.component_wise})"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:

        diff = series_1 - series_2

        if self.component_wise:
            return diff.map(lambda x: np.abs(x))

        else:
            diff_np = diff.all_values(copy=False)

            return TimeSeries.from_times_and_values(
                diff._time_index, np.linalg.norm(diff_np, ord=self.ord, axis=1)
            )


class NLLScorer(NonFittableAnomalyScorer):
    """Parent class for all LikelihoodScorer"""

    def __init__(self, window) -> None:
        super().__init__(window=window)

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        """For each timestamp of the inputs:
            - the parameters of the considered distribution are fitted on the samples of the probabilistic time series
            - the negative log-likelihood of the determinisitc time series values are computed

        If the series are multivariate, the score will be computed on each width independently.

        Parameters
        ----------
        series_1
            A probabilistic time series (number of samples per timestamp must be higher than 1)
        series_2:
            A determinisict time series (number of samples per timestamp must be equal to 1)

        Returns
        -------
        TimeSeries
        """
        np_series_1 = series_1.all_values(copy=False)
        np_series_2 = series_2.all_values(copy=False)

        np_anomaly_scores = []
        for width in range(series_1.width):
            np_anomaly_scores.append(
                self._score_core_likelihood(
                    np_series_1[:, width], np_series_2[:, width].flatten()
                )
            )

        anomaly_scores = TimeSeries.from_times_and_values(
            series_1._time_index, list(zip(*np_anomaly_scores))
        )

        return self.window_adjustment_series(anomaly_scores)

    def window_adjustment_series(
        self,
        series: TimeSeries,
    ) -> TimeSeries:
        """Slides a window of size self.window along the input series, and replaces the value of
        the input time series by the mean of the values contained in the window (past self.window
        and itself points).

        A series of length n will be transformed into a series of length n-self.window.

        Parameters
        ----------
        series
            time series to adjust

        Returns
        -------
        TimeSeries
        """

        if self.window == 1:
            return series
        else:
            values = [
                series[ind : ind + self.window].mean(axis=0).all_values().flatten()[0]
                for (ind, _) in enumerate(series[self.window - 1 :].pd_series())
            ]
            return TimeSeries.from_times_and_values(
                series._time_index[self.window - 1 :], values
            )

    def _expects_probabilistic(self) -> bool:
        return True

    @abstractmethod
    def _score_core_NLlikelihood(self, input_1: Any, input_2: Any) -> Any:
        """For each timestamp, the corresponding distribution is fitted on the probabilistic time-series
        input_1, and returns the negative log-likelihood of the deterministic time-series input_2
        given the distribution.
        """
        pass


class GaussianNLLScorer(NLLScorer):
    """Gaussian negative log-likelihood Scorer

    Source of PDF function and parameters estimation (MLE):
    https://programmathically.com/maximum-likelihood-estimation-for-gaussian-distributions/
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GaussianNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error if std of deterministic_values is 0 (dividing by 0 otherwise)

        return [
            -np.log(
                (1 / np.sqrt(2 * np.pi * x1.std() ** 2))
                * np.exp(-((x2 - x1.mean()) ** 2) / (2 * x1.std() ** 2))
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]


class ExponentialNLLScorer(NLLScorer):
    """Exponential negative log-likelihood Scorer

    Source of PDF function and parameters estimation (MLE):
    https://www.statlect.com/fundamentals-of-statistics/exponential-distribution-maximum-likelihood
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "ExponentialNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        return [
            -np.log(x1.mean() * np.exp(-x1.mean() * x2))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]


class PoissonNLLScorer(NLLScorer):
    """Poisson negative log-likelihood Scorer

    Source of PDF function and parameters estimation (MLE):
    https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "PoissonNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error if values of deterministic_values are not (int and >=0). Required by the factorial function

        return [
            -np.log(np.exp(x1.mean()) * (x1.mean() ** x2) / math.factorial(x2))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]


class LaplaceNLLScorer(NLLScorer):
    """Laplace negative log-likelihood Scorer

    Source of PDF function and parameters estimation (MLE):
    https://en.wikipedia.org/wiki/Laplace_distribution
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "LaplaceNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error when all values are equal to the median -> divide by 0

        return [
            -np.log(
                (1 / (2 * np.abs(x1 - np.median(x1)).mean()))
                * np.exp(
                    -(np.abs(x2 - np.median(x1)) / np.abs(x1 - np.median(x1)).mean())
                )
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]


class CauchyNLLScorer(NLLScorer):
    """Cauchy negative log-likelihood Scorer

    For computational reasons, we opted for the simple method to estimate the parameters:
        - location parameter (x_0): median of the samples
        - scale parameter (gamma): half the sample interquartile range (Q3-Q1)/2

    Source of PDF function and parameters estimation:
    https://en.wikipedia.org/wiki/Cauchy_distribution
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "CauchyNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error when gamma is equal to 0 -> interquartile is equal to 0

        return [
            -np.log(
                (2 / (np.pi * np.subtract(*np.percentile(x1, [75, 25]))))
                * (
                    (np.subtract(*np.percentile(x1, [75, 25])) / 2) ** 2
                    / (
                        ((x2 - np.median(x1)) ** 2)
                        + ((np.subtract(*np.percentile(x1, [75, 25])) / 2) ** 2)
                    )
                )
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]


class GammaNLLScorer(NLLScorer):
    """Gamma negative log-likelihood Scorer

    Wrapped around the gamma scipy.stats functon.
    Source code: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>.
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GammaNLLScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        # TODO: takes a very long time to compute, understand why

        return [
            -gamma.logpdf(x2, *gamma.fit(x1))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
