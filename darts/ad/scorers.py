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

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from darts import TimeSeries
from darts.logging import raise_if, raise_if_not


class AnomalyScorer(ABC):
    "Base class for all anomaly scorers"

    def __init__(self, window: Optional[int] = None) -> None:

        if window is None:
            window = 1

        raise_if_not(
            window > 0,
            f"window must be stricly greater than 0, found {window}",
        )

        self.window = window

    @abstractmethod
    def score(self, input_1: Any, input_2: Any) -> Any:
        pass

    @abstractmethod
    def _score_core(self, input_1: Any, input_2: Any) -> Any:
        pass

    # this function has nothing to do with the AnomalyScorer itself...
    def eval_accuracy_from_scores(
        self,
        anomaly_score: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
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
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score of the anomaly time series
        """

        if metric == "AUC_ROC":
            scoring_fn = roc_auc_score
        elif metric == "AUC_PR":
            scoring_fn = average_precision_score
        else:
            raise ValueError("Argument `metric` must be one of 'AUC_ROC', 'AUC_PR'")

        list_anomaly_score, list_actual_anomalies = self._convert_to_list(
            anomaly_score, actual_anomalies
        )

        sol = []
        for s_score, s_anomalies in zip(list_anomaly_score, list_actual_anomalies):

            # if window > 1, the anomalies will be adjusted so that it can be compared timewise with s_score
            s_anomalies = self.window_adjustment_anomalies(s_anomalies)

            raise_if_not(
                np.array_equal(
                    s_anomalies.values(copy=False),
                    s_anomalies.values(copy=False).astype(bool),
                ),
                "'actual_anomalies' must be a binary time series.",
            )

            self._sanity_check(s_score, s_anomalies)
            s_score, s_anomalies = self._return_intersect(s_score, s_anomalies)

            raise_if(
                s_anomalies.sum(axis=0).values(copy=False).flatten().min() == 0,
                f"'actual_anomalies' does not contain anomalies. {metric} cannot be computed.",
            )

            raise_if(
                s_anomalies.sum(axis=0).values(copy=False).flatten().max()
                == len(s_anomalies),
                f"'actual_anomalies' contains only anomalies. {metric} cannot be computed."
                + ["", f" Think about reducing the window (window={self.window})"][
                    self.window > 1
                ],
            )

            metrics = []
            for width in range(s_score.width):
                metrics.append(
                    scoring_fn(
                        y_true=s_anomalies.all_values(copy=False)[:, width],
                        y_score=s_score.all_values(copy=False)[:, width],
                    )
                )

            if width == 0:
                sol.append(metrics[0])
            else:
                sol.append(metrics)

        if len(sol) == 1:
            return sol[0]
        else:
            return sol

    def _return_intersect(
        self,
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

    def eval_accuracy(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]] = None,
        scoring: str = "AUC_ROC",
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
        scoring:
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score for the time series
        """
        anomaly_score = self.score(series_1, series_2)

        return self.eval_accuracy_from_scores(anomaly_score, actual_anomalies, scoring)

    def _sanity_check(
        self,
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

    def _convert_to_list(
        self,
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

    def window_adjustment_anomalies(
        self,
        series: TimeSeries,
    ) -> TimeSeries:
        """Slides a window of size self.window along the input series, and replaces the value of the
        input time series by the maximum of the values contained in the window.

        The binary time series output represents if there is an anomaly (=1) or not (=0) in the past
        self.window points. The new series will equal the length of the input series
        - self.window. Its first point will start at the first time index of the input time series +
        self.window points.

        Parameters
        ----------
        series:
            Binary time series

        Returns
        -------
        Binary TimeSeries
        """
        if self.window == 1:
            # the process results in replacing every value by itself -> return directly the series
            return series
        else:

            self._sanity_check(series)

            np_series = series.all_values(copy=False)

            values = [
                np_series[ind : ind + self.window].max(axis=0)
                for ind in range(len(np_series) - self.window + 1)
            ]

            return TimeSeries.from_times_and_values(
                series._time_index[self.window - 1 :], values
            )


class NonFittableAnomalyScorer(AnomalyScorer):
    "Base class of anomaly scorers that do not need training."

    def __init__(self) -> None:
        # window for NonFittableAnomalyScorer are always set to None
        super().__init__(window=None)

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
        list_series_1, list_series_2 = self._convert_to_list(series_1, series_2)

        anomaly_scores = []

        for s1, s2 in zip(list_series_1, list_series_2):

            self._sanity_check(s1, s2)
            s1, s2 = self._return_intersect(s1, s2)

            anomaly_scores.append(self._score_core(s1, s2))

        if len(anomaly_scores) == 1:
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

        list_series_1, list_series_2 = self._convert_to_list(series_1, series_2)

        anomaly_scores = []

        if series_2 is None:
            for series in list_series_1:
                self._sanity_check(series)
                anomaly_scores.append(self._score_core(series, None))
        else:
            for (s1, s2) in zip(list_series_1, list_series_2):
                self._sanity_check(s1, s2)
                anomaly_scores.append(self._score_core(s1, s2))

        if len(anomaly_scores) == 1:
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
        list_series_1, list_series_2 = self._convert_to_list(series_1, series_2)

        if series_2 is None:
            for series in list_series_1:
                self._sanity_check(series)
        else:
            for (s1, s2) in zip(list_series_1, list_series_2):
                self._sanity_check(s1, s2)

        self._fit_core(list_series_1, list_series_2)

    def _diff_sequence(
        self, list_series_1: Sequence[TimeSeries], list_series_2: Sequence[TimeSeries]
    ) -> Sequence[TimeSeries]:
        """Calls the function _diff() to every pair (s1,s2) in the list (list_series_1,list_series_2).

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
        series_1, series_2 = self._return_intersect(series_1, series_2)

        if self.reduced_function == "abs_diff":
            return (series_1 - series_2).map(lambda x: np.abs(x))
        elif self.reduced_function == "diff":
            return series_1 - series_2
        else:
            # found an non-existent reduced_function
            raise ValueError(
                f"Metric should be 'diff' or 'abs_diff', found {self.reduced_function}"
            )


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
    ) -> None:

        """
        A Gaussian mixture model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will compute the log-likelihood of each sample.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option to change
        these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned
        into equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter.
        If the series is of length n, width d and the window is set to w, the training phase will generate (n-w+1)/w
        data samples of length d * w. If a list of series is given of length l, each series will be partitioned into
        subsequences, and the results will be concatenated into an array of length l * number of subsequences.

        The Gaussian mixture model will be fitted on the generated subsequences. The model will estimate its parameters
        with the EM algorithm.

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the GaussianMixture model. A window of size w is rolled on the series with
        a stride equal to 1. It is the same window used during the training phase. At each timestamp, the previous w
        values will be used to form a vector of size w * width of the series. The ``score_samples()`` of the Gaussian
        mixture model will be called, with the vector as parameter. It will return the log-likelihood of the vector,
        and the exponential function is applied. The output will be a series of width 1 and length n-w+1, with n being
        the length of the input series. Each value will represent how anomalous the sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

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
        """

        super().__init__(window=window, reduced_function=reduced_function)
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components)

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
        self.model.fit(
            np.concatenate(
                [
                    series.all_values(copy=False).reshape(
                        -1, self.window * series.width
                    )
                    for series in list_series
                ]
            )
        )

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

        np_anomaly_score = np.exp(
            self.model.score_samples(
                np.array(
                    [
                        np.array(series.all_values(copy=False)[i : i + self.window])
                        for i in range(len(series) - self.window + 1)
                    ]
                ).reshape(-1, self.window * series.width)
            )
        )
        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], np_anomaly_score
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
    ) -> None:
        """
        A KMeans model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the minimal distance between the centroid and the sample.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option to change
        these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned into
        equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter. If the series is of length n, width d and the window is set
        to w, the training phase will generate (n-w+1)/w data samples of length d*w. If a list of series is given of
        length l, each series will be partitioned into subsequences, and the results will be concatenated into an array
        of length l * number of subsequences.

        The model KMeans will be fitted on the generated subsequences. The model will find k cluster of dimensions
        equal to the length of the subsequences (d*w).

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the KMeans model. A window of size w is rolled on the series with a stride
        equal to 1. It is the same window used during the training phase. At each timestamp, the previous w values
        will be used to form a vector of size w * width of the series.
        The KMeans model will then retrieve the closest centroid to this vector, and compute the L1 norm between the
        centroid and the vector. The output will be a series of width 1 and length n-w+1, with n being the length of
        the input series. Each value will represent how anomalous the sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

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
        """

        super().__init__(window=window, reduced_function=reduced_function)
        self.k = k
        self.model = KMeans(n_clusters=k)

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
        self.model.fit(
            np.concatenate(
                [
                    s.all_values(copy=False).reshape(-1, self.window * s.width)
                    for s in list_series
                ]
            )
        )

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

        # return distance to the clostest centroid
        np_anomaly_score = self.model.transform(
            np.array(
                [
                    np.array(np_series[i : i + self.window])
                    for i in range(len(series) - self.window + 1)
                ]
            ).reshape(-1, self.window * series.width)
        ).min(axis=1)

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], np_anomaly_score
        )


class WasserteinScorer(FittableAnomalyScorer):
    """WasserteinScorer anomaly score

    Wrapped around the Wassertein scipy.stats functon.
    Source code: <https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L8675-L8749>.
    """

    def __init__(self, window: Optional[int] = None, reduced_function=None) -> None:
        """
        A Wassertein model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the wassertein distance bewteen the training distribution
        and the window sample distribution. Both distributions are 1D.

        TODO:
        - understand better the math behind the Wassertein distance (ex: when the test distribution contains
        only one sample)
        - check if there is an equivalent wassertein distance for d-D distributions (currently only accepts 1D)

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (reduced_function), will be applied to transform the 2 time series into 1.
        Default: "abs_diff"

        Training:

        The input can be a series (univariate or multivariate) or a list of series. All the values will be concatenated
        to form one continuous array. If the series is of length n and width d, the array will be of length n*d.
        If a list of series is given of length l, each series will be reduced to an array, and the l arrays will then
        be concatenated to form a continuous array of length l*d*w.

        The array will be kept in memory, representing the training data distribution.
        In practice, the series or list of series would represent residuals than can be considered independent
        and identically distributed (iid).


        Compute score:

        The input is a series (univariate or multivariate) or a list of series.
        A window of size w (given as a parameter named window) is rolled on the series with a stride equal to 1.
        At each timestamp, the previous w values will be used to form a vector of size w * width of the series.
        The Wassertein distance will be computed between this vector and the train distribution.
        The function will return a float number indicating how different these two distributions are.
        The output will be a series of width 1 and length n-w+1, with n being the length of the input series.
        Each value will represent how anomalous the sample of the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()``
        will be applied independently on each series.

        Parameters
        ----------
        window
            Size of the sliding window that represents the number of samples in the testing distribution to compare
            with the training distribution in the Wassertein function
        reduced_function
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the WasserteinScorer to compute the Wassertein distance on the original series or on its
            residuals (difference between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff()``).
            Default: "abs_diff"
        """

        if window is None:
            window = 10
        super().__init__(window=window, reduced_function=reduced_function)

        raise_if_not(
            self.window > 0,
            "window must be stricly higher than 0,"
            "(preferably higher than 10 as it is the number of samples of the test distribution)",
        )

    def __str__(self):
        return f"WasserteinScorer (window={self.window}, reduced_function={self.reduced_function})"

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
        self.training_data = np.concatenate(
            [s.all_values(copy=False).flatten() for s in list_series]
        )

    def _score_core(
        self, series_1: TimeSeries, series_2: TimeSeries = None
    ) -> TimeSeries:

        if series_2 is None:
            series = series_1
        else:
            series = self._diff(series_1, series_2)

        distance = []
        np_series = series.all_values(copy=False).flatten()

        for i in range(len(series) - self.window + 1):
            distance.append(
                wasserstein_distance(
                    self.training_data,
                    np_series[i * series.width : (i + self.window + 1) * series.width],
                )
            )

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], distance
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

        Training:

        The input can be a series (univariate or multivariate) or a list of series. The series will be partitioned into
        equal size subsequences. If the series is multivariate (width>1), then the subsequence will be of size
        window * width, with window being a given parameter.
        If the series is of length n, width d and the window is set to w, the training phase will generate (n-w+1)/w
        data samples of length d*w. If a list of series is given of length l, each series will be partitioned into
        subsequences, and the results will be concatenated into an array of length l * number of subsequences.

        The model LocalOutlierFactor will be fitted on the generated subsequences.

        Compute score:

        The input is a series (univariate or multivariate) or a list of series. The given series must have the same
        width d as the data used to train the LocalOutlierFactor model. A window of size w is rolled on the series
        with a stride equal to 1. It is the same window used during the training phase. At each timestamp, the previous
        w values will be used to form a vector of size w * width of the series.
        The Local outlier factor model will then return the local deviation of the density of the given vector with
        respect to its neighbors (constituted in the training phase). The output will be a series of width 1 and length
        n-w+1, with n being the length of the input series. Each value will represent how anomalous the sample of
        the w previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series.

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
        """

        super().__init__(window=window, reduced_function=reduced_function)
        self.n_neighbors = n_neighbors
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

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
        self.model.fit(
            np.concatenate(
                [
                    s.all_values(copy=False).reshape(-1, self.window * s.width)
                    for s in list_series
                ]
            )
        )

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
        # return distance to the clostest centroid
        np_anomaly_score = np.abs(
            self.model.score_samples(
                np.array(
                    [
                        np.array(np_series[i : i + self.window])
                        for i in range(len(np_series) - self.window + 1)
                    ]
                ).reshape(-1, self.window * series.width)
            )
        )

        return TimeSeries.from_times_and_values(
            series._time_index[self.window - 1 :], np_anomaly_score
        )


class L2(NonFittableAnomalyScorer):
    """L2 distance metric

    Returns the L2 norm between each timestamps of two series.

    The input can be univariate or multivariate, it will always return a univariate timesereies.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "L2"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        return ((series_1 - series_2) ** 2).sum(axis=1).map(lambda x: np.sqrt(x))


class L1(NonFittableAnomalyScorer):
    """L1 distance metric

    Returns the L1 norm between each timestamps of two series.

    The input can be univariate or multivariate, it will always return a univariate timesereies.
    If the two series are univariate, scorer L1 is equivalent to the scorer AbsDifference.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "L1"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        return (series_1 - series_2).map(lambda x: np.abs(x)).sum(axis=1)


class Difference(NonFittableAnomalyScorer):
    """Difference distance metric

    Returns the difference between each timestamps of two series.

    If the two series are multivariate, it will return a multivariate series.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "Difference"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        return series_1 - series_2


class AbsDifference(NonFittableAnomalyScorer):
    """Absolute difference distance metric

    Returns the absolute difference between each timestamps of two series.

    If the two series are multivariate, it will return a multivariate series.
    If the two series are univariate, scorer AbsDifference is equivalent to the scorer L1.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "AbsDifference"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:
        return (series_1 - series_2).map(lambda x: np.abs(x))
