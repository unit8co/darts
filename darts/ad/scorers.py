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
from typing import Any, Optional, Sequence, Union

import numpy as np
from pyod.models.base import BaseDetector
from scipy.stats import gamma, norm, wasserstein_distance
from sklearn.cluster import KMeans

from darts import TimeSeries
from darts.ad.utils import (
    _check_timeseries_type,
    _intersect,
    _same_length,
    _sanity_check_2series,
    _to_list,
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

    @property
    def _expects_probabilistic(self) -> bool:
        """Checks if the scorer expects a probabilistic predictions.
        By default, returns False. Needs to be overwritten by scorers that do expects
        probabilistic predictions.
        """
        return False

    def _check_probabilistic(self, series: TimeSeries):

        raise_if_not(
            series.is_stochastic,
            f"Scorer {self.__str__()} is expecting a probabilitic timeseries as its first input \
            (number of samples must be higher than 1).",
        )

        #        if self._expects_probabilistic:
        # else:
        #    if series.is_stochastic:
        # TODO: output a warning "The scorer expects a non probabilitic input
        # (num of samples needs to be equal to 1)"
        # median along each time stamp is computed to reduce the number of samples to 1
        #        series = series.quantile_timeseries(quantile=0.5)

        return series

    def _check_deterministic(self, series: TimeSeries):
        # TODO: create a warning rather than an error, and avg on axis 2
        raise_if_not(
            series.is_deterministic,
            f"Scorer {self.__str__()} is expecting a deterministic timeseries as its second input \
        (number of samples must be equal to 1, found number: {series.n_samples}).",
        )

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

    def eval_accuracy_from_prediction(
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
        anomaly_score = self.score_from_prediction(series_1, series_2)

        return eval_accuracy_from_scores(
            anomaly_score, actual_anomalies, self.window, metric
        )


class NonFittableAnomalyScorer(AnomalyScorer):
    "Base class of anomaly scorers that do not need training."

    def __init__(self, window) -> None:
        super().__init__(window=window)

        # indicates if the scorer is trainable or not
        self.trainable = False

    def score_from_prediction(
        self,
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        return self.score(pred_series, actual_series)

    def score(
        self,
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score between the two given time series.

        Parameters
        ----------
        pred_series
            1st time series
        actual_series:
            2nd time series

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series
        """
        list_pred_series, list_actual_series = _to_list(pred_series), _to_list(
            actual_series
        )
        _same_length(list_pred_series, list_actual_series)

        anomaly_scores = []

        for s1, s2 in zip(list_pred_series, list_actual_series):
            _sanity_check_2series(s1, s2)
            s1, s2 = _intersect(s1, s2)
            anomaly_scores.append(self._score_core(s1, s2))

        if (
            len(anomaly_scores) == 1
            and not isinstance(pred_series, Sequence)
            and not isinstance(actual_series, Sequence)
        ):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def eval_accuracy(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series_1: Union[TimeSeries, Sequence[TimeSeries]],
        series_2: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:

        return self.eval_accuracy_from_prediction(
            actual_anomalies, series_1, series_2, metric
        )


class FittableAnomalyScorer(AnomalyScorer):
    "Base class of scorers that do need training."

    def __init__(self, window, diff_fn) -> None:
        super().__init__(window=window)

        # indicates if the scorer is trainable or not
        self.trainable = True

        # indicates if the scorer has been trained yet
        self._fit_called = False

        # function used in ._diff_series() to convert 2 time series into 1
        if diff_fn is None:
            diff_fn = "abs_diff"
        self.diff_fn = diff_fn

    def check_if_fit_called(self):
        """Checks if the scorer has been fitted before calling its `score()` function."""

        raise_if_not(
            self._fit_called,
            f"The Scorer {self.__str__()} has not been fitted yet. Call `fit()` first",
        )

    def _check_window_size(self, series: TimeSeries):
        """Checks if the parameter window is less or equal to the length of the given series"""

        raise_if_not(
            self.window <= len(series),
            f"Window size {self.window} is greater than the targeted series length {len(series)}, \
            must be lower or equal.",
        )

    def eval_accuracy(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:

        """Computes the anomaly score of the given time series, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            time series
        metric
            Optionally, metric function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score for the time series
        """
        anomaly_score = self.score(series)

        print(metric)

        return eval_accuracy_from_scores(
            anomaly_score, actual_anomalies, self.window, metric
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the given series.

        Parameters
        ----------
        series
            Time series

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series
        """

        self.check_if_fit_called()

        list_series = _to_list(series)

        anomaly_scores = []
        for s in list_series:
            _check_timeseries_type(s)
            self._check_deterministic(s)
            anomaly_scores.append(self._score_core(s))

        if len(anomaly_scores) == 1 and not isinstance(series, Sequence):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_from_prediction(
        self,
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the given series.

        If 2 time series are given, a reduced function, given as a parameter in the __init__ method
        (diff_fn), will be applied to transform the 2 time series into 1. Default: "abs_diff"

        Parameters
        ----------
        pred_series
            1st time series
        actual_series
            2nd time series

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series
        """

        self.check_if_fit_called()

        list_pred_series, list_actual_series = _to_list(pred_series), _to_list(
            actual_series
        )
        _same_length(list_pred_series, list_actual_series)

        anomaly_scores = []
        for (s1, s2) in zip(list_pred_series, list_actual_series):
            _sanity_check_2series(s1, s2)
            self._check_deterministic(s1)
            self._check_deterministic(s2)
            anomaly_scores.append(self.score(self._diff_series(s1, s2)))

        if (
            len(anomaly_scores) == 1
            and not isinstance(pred_series, Sequence)
            and not isinstance(actual_series, Sequence)
        ):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        """Fits the scorer on the given time series input.

        Parameters
        ----------
        series
            Time series

        Returns
        -------
        self
            Fitted Scorer.
        """
        list_series = _to_list(series)

        for idx, s in enumerate(list_series):
            _check_timeseries_type(s)
            self._check_deterministic(s)

        self._fit_core(list_series)
        self._fit_called = True

    def fit_from_prediction(
        self,
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        """Fits the scorer on the given time series input.

        If 2 time series are given, a reduced function, given as a parameter in the __init__ method
        (diff_fn), will be applied to transform the 2 time series into 1. Default: "abs_diff"

        Parameters
        ----------
        pred_series
            1st time series
        actual_series
            2nd time series

        Returns
        -------
        self
            Fitted Scorer.
        """
        list_pred_series, list_actual_series = _to_list(pred_series), _to_list(
            actual_series
        )
        _same_length(list_pred_series, list_actual_series)

        list_fit_series = []
        for idx, (s1, s2) in enumerate(zip(list_pred_series, list_actual_series)):
            _sanity_check_2series(s1, s2)
            self._check_deterministic(s1)
            self._check_deterministic(s2)
            list_fit_series.append(self._diff_series(s1, s2))

        self._fit_core(list_fit_series)
        self._fit_called = True

    def _diff_sequence(
        self, list_series_1: Sequence[TimeSeries], list_series_2: Sequence[TimeSeries]
    ) -> Sequence[TimeSeries]:
        """Calls the function _diff_series() on every pair (s1,s2) in the list (list_series_1,list_series_2).

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

        return list(map(self._diff_series, list_series_1, list_series_2))

    def _diff_series(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        """Applies the diff_fn to the two time series. Converts two time series into 1.

        series_1 and series_2 must:
        - have a non empty time intersection
        - be of the same width/dimension W

        Parameters
        ----------
        series_1
            1st time series
        series_2:
            2nd time series

        Returns
        -------
        TimeSeries
            series of width/dimension W
        """
        series_1, series_2 = _intersect(series_1, series_2)

        if self.diff_fn == "abs_diff":
            return (series_1 - series_2).map(lambda x: np.abs(x))
        elif self.diff_fn == "diff":
            return series_1 - series_2
        else:
            # found an non-existent diff_fn
            raise ValueError(
                f"Metric should be 'diff' or 'abs_diff', found {self.diff_fn}"
            )


class PyODScorer(FittableAnomalyScorer):
    "Wrapped around models of PyOD"

    def __init__(
        self,
        model,
        window: Optional[int] = None,
        component_wise: bool = False,
        diff_fn=None,
    ) -> None:

        raise_if_not(
            isinstance(model, BaseDetector),
            f"model must be BaseDetector of the library PyOD, found type: {type(component_wise)}",
        )
        self.model = model

        super().__init__(window=window, diff_fn=diff_fn)

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

    def __str__(self):
        return "PyODScorer model: {}".format(self.model.__str__().split("(")[0])

    def _fit_core(self, list_series: Sequence[TimeSeries]):

        self.width_trained_on = list_series[0].width

        list_np_series = [series.all_values(copy=False) for series in list_series]

        if not self.component_wise:
            self.model.fit(
                np.concatenate(
                    [
                        np.array(
                            [
                                np.array(np_series[i : i + self.window])
                                for i in range(len(np_series) - self.window + 1)
                            ]
                        ).reshape(-1, self.window * len(np_series[0]))
                        for np_series in list_np_series
                    ]
                )
            )
        else:
            models = []
            for width in range(self.width_trained_on):

                model_width = self.model
                model_width.fit(
                    np.concatenate(
                        [
                            np.array(
                                [
                                    np.array(np_series[i : i + self.window, width])
                                    for i in range(len(np_series) - self.window + 1)
                                ]
                            ).reshape(-1, self.window)
                            for np_series in list_np_series
                        ]
                    )
                )
                models.append(model_width)
            self.model = models

    def _score_core(self, series: TimeSeries) -> TimeSeries:

        raise_if_not(
            self.width_trained_on == series.width,
            "Input must have the same width of the data used for training the PyODScorer model {}, found \
            width: {} and {}".format(
                self.model.__str__().split("(")[0], self.width_trained_on, series.width
            ),
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:

            np_anomaly_score.append(
                np.exp(
                    self.model.decision_function(
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
                np_anomaly_score_width = self.model[width].decision_function(
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
        component_wise: bool = False,
        diff_fn=None,
    ) -> None:
        """
        A KMeans model is trained on the training data when the ``fit()`` method is called.
        The ``score()`` method will return the minimal distance between the centroid and the sample.

        TODO: stride in training is equal to w, and in score stride is equal to 1. Give the option to change
        these parameters.

        If 2 time series are given in the ``fit()`` or ``score()`` methods, a reduced function, given as a parameter
        in the __init__ method (diff_fn), will be applied to transform the 2 time series into 1.
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
        diff_fn
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the KMeansScorer to apply KMeans on the original series or on its residuals (difference between
            the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff_series()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        super().__init__(window=window, diff_fn=diff_fn)

        raise_if_not(
            isinstance(component_wise, bool),
            f"component_wise must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

        self.k = k

    def __str__(self):
        return "KMeansScorer"

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
    ):

        self.width_trained_on = list_series[0].width

        list_np_series = [series.all_values(copy=False) for series in list_series]

        if not self.component_wise:
            self.model = KMeans(n_clusters=self.k)
            self.model.fit(
                np.concatenate(
                    [
                        np.array(
                            [
                                np.array(np_series[i : i + self.window])
                                for i in range(len(np_series) - self.window + 1)
                            ]
                        ).reshape(-1, self.window * len(np_series[0]))
                        for np_series in list_np_series
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
                            np.array(
                                [
                                    np.array(np_series[i : i + self.window, width])
                                    for i in range(len(np_series) - self.window + 1)
                                ]
                            ).reshape(-1, self.window)
                            for np_series in list_np_series
                        ]
                    )
                )
                models.append(model)
            self.model = models

    def _score_core(self, series: TimeSeries) -> TimeSeries:

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


class WassersteinScorer(FittableAnomalyScorer):
    """WassersteinScorer anomaly score

    Wrapped around the Wasserstein scipy.stats functon.
    Source code: <https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L8675-L8749>.
    """

    def __init__(
        self, window: Optional[int] = None, component_wise: bool = False, diff_fn=None
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
        in the __init__ method (diff_fn), will be applied to transform the 2 time series into 1.
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
        diff_fn
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the WassersteinScorer to compute the Wasserstein distance on the original series or on its
            residuals (difference between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff_series()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each width/dimension independently (True)
            or by concatenating the width in the considered window to compute one score (False).
            Default: False
        """

        if window is None:
            window = 10
        super().__init__(window=window, diff_fn=diff_fn)

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
        return "WassersteinScorer"

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
    ):

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

    def _score_core(self, series: TimeSeries) -> TimeSeries:

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
        self._check_deterministic(series_1)
        self._check_deterministic(series_2)
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
        return f"Norm (ord={self.ord})"

    def _score_core(
        self,
        series_1: TimeSeries,
        series_2: TimeSeries,
    ) -> TimeSeries:

        self._check_deterministic(series_1)
        self._check_deterministic(series_2)

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
        pred_series: TimeSeries,
        actual_series: TimeSeries,
    ) -> TimeSeries:
        """For each timestamp of the inputs:
            - the parameters of the considered distribution are fitted on the samples of the probabilistic time series
            - the negative log-likelihood of the determinisitc time series values are computed

        If the series are multivariate, the score will be computed on each width independently.

        Parameters
        ----------
        pred_series
            A probabilistic time series (number of samples per timestamp must be higher than 1)
        actual_series:
            A determinisict time series (number of samples per timestamp must be equal to 1)

        Returns
        -------
        TimeSeries
        """
        self._check_probabilistic(pred_series)
        self._check_deterministic(actual_series)

        np_pred_series = pred_series.all_values(copy=False)
        np_actual_series = actual_series.all_values(copy=False)

        np_anomaly_scores = []
        for width in range(pred_series.width):
            np_anomaly_scores.append(
                self._score_core_NLlikelihood(
                    np_pred_series[:, width], np_actual_series[:, width].flatten()
                )
            )

        anomaly_scores = TimeSeries.from_times_and_values(
            pred_series._time_index, list(zip(*np_anomaly_scores))
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
            if x1.std() > 0.01
            else -np.log(
                (1 / np.sqrt(2 * np.pi * 0.06**2))
                * np.exp(-((x2 - x1.mean()) ** 2) / (2 * 0.06**2))
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


class CRPSScorer(NLLScorer):
    """CRPS Scorer"""

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "CRPSScorer"

    def _score_core_NLlikelihood(
        self,
        probabilistic_estimations: np.ndarray,
        deterministic_values: np.ndarray,
    ) -> np.ndarray:

        return [
            x1.std()
            * (
                ((x2 - x1.mean()) / x1.std())
                * (2 * norm.cdf((x2 - x1.mean()) / x1.std()) - 1)
                + 2 * norm.pdf((x2 - x1.mean()) / x1.std())
                - 1 / np.sqrt(np.pi)
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
