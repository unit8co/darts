"""
k-means Scorer
--------------

`k`-means Scorer (k-means clustering) [1]_.
The implementations is wrapped around `scikit-learn
<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/K-means_clustering
"""

from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from darts.ad.scorers.scorers import FittableAnomalyScorer
from darts.logging import raise_if_not
from darts.timeseries import TimeSeries


class KMeansScorer(FittableAnomalyScorer):
    def __init__(
        self,
        window: Optional[int] = None,
        k: int = 2,
        component_wise: bool = False,
        diff_fn="abs_diff",
    ) -> None:
        """
        When calling ``fit(series)``, a moving window is applied, which results in a set of vectors of size W,
        where W is the window size. The KMeans model is trained on these vectors. The ``score(series)`` function
        will apply the same moving window and return the minimal distance between the K centroid and for each
        vector of size W.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two inputs and transform them into one series by applying the function ``diff_fn``
        (default: absolute difference). The resulting series will then be passed to the respective function
        ``fit()`` and ``score()``.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs
        series. If set to True, the model will treat each series dimension independently by fitting a different
        KMeans model for each dimension. If the input series has a dimension of D, the model will train D KMeans
        models. If set to False, the model will concatenate the dimensions in the considered `window` W and compute
        the score using only one trained KMeans model.

        Training with ``fit()``:

        The input can be a series (univariate or multivariate) or a sequence of series. The series will be partitioned
        into equal size subsequences. The subsequence will be of size W * D, with:
            - W being the size of the window given as a parameter `window` (w>0)
            - D being the dimension of the series (D=1 if deterministic)

        For a series of length N, (N-W+1)/W subsequences will be generated. If a list of series is given of length L,
        each series will be partitioned into subsequences, and the results will be concatenated into an array of
        length L * number of subsequences.

        The model KMeans will be fitted on the generated subsequences. The model will find `k` clusters
        in the vector space of dimension equal to the length of the subsequences (D*W).

        If `component_wise` is set to True, the algorithm will be applied to each dimension independently. For each
        dimension, a KMeans model will be trained.

        Compute score with ``score()``:

        The input can be a series (univariate or multivariate) or a sequence of series. The given series must have the
        same dimension D as the data used to train the KMeans model.

        - If the series is multivariate of dimension D:
            - if `component_wise` is set to False: it will return a univariate series (dimension=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if `component_wise` is set to True: it will return a multivariate series of dimension D. Each dimension
            represents the anomaly score of the corresponding component of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        `component_wise`.

        A window of size W is rolled on the series with a stride equal to 1. It is the same size window W used during
        the training phase. At each timestamp, the previous W values will form a vector of size W * D
        of the series (with D being the series dimensions). The KMeans model will then retrieve the closest centroid
        to this vector and compute the euclidean distance between the centroid and the vector. The output will be a
        series of dimension one and length N-W+1, with N being the length of the input series. Each value represents
        how anomalous the sample of the W previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series. The function will return an anomaly score for each series in the list.

        If `component_wise` is set to True, the algorithm will be applied to each dimension independently. The distance
        will be computed between the vector and the closest centroid found by the model trained on the corresponding
        dimension during the training.

        Parameters
        ----------
        window
            Size of the window used to create the subsequences of the series.
        k
            The number of clusters to form as well as the number of centroids to generate by the KMeans model.
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

        raise_if_not(
            type(component_wise) is bool,
            f"'component_wise' must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

        self.k = k

        if component_wise:
            returns_UTS = False
        else:
            returns_UTS = True

        super().__init__(returns_UTS=returns_UTS, window=window, diff_fn=diff_fn)

    def __str__(self):
        return "KMeansScorer"

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
    ):

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
            self.models = models

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
                    self.models[width]
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
