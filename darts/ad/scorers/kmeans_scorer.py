"""
k-means Scorer
--------------

`k`-means Scorer implementing `k`-means clustering [1]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/K-means_clustering
"""


import numpy as np
from sklearn.cluster import KMeans

from darts.ad.scorers.scorers import WindowedAnomalyScorer
from darts.logging import get_logger, raise_if_not

logger = get_logger(__name__)


class KMeansScorer(WindowedAnomalyScorer):
    def __init__(
        self,
        window: int = 1,
        k: int = 8,
        component_wise: bool = False,
        window_agg: bool = True,
        diff_fn="abs_diff",
        **kwargs,
    ) -> None:
        """
        When calling ``fit(series)``, a moving window is applied, which results in a set of vectors of size `W`,
        where `W` is the window size. The `k`-means model is trained on these vectors. The ``score(series)`` function
        applies the same moving window and returns the distance to the closest of the `k` centroids for each
        vector of size `W`.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two series (actual and prediction), and compute a "difference" series by applying the
        function ``diff_fn`` (default: absolute difference). The resulting series is then passed to the
        functions ``fit()`` and ``score()``, respectively.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs
        series. If set to True, the model will treat each component independently by fitting a different
        `k`-means model for each dimension. If set to False, the model concatenates the dimensions in
        each windows of length `W` and computes the score using only one underlying `k`-means model.

        **Training with** ``fit()``:

        The input can be a series (univariate or multivariate) or multiple series. The series will be sliced
        into equal size subsequences. The subsequence will be of size `W` * `D`, with:

        * `W` being the size of the window given as a parameter `window`
        * `D` being the dimension of the series (`D` = 1 if univariate or if `component_wise` is set to True)

        For a series of length `N`, (`N` - `W` + 1)/W subsequences will be generated. If a list of series is given
        of length L, each series will be partitioned into subsequences, and the results will be concatenated into
        an array of length L * number of subsequences of each series.

        The `k`-means model will be fitted on the generated subsequences. The model will find `k` clusters
        in the vector space of dimension equal to the length of the subsequences (`D` * `W`).

        If `component_wise` is set to True, the algorithm will be applied to each dimension independently. For each
        dimension, a `k`-means model will be trained.

        **Computing score with** ``score()``:

        The input can be a series (univariate or multivariate) or a sequence of series. The given series must have the
        same dimension `D` as the data used to train the `k`-means model.

        For each series, if the series is multivariate of dimension `D`:

        * if `component_wise` is set to False: it returns a univariate series (dimension=1). It represents
          the anomaly score of the entire series in the considered window at each timestamp.
        * if `component_wise` is set to True: it returns a multivariate series of dimension `D`. Each dimension
          represents the anomaly score of the corresponding component of the input.

        If the series is univariate, it returns a univariate series regardless of the parameter
        `component_wise`.

        A window of size `W` is rolled on the series with a stride equal to 1. It is the same size window `W` used
        during the training phase.
        Each value in the score series thus represents how anomalous the sample of the `W` previous values is.

        Parameters
        ----------
        window
            Size of the window used to create the subsequences of the series.
        k
            The number of clusters to form as well as the number of centroids to generate by the KMeans model.
        component_wise
            Boolean value indicating if the score needs to be computed for each component independently (True)
            or by concatenating the component in the considered window to compute one score (False).
            Default: False
        window_agg
            Boolean indicating whether the anomaly score for each time step is computed by
            averaging the anomaly scores for all windows this point is included in.
            If False, the anomaly score for each point is the anomaly score of its trailing window.
            Default: True.
        diff_fn
            Optionally, reduction function to use if two series are given. It will transform the two series into one.
            This allows the KMeansScorer to apply KMeans on the original series or on its residuals (difference
            between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff_series()``).
            Default: "abs_diff"
        kwargs
            Additional keyword arguments passed to the internal scikit-learn KMeans model(s).
        """

        raise_if_not(
            type(component_wise) is bool,
            f"Parameter `component_wise` must be Boolean, found type: {type(component_wise)}.",
            logger,
        )
        self.component_wise = component_wise

        self.kmeans_kwargs = kwargs
        self.kmeans_kwargs["n_clusters"] = k
        # stop warning about default value of "n_init" changing from 10 to "auto" in sklearn 1.4
        if "n_init" not in self.kmeans_kwargs:
            self.kmeans_kwargs["n_init"] = 10

        self.model = KMeans(**self.kmeans_kwargs)

        super().__init__(
            univariate_scorer=(not component_wise),
            window=window,
            diff_fn=diff_fn,
            window_agg=window_agg,
        )

    def __str__(self):
        return "k-means Scorer"

    def _model_score_method(self, model, data: np.ndarray) -> np.ndarray:
        """Wrapper around model inference method"""
        # only return the closest distance out of the k ones (k centroids)
        return model.transform(data).min(axis=1)
