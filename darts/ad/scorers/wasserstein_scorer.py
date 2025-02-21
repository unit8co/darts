"""
Wasserstein Scorer
-----

Wasserstein Scorer (distance function defined between probability distributions) [1]_.
The implementations is wrapped around `scipy.stats
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Wasserstein_metric
"""

from collections.abc import Sequence

import numpy as np
from scipy.stats import wasserstein_distance

from darts import metrics
from darts.ad.scorers.scorers import WindowedAnomalyScorer
from darts.logging import get_logger
from darts.metrics.metrics import METRIC_TYPE
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class WassersteinScorer(WindowedAnomalyScorer):
    def __init__(
        self,
        window: int = 10,
        component_wise: bool = False,
        window_agg: bool = True,
        diff_fn: METRIC_TYPE = metrics.ae,
    ) -> None:
        """Wasserstein Scorer

        When calling `fit(series)`, a moving window is applied, which results in a set of vectors of size `W`,
        where `W` is the window size. These vectors are kept in memory, representing the training
        distribution. The `score(series)` function will apply the same moving window.
        The Wasserstein distance is computed between the training distribution and each vector,
        resulting in an anomaly score.

        Alternatively, the scorer has the functions `fit_from_prediction()` and `score_from_prediction()`.
        Both require two series (actual and prediction), and compute a "difference" series by applying the
        function `diff_fn` (default: absolute difference). The resulting series is then passed to the
        functions `fit()` and `score()`, respectively.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs
        series. If set to `True`, the model will treat each series dimension independently. If set to `False`, the model
        concatenates the dimensions in each windows of length `W` and computes a single score for all dimensions.

        **Training with** `fit()`:

        The input can be a series (univariate or multivariate) or multiple series. The series will be partitioned
        into equal size subsequences. Each subsequence has size `W * D` (features), where:

        - `W` is the size of the window given as a parameter `window`
        - `D` is the dimension of the series (`D` = 1 if univariate or if `component_wise` is set to `True`)

        For a series of length `N`, `(N - W + 1)` subsequences will be generated. The final array `X` passed to the
        underlying scorer has shape `(N - W + 1, W * D)`; or in other terms (number of samples, number of features).
        If a list of series is given of length L, each series `i` is partitioned, and all `X_i` are concatenated along
        the sample axis.

        The arrays will be kept in memory, representing the training data distribution.
        In practice, the series or list of series can for instance represent residuals than can be
        considered independent and identically distributed (iid).

        If `component_wise` is set to `True`, the algorithm will be applied to each dimension independently. For each
        dimension, a PyOD model will be trained.

        **Computing score with** `score()`:

        The input can be a series (univariate or multivariate) or a sequence of series. The given series must have the
        same dimension `D` as the data used to train the PyOD model.

        For each series, if the series is multivariate of dimension `D`:

        - if `component_wise` is set to `False`: it returns a univariate series (dimension=1). It represents
          the anomaly score of the entire series in the considered window at each timestamp.
        - if `component_wise` is set to `True`: it returns a multivariate series of dimension `D`. Each dimension
          represents the anomaly score of the corresponding component of the input.

        If the series is univariate, it returns a univariate series regardless of the parameter
        `component_wise`.

        A window of size `W` is rolled on the series with a stride equal to 1. It is the same size window `W` used
        during the training phase.
        Each value in the score series thus represents how anomalous the sample of the `W` previous values is.

        Parameters
        ----------
        window
            Size of the sliding window that represents the number of samples in the testing distribution to compare
            with the training distribution in the Wasserstein function
        component_wise
            Boolean value indicating if the score needs to be computed for each component independently (`True`)
            or by concatenating the component in the considered window to compute one score (`False`).
            Default: `False`.
        window_agg
            Boolean indicating whether the anomaly score for each time step is computed by
            averaging the anomaly scores for all windows this point is included in.
            If `False`, the anomaly score for each point is the anomaly score of its trailing window.
            Default: `True`.
        diff_fn
            The differencing function to use to transform the predicted and actual series into one series.
            The scorer is then applied to this series. Must be one of Darts per-time-step metrics (e.g.,
            :func:`~darts.metrics.metrics.ae` for the absolute difference, :func:`~darts.metrics.metrics.err` for the
            difference, :func:`~darts.metrics.metrics.se` for the squared difference, ...).
            By default, uses the absolute difference (:func:`~darts.metrics.metrics.ae`).
        """

        # TODO:
        #     - understand better the math behind the Wasserstein distance when the test distribution contains
        #     only one sample
        #     - check if there is an equivalent Wasserstein distance for d-D distributions (currently only accepts 1D)

        if type(window) is int:  # noqa: E721
            if window > 0 and window < 10:
                logger.warning(
                    f"The `window` parameter WassersteinScorer is smaller than 10 (w={window})."
                    + " The value represents the window length rolled on the series given as"
                    + " input in the `score` function. At each position, the w values will"
                    + " constitute a subset, and the Wasserstein distance between the subset"
                    + " and the train distribution will be computed. To better represent the"
                    + " constituted test distribution, the window parameter should be larger"
                    + " than 10."
                )
        super().__init__(
            is_univariate=(not component_wise),
            window=window,
            window_agg=window_agg,
            diff_fn=diff_fn,
        )

    def __str__(self):
        return "WassersteinScorer"

    def _fit_core(self, series: Sequence[TimeSeries], *args, **kwargs):
        """The training values are considered as the scorer model"""
        self.model = np.concatenate([s.all_values(copy=False) for s in series]).squeeze(
            -1
        )

        if self.is_univariate or series[0].width == 1:
            self.model = self.model.flatten()

    def _model_score_method(self, model, data: np.ndarray) -> np.ndarray:
        """Wrapper around model inference method"""
        return [wasserstein_distance(model, window_samples) for window_samples in data]
