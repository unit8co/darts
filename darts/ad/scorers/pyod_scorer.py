"""
PyOD Scorer
-----

This scorer can wrap around detection algorithms of PyOD.
`PyOD https://pyod.readthedocs.io/en/latest/#`_.
"""

import numpy as np
from pyod.models.base import BaseDetector

from darts import metrics
from darts.ad.scorers.scorers import WindowedAnomalyScorer
from darts.logging import get_logger, raise_if_not
from darts.metrics.metrics import METRIC_TYPE

logger = get_logger(__name__)


class PyODScorer(WindowedAnomalyScorer):
    def __init__(
        self,
        model: BaseDetector,
        window: int = 1,
        component_wise: bool = False,
        window_agg: bool = True,
        diff_fn: METRIC_TYPE = metrics.ae,
    ) -> None:
        """PyOD Scorer

        When calling ``fit(series)``, a moving window is applied, which results in a set of vectors of size `W`,
        where `W` is the window size. The PyODScorer model is trained on these vectors. The ``score(series)``
        function will apply the same moving window and return the predicted raw anomaly score of each vector.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two series (actual and prediction), and compute a "difference" series by applying the
        function ``diff_fn`` (default: absolute difference). The resulting series is then passed to the
        functions ``fit()`` and ``score()``, respectively.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs
        series. If set to `True`, the model will treat each series dimension independently by fitting a different
        PyODScorer model for each dimension. If set to `False`, the model concatenates the dimensions in
        each windows of length `W` and compute the score using only one underlying PyODScorer model.

        **Training with** ``fit()``:

        The input can be a series (univariate or multivariate) or multiple series. The series will be partitioned
        into equal size subsequences. Each subsequence has size `W * D` (features), where:

        - `W` is the size of the window given as a parameter `window`
        - `D` is the dimension of the series (`D` = 1 if univariate or if `component_wise` is set to `True`)

        For a series of length `N`, `(N - W + 1)` subsequences will be generated. The final array `X` passed to the
        underlying scorer has shape `(N - W + 1, W * D)`; or in other terms (number of samples, number of features).
        If a list of series is given of length L, each series `i` is partitioned, and all `X_i` are concatenated along
        the sample axis.

        The PyOD model will be fitted on the generated subsequences.

        If `component_wise` is set to `True`, the algorithm will be applied to each dimension independently. For each
        dimension, a PyOD model will be trained.

        **Computing score with** ``score()``:

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
        model
            The (fitted) PyOD BaseDetector model.
        window
            Size of the window used to create the subsequences of the series.
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

        raise_if_not(
            isinstance(model, BaseDetector),
            f"model must be a PyOD BaseDetector, found type: {type(model)}",
            logger,
        )
        self.model = model
        super().__init__(
            is_univariate=(not component_wise),
            window=window,
            window_agg=window_agg,
            diff_fn=diff_fn,
        )

    def __str__(self):
        return "PyODScorer (model {})".format(self.model.__str__().split("(")[0])

    def _model_score_method(self, model, data: np.ndarray) -> np.ndarray:
        """Wrapper around model inference method"""
        return model.decision_function(data)
