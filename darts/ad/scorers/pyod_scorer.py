"""
PyODScorer
-----

This scorer can wrap around detection algorithms of PyOD.
`PyOD https://pyod.readthedocs.io/en/latest/#`_.
"""

from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pyod.models.base import BaseDetector

from darts.ad.scorers.scorers import FittableAnomalyScorer
from darts.logging import get_logger, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class PyODScorer(FittableAnomalyScorer):
    def __init__(
        self,
        model: BaseDetector,
        window: int = 1,
        component_wise: bool = False,
        diff_fn="abs_diff",
    ) -> None:
        """
        When calling ``fit(series)``, a moving window is applied, which results in a set of vectors of size `W`,
        where `W` is the window size. The PyODScorer model is trained on these vectors. The ``score(series)``
        function will apply the same moving window and return the predicted raw anomaly score of each vector.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two series (actual and prediction), and compute a "difference" series by applying the
        function ``diff_fn`` (default: absolute difference). The resulting series is then passed to the
        functions ``fit()`` and ``score()``, respectively.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs
        series. If set to True, the model will treat each series dimension independently by fitting a different
        PyODScorer model for each dimension. If set to False, the model concatenates the dimensions in
        each windows of length `W` and compute the score using only one underlying PyODScorer model.

        **Training with** ``fit()``:

        The input can be a series (univariate or multivariate) or multiple series. The series will be partitioned
        into equal size subsequences. The subsequence will be of size `W` * `D`, with:

        * `W` being the size of the window given as a parameter `window`
        * `D` being the dimension of the series (`D` = 1 if univariate or if `component_wise` is set to True)

        For a series of length `N`, (`N` - `W` + 1)/W subsequences will be generated. If a list of series is given
        of length L, each series will be partitioned into subsequences, and the results will be concatenated into
        an array of length L * number of subsequences of each series.

        The PyOD model will be fitted on the generated subsequences.

        If `component_wise` is set to True, the algorithm will be applied to each dimension independently. For each
        dimension, a PyOD model will be trained.

        **Computing score with** ``score()``:

        The input can be a series (univariate or multivariate) or a sequence of series. The given series must have the
        same dimension `D` as the data used to train the PyOD model.

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
        model
            The (fitted) PyOD BaseDetector model.
        window
            Size of the window used to create the subsequences of the series.
        diff_fn
            Optionally, reduced function to use if two series are given. It will transform the two series into one.
            This allows the KMeansScorer to apply PyODScorer on the original series or on its residuals (difference
            between the prediction and the original series).
            Must be one of "abs_diff" and "diff" (defined in ``_diff_series()``).
            Default: "abs_diff"
        component_wise
            Boolean value indicating if the score needs to be computed for each component independently (True)
            or by concatenating the component in the considered window to compute one score (False).
            Default: False
        """

        raise_if_not(
            isinstance(model, BaseDetector),
            f"model must be a PyOD BaseDetector, found type: {type(model)}",
        )
        self.model = model

        raise_if_not(
            type(component_wise) is bool,
            f"Parameter `component_wise` must be Boolean, found type: {type(component_wise)}.",
        )
        self.component_wise = component_wise

        super().__init__(
            univariate_scorer=(not component_wise), window=window, diff_fn=diff_fn
        )

    def __str__(self):
        return "PyODScorer (model {})".format(self.model.__str__().split("(")[0])

    def _fit_core(self, list_series: Sequence[TimeSeries]):

        list_np_series = [series.all_values(copy=False) for series in list_series]

        # TODO: can we factorize code in common bteween PyODScorer and KMeansScorer?

        if not self.component_wise:
            self.model.fit(
                np.concatenate(
                    [
                        sliding_window_view(ar, window_shape=self.window, axis=0)
                        .transpose(0, 3, 1, 2)
                        .reshape(-1, self.window * len(ar[0]))
                        for ar in list_np_series
                    ]
                )
            )
        else:
            models = []
            for component_idx in range(self.width_trained_on):

                model_width = self.model
                model_width.fit(
                    np.concatenate(
                        [
                            sliding_window_view(
                                ar[:, component_idx], window_shape=self.window, axis=0
                            )
                            .transpose(0, 2, 1)
                            .reshape(-1, self.window)
                            for ar in list_np_series
                        ]
                    )
                )
                models.append(model_width)
            self.models = models

    def _score_core(self, series: TimeSeries) -> TimeSeries:

        raise_if_not(
            self.width_trained_on == series.width,
            "Input must have the same number of components as the data used for training"
            + " the PyODScorer model {},".format(self.model.__str__().split("(")[0])
            + f" found number of components equal to {series.width} and expected "
            + f"{self.width_trained_on}.",
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:

            np_anomaly_score.append(
                self.model.decision_function(
                    sliding_window_view(np_series, window_shape=self.window, axis=0)
                    .transpose(0, 3, 1, 2)
                    .reshape(-1, self.window * series.width)
                )
            )
        else:

            for component_idx in range(self.width_trained_on):
                score = self.models[component_idx].decision_function(
                    sliding_window_view(
                        np_series[:, component_idx],
                        window_shape=self.window,
                        axis=0,
                    )
                    .transpose(0, 2, 1)
                    .reshape(-1, self.window)
                )

                np_anomaly_score.append(score)

        return TimeSeries.from_times_and_values(
            series.time_index[self.window - 1 :], list(zip(*np_anomaly_score))
        )
