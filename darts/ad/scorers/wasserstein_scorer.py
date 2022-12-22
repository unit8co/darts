"""
WassersteinScorer
-----

Wasserstein Scorer (distance function defined between probability distributions) [1]_.
The implementations is wrapped around `scipy.stats
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>`_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Wasserstein_metric
"""

from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import wasserstein_distance

from darts.ad.scorers.scorers import FittableAnomalyScorer
from darts.logging import get_logger, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class WassersteinScorer(FittableAnomalyScorer):
    def __init__(
        self,
        window: int = 10,
        component_wise: bool = False,
        diff_fn="abs_diff",
    ) -> None:
        """
        When calling ``fit(series)``, the series will be kept in memory and is considered as a subset of samples
        representing the training 1-D distribution. When calling ``score(series)``, a moving window is applied on
        the series, which results in a set of vectors of size `W`, where `W` is the window size.
        The Wasserstein distance will be computed between the training distribution and each vector,
        resulting in an anomaly score.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two inputs and transform them into one series by applying the function ``diff_fn``
        (default: absolute difference). The resulting series will then be passed to the respective function
        ``fit()`` and ``score()``.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs series.
        If set to True, the model will treat each series dimension independently. If set to False, the model will
        concatenate the dimension in the considered `window` W and compute the score.

        **Training with** ``fit()``:

        The input can be a series (univariate or multivariate) or multiple series.

        In case of a single series of length `N` and dimension `D`, the components are concatenated in an array of
        length `N` * `D` (if `component_wise` is False) or `D` arrays of length `N` (if `component_wise` is True).

        If a sequence of series is given of length `L`, their underlying arrays will be concatenated to
        form a continuous array of length `L` * `D` * `N` (if `component_wise` is False) or `D` arrays of length
        `L` * `N` (if `component_wise` is True).

        The arrays will be kept in memory, representing the training data distribution.
        In practice, the series or list of series can for instance represent residuals than can be
        considered independent and identically distributed (iid).

        **Computing score with** ``score()``:

        The input is a series (univariate or multivariate) or a sequence of series.

        For each series, if the series is multivariate of dimension `D`:

        * if `component_wise` is set to False: it will return a univariate series representing
          the anomaly score of the entire series in the considered window at each timestamp.
        * if `component_wise` is set to True: it will return a multivariate series of dimension D. Each dimension
          represents the anomaly score of the corresponding dimension of the input.

        If the series is univariate, it will return a univariate series regardless of the parameter
        `component_wise`.

        A window of size `W` (given as a parameter named `window`) is rolled on the series (with a stride of 1).
        At each timestamp, the previous `W` values are be used to form a subset of `W` * `D` elements, with `D`
        being the dimension of the series. The subset values are considered to be observed from the same (empirical)
        distribution. The Wasserstein distance will be computed between this subset and the train distribution. The
        function will return a scalar indicating how different these two distributions are. The output will be
        a series of dimension one and length `N` - `W`+1, with `N` being the length of the input series. Each value will
        represent how anomalous the sample of the `D` previous values is.

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
            Boolean value indicating if the score needs to be computed for each component independently (True)
            or by concatenating the component in the considered window to compute one score (False).
            Default: False

        """

        # TODO:
        #     - understand better the math behind the Wasserstein distance when the test distribution contains
        #     only one sample
        #     - check if there is an equivalent Wasserstein distance for d-D distributions (currently only accepts 1D)

        if type(window) is int:
            if window > 0 and window < 10:
                logger.warning(
                    f"The `window` parameter WassersteinScorer is smaller than 10 (w={window})."
                    + " The value represents the window length rolled on the series given as"
                    + " input in the ``score`` function. At each position, the w values will"
                    + " constitute a subset, and the Wasserstein distance between the subset"
                    + " and the train distribution will be computed. To better represent the"
                    + " constituted test distribution, the window parameter should be larger"
                    + " than 10."
                )

        raise_if_not(
            type(component_wise) is bool,
            f"Parameter `component_wise` must be Boolean, found type: {type(component_wise)}.",
        )
        self.component_wise = component_wise

        super().__init__(
            univariate_scorer=(not component_wise), window=window, diff_fn=diff_fn
        )

    def __str__(self):
        return "WassersteinScorer"

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
    ):
        self.training_data = np.concatenate(
            [s.all_values(copy=False) for s in list_series]
        ).squeeze(-1)

        if not self.component_wise:
            self.training_data = self.training_data.flatten()

    def _score_core(self, series: TimeSeries) -> TimeSeries:
        raise_if_not(
            self.width_trained_on == series.width,
            "Input must have the same number of components as the data used for"
            + " training the Wasserstein model, found number of components equal"
            + f" to {series.width} and expected {self.width_trained_on}.",
        )

        np_series = series.all_values(copy=False)
        np_anomaly_score = []

        if not self.component_wise:
            np_anomaly_score = [
                wasserstein_distance(self.training_data, window_samples)
                for window_samples in sliding_window_view(
                    np_series, window_shape=self.window, axis=0
                )
                .transpose(0, 3, 1, 2)
                .reshape(-1, self.window * series.width)
            ]

            return TimeSeries.from_times_and_values(
                series.time_index[self.window - 1 :], np_anomaly_score
            )

        else:
            for component_idx in range(self.width_trained_on):
                score = [
                    wasserstein_distance(
                        self.training_data[component_idx, :], window_samples
                    )
                    for window_samples in sliding_window_view(
                        np_series[:, component_idx],
                        window_shape=self.window,
                        axis=0,
                    )
                    .transpose(0, 2, 1)
                    .reshape(-1, self.window)
                ]

                np_anomaly_score.append(score)

            return TimeSeries.from_times_and_values(
                series.time_index[self.window - 1 :], list(zip(*np_anomaly_score))
            )
