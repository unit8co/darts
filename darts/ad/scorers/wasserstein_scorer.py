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

from typing import Optional, Sequence

import numpy as np
from scipy.stats import wasserstein_distance

from darts.ad.scorers.scorers import FittableAnomalyScorer
from darts.logging import get_logger, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class WassersteinScorer(FittableAnomalyScorer):
    def __init__(
        self, window: Optional[int] = None, component_wise: bool = False, diff_fn=None
    ) -> None:
        """
        TODO:
            - understand better the math behind the Wasserstein distance when the test distribution contains
            only one sample
            - check if there is an equivalent Wasserstein distance for d-D distributions (currently only accepts 1D)

        When calling ``fit(series)``, the series will be kept in memory and is considered as a subset of samples
        representing the training 1D distribution. When calling ``score(series)``, a moving window is applied on
        the series, which results in a set of vectors of size W, where W is the window size. The Wasserstein distance
        will be computed between the training distribution and each vector, resulting in an anomaly score.

        Alternatively, the scorer has the functions ``fit_from_prediction()`` and ``score_from_prediction()``.
        Both require two inputs and transform them into one series by applying the function ``diff_fn``
        (default: absolute difference). The resulting series will then be passed to the respective function
        ``fit()`` and ``score()``.

        `component_wise` is a boolean parameter indicating how the model should behave with multivariate inputs series.
        If set to True, the model will treat each series dimension independently. If set to False, the model will
        concatenate the dimension in the considered `window` W and compute the score.

        Training with ``fit()``:

        The input can be a series (univariate or multivariate) or a list of series. The element of a list will be
        concatenated to form one continuous array (by definition, each element have the same dimensions).

        If the series is of length N and dimension D, the array will be of length N*D. If `component_wise` is True,
        each dimension D is treated independently, and the data is stored in a list of size d. Each element is an array
        of length N.

        If a sequence of series is given of length L, each series of the sequence will be reduced to an array, and the
        L arrays will then be concatenated to form a continuous array of length L*D*N. If `component_wise` is True,
        the data is stored in a list of size D. Each element is an array of length L*N.

        The array will be kept in memory, representing the training data distribution. In practice, the series or list
        of series would represent residuals than can be considered independent and identically distributed (iid).

        Compute score with ``score()``:

        The input is a series (univariate or multivariate) or a list of series.

        - If the series is multivariate of dimension D:
            - if `component_wise` is set to False: it will return a univariate series (dimension=1). It represents
            the anomaly score of the entire series in the considered window at each timestamp.
            - if `component_wise` is set to True: it will return a multivariate series of dimension D. Each dimension
            represents the anomaly score of the corresponding dimension of the input.

        - If the series is univariate, it will return a univariate series regardless of the parameter
        `component_wise`.

        A window of size W (given as a parameter named `window`) is rolled on the series with a stride equal to 1.
        At each timestamp, the previous W values will be used to form a subset of W * D elements, with D being the
        dimension of the series. The subset values are considered to be observed from the same (empirical)
        distribution. The Wasserstein distance will be computed between this subset and the train distribution. The
        function will return a float number indicating how different these two distributions are. The output will be
        a series of dimension one and length N-W+1, with N being the length of the input series. Each value will
        represent how anomalous the sample of the D previous values is.

        If a list is given, a for loop will iterate through the list, and the function ``_score_core()`` will be
        applied independently on each series. The function will return an anomaly score for each series in the list.

        If `component_wise` is set to True, the algorithm will be applied to each dimension independently,
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

        if type(window) is int:
            if window > 0 and window < 10:
                logger.warning(
                    f"The window parameter WassersteinScorer is smaller than 10 (w={window}). \
                The value represents the window length rolled on the series given as input in \
                the ``score`` function. At each position, the w values will constitute a subset, \
                and the Wasserstein distance between the subset and the train distribution \
                will be computed. To better represent the constituted test distribution, \
                the window parameter should be larger than 10."
                )

        raise_if_not(
            type(component_wise) is bool,
            f"'component_wise' must be Boolean, found type: {type(component_wise)}",
        )
        self.component_wise = component_wise

        if component_wise:
            returns_UTS = False
        else:
            returns_UTS = True

        super().__init__(returns_UTS=returns_UTS, window=window, diff_fn=diff_fn)

    def __str__(self):
        return "WassersteinScorer"

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
    ):

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
