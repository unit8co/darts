"""
Norm Scorer
-----------

Norm anomaly score (of given order) [1]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Norm_(mathematics)
"""

import numpy as np

from darts.ad.scorers.scorers import NonFittableAnomalyScorer
from darts.logging import raise_if_not
from darts.timeseries import TimeSeries


class NormScorer(NonFittableAnomalyScorer):
    def __init__(self, ord=None, component_wise: bool = False) -> None:
        """
        Returns the elementwise norm of a given order between two series' values.

        If `component_wise` is False, the norm is computed between vectors
        made of the series' components (one norm per timestamp).

        If `component_wise` is True, for any `ord` this effectively amounts to computing the absolute
        value of the difference.

        The scoring function expects two series.

        If the two series are multivariate of width `w`:

        * if `component_wise` is set to False: it returns a univariate series (width=1).
        * if `component_wise` is set to True: it returns a multivariate series of width `w`.

        If the two series are univariate, it returns a univariate series regardless of the parameter
        `component_wise`.

        Parameters
        ----------
        ord
            Order of the norm. Options are listed under 'Notes' at:
            <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>.
            Default: None
        component_wise
            Whether to compare components of the two series in isolation (True), or jointly (False).
            Default: False
        """

        raise_if_not(
            type(component_wise) is bool,
            f"`component_wise` must be Boolean, found type: {type(component_wise)}.",
        )

        self.ord = ord
        self.component_wise = component_wise
        super().__init__(univariate_scorer=(not component_wise), window=1)

    def __str__(self):
        return f"Norm (ord={self.ord})"

    def _score_core_from_prediction(
        self,
        actual_series: TimeSeries,
        pred_series: TimeSeries,
    ) -> TimeSeries:

        self._assert_deterministic(actual_series, "actual_series")
        self._assert_deterministic(pred_series, "pred_series")

        diff = actual_series - pred_series

        if self.component_wise:
            return diff.map(lambda x: np.abs(x))

        else:
            diff_np = diff.all_values(copy=False)

            return TimeSeries.from_times_and_values(
                diff.time_index, np.linalg.norm(diff_np, ord=self.ord, axis=1)
            )
