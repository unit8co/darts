"""
Norm Scorer
-----------

Norm anomaly score (of given order) [1]_.

References
----------
.. [1] https://en.wikipedia.org/wiki/Norm_(mathematics)
"""

from typing import Union

import numpy as np
import pandas as pd

from darts.ad.scorers.scorers import AnomalyScorer
from darts.timeseries import TimeSeries


class NormScorer(AnomalyScorer):
    """Norm Scorer"""

    def __init__(self, ord=None, component_wise: bool = False) -> None:
        """
        Returns the element-wise norm of a given order between two series' values.

        If `component_wise` is `False`, the norm is computed between vectors
        made of the series' components (one norm per timestamp).

        If `component_wise` is `True`, for any `ord` this effectively amounts to computing the absolute
        value of the difference.

        The scoring function expects two series.

        If the two series are multivariate of width `w`:

        - if `component_wise` is set to `False`: it returns a univariate series (width=1).
        - if `component_wise` is set to `True`: it returns a multivariate series of width `w`.

        If the two series are univariate, it returns a univariate series regardless of the parameter
        `component_wise`.

        Parameters
        ----------
        ord
            Order of the norm. Options are listed under 'Notes' at:
            <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>.
            Default: `None`
        component_wise
            Whether to compare components of the two series in isolation (`True`), or jointly (`False`).
            Default: `False`
        """
        self.ord = ord
        super().__init__(is_univariate=(not component_wise), window=1)

    def __str__(self):
        return f"Norm (ord={self.ord})"

    def _score_core_from_prediction(
        self,
        actual_vals: np.ndarray,
        pred_vals: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        actual_vals = self._extract_deterministic_values(actual_vals, "actual_series")
        pred_vals = self._extract_deterministic_values(pred_vals, "pred_series")
        diff = actual_vals - pred_vals
        if not self.is_univariate:
            diff = np.abs(diff)
        else:
            diff = np.linalg.norm(diff, ord=self.ord, axis=1)
        return TimeSeries.from_times_and_values(values=diff, times=time_index)
