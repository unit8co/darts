"""
Difference Scorer
-----------------

This scorer simply computes the elementwise difference
between two series. If the two series are multivariate, it
returns a multivariate series.
"""

import numpy as np

from darts.ad.scorers.scorers import AnomalyScorer


class DifferenceScorer(AnomalyScorer):
    def __init__(self) -> None:
        """Difference Scorer"""
        super().__init__(is_univariate=False, window=1)

    def __str__(self):
        return "Difference"

    def _score_core_from_prediction(
        self,
        vals: np.ndarray,
        pred_vals: np.ndarray,
    ) -> np.ndarray:
        vals = self._extract_deterministic_values(vals, "series")
        pred_vals = self._extract_deterministic_values(pred_vals, "pred_series")
        return vals - pred_vals
