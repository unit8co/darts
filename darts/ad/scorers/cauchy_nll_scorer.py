"""
NLL Cauchy Scorer
-----------------

Cauchy distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Cauchy distribution estimated from the stochastic prediction.
"""

import numpy as np
from scipy.stats import cauchy

from darts.ad.scorers.scorers import NLLScorer


class CauchyNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "CauchyNLLScorer"

    def _score_core_nllikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        params = np.apply_along_axis(cauchy.fit, axis=1, arr=probabilistic_estimations)
        return -cauchy.logpdf(
            deterministic_values, loc=params[:, 0], scale=params[:, 1]
        )
