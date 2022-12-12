"""
ExponentialNLLScorer
-----

Exponential negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Exponential distribution
<https://www.statlect.com/fundamentals-of-statistics/exponential-distribution-maximum-likelihood>`_.
"""

from typing import Optional

import numpy as np

from darts.ad.scorers.scorers import NLLScorer


class ExponentialNLLScorer(NLLScorer):
    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "ExponentialNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: vectorize

        return [
            -np.log(x1.mean() * np.exp(-x1.mean() * x2))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
