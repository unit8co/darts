"""
ExponentialNLLScorer
-----

Exponential negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Exponential distribution
<https://www.statlect.com/fundamentals-of-statistics/exponential-distribution-maximum-likelihood>`_.
"""

import numpy as np
from scipy.stats import expon

from darts.ad.scorers.scorers import NLLScorer


class ExponentialNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "ExponentialNLLScorer"

    def _score_core_nllikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: vectorize

        return [
            -expon.logpdf(x2, *expon.fit(x1))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
