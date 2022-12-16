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

        mu = np.mean(probabilistic_estimations, axis=1)
        return -expon.logpdf(deterministic_values, scale=mu)
