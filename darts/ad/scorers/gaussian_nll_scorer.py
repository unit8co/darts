"""
NLL Gaussian Scorer
-------------------

Gaussian negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Gaussian distribution estimated from the stochastic predictions.
"""

import numpy as np
from scipy.stats import norm

from darts.ad.scorers.scorers import NLLScorer


class GaussianNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GaussianNLLScorer"

    def _score_core_nllikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        mu = np.mean(probabilistic_estimations, axis=1)
        std = np.std(probabilistic_estimations, axis=1)
        return -norm.logpdf(deterministic_values, loc=mu, scale=std)
