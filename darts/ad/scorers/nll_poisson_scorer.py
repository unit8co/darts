"""
NLL Poisson Scorer
------------------

Poisson distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Poisson distribution estimated from the stochastic prediction.
"""

import numpy as np
from scipy.stats import poisson

from darts.ad.scorers.scorers import NLLScorer


class PoissonNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "PoissonNLLScorer"

    def _score_core_nllikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        mu = np.mean(probabilistic_estimations, axis=1)
        return -poisson.logpmf(deterministic_values, mu=mu)
