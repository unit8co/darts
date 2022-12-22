"""
NLL Exponential Scorer
----------------------

Exponential distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under an Exponential distribution estimated from the stochastic prediction.
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

        # This is the ML estimate for 1/lambda, which is what scipy expects as scale.
        mu = np.mean(probabilistic_estimations, axis=1)

        # This is ML estimate for the loc - see:
        # https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/stats/_continuous_distns.py#L1705
        loc = np.min(probabilistic_estimations, axis=1)

        return -expon.logpdf(deterministic_values, scale=mu, loc=loc)
