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
    """NLL Gaussian Scorer"""

    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GaussianNLLScorer"

    def _score_core_nllikelihood(
        self, vals: np.ndarray, pred_vals: np.ndarray
    ) -> np.ndarray:
        mu = np.mean(pred_vals, axis=1)
        std = np.std(pred_vals, axis=1)
        return -norm.logpdf(vals, loc=mu, scale=std)
