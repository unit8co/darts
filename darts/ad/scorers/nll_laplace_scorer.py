"""
NLL Laplace Scorer
------------------

Laplace distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Laplace distribution estimated from the stochastic prediction.
"""

import numpy as np
from scipy.stats import laplace

from darts.ad.scorers.scorers import NLLScorer


class LaplaceNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "LaplaceNLLScorer"

    def _score_core_nllikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # ML estimate for the Laplace loc
        loc = np.median(probabilistic_estimations, axis=1)

        # ML estimate for the Laplace scale
        # see: https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy
        # /stats/_continuous_distns.py#L4846
        scale = (
            np.sum(np.abs(probabilistic_estimations.T - loc), axis=0).T
            / probabilistic_estimations.shape[1]
        )

        return -laplace.logpdf(deterministic_values, loc=loc, scale=scale)
