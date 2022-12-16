"""
LaplaceNLLScorer
-----

Laplace negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Laplace distribution
<https://en.wikipedia.org/wiki/Laplace_distribution>`_.
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

        # TODO: vectorize

        median = np.median(probabilistic_estimations, axis=1)

        return -laplace.logpdf(deterministic_values, loc=median)
