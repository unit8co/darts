"""
CauchyNLLScorer
-----

Cauchy negative log-likelihood Scorer.
Source of PDF function and parameters estimation: `Cauchy distribution
<https://en.wikipedia.org/wiki/Cauchy_distribution>`_
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

        median = np.median(probabilistic_estimations, axis=1)
        return -cauchy.logpdf(deterministic_values, median)
