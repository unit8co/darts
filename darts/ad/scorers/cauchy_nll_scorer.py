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

        # TODO: vectorize

        return [
            -cauchy.logpdf(x2, *cauchy.fit(x1))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
