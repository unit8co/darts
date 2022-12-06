"""
PoissonNLLScorer
-----

Poisson negative log-likelihood Scorer
Source of PDF function and parameters estimation (MLE):  `Poisson distribution
<https://www.statlect.com/fundamentals-of-statistics/Poisson-distribution-maximum-likelihood>`_.
"""

import math
from typing import Optional

import numpy as np

from darts.ad.scorers.scorers import NLLScorer


class PoissonNLLScorer(NLLScorer):
    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "PoissonNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error if values of deterministic_values are not (int and >=0). Required by the factorial function

        return [
            -np.log(np.exp(x1.mean()) * (x1.mean() ** x2) / math.factorial(x2))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
