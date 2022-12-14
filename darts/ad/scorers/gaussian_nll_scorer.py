"""
GaussianNLLScorer
-----

Gaussian negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Gaussian distribution
<https://programmathically.com/maximum-likelihood-estimation-for-gaussian-distributions/>`_.
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

        # TODO: vectorize

        return [
            -norm.logpdf(value, scale=std, loc=mean)
            for value, (mean, std) in zip(
                deterministic_values, list(map(norm.fit, probabilistic_estimations))
            )
        ]
