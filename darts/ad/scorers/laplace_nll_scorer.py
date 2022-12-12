"""
LaplaceNLLScorer
-----

Laplace negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Laplace distribution
<https://en.wikipedia.org/wiki/Laplace_distribution>`_.
"""

from typing import Optional

import numpy as np

from darts.ad.scorers.scorers import NLLScorer


class LaplaceNLLScorer(NLLScorer):
    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "LaplaceNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error when all values are equal to the median -> divide by 0

        # TODO: vectorize

        return [
            -np.log(
                (1 / (2 * np.abs(x1 - np.median(x1)).mean()))
                * np.exp(
                    -(np.abs(x2 - np.median(x1)) / np.abs(x1 - np.median(x1)).mean())
                )
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
