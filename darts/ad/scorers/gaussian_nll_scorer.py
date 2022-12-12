"""
GaussianNLLScorer
-----

Gaussian negative log-likelihood Scorer.
Source of PDF function and parameters estimation (MLE):  `Gaussian distribution
<https://programmathically.com/maximum-likelihood-estimation-for-gaussian-distributions/>`_.
"""

from typing import Optional

import numpy as np

from darts.ad.scorers.scorers import NLLScorer


class GaussianNLLScorer(NLLScorer):
    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GaussianNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error if std of deterministic_values is 0 (dividing by 0 otherwise)

        # TODO: vectorize

        return [
            -np.log(
                (1 / np.sqrt(2 * np.pi * x1.std() ** 2))
                * np.exp(-((x2 - x1.mean()) ** 2) / (2 * x1.std() ** 2))
            )
            if x1.std() > 0.01
            else -np.log(
                (1 / np.sqrt(2 * np.pi * 0.06**2))
                * np.exp(-((x2 - x1.mean()) ** 2) / (2 * 0.06**2))
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
