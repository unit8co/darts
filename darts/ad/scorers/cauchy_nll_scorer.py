"""
CauchyNLLScorer
-----

Cauchy negative log-likelihood Scorer.
Source of PDF function and parameters estimation: `Cauchy distribution
<https://en.wikipedia.org/wiki/Cauchy_distribution>`_
"""

from typing import Optional

import numpy as np

from darts.ad.scorers.scorers import NLLScorer


class CauchyNLLScorer(NLLScorer):
    """
    For computational reasons, we opted for the simple method to estimate the parameters:
        - location parameter (x_0): median of the samples
        - scale parameter (gamma): half the sample interquartile range (Q3-Q1)/2
    """

    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "CauchyNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: raise error when gamma is equal to 0 -> interquartile is equal to 0

        # TODO: vectorize

        return [
            -np.log(
                (2 / (np.pi * np.subtract(*np.percentile(x1, [75, 25]))))
                * (
                    (np.subtract(*np.percentile(x1, [75, 25])) / 2) ** 2
                    / (
                        ((x2 - np.median(x1)) ** 2)
                        + ((np.subtract(*np.percentile(x1, [75, 25])) / 2) ** 2)
                    )
                )
            )
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
