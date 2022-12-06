"""
GammaNLLScorer
-----

Gamma negative log-likelihood Scorer.
The implementations is wrapped around `scipy.stats
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>`_.
"""

from typing import Optional

import numpy as np
from scipy.stats import gamma

from darts.ad.scorers.scorers import NLLScorer


class GammaNLLScorer(NLLScorer):
    def __init__(self, window: Optional[int] = None) -> None:
        super().__init__(window=window)

    def __str__(self):
        return "GammaNLLScorer"

    def _score_core_NLlikelihood(
        self,
        deterministic_values: np.ndarray,
        probabilistic_estimations: np.ndarray,
    ) -> np.ndarray:

        # TODO: takes a very long time to compute, understand why

        return [
            -gamma.logpdf(x2, *gamma.fit(x1))
            for (x1, x2) in zip(probabilistic_estimations, deterministic_values)
        ]
