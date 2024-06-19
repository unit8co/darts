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
        """NLL Laplace Scorer

        Parameters
        ----------
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        """
        super().__init__(window=window)

    def __str__(self):
        return "LaplaceNLLScorer"

    def _score_core_nllikelihood(
        self, vals: np.ndarray, pred_vals: np.ndarray
    ) -> np.ndarray:
        # ML estimate for the Laplace loc
        loc = np.median(pred_vals, axis=1)
        # ML estimate for the Laplace scale
        # see: https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy
        # /stats/_continuous_distns.py#L4846
        scale = np.sum(np.abs(pred_vals.T - loc), axis=0).T / pred_vals.shape[1]
        return -laplace.logpdf(vals, loc=loc, scale=scale)
