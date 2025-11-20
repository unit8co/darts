"""
NLL Exponential Scorer
----------------------

Exponential distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under an Exponential distribution estimated from the stochastic prediction.
"""

import numpy as np
from scipy.stats import expon

from darts.ad.scorers.scorers import NLLScorer


class ExponentialNLLScorer(NLLScorer):
    def __init__(self, window: int = 1) -> None:
        """NLL Exponential Scorer

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
        return "ExponentialNLLScorer"

    def _score_core_nllikelihood(
        self, vals: np.ndarray, pred_vals: np.ndarray
    ) -> np.ndarray:
        # This is the ML estimate for 1/lambda, which is what scipy expects as scale.
        mu = np.mean(pred_vals, axis=1)
        # This is ML estimate for the loc - see:
        # https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/stats/_continuous_distns.py#L1705
        loc = np.min(pred_vals, axis=1)
        return -expon.logpdf(vals, scale=mu, loc=loc)
