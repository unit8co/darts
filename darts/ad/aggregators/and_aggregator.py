"""
AndAggregator

Aggregator that identifies a time point as anomalous only if it is
included in all the input anomaly lists.
"""

import numpy as np

from darts.ad.aggregators.aggregators import NonFittableAggregator


class AndAggregator(NonFittableAggregator):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "AndAggregator"

    def _predict_core(self, np_series: np.ndarray, width: int) -> np.ndarray:

        return [0 if 0 in timestamp else 1 for timestamp in np_series]
