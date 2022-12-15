"""
Or Aggregator
-------------

Aggregator that identifies a time point as anomalous as long as it is
included in one of the input anomaly lists.
"""

import numpy as np

from darts.ad.aggregators.aggregators import NonFittableAggregator


class OrAggregator(NonFittableAggregator):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "OrAggregator"

    def _predict_core(self, np_series: np.ndarray, width: int) -> np.ndarray:
        # TODO vectorize
        return [1 if timestamp.sum() >= 1 else 0 for timestamp in np_series]
