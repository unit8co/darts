"""
Or Aggregator
-------------

Aggregator that identifies a time point as anomalous as long as it is
included in one of the input anomaly lists.
"""


from typing import Sequence

from darts import TimeSeries
from darts.ad.aggregators.aggregators import NonFittableAggregator


class OrAggregator(NonFittableAggregator):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return "OrAggregator"

    def _predict_core(self, list_series: Sequence[TimeSeries]) -> Sequence[TimeSeries]:

        return [
            series.sum(axis=1).map(lambda x: (x > 0).astype(float))
            for series in list_series
        ]
