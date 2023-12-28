"""
OR Aggregator
-------------

Aggregator that identifies a time step as anomalous if any of the components
is flagged as anomalous (logical OR).
"""


from typing import Sequence

from darts import TimeSeries
from darts.ad.aggregators.aggregators import Aggregator
from darts.utils.utils import _parallel_apply


class OrAggregator(Aggregator):
    def __init__(self, n_jobs: int = 1) -> None:
        super().__init__()

        self._n_jobs = n_jobs

    def __str__(self) -> str:
        return "OrAggregator"

    def _predict_core(
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        def _compononents_or(s: TimeSeries, _):
            return s.sum(axis=1).map(lambda x: (x > 0).astype(s.dtype))

        return _parallel_apply(
            zip(series, [None] * len(series)),
            _compononents_or,
            n_jobs=1,
            fn_args=args,
            fn_kwargs=kwargs,
        )
