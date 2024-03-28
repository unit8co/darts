"""
AND Aggregator
--------------

Aggregator that identifies a time step as anomalous if all the components
are flagged as anomalous (logical AND).
"""

from typing import Sequence

from darts import TimeSeries
from darts.ad.aggregators.aggregators import Aggregator
from darts.utils.utils import _parallel_apply


class AndAggregator(Aggregator):
    def __init__(self, n_jobs: int = 1) -> None:
        super().__init__()

        self._n_jobs = n_jobs

    def __str__(self) -> str:
        return "AndAggregator"

    def _predict_core(
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        def _compononents_and(s: TimeSeries):
            return s.sum(axis=1).map(lambda x: (x >= s.width).astype(s.dtype))

        return _parallel_apply(
            [(s,) for s in series],
            _compononents_and,
            n_jobs=1,
            fn_args=args,
            fn_kwargs=kwargs,
        )
