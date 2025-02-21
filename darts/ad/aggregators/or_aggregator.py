"""
OR Aggregator
-------------
"""

from collections.abc import Sequence

from darts import TimeSeries
from darts.ad.aggregators.aggregators import Aggregator
from darts.utils.utils import _parallel_apply


class OrAggregator(Aggregator):
    def __init__(self, n_jobs: int = 1) -> None:
        """OR Aggregator

        Aggregator that identifies a time step as anomalous if any of the components
        is flagged as anomalous (logical OR).

        Parameters
        ----------
        n_jobs
            The number of jobs to run in parallel. Defaults to `1` (sequential). Setting the parameter to `-1` means
            using all the available processors.
        """
        super().__init__()

        self._n_jobs = n_jobs

    def __str__(self) -> str:
        return "OrAggregator"

    def _predict_core(
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        def _compononents_or(s: TimeSeries):
            return TimeSeries.from_times_and_values(
                times=s.time_index,
                values=(s.all_values(copy=False).sum(axis=1) > 0).astype(s.dtype),
                columns=["components_sum"],
            )

        return _parallel_apply(
            [(s,) for s in series],
            _compononents_or,
            n_jobs=1,
            fn_args=args,
            fn_kwargs=kwargs,
        )
