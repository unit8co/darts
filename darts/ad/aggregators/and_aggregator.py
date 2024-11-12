"""
AND Aggregator
--------------
"""

from collections.abc import Sequence

from darts import TimeSeries
from darts.ad.aggregators.aggregators import Aggregator
from darts.utils.utils import _parallel_apply


class AndAggregator(Aggregator):
    def __init__(self, n_jobs: int = 1) -> None:
        """AND Aggregator

        Aggregator that identifies a time step as anomalous if all the components are flagged as anomalous
        (logical AND).

        Parameters
        ----------
        n_jobs
            The number of jobs to run in parallel. Defaults to `1` (sequential). Setting the parameter to `-1` means
            using all the available processors.
        """
        super().__init__()
        self._n_jobs = n_jobs

    def __str__(self) -> str:
        return "AndAggregator"

    def _predict_core(
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        def _compononents_and(s: TimeSeries):
            return TimeSeries.from_times_and_values(
                times=s.time_index,
                values=(s.all_values(copy=False).sum(axis=1) >= s.width).astype(
                    s.dtype
                ),
                columns=["components_sum"],
            )

        return _parallel_apply(
            [(s,) for s in series],
            _compononents_and,
            n_jobs=1,
            fn_args=args,
            fn_kwargs=kwargs,
        )
