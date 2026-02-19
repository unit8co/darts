"""
Missing Values Filler
---------------------
"""

from collections.abc import Mapping
from typing import Any

from darts import TimeSeries
from darts.dataprocessing.transformers.base_data_transformer import BaseDataTransformer
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils.missing_values import fill_missing_values

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer):
    def __init__(
        self,
        fill: str | float = "auto",
        name: str = "MissingValuesFiller",
        n_jobs: int = 1,
        verbose: bool = False,
        columns: str | list[str] | None = None,
    ):
        """Data transformer to fill missing values from a (sequence of) deterministic ``TimeSeries``.

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the :func:`pd.Dataframe.interpolate()` method.
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.dataprocessing.transformers import MissingValuesFiller
        >>> values = np.arange(start=0, stop=1, step=0.25)
        >>> values[1:3] = np.nan
        >>> series = TimeSeries.from_values(values)
        >>> transformer = MissingValuesFiller()
        >>> series_filled = transformer.transform(series)
        >>> print(series_filled.values())
        [[0.  ]
         [0.25]
         [0.5 ]
         [0.75]]
        """
        raise_if_not(
            isinstance(fill, str) or isinstance(fill, float),
            "`fill` should either be a string or a float",
            logger,
        )
        raise_if(
            isinstance(fill, str) and fill != "auto",
            "invalid string for `fill`: can only be set to 'auto'",
            logger,
        )
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self._fill = fill
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, columns=columns)

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        return fill_missing_values(series, params["fixed"]["_fill"], **kwargs)
