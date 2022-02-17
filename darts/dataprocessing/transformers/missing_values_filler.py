"""
Missing Values Filler
---------------------
"""

from typing import List, Sequence, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils.missing_values import fill_missing_values

from .base_data_transformer import BaseDataTransformer

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer):
    def __init__(
        self,
        fill: Union[str, float] = "auto",
        name: str = "MissingValuesFiller",
        n_jobs: int = 1,
        verbose: bool = False,
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
        >>> values = np.arange(start=0, stop=1, step=0.1)
        >>> values[5:8] = np.nan
        >>> series = TimeSeries.from_values(values)
        >>> transformer = MissingValuesFiller()
        >>> series_filled = transformer.transform(series)
        >>> print(series_filled)
        <TimeSeries (DataArray) (time: 10, component: 1, sample: 1)>
        array([[[0. ]],
            [[0.1]],
            [[0.2]],
            [[0.3]],
            [[0.4]],
            [[0.5]],
            [[0.6]],
            [[0.7]],
            [[0.8]],
            [[0.9]]])
        Coordinates:
        * time       (time) int64 0 1 2 3 4 5 6 7 8 9
        * component  (component) object '0'
        Dimensions without coordinates: sample
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

        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self._fill = fill

    @staticmethod
    def ts_transform(
        series: TimeSeries, fill: Union[str, float], **kwargs
    ) -> TimeSeries:
        return fill_missing_values(series, fill, **kwargs)

    def transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        # adding the fill param
        return super().transform(series, self._fill, *args, **kwargs)
