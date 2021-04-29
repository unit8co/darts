"""
Missing Values Filler
---------------------
"""
from typing import Union
from darts import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.missing_values import fill_missing_values
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer):
    def __init__(self,
                 fill: Union[str, float] = 'auto',
                 name: str = "MissingValuesFiller",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to fill missing values from a (sequence of) TimeSeries

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the `pandas.Dataframe.interpolate()` method.
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel (in case the transformer is handling a Sequence[TimeSeries]).
            Defaults to `1` (sequential). `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """
        raise_if_not(isinstance(fill, str) or isinstance(fill, float),
                     "`fill` should either be a string or a float",
                     logger)
        raise_if(isinstance(fill, str) and fill != 'auto',
                 "invalid string for `fill`: can only be set to 'auto'",
                 logger)

        def _mvf_ts_transform(series: TimeSeries, **kwargs) -> TimeSeries:
            return fill_missing_values(series, fill, **kwargs)

        super().__init__(ts_transform=_mvf_ts_transform, name=name, n_jobs=n_jobs, verbose=verbose)
