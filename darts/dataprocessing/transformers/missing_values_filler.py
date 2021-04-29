"""
Missing Values Filler
---------------------
"""
from typing import Union, Sequence, Iterator
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.missing_values import fill_missing_values

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer):
    def __init__(self,
                 fill: Union[str, float] = 'auto',
                 name: str = "MissingValuesFiller",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to fill missing values from a (sequence of) time series

        Parameters
        ----------
        fill
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the `pandas.Dataframe.interpolate()` method.
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        raise_if_not(isinstance(fill, str) or isinstance(fill, float),
                     "`fill` should either be a string or a float",
                     logger)
        raise_if(isinstance(fill, str) and fill != 'auto',
                 "invalid string for `fill`: can only be set to 'auto'",
                 logger)

        def mvf_ts_transform(series, **kwargs):
            return fill_missing_values(series, self._fill, **kwargs)

        super().__init__(ts_transform=mvf_ts_transform, name=name, n_jobs=n_jobs, verbose=verbose)
        self._fill = fill
