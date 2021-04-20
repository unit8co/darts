"""
Missing Values Filler
---------------------
"""
from typing import Union, Sequence
from joblib import Parallel, delayed

from darts.dataprocessing.transformers import BaseDataTransformer
from darts.utils.missing_values import fill_missing_values

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils import _build_tqdm_iterator

logger = get_logger(__name__)


class MissingValuesFiller(BaseDataTransformer[TimeSeries]):
    def __init__(self,
                 fill: Union[str, float] = 'auto',
                 name: str = "MissingValuesFiller",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to fill missing values from time series

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

        super().__init__(name)
        self._fill = fill
        self._n_jobs = n_jobs
        self._verbose = verbose

    def transform(self,
                  data: Union[TimeSeries, Sequence[TimeSeries]],
                  **interpolate_kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().transform(data)

        if isinstance(data, TimeSeries):
            return fill_missing_values(data, self._fill, **interpolate_kwargs)
        else:
            def map_ts(ts):
                return fill_missing_values(ts, self._fill, **interpolate_kwargs)

            iterator = _build_tqdm_iterator(data,
                                            verbose=self._verbose,
                                            desc="Applying missing value filler")

            transformed_data = Parallel(n_jobs=self._n_jobs)(delayed(map_ts)(ts) for ts in iterator)

            return transformed_data
