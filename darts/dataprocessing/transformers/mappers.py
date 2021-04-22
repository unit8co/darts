"""
Mappers
-------
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from typing import Callable, Union, Sequence

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger
from darts.utils import _build_tqdm_iterator

logger = get_logger(__name__)


class Mapper(BaseDataTransformer[TimeSeries]):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "Mapper",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to apply a function to a (sequence of) time series (similar to calling `series.map()`)

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        super().__init__(name, n_jobs=n_jobs, verbose=verbose)
        self._fn = fn

    def transform(self,
                  data: Union[TimeSeries, Sequence[TimeSeries]],
                  *args,
                  **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().transform(data)

        if isinstance(data, TimeSeries):
            return data.map(self._fn)
        else:
            def map_ts(ts):
                return ts.map(self._fn)

            iterator = _build_tqdm_iterator(data,
                                            verbose=self._verbose,
                                            desc="Applying {}".format(self.name))

            transformed_data = Parallel(n_jobs=self._n_jobs)(delayed(map_ts)(ts) for ts in iterator)

            return transformed_data


class InvertibleMapper(InvertibleDataTransformer[TimeSeries]):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 inverse_fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "InvertibleMapper",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to apply a function and its inverse to a time series (similar to calling `series.map()`)

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
        inverse_fn
            Similarly to `fn`, either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
            `inverse_fn` should be such that `inverse_fn(fn(x)) == x`
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print progress
        """
        super().__init__(name)
        self._fn = fn
        self._inverse_fn = inverse_fn
        self._n_jobs = n_jobs
        self._verbose = verbose

    def transform(self,
                  data: Union[TimeSeries, Sequence[TimeSeries]],
                  *args,
                  **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().transform(data)

        if isinstance(data, TimeSeries):
            return data.map(self._fn)
        else:
            def map_ts(series):
                return series.map(self._fn)

            iterator = _build_tqdm_iterator(data,
                                            verbose=self._verbose,
                                            desc="{}: tranform".format(self.name))

            transformed_data = Parallel(n_jobs=self._n_jobs)(delayed(map_ts)(ts) for ts in iterator)

            return transformed_data

    def inverse_transform(self,
                          data: Union[TimeSeries, Sequence[TimeSeries]],
                          *args,
                          **kwargs) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().inverse_transform(data, *args, *kwargs)

        if isinstance(data, TimeSeries):
            return data.map(self._inverse_fn)
        else:
            def inverse_map_ts(ts):
                return ts.map(self._inverse_fn)

            iterator = _build_tqdm_iterator(data,
                                            verbose=self._verbose,
                                            desc="{}: inverse".format(self.name))

            transformed_data = Parallel(n_jobs=self._n_jobs)(delayed(inverse_map_ts)(ts) for ts in iterator)

            return transformed_data
