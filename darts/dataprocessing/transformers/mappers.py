"""
Mappers
-------
"""
import numpy as np
import pandas as pd

from typing import Callable, Union, Sequence, List

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger


logger = get_logger(__name__)


class Mapper(BaseDataTransformer):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "Mapper",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to apply a function to a (`Sequence` of) `TimeSeries` (similar to calling `series.map()`)

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
        name
            A specific name for the transformer
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """

        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self._fn = fn

    @staticmethod
    def ts_transform(series: TimeSeries, fn) -> TimeSeries:
        return series.map(fn)

    def transform(self,
                  series: Union[TimeSeries, Sequence[TimeSeries]],
                  *args, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        return super().transform(series, *args, fn=self._fn)


class InvertibleMapper(InvertibleDataTransformer):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 inverse_fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "InvertibleMapper",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to apply a function and its inverse to a (`Sequence` of) `TimeSeries` (similar to calling
        `series.map()`)

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
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """

        super().__init__(name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)
        self._fn = fn
        self._inverse_fn = inverse_fn

    @staticmethod
    def ts_transform(series: TimeSeries,
                     fn: Union[Callable[[np.number], np.number],
                               Callable[[pd.Timestamp, np.number], np.number]]) -> TimeSeries:
        return series.map(fn)

    @staticmethod
    def ts_inverse_transform(series: TimeSeries,
                             inverse_fn: Union[Callable[[np.number], np.number],
                                               Callable[[pd.Timestamp, np.number], np.number]]) -> TimeSeries:
        return series.map(inverse_fn)

    def transform(self,
                  series: Union[TimeSeries, Sequence[TimeSeries]],
                  *args, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        # adding the fn param
        return super().transform(series, self._fn, *args, **kwargs)

    def inverse_transform(self,
                          series: Union[TimeSeries, Sequence[TimeSeries]],
                          *args, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        # adding the inverse_fn param
        return super().inverse_transform(series, inverse_fn=self._inverse_fn, *args, **kwargs)
