"""
Mappers
-------
"""
import numpy as np
import pandas as pd

from typing import Callable, Union, Sequence, Iterator

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

        def mapper_ts_transform(series: TimeSeries) -> TimeSeries:
            return series.map(fn)

        super().__init__(ts_transform=mapper_ts_transform, name=name, n_jobs=n_jobs, verbose=verbose)

    def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator:
        return zip(series)


class InvertibleMapper(InvertibleDataTransformer):
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

        def mapper_ts_transform(series: TimeSeries) -> TimeSeries:
            return series.map(fn)

        def mapper_ts_inverse_transform(series: TimeSeries) -> TimeSeries:
            return series.map(inverse_fn)

        super().__init__(ts_transform=mapper_ts_transform,
                         ts_inverse_transform=mapper_ts_inverse_transform,
                         name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)
