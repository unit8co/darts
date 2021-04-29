"""
Mappers
-------
"""
import numpy as np
import pandas as pd

from typing import Callable, Union

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
        Data transformer to apply a function to a (sequence of) TimeSeries (similar to calling `series.map()`)

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
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

        def _mapper_ts_transform(series: TimeSeries) -> TimeSeries:
            return series.map(fn)

        super().__init__(ts_transform=_mapper_ts_transform, name=name, n_jobs=n_jobs, verbose=verbose)


class InvertibleMapper(InvertibleDataTransformer):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 inverse_fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "InvertibleMapper",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Data transformer to apply a function and its inverse to a (sequence of) TimeSeries (similar to calling
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
            The number of jobs to run in parallel (in case the transformer is handling a Sequence[TimeSeries]).
            Defaults to `1` (sequential). `-1` means using all the available processors.
            Note: for small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """

        def _mapper_ts_transform(series: TimeSeries) -> TimeSeries:
            return series.map(fn)

        def _mapper_ts_inverse_transform(series: TimeSeries) -> TimeSeries:
            return series.map(inverse_fn)

        super().__init__(ts_transform=_mapper_ts_transform,
                         ts_inverse_transform=_mapper_ts_inverse_transform,
                         name=name,
                         n_jobs=n_jobs,
                         verbose=verbose)
