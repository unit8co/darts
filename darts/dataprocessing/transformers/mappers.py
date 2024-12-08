"""
Mapper and InvertibleMapper
---------------------------
"""

from collections.abc import Mapping
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from darts.dataprocessing.transformers.base_data_transformer import BaseDataTransformer
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.logging import get_logger
from darts.timeseries import TimeSeries

logger = get_logger(__name__)

MapperFn = Union[
    Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]
]


class Mapper(BaseDataTransformer):
    def __init__(
        self,
        fn: Union[
            Callable[[np.number], np.number],
            Callable[[pd.Timestamp, np.number], np.number],
        ],
        name: str = "Mapper",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        Data transformer to apply a custom function to a (sequence of) ``TimeSeries``
        (similar to calling :func:`TimeSeries.map()` on each series).

        The mapper takes care of parallelizing the operations on multiple series over
        multiple processors.

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`.
        name
            A specific name for the transformer.
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
        >>> from darts.dataprocessing.transformers import InvertibleMapper
        >>> series = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3]))
        >>> transformer = InvertibleMapper(np.log10, lambda x: 10**x)
        >>> series_transformed = transformer.transform(series)
        >>> print(series_transformed)
        <TimeSeries (DataArray) (time: 4, component: 1, sample: 1)>
        array([[[0.]],
            [[1.]],
            [[2.]],
            [[3.]]])
        Coordinates:
        * time       (time) int64 0 1 2 3
        * component  (component) <U1 '0'
        Dimensions without coordinates: sample
        """
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self._fn = fn
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)

    @staticmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        return series.map(params["fixed"]["_fn"])


class InvertibleMapper(InvertibleDataTransformer):
    def __init__(
        self,
        fn: Union[
            Callable[[np.number], np.number],
            Callable[[pd.Timestamp, np.number], np.number],
        ],
        inverse_fn: Union[
            Callable[[np.number], np.number],
            Callable[[pd.Timestamp, np.number], np.number],
        ],
        name: str = "InvertibleMapper",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """
        Data transformer to apply a custom function and its inverse to a (sequence of) ``TimeSeries``
        (similar to calling :func:`TimeSeries.map()` on each series).

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`.
        inverse_fn
            Similarly to `fn`, either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`.
            `inverse_fn` should be such that ``inverse_fn(fn(x)) == x``.
        name
            A specific name for the transformer.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress

        Examples
        --------
        >>> import numpy as np
        >>> from darts import TimeSeries
        >>> from darts.dataprocessing.transformers import Mapper
        >>> series = TimeSeries.from_values(np.array([1e0, 1e1, 1e2, 1e3]))
        >>> transformer = Mapper(np.log10)
        >>> series_transformed = transformer.transform(series)
        >>> print(series_transformed)
        <TimeSeries (DataArray) (time: 4, component: 1, sample: 1)>
        array([[[0.]],
            [[1.]],
            [[2.]],
            [[3.]]])
        Coordinates:
        * time       (time) int64 0 1 2 3
        * component  (component) <U1 '0'
        Dimensions without coordinates: sample
        >>> series_restaured = transformer.inverse_transform(series_transformed)
        >>> print(series_restaured)
        <TimeSeries (DataArray) (time: 4, component: 1, sample: 1)>
        array([[[   1.]],
            [[  10.]],
            [[ 100.]],
            [[1000.]]])
        Coordinates:
        * time       (time) int64 0 1 2 3
        * component  (component) <U1 '0'
        Dimensions without coordinates: sample
        """

        self._fn = fn
        self._inverse_fn = inverse_fn
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Mapping[str, MapperFn]]
    ) -> TimeSeries:
        return series.map(params["fixed"]["_fn"])

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Mapping[str, MapperFn]]
    ) -> TimeSeries:
        return series.map(params["fixed"]["_inverse_fn"])
