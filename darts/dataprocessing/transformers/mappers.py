"""
Mappers
-------
"""
import numpy as np
import pandas as pd

from typing import Callable, Optional, Union, Sequence
from inspect import signature

from darts.timeseries import TimeSeries
from darts.dataprocessing import Validator
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer
from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


class Mapper(BaseDataTransformer[TimeSeries]):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "Mapper",
                 validators: Optional[Sequence[Validator]] = None):
        """
        Data transformer to apply a function to a time series (similar to calling `series.map()`)

        Parameters
        ----------
        fn
            Either a function which takes a value and returns a value ie. f(x) = y
            Or a function which takes a value and its timestamp and returns a value ie. f(timestamp, x) = y
        name
            A specific name for the transformer
        validators
            Sequence of validators that will be called before transform()
        """
        if not isinstance(fn, Callable):
            raise_log(TypeError("fn should be callable"), logger)
        if len(signature(fn).parameters) not in [1, 2]:
            raise_log(TypeError("fn must either take one or two parameters"))

        super().__init__(name=name, validators=validators)
        self._fn = fn

    def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
        super().transform(data)
        return data.map(self._fn)


class InvertibleMapper(InvertibleDataTransformer[TimeSeries]):
    def __init__(self,
                 fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 inverse_fn: Union[Callable[[np.number], np.number], Callable[[pd.Timestamp, np.number], np.number]],
                 name: str = "InvertibleMapper",
                 validators: Optional[Sequence[Validator]] = None):
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
        validators
            Sequence of validators that will be called before transform()
        """
        super().__init__(name=name, validators=validators)
        self._fn = fn
        self._inverse_fn = inverse_fn

    def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
        super().transform(data)
        return data.map(self._fn)

    def inverse_transform(self, data: TimeSeries, *args, **kwargs):
        super().inverse_transform(data, *args, *kwargs)
        return data.map(self._inverse_fn)
