"""
Data Processing Utils
-------------------
"""
import numpy as np

from typing import TypeVar, Callable, Optional, Tuple, List, Any, Dict, Type

from darts import TimeSeries
from darts.logging import raise_if, get_logger
from darts.dataprocessing import BaseDataTransformer, InvertibleDataTransformer, FittableDataTransformer

logger = get_logger(__name__)
T = TypeVar('T')
_callable_with_args_kwargs = Callable[[T, Tuple[Any, ...], Dict[str, any]], T]


def _create_wrapper_data_transformer(parent_classes: List[Type[BaseDataTransformer[T]]],
                                     transform: _callable_with_args_kwargs,
                                     inverse_transform: Optional[_callable_with_args_kwargs] = None,
                                     fit: Optional[_callable_with_args_kwargs] = None,
                                     name: str = "WrapperDataTransformer"):
    class _WrapperDataTransformer(*parent_classes):
        def __init__(self,
                     transform: _callable_with_args_kwargs,
                     inverse_transform: Optional[_callable_with_args_kwargs] = None,
                     fit: Optional[_callable_with_args_kwargs] = None,
                     name: str = "WrapperDataTransformer"):
            raise_if(transform is None, "transform cannot be None", logger)
            raise_if(InvertibleDataTransformer in parent_classes and inverse_transform is None,
                     "WrapperDataTransformer marked as invertible but no inverse_transform() function was provided",
                     logger)
            raise_if(FittableDataTransformer in parent_classes and fit is None,
                     "WrapperDataTransfomer marked as fittable but no fit() function was provided")

            super().__init__(name=name, validators=None)  # TODO: Add possibility to specify validators ?

            self._transform = transform
            self._inverse_transform = inverse_transform
            self._fit = fit

            self._name = name

        def transform(self, data: T, *args, **kwargs) -> T:
            return self._transform(data, *args, **kwargs)

        if InvertibleDataTransformer in parent_classes:
            def inverse_transform(self, data: T, *args, **kwargs) -> T:
                return self._inverse_transform(data, *args, **kwargs)

        if FittableDataTransformer in parent_classes:
            def fit(self, data: T) -> 'Type[BaseDataTransformer[T]]':
                self._fit(data)
                return self

    return _WrapperDataTransformer(transform, inverse_transform, fit, name)


def _get_parent_classes(inverse_transform: Optional[_callable_with_args_kwargs] = None,
                        fit: Optional[_callable_with_args_kwargs] = None):
    parent_classes = []
    if inverse_transform is not None:
        parent_classes.append(InvertibleDataTransformer)
    if fit is not None:
        parent_classes.append(FittableDataTransformer)

    if not parent_classes:
        parent_classes.append(BaseDataTransformer)

    return parent_classes


def data_transformer_from_ts_functions(transform: _callable_with_args_kwargs[TimeSeries],
                                       inverse_transform: Optional[_callable_with_args_kwargs[TimeSeries]] = None,
                                       fit: Optional[Callable[[TimeSeries], Any]] = None,
                                       name: str = "FromTSWrappedDataTransformer") -> Type[BaseDataTransformer[TimeSeries]]:  # noqa: E501
    """
    Utility function to create a data transformer from functions taking as input TimeSeries.
    All functions except transform are optional.

    Parameters
    ----------
    transform
        Function taking TimeSeries as an input and returning TimeSeries.
    inverse_transform
        Function taking TimeSeries as an input and returning TimeSeries.
    fit
        Function taking TimeSeries as an input.
    name
        Data transformer's name will be set to this.

    Returns
    -------
    Type[BaseDataTransformer[TimeSeries]]
        Data transformer created from the provided functions.
    """
    return _create_wrapper_data_transformer(_get_parent_classes(inverse_transform, fit),
                                            transform,
                                            inverse_transform,
                                            fit,
                                            name)


def data_transformer_from_values_functions(transform: _callable_with_args_kwargs[np.ndarray],
                                           inverse_transform: Optional[_callable_with_args_kwargs[np.ndarray]] = None,
                                           fit: Optional[Callable[[np.ndarray], Any]] = None,
                                           name: str = "FromValuesWrappedDataTransformer") -> Type[BaseDataTransformer[TimeSeries]]:  # noqa: E501
    """
    Utility function to create a data transformer from functions taking as input value series.
    All functions except transform are optional.

    Parameters
    ----------
    transform
        Function taking ndarray as an input and returning ndarray.
    inverse_transform
        Function taking ndarray as an input and returning ndarray.
    fit
        Function taking ndarray as an input.
    name
        Data transformer's name will be set to this.

    Returns
    -------
    Type[BaseDataTransformer[TimeSeries]]
        Data transformer created from the provided functions.
    """

    def apply_to_values(f: _callable_with_args_kwargs[np.ndarray], returns: bool = True):
        if f is None:
            return None

        def func(ts: TimeSeries, *args, **kwargs) -> TimeSeries:
            if returns:
                return TimeSeries.from_times_and_values(ts.time_index(), f(ts.values(), *args, **kwargs),
                                                        freq=ts.freq_str())
            f(ts.values(), *args, **kwargs)

        return func

    transform = apply_to_values(transform)
    inverse_transform = apply_to_values(inverse_transform)
    fit = apply_to_values(fit, returns=False)

    return _create_wrapper_data_transformer(_get_parent_classes(inverse_transform, fit),
                                            transform,
                                            inverse_transform,
                                            fit,
                                            name)
