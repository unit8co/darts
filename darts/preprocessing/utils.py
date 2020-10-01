"""
Preprocessing Utils
-------------------
"""
import numpy as np

from typing import TypeVar, Callable, Optional, Tuple, List, Any, Dict, Type

from darts import TimeSeries
from darts.logging import raise_if, get_logger
from darts.preprocessing import BaseTransformer, InvertibleTransformer, FittableTransformer

logger = get_logger(__name__)
T = TypeVar('T')
_callable_with_args_kwargs = Callable[[T, Tuple[Any, ...], Dict[str, any]], T]


def _create_wrapper_transformer(parent_classes: List[Type[BaseTransformer[T]]],
                                transform: _callable_with_args_kwargs,
                                inverse_transform: Optional[_callable_with_args_kwargs] = None,
                                fit: Optional[_callable_with_args_kwargs] = None,
                                name: str = "WrapperTransformer"):
    class _WrapperTransformer(*parent_classes):
        def __init__(self,
                     transform: _callable_with_args_kwargs,
                     inverse_transform: Optional[_callable_with_args_kwargs] = None,
                     fit: Optional[_callable_with_args_kwargs] = None,
                     name: str = "WrapperTransformer"):
            raise_if(transform is None, "transform cannot be None", logger)
            raise_if(InvertibleTransformer in parent_classes and inverse_transform is None,
                     "WrapperTransformer marked as invertible but no inverse_transform() function was provided",
                     logger)
            raise_if(FittableTransformer in parent_classes and fit is None,
                     "WrapperTransfomer marked as fittable but no fit() function was provided")

            super().__init__(name=name, validators=None)  # TODO: Add possibility to specify validators ?

            self._transform = transform
            self._inverse_transform = inverse_transform
            self._fit = fit

            self._name = name

        def transform(self, data: T, *args, **kwargs) -> T:
            return self._transform(data, *args, **kwargs)

        if InvertibleTransformer in parent_classes:
            def inverse_transform(self, data: T, *args, **kwargs) -> T:
                return self._inverse_transform(data, *args, **kwargs)

        if FittableTransformer in parent_classes:
            def fit(self, data: T) -> 'Type[BaseTransformer[T]]':
                self._fit(data)
                return self

    return _WrapperTransformer(transform, inverse_transform, fit, name)


def _get_parent_classes(inverse_transform: Optional[_callable_with_args_kwargs] = None,
                        fit: Optional[_callable_with_args_kwargs] = None):
    parent_classes = []
    if inverse_transform is not None:
        parent_classes.append(InvertibleTransformer)
    if fit is not None:
        parent_classes.append(FittableTransformer)

    if not parent_classes:
        parent_classes.append(BaseTransformer)

    return parent_classes


def transformer_from_ts_functions(transform: _callable_with_args_kwargs[TimeSeries],
                                  inverse_transform: Optional[_callable_with_args_kwargs[TimeSeries]] = None,
                                  fit: Optional[Callable[[TimeSeries], Any]] = None,
                                  name: str = "FromTSWrappedTransformer") -> Type[BaseTransformer[TimeSeries]]:
    """
    Utility function to create transformer from functions taking as input TimeSeries. All functions except transform
    are optional.

    Parameters
    ----------
    transform
        Function taking TimeSeries as an input  and returning TimeSeries.
    inverse_transform
        Function taking TimeSeries as an input and returning TimeSeries.
    fit
        Function taking TimeSeries as an input.
    name
        Transformer name will be set to this.

    Returns
    -------
    Type[BaseTransformer[TimeSeries]]
        Transformer created from the provided functions.
    """
    return _create_wrapper_transformer(_get_parent_classes(inverse_transform, fit),
                                       transform,
                                       inverse_transform,
                                       fit,
                                       name)


def transformer_from_values_functions(transform: _callable_with_args_kwargs[np.ndarray],
                                      inverse_transform: Optional[_callable_with_args_kwargs[np.ndarray]] = None,
                                      fit: Optional[Callable[[np.ndarray], Any]] = None,
                                      name: str = "FromValuesWrappedTransformer") -> Type[BaseTransformer[TimeSeries]]:
    """
    Utility function to create transformer from functions taking as input value series. All functions except transform
    are optional.

    Parameters
    ----------
    transform
        Function taking ndarray as an input and returning ndarray.
    inverse_transform
        Function taking ndarray as an input and returning ndarray.
    fit
        Function taking ndarray as an input.
    name
        Transformer name will be set to this.

    Returns
    -------
    Type[BaseTransformer[TimeSeries]]
        Transformer created from the provided functions.
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

    return _create_wrapper_transformer(_get_parent_classes(inverse_transform, fit),
                                       transform,
                                       inverse_transform,
                                       fit,
                                       name)
