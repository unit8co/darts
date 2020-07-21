"""
Preprocessing Utils
-------------------
"""
import numpy as np

from typing import TypeVar, Callable, Optional, Tuple, Any, Dict

from darts import TimeSeries
from darts.logging import raise_if_not, raise_if, get_logger
from darts.preprocessing.base_transformer import BaseTransformer

logger = get_logger(__name__)
T = TypeVar('T')
_callable_with_args_kwargs = Callable[[T, Tuple[Any, ...], Dict[str, any]], T]


class _WrapperTransformer(BaseTransformer[T]):
    def __init__(self,
                 transform: _callable_with_args_kwargs,
                 inverse_transform: Optional[_callable_with_args_kwargs] = None,
                 predict: Optional[_callable_with_args_kwargs] = None,
                 fit: Optional[_callable_with_args_kwargs] = None,
                 name: str = "WrapperTransformer"):
        raise_if(transform is None, "transform cannot be None", logger)

        _can_predict = predict is not None
        _fittable = fit is not None
        _reversible = inverse_transform is not None
        super().__init__(can_predict=_can_predict, fittable=_fittable, reversible=_reversible)

        self._transform = transform
        self._inverse_transform = inverse_transform
        self._fit = fit
        self._predict = predict

        self._name = name

    @property
    def name(self):
        return self._name

    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        raise_if_not(self.reversible,
                     f"inverse_transform not implemented for transformer {self.name}", logger)
        return self._inverse_transform(data, *args, **kwargs)

    def predict(self, data: T, *args, **kwargs) -> T:
        raise_if_not(self.can_predict,
                     f"predict not implemented for transformer {self.name}", logger)
        return self._predict(data, *args, **kwargs)

    def fit(self, data: T) -> 'BaseTransformer[T]':
        raise_if_not(self.fittable,
                     f"fit not implemented for transformer {self.name}", logger)
        self._fit(data)
        return self

    def transform(self, data: T, *args, **kwargs) -> T:
        return self._transform(data, *args, **kwargs)


def transformer_from_ts_functions(transform: _callable_with_args_kwargs[TimeSeries],
                                  inverse_transform: Optional[_callable_with_args_kwargs[TimeSeries]] = None,
                                  predict: Optional[_callable_with_args_kwargs[TimeSeries]] = None,
                                  fit: Optional[Callable[[TimeSeries], Any]] = None,
                                  name: str = "WrappedTransformer") -> BaseTransformer[TimeSeries]:
    """
    Utility function to create transformer from functions taking as input TimeSeries. All functions except transform
    are optional.

    Parameters
    ----------
    transform
        Function taking TimeSeries as an input  and returning TimeSeries.
    inverse_transform
        Function taking TimeSeries as an input and returning TimeSeries.
    predict
        Function taking TimeSeries as an input and returning TimeSeries.
    fit
        Function taking TimeSeries as an input.
    name
        Transformer name will be set to this.

    Returns
    -------
    BaseTransformer[T]
        Transformer created from passed function.
    """
    return _WrapperTransformer(transform, inverse_transform, predict, fit, name)


def transformer_from_values_functions(transform: _callable_with_args_kwargs[np.ndarray],
                                      inverse_transform: Optional[_callable_with_args_kwargs[np.ndarray]] = None,
                                      predict: Optional[_callable_with_args_kwargs[np.ndarray]] = None,
                                      fit: Optional[Callable[[np.ndarray], Any]] = None,
                                      name: str = "WrappedTransformer") -> BaseTransformer[TimeSeries]:
    """
    Utility function to create transformer from functions taking as input value series. All functions except transform
    are optional.

    Parameters
    ----------
    transform
        Function taking ndarray as an input and returning ndarray.
    inverse_transform
        Function taking ndarray as an input and returning ndarray.
    predict
        Function taking ndarray as an input and returning ndarray.
    fit
        Function taking ndarray as an input.
    name
        Transformer name will be set to this.

    Returns
    -------
    BaseTransformer[T]
        Transformer created from passed function.
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

    return _WrapperTransformer(
        apply_to_values(transform),
        apply_to_values(inverse_transform),
        apply_to_values(predict),
        apply_to_values(fit, returns=False),
        name
    )
