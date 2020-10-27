"""
Fittable Data Transformer
-------------------------
"""

from abc import abstractmethod
from typing import TypeVar

from darts.logging import get_logger, raise_if_not
from darts.dataprocessing.transformers import BaseDataTransformer

logger = get_logger(__name__)
T = TypeVar('T')


class FittableDataTransformer(BaseDataTransformer[T]):
    def __init__(self,
                 name: str = "FittableDataTransformer"):

        """
        Abstract class for data transformers implementing a fit method. All deriving classes must implement
        `fit()` and `transform()`.

        names
            The data transformer's name
        """
        super().__init__(name)
        self._fit_called = False

    @abstractmethod
    def fit(self, data: T) -> 'FittableDataTransformer[T]':
        """
        Fit the data transformer to data.
        Not implemented in base class and has to be implemented by deriving classes.

        Parameters
        ----------
        data
            Object on which data transformer will be fitted.

        Returns
        -------
        BaseDataTransformer[T]
            Fitted data transformer (typically would be self)
        """
        self._fit_called = True

    @abstractmethod
    def transform(self, data: T, *args, **kwargs) -> T:
        """
        Transform the data. Not implemented in base class.
        Will raise an error if called before a call to `fit()`

        Parameters
        ----------
        data
            Object which will be inverse transformed.
        args
            Additional positional arguments for the `transform()` method
        kwargs
            Additional keyword arguments for the `transform()` method

        Returns
        -------
        T
            Transformed data.
        """
        raise_if_not(self._fit_called, "fit() must have been called before transform()", logger)
        super().transform(data, args, kwargs)

    def fit_transform(self, data: T, *args, **kwargs) -> T:
        """
        Fit the transformer to data and then transform data.

        Parameters
        ----------
        data
            Object used to fit and transform.
        args
            Additional positional arguments for the `transform()` method
        kwargs
            Additional keyword arguments for the `transform()` method

        Returns
        -------
        T
            Transformed data.
        """
        return self.fit(data).transform(data, *args, **kwargs)
