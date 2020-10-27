"""
Base Data Transformer
---------------------
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class BaseDataTransformer(Generic[T], ABC):
    def __init__(self,
                 name: str = "BaseDataTransformer"):
        """
        Abstract class for data transformers. All deriving classes have to implement only one function `transform`.
        Data transformers requiring to be fit first before calling `transform()` should derive
        from `FittableDataTransformer` instead.
        Data transformers which are invertible should derive from ´InvertibleDataTransformer´ instead.

        Parameters
        ----------
        names
            The data transformer's name
        """
        self._name = name

    @abstractmethod
    def transform(self, data: T, *args, **kwargs) -> T:
        """
        Transform the data.
        Not implemented in base class and has to be implemented by deriving classes.

        Parameters
        ----------
        data
            Object which will be transformed.
        args
            Additional positional arguments for the `transform` method
        kwargs
            Additional keyword arguments for the `transform` method

        Returns
        -------
        T
            Transformed data.
        """
        pass

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of data transformer.
        """
        return self._name

    def __str__(self):
        return self._name

    def __repr__(self):
        return self.__str__()
