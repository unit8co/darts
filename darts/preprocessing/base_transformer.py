"""
Base Transformer
----------------
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from darts.logging import raise_if_not, get_logger
from darts.preprocessing.validator import Validator

logger = get_logger(__name__)
T = TypeVar('T')


class BaseTransformer(Generic[T], ABC):
    def __init__(self,
                 reversible: bool = False,
                 fittable: bool = False,
                 name: str = "BaseTransformer",
                 validators: List[Validator] = []):
        """
        Abstract class for transformers. All deriving classes have to implement only one function `transform`.
        It also has `inverse_transform` and `fit` left unimplemented. If a child of this class implements
        any of these methods, it should mark them with the appropriate property (`reversible`, `fittable`).

        Parameters
        ----------
        reversible
            Flag indicating whether this transformer implements inverse_transform
        fittable
            Flag indicating whether this transformer implements fit
        names
            The transformer's name
        validators
            List of validators that will be called before transform
        args
            Additional positional arguments
        kwargs
            Additional keyword arguments
        """
        self._reversible = reversible
        self._fittable = fittable
        self._name = name
        self._validators = validators

    @property
    def reversible(self) -> bool:
        """
        Returns
        -------
        bool
            Whether transformer has inverse_transform.
        """
        return self._reversible

    @property
    def fittable(self) -> bool:
        """
        Returns
        -------
        bool
            Whether transformer can be fitted.
        """
        return self._fittable

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of transformer.
        """
        return self._name

    def validate(self, data: T) -> bool:
        """
        Validate data using validators set at init. Will raise an exception if validation fails
        potentially with reason/explanation why.

        Parameters
        ----------
        data
            Object on which validation functions will be run.

        Returns
        -------
        True
            If successful.
        """
        for validator in self._validators:
            raise_if_not(validator(data), f"Validation failed for {self.name}\n{validator.reason}", logger)
        return True

    @abstractmethod
    def transform(self, data: T, *args, **kwargs) -> T:
        """
        Transform data. Not implemented in base class and has to be implemented by deriving classes.

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

    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        """
        Inverse transformation of data. Not implemented in base class.

        Parameters
        ----------
        data
            Object which will be inverse transformed.
        args
            Additional positional arguments for the `inverse_transform` method
        kwargs
            Additional keyword arguments for the `inverse_transform` method

        Returns
        -------
        T
            Inverse transformed data.
        """
        raise NotImplementedError(f"inverse_transform not implemented for transformer {self.name}")

    def fit(self, data: T) -> 'BaseTransformer[T]':
        """
        Function which will fit transformer. Not implemented in base class.

        Parameters
        ----------
        data
            Object on which transformer will be fitted.

        Returns
        -------
        BaseTransformer[T]
            Fitted transformer (typically would be self)
        """
        raise NotImplementedError(f"fit not implemented for transformer {self.name}")

    def fit_transform(self, data: T, *args, **kwargs) -> T:
        """
        First fit transformer with data and then perform transformation on data.

        Parameters
        ----------
        data
            Object used to fit and transform.
        args
            Additional positional arguments for the `transform` method
        kwargs
            Additional keyword arguments for the `transform` method

        Returns
        -------
        T
            Transformed data.
        """
        return self.fit(data).transform(data, *args, **kwargs)

    def __str__(self):
        return self._name

    def __repr__(self):
        return self.__str__()
