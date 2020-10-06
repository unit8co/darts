"""
Base Data Transformer
---------------------
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

from darts.logging import raise_if, get_logger
from darts.dataprocessing import Validator

logger = get_logger(__name__)
T = TypeVar('T')


class BaseDataTransformer(Generic[T], ABC):
    def __init__(self,
                 name: str = "BaseDataTransformer",
                 validators: Optional[List[Validator]] = None):
        """
        Abstract class for data transformers. All deriving classes have to implement only one function `transform`.
        Data transformers requiring to be fit first before calling `transform()` should derive
        from `FittableDataTransformer` instead.
        Data transformers which are invertible should derive from ´InvertibleDataTransformer´ instead.

        Parameters
        ----------
        names
            The data transformer's name
        validators
            List of validators that will be called before transform()
        """
        self._name = name

        if validators is None:
            validators = []

        self._validators = validators

    @abstractmethod
    def transform(self, data: T, *args, **kwargs) -> T:
        """
        Perform validation and transform data.
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
        self._validate(data)

    def _validate(self, data: T) -> bool:
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
        # Collect all validation errors (if any) and throw them all at once
        fail_reasons = []
        for validator in self._validators:
            if not validator(data):
                fail_reasons.append(validator.reason)

        raise_if(len(fail_reasons) != 0,
                 f"Validation failed for {self.name}, reason(s):\n" + '\n'.join(fail_reasons),
                 logger)
        return True

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
