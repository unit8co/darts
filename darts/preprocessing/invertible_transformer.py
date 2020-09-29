from abc import abstractmethod
from typing import TypeVar, List, Optional

from darts.logging import get_logger, raise_if_not
from darts.preprocessing.validator import Validator
from darts.preprocessing.base_transformer import BaseTransformer

logger = get_logger(__name__)
T = TypeVar('T')


class InvertibleTransformer(BaseTransformer[T]):

    def __init__(self,
                 name: str = "InvertibleTransformer",
                 validators: Optional[List[Validator]] = None):

        """
        Abstract class for transformers implementing a fit method. All deriving classes must implement
        `transform()` and `inverse_transform()`.

        Parameters
        ----------
        names
            The transformer's name
        validators
            List of validators that will be called before transform() and inverse_transform()
        """
        super().__init__(name, validators)

    @abstractmethod
    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        """
        Perform validation and inverse transformation of data. Not implemented in base class.

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
        # TODO: this is a little hacky… And wouldn't we want to enforce "transform() called
        # before inverse_transform()" instead (might be too restrictive…)
        if hasattr(self, "_fit_called"):
            raise_if_not(self._fit_called, "fit() must have been called before inverse_transform()", logger)

        super()._validate(data)
