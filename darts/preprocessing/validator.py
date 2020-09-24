"""
Validator
--------------
"""
from typing import TypeVar, Generic, Callable

from darts.logging import get_logger

logger = get_logger(__name__)
T = TypeVar('T')


class Validator(Generic[T]):
    def __init__(self, validation_function: Callable[[T], bool], reason: str = ""):
        """
        Simple wrapper for validation purposes.

        Parameters
        ----------
        validate_func
            Validation function, returning True means validation passed
        reason
            Optional string explaining what was checked during validation
        """
        self._func = validation_function
        self.reason = reason

    def __call__(self, data: T) -> bool:
        """
        Calls validation function on data.

        Parameters
        ----------
        data
            object that will be validated
        Returns
        -------
        bool
            whether validation was successful.
        """
        return self._func(data)
