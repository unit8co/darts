"""
Validator
--------------
"""
from typing import TypeVar, Generic, Callable, Optional

from darts.logging import get_logger

logger = get_logger(__name__)
T = TypeVar('T')


class Validator(Generic[T]):
    def __init__(self, validate_func: Callable[[T], bool], reason: Optional[str] = None):
        """
        Simple wrapper for validation purposes.

        Parameters
        ----------
        validate_func
            Validation function, returning True means validation passed
        reason
            Optional string explaining what was checked during validation
        """
        self._func = validate_func
        self._reason = reason

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

    @property
    def reason(self) -> Optional[str]:
        """
        Returns
        -------
        Optional[str]
            Potential explanation what was checked during validation
        """
        return self._reason
