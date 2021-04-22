"""
Invertible Data Transformer
---------------------------
"""
from abc import abstractmethod
from typing import TypeVar

from darts.logging import get_logger, raise_if_not
from darts.dataprocessing.transformers import BaseDataTransformer

logger = get_logger(__name__)
T = TypeVar('T')


class InvertibleDataTransformer(BaseDataTransformer[T]):

    def __init__(self,
                 name: str = "InvertibleDataTransformer",
                 n_jobs: int = 1,
                 verbose: bool = False):

        """
        Abstract class for data transformers implementing a fit method. All deriving classes must implement
        `transform()` and `inverse_transform()`.

        Parameters
        ----------
        names
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print operations progress
        """
        super().__init__(name, n_jobs=n_jobs, verbose=verbose)

    @abstractmethod
    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        """
        Perform inverse transformation of the data. Not implemented in base class.

        Parameters
        ----------
        data
            Object which will be inverse transformed.
        args
            Additional positional arguments for the `inverse_transform()` method
        kwargs
            Additional keyword arguments for the `inverse_transform()` method

        Returns
        -------
        T
            Inverse transformed data.
        """
        if hasattr(self, "_fit_called"):
            raise_if_not(self._fit_called, "fit() must have been called before inverse_transform()", logger)
