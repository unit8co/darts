"""
Base Data Transformer
---------------------
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from darts.logging import raise_if_not

T = TypeVar('T')


class BaseDataTransformer(Generic[T], ABC):
    def __init__(self,
                 name: str = "BaseDataTransformer",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Abstract class for data transformers. All deriving classes have to implement only one function `transform`.
        Data transformers requiring to be fit first before calling `transform()` should derive
        from `FittableDataTransformer` instead.
        Data transformers which are invertible should derive from ´InvertibleDataTransformer´ instead.

        Parameters
        ----------
        names
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Defaults to `1`. `-1` means using all processors
        verbose
            Optionally, whether to print operations progress
        """
        self._name = name
        self._verbose = verbose
        self._n_jobs = 1

    def set_verbose(self, value: bool):
        """
        Setter for the verbosity status. True for enabling the datailed report about scaler's operation progress, False
        for no additional information

        Parameters
        ----------
        value
            New verbosity status

        """
        raise_if_not(isinstance(value, bool), "Verbosity status must be a boolean.")

        self._verbose = value

    def set_n_jobs(self, value: int):
        """
        Sets the number of cores that will be used by the transformer while processing multiple time series. Set to `-1` 
        for using all the available cores.
        """

        raise_if_not(isinstance(value, int), "n_jobs must be an integer")
        self._n_jobs = value

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
