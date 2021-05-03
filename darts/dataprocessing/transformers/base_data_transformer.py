"""
Base Data Transformer
---------------------
"""
from typing import Sequence, Union, Iterator, Tuple, List
from darts.logging import raise_if_not
from darts.utils import _parallel_apply, _build_tqdm_iterator
from darts import TimeSeries
from abc import ABC, abstractmethod


class BaseDataTransformer(ABC):
    def __init__(self,
                 name: str = "BaseDataTransformer",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Abstract class for data transformers. All the deriving classes have to implement the static method
        `ts_transform()`. The class offers the method `transform()`, for applying a transformation to a `TimeSeries` or
        `Sequence[TimeSeries]`. This class takes care of parallelizing the transformation of multiple `TimeSeries`
        when possible.

        Data transformers requiring to be fit first before calling `transform()` should derive
        from `FittableDataTransformer` instead.
        Data transformers that are invertible should derive from ´InvertibleDataTransformer´ instead.

        Note: the `ts_transform()` method is designed to be a static method instead of a instance method to allow an
        efficient parallelisation also when the scaler instance is storing a non-negligible amount of data. Using
        an instance method would imply copying the instance's data through multiple processes, which can easily
        introduce a bottleneck and nullify parallelisation benefits.

        Parameters
        ----------
        name
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a Sequence[TimeSeries] is passed
            as input to a method, parallelising operations regarding different TimeSeries. Defaults to `1` (sequential).
            Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """
        self._name = name
        self._verbose = verbose
        self._n_jobs = n_jobs

    def set_verbose(self, value: bool):
        """
        Setter for the verbosity status. `True` for enabling the detailed report about scaler's operation progress,
        `False` for no additional information

        Parameters
        ----------
        value
            New verbosity status

        """
        raise_if_not(isinstance(value, bool), "Verbosity status must be a boolean.")

        self._verbose = value

    def set_n_jobs(self, value: int):
        """
        Sets the number of cores that will be used by the transformer while processing multiple `TimeSeries`. Set to
        `-1` for using all the available cores.

        Parameters
        ----------
        value
            New n_jobs value

        """

        raise_if_not(isinstance(value, int), "n_jobs must be an integer")
        self._n_jobs = value

    @staticmethod
    @abstractmethod
    def ts_transform(series: TimeSeries) -> TimeSeries:
        """
        The function that will be applied to each `TimeSeries` object once the `transform()` function is called. The
        function must take as first argument a `TimeSeries` object, and return the transformed `TimeSeries` object.
        If more parameters are added as input in the derived classes, the `_transform_iterator()` should be
        redefined accordingly, to yield the necessary arguments to this function (See `_transform_iterator()` for
        further details).

        This method is not implemented in the base class and must be implemented in the deriving classes.

        Note: this method is designed to be a static method instead of instance methods to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.

        Parameters
        ----------
        series (TimeSeries)
            `TimeSeries` which will be transformed.
        """
        pass

    def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries]]:
        """
        Returns an `Iterator` object with tuples of inputs for each single call to `ts_transform()`.
        Additional `args` and `kwargs` from `transform()` (constant across all the calls to `ts_transform()`)
        are already forwarded, and thus don't need to be included in this generator.

        The basic implementation of this method returns `zip(series)`, i.e., a generator of single-valued tuples,
        each containing one `TimeSeries` object.

        Parameters
        ----------
        series (Sequence[TimeSeries])
            `Sequence` of `TimeSeries` received in input.

        Returns
        -------
        Iterator[Tuple[TimeSeries]]
            An iterator containing tuples of inputs for the `ts_transform()` method.

        Examples
        ________

        class IncreasingAdder(BaseDataTransformer):
            def __init__(self):
                super().__init__()

            @staticmethod
            def ts_transform(series: TimeSeries, n: int) -> TimeSeries:
                return series + n

            def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                return zip(series, (i for i in range(len(series))))

        """
        return zip(series)

    def transform(self,
                  series: Union[TimeSeries, Sequence[TimeSeries]],
                  *args, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Transform the data. In case a `Sequence` is passed as input data, this function takes care of
        parallelising the transformation of multiple series in the sequence at the same time.

        Parameters
        ----------
        series
            `TimeSeries` or `Sequence` of `TimeSeries` which will be transformed.
        args
            Additional positional arguments for each `ts_transform()` method call
        kwargs
            Additional keyword arguments for each `ts_transform()` method call

        Returns
        -------
        Union[TimeSeries, List[TimeSeries]]
            Transformed data.
        """

        desc = "Transform ({})".format(self._name)

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        input_iterator = _build_tqdm_iterator(self._transform_iterator(data),
                                              verbose=self._verbose,
                                              desc=desc,
                                              total=len(data))

        transformed_data = _parallel_apply(input_iterator, self.__class__.ts_transform,
                                           self._n_jobs, args, kwargs)

        return transformed_data[0] if isinstance(series, TimeSeries) else transformed_data

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
