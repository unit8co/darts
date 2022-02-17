"""
Invertible Data Transformer Base Class
--------------------------------------
"""

from abc import abstractmethod
from typing import Iterator, List, Sequence, Tuple, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.utils import _build_tqdm_iterator, _parallel_apply

from .base_data_transformer import BaseDataTransformer

logger = get_logger(__name__)


class InvertibleDataTransformer(BaseDataTransformer):
    def __init__(
        self,
        name: str = "InvertibleDataTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
    ):

        """Abstract class for invertible transformers.

        All the deriving classes have to implement the static methods
        :func:`ts_transform()` and :func:`ts_inverse_transform()`.
        This class takes care of parallelizing the transformation on multiple ``TimeSeries`` when possible.

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

        Notes
        -----
        Note: the :func:`ts_transform()` and :func:`ts_inverse_transform()` methods are designed to be
        static methods instead of instance methods to allow an efficient parallelisation also when the
        scaler instance is storing a non-negligible amount of data. Using instance methods would imply
        copying the instance's data through multiple processes, which can easily introduce a bottleneck
        and nullify parallelisation benefits.
        """
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)

    @staticmethod
    @abstractmethod
    def ts_inverse_transform(series: TimeSeries) -> TimeSeries:
        """The function that will be applied to each series when :func:`inverse_transform` is called.

        The function must take as first argument a ``TimeSeries`` object, and return the transformed
        ``TimeSeries`` object. Additional parameters can be added if necessary, but in this case,
        :func:`_inverse_transform_iterator()` should be redefined accordingly, to yield the necessary
        arguments to this function (See :func:`_inverse_transform_iterator()` for further details)

        This method is not implemented in the base class and must be implemented in the deriving classes.

        Parameters
        ----------
        series (TimeSeries)
            TimeSeries which will be transformed.

        Notes
        -----
        This method is designed to be a static method instead of instance methods to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries]]:
        """Return an `Iterator` object with tuples of inputs for each single call to :func:`ts_inverse_transform()`.

        Additional `args` and `kwargs` from :func:`inverse_transform()` (that don't change across the calls to
        :func:`ts_inverse_transform()`) are already forwarded, and thus don't need to be included in this generator.

        The basic implementation of this method returns ``zip(series)``, i.e., a generator of single-valued tuples,
        each containing one TimeSeries object.

        Parameters
        ----------
        series (Sequence[TimeSeries])
            Sequence of TimeSeries received in input.

        Returns
        -------
        Iterator[Tuple[TimeSeries]]
            An iterator containing tuples of inputs for the :func:`ts_inverse_transform()` method.

        Examples
        ________

        class IncreasingAdder(InvertibleDataTransformer):
            def __init__(self):
                super().__init__(ts_transform=my_ts_transform,
                                 ts_inverse_transform=my_ts_inverse_transform)

            @staticmethod
            def ts_transform(series: TimeSeries, n: int) -> TimeSeries:
                return series + n

            @staticmethod
            def ts_inverse_transform(series: TimeSeries, n: int) -> TimeSeries:
                return series - n

            def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                return zip(series, (i for i in range(len(series))))

            def _inverse_transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                return zip(series, (i for i in range(len(series))))
        """
        return zip(series)

    def inverse_transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """Inverse-transform a (sequence of) series.

        In case a sequence is passed as input data, this function takes care of
        parallelising the transformation of multiple series in the sequence at the same time.

        Parameters
        ----------
        series
            the (sequence of) series be inverse-transformed.
        args
            Additional positional arguments for the :func:`ts_inverse_transform()` method
        kwargs
            Additional keyword arguments for the :func:`ts_inverse_transform()` method

            component_mask : Optional[np.ndarray] = None
                Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies
                which components of the underlying `series` the Scaler should consider.

        Returns
        -------
        Union[TimeSeries, List[TimeSeries]]
            Inverse transformed data.
        """
        if hasattr(self, "_fit_called"):
            raise_if_not(
                self._fit_called,
                "fit() must have been called before inverse_transform()",
                logger,
            )

        desc = f"Inverse ({self._name})"

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        input_iterator = _build_tqdm_iterator(
            self._inverse_transform_iterator(data),
            verbose=self._verbose,
            desc=desc,
            total=len(data),
        )

        transformed_data = _parallel_apply(
            input_iterator,
            self.__class__.ts_inverse_transform,
            self._n_jobs,
            args,
            kwargs,
        )

        return (
            transformed_data[0] if isinstance(series, TimeSeries) else transformed_data
        )
