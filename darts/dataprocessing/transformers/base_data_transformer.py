"""
Data Transformer Base Class
---------------------------
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not
from darts.utils import _build_tqdm_iterator, _parallel_apply

logger = get_logger(__name__)


class BaseDataTransformer(ABC):
    def __init__(
        self, name: str = "BaseDataTransformer", n_jobs: int = 1, verbose: bool = False
    ):
        """Abstract class for data transformers.

        All the deriving classes have to implement the static method :func:`ts_transform`.
        The class offers the method :func:`transform()`, for applying a transformation to a ``TimeSeries`` or
        ``Sequence[TimeSeries]``. This class takes care of parallelizing the transformation of multiple ``TimeSeries``
        when possible.

        Data transformers requiring to be fit first before calling :func:`transform` should derive
        from :class:`.FittableDataTransformer` instead.
        Data transformers that are invertible should derive from :class:`.InvertibleDataTransformer` instead.

        Note: the :func:`ts_transform` method is designed to be a static method instead of a instance method to allow an
        efficient parallelisation also when the scaler instance is storing a non-negligible amount of data. Using
        an instance method would imply copying the instance's data through multiple processes, which can easily
        introduce a bottleneck and nullify parallelisation benefits.

        Parameters
        ----------
        name
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]``
            is passed as input to a method, parallelising operations regarding different TimeSeries.
            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress
        """
        self._name = name
        self._verbose = verbose
        self._n_jobs = n_jobs

    def set_verbose(self, value: bool):
        """Set the verbosity status.

        `True` for enabling the detailed report about scaler's operation progress,
        `False` for no additional information.

        Parameters
        ----------
        value
            New verbosity status
        """
        raise_if_not(isinstance(value, bool), "Verbosity status must be a boolean.")

        self._verbose = value

    def set_n_jobs(self, value: int):
        """Set the number of processors to be used by the transformer while processing multiple ``TimeSeries``.

        Parameters
        ----------
        value
            New n_jobs value.  Set to `-1` for using all the available cores.
        """

        raise_if_not(isinstance(value, int), "n_jobs must be an integer")
        self._n_jobs = value

    @staticmethod
    @abstractmethod
    def ts_transform(series: TimeSeries) -> TimeSeries:
        """The function that will be applied to each series when :func:`transform()` is called.

        The function must take as first argument a ``TimeSeries`` object, and return the transformed ``TimeSeries``
        object. If more parameters are added as input in the derived classes, the ``_transform_iterator()`` should be
        redefined accordingly, to yield the necessary arguments to this function (See ``_transform_iterator()`` for
        further details).

        This method is not implemented in the base class and must be implemented in the deriving classes.

        Parameters
        ----------
        series
            series to be transformed.

        Notes
        -----
        This method is designed to be a static method instead of instance method to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries]]:
        """
        Return an ``Iterator`` object with tuples of inputs for each single call to :func:`ts_transform()`.
        Additional `args` and `kwargs` from :func:`transform()` (constant across all the calls to
        :func:`ts_transform()`) are already forwarded, and thus don't need to be included in this generator.

        The basic implementation of this method returns ``zip(series)``, i.e., a generator of single-valued tuples,
        each containing one ``TimeSeries`` object.

        Parameters
        ----------
        series
            Sequence of series received in input.

        Returns
        -------
        Iterator[Tuple[TimeSeries]]
            An iterator containing tuples of inputs for the :func:`ts_transform` method.

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

    def transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """Transform a (sequence of) of series.

        In case a ``Sequence`` is passed as input data, this function takes care of
        parallelising the transformation of multiple series in the sequence at the same time.

        Parameters
        ----------
        series
            (sequence of) series to be transformed.
        args
            Additional positional arguments for each :func:`ts_transform()` method call
        kwargs
            Additional keyword arguments for each :func:`ts_transform()` method call

        Returns
        -------
        Union[TimeSeries, List[TimeSeries]]
            Transformed data.
        """

        desc = f"Transform ({self._name})"

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        input_iterator = _build_tqdm_iterator(
            self._transform_iterator(data),
            verbose=self._verbose,
            desc=desc,
            total=len(data),
        )

        transformed_data = _parallel_apply(
            input_iterator, self.__class__.ts_transform, self._n_jobs, args, kwargs
        )

        return (
            transformed_data[0] if isinstance(series, TimeSeries) else transformed_data
        )

    @staticmethod
    def _reshape_in(
        series: TimeSeries,
        component_mask: Optional[np.ndarray] = None,
        flatten: Optional[bool] = True,
    ) -> np.ndarray:
        """Extracts specified components from series and reshapes these values into an appropriate input shape
        for a transformer. If `flatten=True`, the output is a 2-D matrix where each row corresponds to a timestep,
        each column corresponds to a component (dimension) of the series, and the columns' values are the flattened
        values over all samples. Conversely, if `flatten=False`, the output is a 3-D matrix (i.e. timesteps along
        the zeroth axis, components along the first axis, and samples along the second axis).

        Parameters
        ----------
        series
            input TimeSeries to be fed into transformer.
        component_mask
            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components to
            extract from `series`.
        flatten
            Optionally, bool specifying whether the samples for each component extracted by `component_mask`
            should be flattened into a single column.

        """

        if component_mask is None:
            component_mask = np.ones(series.n_components, dtype=bool)

        raise_if_not(
            isinstance(component_mask, np.ndarray) and component_mask.dtype == bool,
            "`component_mask` must be a boolean np.ndarray`",
            logger,
        )
        raise_if_not(
            series.width == len(component_mask),
            "mismatch between number of components in `series` and length of `component_mask`",
            logger,
        )

        vals = series.all_values(copy=False)[:, component_mask, :]

        if flatten:
            vals = np.stack(
                [vals[:, i, :].reshape(-1) for i in range(component_mask.sum())], axis=1
            )

        return vals

    @staticmethod
    def _reshape_out(
        series: TimeSeries,
        vals: np.ndarray,
        component_mask: Optional[np.ndarray] = None,
        flatten: Optional[bool] = True,
    ) -> np.ndarray:
        """Reshapes the 2-D or 3-D matrix coming out of a transformer into a 3-D matrix suitable to build a TimeSeries,
        and adds back components previously removed by `component_mask` in `_reshape_in` method.

        If `flatten=True`, the output is built by taking each column of the 2-D input matrix (the flattened components)
        and reshaping them to (len(series), n_samples), then stacking them on 2nd axis. Conversely, if `flatten=False`,
        the shape of the 3-D input matrix is left unchanged.

        Parameters
        ----------
        series
            input TimeSeries that was fed into transformer.
        vals:
            transformer output
        component_mask
            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components were extracted
            from `series`. If given, insert `vals` back into the columns of the original array.
        flatten
            Optionally, bool specifying whether `series` is a 2-D matrix. Should match value of `flatten` argument
            provided to `_reshape_in` method.
        """

        raise_if_not(
            component_mask is None
            or isinstance(component_mask, np.ndarray)
            and component_mask.dtype == bool,
            "If `component_mask` is given, must be a boolean np.ndarray`",
            logger,
        )

        series_width = series.width if component_mask is None else component_mask.sum()
        if flatten:
            reshaped = np.stack(
                [vals[:, i].reshape(-1, series.n_samples) for i in range(series_width)],
                axis=1,
            )
        else:
            reshaped = vals

        if component_mask is None:
            return reshaped

        raise_if_not(
            series.width == len(component_mask),
            "mismatch between number of components in `series` and length of `component_mask`",
            logger,
        )

        series_vals = series.all_values(copy=True)
        series_vals[:, component_mask, :] = reshaped
        return series_vals

    @property
    def name(self):
        """Name of the data transformer."""
        return self._name

    def __str__(self):
        return self._name

    def __repr__(self):
        return self.__str__()
