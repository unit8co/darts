"""
Invertible Data Transformer Base Class
--------------------------------------
"""

from abc import abstractmethod
from typing import Any, List, Mapping, Sequence, Union

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
        parallel_params: Union[bool, Sequence[str]] = False,
        mask_components: bool = True,
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
        parallel_params
            Optionally, specifies which fixed parameters (i.e. the attributes initialized in the child-most
            class's `__init__`) take on different values for different parallel jobs. Fixed parameters specified
            by `parallel_params` are assumed to be a `Sequence` of values that should be used for that parameter
            in each parallel job; the length of this `Sequence` should equal the number of parallel jobs. If
            `parallel_params=True`, every fixed parameter will take on a different value for each
            parallel job. If `parallel_params=False`, every fixed parameter will take on the same value for
            each parallel job. If `parallel_params` is a `Sequence` of fixed attribute names, only those
            attribute names specified will take on different values between different parallel jobs.
        mask_components
            Optionally, whether or not to automatically apply any provided `component_mask`s to the
            `TimeSeries` inputs passed to `transform`, `fit`, `inverse_transform`, or `fit_transform`.
            If `True`, any specified `component_mask` will be applied to each input timeseries
            before passing them to the called method; the masked components will also be automatically
            'unmasked' in the returned `TimeSeries`. If `False`, then `component_mask` (if provided) will
            be passed as a keyword argument, but won't automatically be applied to the input timeseries.
            See `apply_component_mask` method of `BaseDataTransformer` for further details.

        Notes
        -----
        Note: the :func:`ts_transform()` and :func:`ts_inverse_transform()` methods are designed to be
        static methods instead of instance methods to allow an efficient parallelisation also when the
        scaler instance is storing a non-negligible amount of data. Using instance methods would imply
        copying the instance's data through multiple processes, which can easily introduce a bottleneck
        and nullify parallelisation benefits.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=mask_components,
        )

    @staticmethod
    @abstractmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any]
    ) -> TimeSeries:
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
                which components of the underlying `series` the inverse transform should consider.

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

        # Take note of original input for unmasking purposes:
        if isinstance(series, TimeSeries):
            input_series = [series]
            data = [series]
        else:
            input_series = series
            data = series

        if self._mask_components:
            mask = kwargs.pop("component_mask", None)
            data = [self.apply_component_mask(ts, mask, return_ts=True) for ts in data]

        input_iterator = _build_tqdm_iterator(
            zip(data, self._get_params(n_timeseries=len(data))),
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

        if self._mask_components:
            unmasked = []
            for ts, transformed_ts in zip(input_series, transformed_data):
                unmasked.append(self.unapply_component_mask(ts, transformed_ts, mask))
            transformed_data = unmasked

        return (
            transformed_data[0] if isinstance(series, TimeSeries) else transformed_data
        )
