"""
Fittable Data Transformer Base Class
------------------------------------
"""

from abc import abstractmethod
from typing import Any, Generator, Iterator, List, Mapping, Sequence, Tuple, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not
from darts.utils import _build_tqdm_iterator, _parallel_apply

from .base_data_transformer import BaseDataTransformer

logger = get_logger(__name__)


class FittableDataTransformer(BaseDataTransformer):
    def __init__(
        self,
        name: str = "FittableDataTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
        parallel_params: Union[bool, Sequence[str]] = False,
        mask_components: bool = True,
    ):

        """Base class for fittable transformers.

        All the deriving classes have to implement the static methods
        :func:`ts_transform()` and :func:`ts_fit()`. The fitting and transformation functions must
        be passed during the transformer's initialization. This class takes care of parallelizing
        operations involving multiple ``TimeSeries`` when possible.

        Parameters
        ----------
        name
            The data transformer's name
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
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
        The :func:`ts_transform()` and :func:`ts_fit()` methods are designed to be static methods instead of instance
        methods to allow an efficient parallelisation also when the scaler instance is storing a non-negligible
        amount of data. Using instance methods would imply copying the instance's data through multiple processes, which
        can easily introduce a bottleneck and nullify parallelisation benefits.
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=mask_components,
        )

        self._fit_called = False
        self.fitted_params = None  # stores the fitted parameters/objects

    @staticmethod
    @abstractmethod
    def ts_fit(series: TimeSeries, params: Mapping[str, Any]):
        """The function that will be applied to each series when :func:`fit` is called.

        The function must take as first argument a ``TimeSeries`` object, and return an object containing information
        regarding the fitting phase (e.g., parameters, or external transformers objects). All these parameters will
        be stored in ``self._fitted_params``, which can be later used during the transformation step.

        This method is not implemented in the base class and must be implemented in the deriving classes.

        If more parameters are added as input in the derived classes, :func:`_fit_iterator()`
        should be redefined accordingly, to yield the necessary arguments to this function (See
        :func:`_fit_iterator()` for further details)

        Parameters
        ----------
        series (TimeSeries)
            `TimeSeries` against which the scaler will be fit.

        Notes
        -----
        This method is designed to be a static method instead of instance methods to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def _fit_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries]]:
        """
        Return an ``Iterator`` object with tuples of inputs for each single call to :func:`ts_fit()`.
        Additional `args` and `kwargs` from :func:`fit()` (that don't change across the calls to :func:`ts_fit()`)
        are already forwarded, and thus don't need to be included in this generator.

        The basic implementation of this method returns ``zip(series)``, i.e., a generator of single-valued tuples,
        each containing one ``TimeSeries`` object.

        Parameters
        ----------
        series (Sequence[TimeSeries])
            sequence of series received in input.

        Returns
        -------
        Iterator[Tuple[TimeSeries]]
            An iterator containing tuples of inputs for the :func:`ts_fit()` method.

        Examples
        ________

        class IncreasingAdder(FittableDataTransformer):
            def __init__(self):
                super().__init__()

            @staticmethod
            def ts_transform(series: TimeSeries, n: int) -> TimeSeries:
                return series + n

            @staticmethod
            def ts_fit(series: TimeSeries, m: float) -> TimeSeries:
                return max(series.first_value(), m)

            def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, float]]:
                # the increased quantity is the max between 0 and the first value in the series (stored into
                # self._fitted params)
                return zip(series, self._fitted_params)

            def _fit_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple[TimeSeries, int]]:
                # the second generator is setting the m parameter of ts_fit() to 0 for each TimeSeries
                return zip(series, (0 for i in range(len(series))))

        """

        params = self._get_params(n_timeseries=len(series), include_fitted_params=False)
        return zip(series, params)

    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> "FittableDataTransformer":
        """Fit the transformer to the provided series or sequence of series.

        Fit the data and store the fitting parameters into ``self._fitted_params``. If a sequence is passed as input
        data, this function takes care of parallelising the fitting of multiple series in the sequence at the same time
        (in this case ``self._fitted_params`` will contain an array of fitted params, one for each series).

        Parameters
        ----------
        series
            (sequence of) series to fit the transformer on.
        args
            Additional positional arguments for the :func:`ts_fit` method
        kwargs
            Additional keyword arguments for the :func:`ts_fit` method

            component_mask : Optional[np.ndarray] = None
                Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which
                components of the underlying `series` the transform should be fitted to.

        Returns
        -------
        FittableDataTransformer
            Fitted transformer.
        """
        self._fit_called = True

        desc = f"Fitting ({self._name})"

        if isinstance(series, TimeSeries):
            data = [series]
        else:
            data = series

        if self._mask_components:
            mask = kwargs.pop("component_mask", None)
            data = [self.apply_component_mask(ts, mask, return_ts=True) for ts in data]

        input_iterator = _build_tqdm_iterator(
            self._fit_iterator(data), verbose=self._verbose, desc=desc, total=len(data)
        )

        self._fitted_params = _parallel_apply(
            input_iterator, self.__class__.ts_fit, self._n_jobs, args, kwargs
        )

        return self

    def fit_transform(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], *args, **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """Fit the transformer to the (sequence of) series and return the transformed input.

        Parameters
        ----------
        series
            the (sequence of) series to transform.
        args
            Additional positional arguments for the :func:`ts_transform` method
        kwargs
            Additional keyword arguments for the :func:`ts_transform` method:

            component_mask : Optional[np.ndarray] = None
                Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which
                components of the underlying `series` the transform should be fitted and applied to.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Transformed data.
        """

        component_mask = kwargs.pop("component_mask", None)
        return self.fit(series, component_mask=component_mask).transform(
            series, *args, component_mask=component_mask, **kwargs
        )

    def _get_params(
        self, n_timeseries: int, include_fitted_params: bool = True
    ) -> Generator[Mapping[str, Any], None, None]:
        """
        Overrides `_get_params` of `BaseDataTransformer`. Creates generator of dictionaries containing
        both the fixed parameter values (i.e. attributes defined in the child-most class), as well as
        the fitted parameter values (only if `include_fitted_params=True`). Those fixed parameters
        specified by `parallel_params` are given different values over each of the parallel jobs;
        every fitted parameter is given a different value for each parallel job (since `self.fit`
        returns a `Sequence` containing one fitted parameter value of each parallel job). Called by
        `_transform_iterator` and `_inverse_transform_iterator`.
        """
        self._check_fixed_params(n_timeseries)
        fitted_params = self._get_fitted_params(n_timeseries, include_fitted_params)

        def params_generator(
            n_timeseries, fixed_params, fitted_params, parallel_params
        ):
            fixed_params_copy = fixed_params.copy()
            for i in range(n_timeseries):
                for key in parallel_params:
                    fixed_params_copy[key] = fixed_params[key][i]
                params = {}
                if fixed_params_copy:
                    params["fixed"] = fixed_params_copy
                if fitted_params:
                    params["fitted"] = fitted_params[i]
                if not params:
                    params = None
                yield params

        return params_generator(
            n_timeseries, self._fixed_params, fitted_params, self._parallel_params
        )

    def _get_fitted_params(
        self, n_timeseries: int, include_fitted_params: bool
    ) -> Sequence[Any]:
        """
        Returns `self._fitted_params` if `include_fitted_params=True`, otherwise returns an empty
        tuple. If `include_fitted_params=True`, also checks that `self._fitted_params`, which is a
        sequence of values, contains exactly `n_timeseries` values; if not, a `ValueError` is thrown.
        """
        if include_fitted_params:
            raise_if_not(
                self._fit_called,
                ("Must call `fit` before calling `transform`/`inverse_transform`."),
            )
            fitted_params = self._fitted_params
        else:
            fitted_params = tuple()
        if fitted_params:
            raise_if(
                n_timeseries > len(fitted_params),
                (
                    f"{n_timeseries} TimeSeries were provided "
                    f"but only {len(fitted_params)} TimeSeries "
                    f"were specified upon training {self.name}."
                ),
            )
        return fitted_params
