"""
Fittable Data Transformer Base Class
------------------------------------
"""

from abc import abstractmethod
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.base_data_transformer import (
    BaseDataTransformer,
    component_masking,
)
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply

logger = get_logger(__name__)


class FittableDataTransformer(BaseDataTransformer):
    def __init__(
        self,
        name: str = "FittableDataTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
        parallel_params: Union[bool, Sequence[str]] = False,
        mask_components: bool = True,
        global_fit: bool = False,
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
            Optionally, whether to automatically apply any provided `component_mask`s to the
            `TimeSeries` inputs passed to `transform`, `fit`, `inverse_transform`, or `fit_transform`.
            If `True`, any specified `component_mask` will be applied to each input timeseries
            before passing them to the called method; the masked components will also be automatically
            'unmasked' in the returned `TimeSeries`. If `False`, then `component_mask` (if provided) will
            be passed as a keyword argument, but won't automatically be applied to the input timeseries.
            See `apply_component_mask` method of `BaseDataTransformer` for further details.
        global_fit
            Optionally, whether all `TimeSeries` passed to the `fit()` method should be used to fit
            a *single* set of parameters, or if a different set of parameters should be independently fitted
            to each provided `TimeSeries`. If `True`, then a `Sequence[TimeSeries]` is passed to `ts_fit`
            and a single set of parameters is fitted using all provided `TimeSeries`. If `False`, then
            each `TimeSeries` is individually passed to `ts_fit`, and a different set of fitted parameters
            if yielded for each of these fitting operations. See `ts_fit` for further details.

        Notes
        -----
        If `global_fit` is `False` and `fit` is called with a `Sequence` containing `n` different `TimeSeries`,
        then `n` sets of parameters will be fitted. When `transform` and/or `inverse_transform` is subsequently
        called with a `Series[TimeSeries]`, the `i`th set of fitted parameter values will be passed to
        `ts_transform`/`ts_inverse_transform` to transform the `i`th `TimeSeries` in this sequence. Conversely,
        if `global_fit` is `True`, then only a single set of fitted values will be produced when `fit` is
        provided with a `Sequence[TimeSeries]`. Consequently, if a `Sequence[TimeSeries]` is then passed to
        `transform`/`inverse_transform`, each of these `TimeSeries` will be transformed using the exact same set
        of fitted parameters.

        Note that if an invertible *and* fittable data transformer is to be globally fitted, the data transformer
        class should first inherit from `FittableDataTransformer` and then from `InvertibleDataTransformer`. In
        other words, `MyTransformer(FittableDataTransformer, InvertibleDataTransformer)` is correct, but
        `MyTransformer(InvertibleDataTransformer, FittableDataTransformer)` is **not**. If this is not implemented
        correctly, then the `global_fit` parameter will not be correctly passed to `FittableDataTransformer`'s
        constructor.

        The :func:`ts_transform()` and :func:`ts_fit()` methods are designed to be static methods instead of instance
        methods to allow an efficient parallelisation also when the scaler instance is storing a non-negligible
        amount of data. Using instance methods would imply copying the instance's data through multiple processes, which
        can easily introduce a bottleneck and nullify parallelisation benefits.

        Example
        --------
        >>> from darts.dataprocessing.transformers import FittableDataTransformer
        >>> from darts.utils.timeseries_generation import linear_timeseries
        >>>
        >>> class SimpleRangeScaler(FittableDataTransformer):
        >>>
        >>>     def __init__(self, scale, position):
        >>>         self._scale = scale
        >>>         self._position = position
        >>>         super().__init__()
        >>>
        >>>     @staticmethod
        >>>     def ts_transform(series, params):
        >>>         vals = series.all_values(copy=False)
        >>>         fit_params = params['fitted']
        >>>         unit_scale = (vals - fit_params['position'])/fit_params['scale']
        >>>         fix_params = params['fixed']
        >>>         rescaled = fix_params['_scale'] * unit_scale + fix_params['_position']
        >>>         return series.from_values(rescaled)
        >>>
        >>>     @staticmethod
        >>>     def ts_fit(series, params):
        >>>         vals = series.all_values(copy=False)
        >>>         scale = vals.max() - vals.min()
        >>>         position = vals[0]
        >>>         return {'scale': scale, 'position': position}
        >>>
        >>> series = linear_timeseries(length=5, start_value=1, end_value=5)
        >>> print(series)
        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>
        array([[[1.]],

            [[2.]],

            [[3.]],

            [[4.]],

            [[5.]]])
        Coordinates:
        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05
        * component  (component) object 'linear'
        Dimensions without coordinates: sample
        Attributes:
            static_covariates:  None
            hierarchy:          None
        >>> series = SimpleRangeScaler(scale=2, position=-1).fit_transform(series)
        >>> print(series)
        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>
        array([[[-1. ]],

            [[-0.5]],

            [[ 0. ]],

            [[ 0.5]],

            [[ 1. ]]])
        Coordinates:
        * time       (time) int64 0 1 2 3 4
        * component  (component) <U1 '0'
        Dimensions without coordinates: sample
        Attributes:
            static_covariates:  None
            hierarchy:          None
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=mask_components,
        )

        self._fit_called = False
        self._fitted_params = None  # stores the fitted parameters/objects
        self._global_fit = global_fit

    @classmethod
    @component_masking
    def _ts_fit(cls, *args, **kwargs):
        """Applies component masking to `ts_fit`."""
        return cls.ts_fit(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        params: Mapping[str, Any],
        *args,
        **kwargs,
    ):
        """The function that will be applied to each series when :func:`fit` is called.

        If the `global_fit` attribute is set to `False`, then `ts_fit` should accept a `TimeSeries` as a first
        argument and return a set of parameters that are fitted to this individual `TimeSeries`. Conversely, if the
        `global_fit` attribute is set to `True`, then `ts_fit` should accept a `Sequence[TimeSeries]` and
        return a set of parameters that are fitted to *all* of the provided `TimeSeries`. All these parameters will
        be stored in ``self._fitted_params``, which can be later used during the transformation step.

        Regardless of whether the `global_fit` attribute is set to `True` or `False`, `ts_fit` should also accept
        a dictionary of fixed parameter values as a second argument (i.e. `params['fixed'] contains the fixed
        parameters of the data transformer).

        Any additional positional and/or keyword arguments passed to the `fit` method will be passed as
        positional/keyword arguments to `ts_fit`.

        This method is not implemented in the base class and must be implemented in the deriving classes.

        If more parameters are added as input in the derived classes, :func:`_fit_iterator()`
        should be redefined accordingly, to yield the necessary arguments to this function (See
        :func:`_fit_iterator()` for further details)

        Parameters
        ----------
        series (Union[TimeSeries, Sequence[TimeSeries]])
            `TimeSeries` against which the scaler will be fit.

        Notes
        -----
        This method is designed to be a static method instead of instance methods to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        *args,
        component_mask: Optional[np.array] = None,
        **kwargs,
    ) -> "FittableDataTransformer":
        """Fits transformer to a (sequence of) `TimeSeries` by calling the user-implemented `ts_fit` method.

        The fitted parameters returned by `ts_fit` are stored in the ``self._fitted_params`` attribute.
        If a `Sequence[TimeSeries]` is passed as the `series` data, then one of two outcomes will occur:
            1. If the `global_fit` attribute was set to `False`, then a different set of parameters will be
            individually fitted to each `TimeSeries` in the `Sequence`. In this case, this function automatically
            parallelises this fitting process over all of the multiple `TimeSeries` that have been passed.
            2. If the `global_fit` attribute was set to `True`, then all of the `TimeSeries` objects will be used
            fit a single set of parameters.

        Parameters
        ----------
        series
            (sequence of) series to fit the transformer on.
        args
            Additional positional arguments for the :func:`ts_fit` method
        component_mask : Optional[np.ndarray] = None
            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which
            components of the underlying `series` the transform should be fitted to.
        kwargs
            Additional keyword arguments for the :func:`ts_fit` method

        Returns
        -------
        FittableDataTransformer
            Fitted transformer.
        """
        self._fit_called = True

        desc = f"Fitting ({self._name})"

        if isinstance(series, TimeSeries):
            data = [series]
            transformer_selector = [0]
        else:
            data = series
            transformer_selector = range(len(series))

        params_iterator = self._get_params(
            transformer_selector=transformer_selector, calling_fit=True
        )
        fit_iterator = (
            zip(data, params_iterator)
            if not self._global_fit
            else zip([data], params_iterator)
        )
        n_jobs = len(data) if not self._global_fit else 1
        input_iterator = _build_tqdm_iterator(
            fit_iterator, verbose=self._verbose, desc=desc, total=n_jobs
        )

        # apply component masking to the fit method
        kwargs["mask_components"] = self._mask_components
        kwargs["mask_components_apply_only"] = True
        kwargs["component_mask"] = component_mask

        self._fitted_params = _parallel_apply(
            input_iterator, self._ts_fit, self._n_jobs, args, kwargs
        )
        return self

    def transform(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        *args,
        component_mask: Optional[np.array] = None,
        series_idx: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> Union[TimeSeries, list[TimeSeries]]:
        return super().transform(
            series=series,
            *args,
            component_mask=component_mask,
            series_idx=series_idx if not self._global_fit else None,
            **kwargs,
        )

    def fit_transform(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        *args,
        component_mask: Optional[np.array] = None,
        **kwargs,
    ) -> Union[TimeSeries, list[TimeSeries]]:
        """Fit the transformer to the (sequence of) series and return the transformed input.

        Parameters
        ----------
        series
            the (sequence of) series to transform.
        args
            Additional positional arguments passed to the :func:`ts_transform` and :func:`ts_fit` methods.
        component_mask : Optional[np.ndarray] = None
            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which
            components of the underlying `series` the transform should be fitted and applied to.
        kwargs
            Additional keyword arguments passed to the :func:`ts_transform` and :func:`ts_fit` methods.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Transformed data.
        """
        return self.fit(
            series, *args, component_mask=component_mask, **kwargs
        ).transform(series, *args, component_mask=component_mask, **kwargs)

    def _get_params(
        self,
        transformer_selector: Iterable,
        calling_fit: bool = False,
        series_specified: bool = False,
    ) -> Generator[Mapping[str, Any], None, None]:
        """
        Overrides `_get_params` of `BaseDataTransformer`. Creates generator of dictionaries containing
        both the fixed parameter values (i.e. attributes defined in the child-most class), as well as
        the fitted parameter values (only if `calling_fit = False`). Those fixed parameters
        specified by `parallel_params` are given different values over each of the parallel jobs;
        every fitted parameter is given a different value for each parallel job (since `self.fit`
        returns a `Sequence` containing one fitted parameter value of each parallel job). Called by
        `transform` and `inverse_transform`.
        """
        # Call `_check_fixed_params` of `BaseDataTransformer`:
        self._check_fixed_params(transformer_selector)
        fitted_params = self._get_fitted_params(
            transformer_selector, calling_fit, series_specified=series_specified
        )

        def params_generator(
            transformer_selector_,
            fixed_params,
            fitted_params,
            parallel_params,
            global_fit,
        ):
            fixed_params_copy = fixed_params.copy()
            for i in transformer_selector_:
                for key in parallel_params:
                    fixed_params_copy[key] = fixed_params[key][i]
                params = {}
                if fixed_params_copy:
                    params["fixed"] = fixed_params_copy
                if fitted_params:
                    params["fitted"] = (
                        fitted_params[0] if global_fit else fitted_params[i]
                    )
                if not params:
                    params = None
                yield params

        transformer_selector_ = (
            transformer_selector if not (calling_fit and self._global_fit) else [0]
        )

        return params_generator(
            transformer_selector_,
            self._fixed_params,
            fitted_params,
            self._parallel_params,
            self._global_fit,
        )

    def _get_fitted_params(
        self,
        transformer_selector: Iterable,
        calling_fit: bool,
        series_specified: bool = False,
    ) -> Sequence[Any]:
        """
        Returns `self._fitted_params` if `calling_fit = False`, otherwise returns an empty
        tuple. If `calling_fit = False`, also checks that `self._fitted_params`, which is a
        sequence of values, contains exactly `transformer_selector` values; if not, a `ValueError` is thrown.
        """
        if not calling_fit:
            if not self._fit_called:
                raise_log(
                    ValueError(
                        "Must call `fit` before calling `transform`/`inverse_transform`."
                    ),
                    logger=logger,
                )
            fitted_params = self._fitted_params
        else:
            fitted_params = tuple()
        if not self._global_fit and fitted_params:
            n_timeseries_ = max(transformer_selector) + 1
            if n_timeseries_ > len(fitted_params):
                raise_log(
                    ValueError(
                        f"{n_timeseries_} TimeSeries were provided "
                        f"but only {len(fitted_params)} TimeSeries "
                        f"were specified upon training {self.name}."
                    ),
                    logger=logger,
                )
            elif n_timeseries_ < len(fitted_params) and not series_specified:
                logger.warning(
                    f"Only {n_timeseries_} TimeSeries (lists) were provided "
                    f"which is lower than the number of series (n={len(fitted_params)}) "
                    f"used to fit {self.name}. This can result in a mismatch between the "
                    f"series and the underlying transformers."
                )
        return fitted_params
