"""
Data Transformer Base Class
---------------------------
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Optional, Union

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils import _build_tqdm_iterator, _parallel_apply
from darts.utils.ts_utils import SeriesType, get_series_seq_type, series2seq

logger = get_logger(__name__)


def component_masking(transformer_method):
    """Applies component masking to the series fed to any `transform` method, and then reverts the masking for the
    final output series.
    """

    @wraps(transformer_method)
    def transform_wrapper(
        cls, series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        # `mask_components` and `component_mask` must be in `kwargs`
        mask_components = kwargs.pop("mask_components")
        mask_components_apply_only = kwargs.pop("mask_components_apply_only")
        component_mask = kwargs.pop("component_mask")

        # remove non-transform columns
        if mask_components and component_mask is not None:
            series_proc = BaseDataTransformer.apply_component_mask(
                series, component_mask, return_ts=True
            )
        else:
            series_proc = series
            if component_mask is not None:
                kwargs["component_mask"] = component_mask

        out = transformer_method(cls, series_proc, params, *args, **kwargs)

        # add back non-transformed columns
        if mask_components and not mask_components_apply_only:
            out = BaseDataTransformer.unapply_component_mask(
                series, out, component_mask
            )
        return out

    return transform_wrapper


class BaseDataTransformer(ABC):
    def __init__(
        self,
        name: str = "BaseDataTransformer",
        n_jobs: int = 1,
        verbose: bool = False,
        parallel_params: Union[bool, Sequence[str]] = False,
        mask_components: bool = True,
    ):
        """Abstract class for data transformers.

        All the deriving classes have to implement the static method :func:`ts_transform`; this implemented
        method can then be applied to ``TimeSeries`` or ``Sequence[TimeSeries]`` inputs by calling the
        :func:`transform()` method. Internally, :func:`transform()` parallelizes func:`ts_transform` over all
        of the ``TimeSeries`` inputs passed to it. See the func:`ts_transform` method documentation
        for further details on how to implement this method in a user-defined class.

        Data transformers requiring to be fit first before calling :func:`transform` should derive
        from :class:`.FittableDataTransformer` instead. Data transformers that are invertible should derive
        from :class:`.InvertibleDataTransformer` instead. Transformers which are both fittable and invertible
        should inherit from both :class:`.FittableDataTransformer` and :class:`.InvertibleDataTransformer`.

        All Data Transformers can store *fixed parameters* that are automatically passed to func:`ts_transform`;
        the fixed parameters of a data transformer object are taken to be all those attributes defined
        in the `__init__` method of the child-most class *before* `super().__init__` is called. The fixed parameter
        values can then be accessed within the func:`ts_transform` method through the `params` dictionary
        argument. More specifically, `params['fixed']` stores a dictionary with all of the fixed parameter
        values, where the keys are simply the attribute names of each fixed parameter (e.g. the `self._my_param`
        fixed parameter attribute is accessed through `params['fixed']['_my_param']`).

        Data Transformers which inherit from :class:`.FittableDataTransformer` can also store fitted parameters
        alongside fixed parameters; please refer to the :class:`.FittableDataTransformer` documentation for
        further details.

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
            See `apply_component_mask` for further details.

        Example
        --------
        >>> from darts.dataprocessing.transformers import BaseDataTransformer
        >>> from darts.utils.timeseries_generation import linear_timeseries
        >>>
        >>> class SimpleTransform(BaseDataTransformer):
        >>>
        >>>         def __init__(self, a):
        >>>             self._a = a
        >>>             super().__init__()
        >>>
        >>>         @staticmethod
        >>>         def ts_transform(series, params, **kwargs):
        >>>             a = params['fixed']['_a']
        >>>             b = kwargs.pop('b')
        >>>             return a*series + b
        >>>
        >>> series = linear_timeseries(length=5)
        >>> print(series)
        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>
        array([[[0.  ]],

            [[0.25]],

            [[0.5 ]],

            [[0.75]],

            [[1.  ]]])
        Coordinates:
        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05
        * component  (component) object 'linear'
        Dimensions without coordinates: sample
        Attributes:
            static_covariates:  None
            hierarchy:          None
        >>> series = SimpleTransform(a=2).transform(series, b=3)
        >>> print(series)
        <TimeSeries (DataArray) (time: 5, component: 1, sample: 1)>
        array([[[3. ]],

            [[3.5]],

            [[4. ]],

            [[4.5]],

            [[5. ]]])
        Coordinates:
        * time       (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05
        * component  (component) object 'linear'
        Dimensions without coordinates: sample
        Attributes:
            static_covariates:  None
            hierarchy:          None
        """
        # Assume `super().__init__` called at *very end* of
        # child-most class's `__init__`, so `vars(self)` contains
        # only attributes of this child-most class:
        self._fixed_params = vars(self).copy()
        # If `parallel_params = True`, but not a list of key values:
        if parallel_params and not isinstance(parallel_params, Sequence):
            parallel_params = self._fixed_params.keys()
        # If `parallel_params = False`:
        elif not parallel_params:
            parallel_params = tuple()
        self._parallel_params = parallel_params
        self._mask_components = mask_components
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
        if not isinstance(value, bool):
            raise_log(ValueError("Verbosity status must be a boolean."), logger=logger)

        self._verbose = value

    def set_n_jobs(self, value: int):
        """Set the number of processors to be used by the transformer while processing multiple ``TimeSeries``.

        Parameters
        ----------
        value
            New n_jobs value.  Set to `-1` for using all the available cores.
        """
        if not isinstance(value, int):
            raise_log(ValueError("n_jobs must be an integer"), logger=logger)
        self._n_jobs = value

    @classmethod
    @component_masking
    def _ts_transform(cls, *args, **kwargs):
        """Applies component masking to `ts_transform`."""
        return cls.ts_transform(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
        """The function that will be applied to each series when :func:`transform()` is called.

        This method is not implemented in the base class and must be implemented in the deriving classes.

        The function must take as first argument a ``TimeSeries`` object and, as a second argument, a
        dictionary containing the fixed and/or fitted parameters of the transformation; this function
        should then return a transformed ``TimeSeries`` object.

        The `params` dictionary *can* contain up to two keys:
            1. `params['fixed']` stores the fixed parameters of the transformation (i.e. attributed
            defined in the `__init__` method of the child-most class *before* `super().__init__` is called);
            `params['fixed']` is a dictionary itself, whose keys are the names of the fixed parameter
            attributes. For example, if `_my_fixed_param` is defined as an attribute in the child-most
            class, then this fixed parameter value can be accessed through `params['fixed']['_my_fixed_param']`.
            2. If the transform inherits from the :class:`.FittableDataTransformer` class, then `params['fitted']`
            will store the fitted parameters of the transformation; the fitted parameters are simply the output(s)
            returned by the `ts_fit` function, whatever those output(s) may be. See :class:`.FittableDataTransformer`
            for further details about fitted parameters.

        Any positional/keyword argument supplied to the `transform` method are passed as positional/keyword arguments
        to `ts_transform`; hence, `ts_transform` should also accept `*args` and/or `**kwargs` if positional/keyword
        arguments are passed to `transform`. Note that if the `mask_components` attribute of `BaseDataTransformer`
        is set to `False`, then the `component_mask` provided to `transform` will be passed as an additional keyword
        argument to `ts_transform`.

        The `BaseDataTransformer` class includes some helper methods which may prove useful when implementing a
        `ts_transform` function:
            1. The `apply_component_mask` and `unapply_component_mask` methods, which apply and 'unapply'
            `component_mask`s to a `TimeSeries` respectively; these methods are automatically called in `transform`
            if the `mask_component` attribute of `BaseDataTransformer` is set to `True`, but you may want to manually
            call them if you set `mask_components` to `False` and wish to manually specify how `component_mask`s are
            applied to a `TimeSeries`.
            2. The `stack_samples` method, which stacks all the samples in a `TimeSeries` along
            the component axis, so that the `TimeSeries` goes from shape `(n_timesteps, n_components, n_samples)` to
            shape `(n_timesteps, n_components * n_samples)`. This stacking is useful if a pointwise transform is being
            implemented (i.e. transforming the value at time `t` depends only on the value of the series at that
            time `t`). Once transformed, the stacked `TimeSeries` can be 'unstacked' using the `unstack_samples` method.

        Parameters
        ----------
        series
            series to be transformed.
        params
            Dictionary containing the parameters of the transformation function. Fixed parameters
            (i.e. attributes defined in the child-most class of the transformation prior to
            calling `super.__init__()`) are stored under the `'fixed'` key. If the transformation
            inherits from the `FittableDataTransformer` class, then the fitted parameters of the
            transformation (i.e. the values returned by `ts_fit`) are stored under the
            `'fitted'` key.
        args
            Any poisitional arguments provided in addition to `series` when
        kwargs
            Any additional keyword arguments provided to `transform`. Note that if the `mask_component`
            attribute of `BaseDataTransformer` is set to `False`, then `component_mask` will
            be passed as a keyword argument.

        Notes
        -----
        This method is designed to be a static method instead of instance method to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def transform(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        *args,
        component_mask: Optional[np.array] = None,
        series_idx: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> Union[TimeSeries, list[TimeSeries]]:
        """Transforms a (sequence of) of series by calling the user-implemeneted `ts_transform` method.

        In case a ``Sequence[TimeSeries]`` is passed as input data, this function takes care of
        parallelising the transformation of multiple series in the sequence at the same time. Additionally,
        if the `mask_components` attribute was set to `True` when instantiating `BaseDataTransformer`,
        then any provided `component_mask`s will be automatically applied to each input `TimeSeries`;
        please refer to 'Notes' for further details on component masking.

        Any additionally specified `*args` and `**kwargs` are automatically passed to `ts_transform`.

        Parameters
        ----------
        series
            (sequence of) series to be transformed.
        args
            Additional positional arguments for each :func:`ts_transform()` method call
        component_mask : Optional[np.ndarray] = None
            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies which
            components of the underlying `series` the transform should consider. If the `mask_components`
            attribute was set to `True` when instantiating `BaseDataTransformer`, then the component mask
            will be automatically applied to each `TimeSeries` input. Otherwise, `component_mask` will be
            provided as an addition keyword argument to `ts_transform`. See 'Notes' for further details.
        series_idx
            Optionally, the index(es) of each series corresponding to their positions within the series used to fit
            the transformer (to retrieve the appropriate transformer parameters).
        kwargs
            Additional keyword arguments for each :func:`ts_transform()` method call

        Returns
        -------
        Union[TimeSeries, List[TimeSeries]]
            Transformed data.

        Notes
        -----
        If the `mask_components` attribute was set to `True` when instantiating `BaseDataTransformer`,
        then any provided `component_mask`s will be automatically applied to each `TimeSeries` input to
        transform; `component_mask`s are simply boolean arrays of shape `(series.n_components,)` that
        specify which components of each `series` should be transformed using `ts_transform` and which
        components should not. If `component_mask[i]` is `True`, then the `i`th component of each
        `series` will be transformed by `ts_transform`. Conversely, if `component_mask[i]` is `False`,
        the `i`th component will be removed from each `series` before being passed to `ts_transform`;
        after transforming this masked series, the untransformed `i`th component will be 'added back'
        to the output. Note that automatic `component_mask`ing can only be performed if the `ts_transform`
        does *not* change the number of timesteps in each series; if this were to happen, then the transformed
        and untransformed components are unable to be concatenated back together along the component axis.

        If `mask_components` was set to `False` when instantiating `BaseDataTransformer`, then any provided
        `component_masks` will be passed as a keyword argument `ts_transform`; the user can then manually specify
        how the `component_mask` should be applied to each series.
        """

        desc = f"Transform ({self._name})"

        # Take note of original input for unmasking purposes:
        series_specified = series_idx is not None
        if isinstance(series, TimeSeries):
            data = [series]
            if series_specified:
                transformer_selector = self._process_series_idx(series_idx)
            else:
                transformer_selector = [0]
        else:
            data = series
            if series_specified:
                transformer_selector = self._process_series_idx(series_idx)
            else:
                transformer_selector = range(len(series))

        input_iterator = _build_tqdm_iterator(
            zip(
                data,
                self._get_params(
                    transformer_selector=transformer_selector,
                    series_specified=series_specified,
                ),
            ),
            verbose=self._verbose,
            desc=desc,
            total=len(data),
        )

        # apply & unapply component masking to the transform method
        kwargs["mask_components"] = self._mask_components
        kwargs["mask_components_apply_only"] = False
        kwargs["component_mask"] = component_mask

        transformed_data = _parallel_apply(
            input_iterator, self._ts_transform, self._n_jobs, args, kwargs
        )
        return (
            transformed_data[0] if isinstance(series, TimeSeries) else transformed_data
        )

    def _get_params(
        self,
        transformer_selector: Iterable,
        series_specified: bool = False,
    ) -> Generator[Mapping[str, Any], None, None]:
        """
        Creates generator of dictionaries containing fixed parameter values
        (i.e. attributes defined in the child-most class). Those fixed parameters
        specified by `parallel_params` are given different values over each of the
        parallel jobs. Called by `transform` and `inverse_transform`,
        if `Transformer` does *not* inherit from `FittableTransformer`.
        """
        self._check_fixed_params(transformer_selector)

        def params_generator(transformer_selector, fixed_params, parallel_params):
            fixed_params_copy = fixed_params.copy()
            for i in transformer_selector:
                for key in parallel_params:
                    fixed_params_copy[key] = fixed_params[key][i]
                if fixed_params_copy:
                    params = {"fixed": fixed_params_copy}
                else:
                    params = None
                yield params
            return None

        return params_generator(
            transformer_selector, self._fixed_params, self._parallel_params
        )

    def _check_fixed_params(self, transformer_selector: Iterable) -> None:
        """
        Raises `ValueError` if `self._parallel_params` specifies a `key` in
        `self._fixed_params` that should be distributed, but
        `len(self._fixed_params[key])` does not equal to the number of time series
        (the maximum value + 1 from `transformer_selector`).
        """
        for key in self._parallel_params:
            n_timeseries_ = max(transformer_selector) + 1
            if n_timeseries_ > len(self._fixed_params[key]):
                raise_log(
                    ValueError(
                        f"{n_timeseries_} TimeSeries were provided "
                        f"but only {len(self._fixed_params[key])} {key} values "
                        f"were specified upon initialising {self.name}."
                    ),
                    logger=logger,
                )
            elif n_timeseries_ < len(self._fixed_params[key]):
                logger.warning(
                    f"Only {n_timeseries_} TimeSeries were provided "
                    f"which is lower than the number of {key} values "
                    f"(n={len(self._fixed_params[key])}) that were specified "
                    f"upon initialising {self.name}."
                )
        return None

    @staticmethod
    def _process_series_idx(series_idx: Union[int, Sequence[int]]) -> Sequence[int]:
        """Convert the `series_idx` to a Sequence[int].

        Note: the validity of the entries in series_idx is checked in _get_params().
        """
        return [series_idx] if isinstance(series_idx, int) else series_idx

    @staticmethod
    def apply_component_mask(
        series: TimeSeries,
        component_mask: Optional[np.ndarray] = None,
        return_ts: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries], np.ndarray, Sequence[np.ndarray]]:
        """
        Extracts components specified by `component_mask` from `series`

        Parameters
        ----------
        series
            input TimeSeries to be fed into transformer.
        component_mask
            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components to
            extract from `series`. The `i`th component of `series` is kept only if `component_mask[i] = True`.
            If not specified, no masking is performed.
        return_ts
            Optionally, specifies that a `TimeSeries` should be returned, rather than an `np.ndarray`.

        Returns
        -------
        masked
            `TimeSeries` (if `return_ts = True`) or `np.ndarray` (if `return_ts = False`) with only those components
            specified by `component_mask` remaining.

        """
        sequence_type_in = get_series_seq_type(series)
        called_with_single_series = sequence_type_in == SeriesType.SINGLE
        series = series2seq(series)

        if component_mask is None:
            if return_ts:
                out = series.copy()
            else:
                out = [series_.all_values() for series_ in series]
            return out[0] if called_with_single_series else out

        if not (
            isinstance(component_mask, np.ndarray) and component_mask.dtype == bool
        ):
            raise_log(
                ValueError(
                    f"`component_mask` must be a boolean `np.ndarray`, not a {type(component_mask)}."
                ),
                logger=logger,
            )

        out = []
        for series_ in series:
            if not series_.width == len(component_mask):
                raise_log(
                    ValueError(
                        "mismatch between number of components in `series` and length of `component_mask`"
                    ),
                    logger=logger,
                )
            if return_ts:
                out_ = series_[series_.columns[component_mask].tolist()]
            else:
                out_ = series_.all_values(copy=False)[:, component_mask, :]
            out.append(out_)
        return out[0] if called_with_single_series else out

    @staticmethod
    def unapply_component_mask(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        vals: Union[np.ndarray, Sequence[np.ndarray], TimeSeries, Sequence[TimeSeries]],
        component_mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Sequence[np.ndarray], TimeSeries, Sequence[TimeSeries]]:
        """
        Adds back components previously removed by `component_mask` in `apply_component_mask` method.

        Parameters
        ----------
        series
            input TimeSeries that was fed into transformer.
        vals:
            `np.ndarray` or `TimeSeries` to 'unmask'
        component_mask
            Optionally, np.ndarray boolean mask of shape (n_components, 1) specifying which components were extracted
            from `series`. If given, insert `vals` back into the columns of the original array. If not specified,
            nothing is 'unmasked'.

        Returns
        -------
        unmasked
            `TimeSeries` (if `vals` is a `TimeSeries`) or `np.ndarray` (if `vals` is an `np.ndarray`) with those
            components previously removed by `component_mask` now 'added back'.
        """
        if component_mask is None:
            return vals

        if not (
            isinstance(component_mask, np.ndarray) and component_mask.dtype == bool
        ):
            raise_log(
                ValueError(
                    "If `component_mask` is given, must be a boolean np.ndarray`"
                ),
                logger=logger,
            )

        sequence_type_in = get_series_seq_type(series)
        called_with_single_series = sequence_type_in == SeriesType.SINGLE
        series = series2seq(series)
        if called_with_single_series:
            vals = [vals]

        out = []
        for series_, vals_ in zip(series, vals):
            if not series_.width == len(component_mask):
                raise_log(
                    ValueError(
                        "mismatch between number of components in `series` and length of `component_mask`"
                    ),
                    logger=logger,
                )
            unmasked = series_.all_values()
            if isinstance(vals_, TimeSeries):
                unmasked[:, component_mask, :] = vals_.all_values()
                # Remove timepoints not present in transformed data:
                unmasked = series_.slice_intersect(vals_).with_values(unmasked)
            else:
                unmasked[:, component_mask, :] = vals_

            out.append(unmasked)
        return out[0] if called_with_single_series else out

    @staticmethod
    def stack_samples(vals: Union[np.ndarray, TimeSeries]) -> np.ndarray:
        """
        Creates an array of shape `(n_timesteps * n_samples, n_components)` from
        either a `TimeSeries` or the `array_values` of a `TimeSeries`.

        Each column of the returned array corresponds to a component (dimension)
        of the series and is formed by concatenating all of the samples associated
        with that component together. More specifically, the `i`th column is formed
        by concatenating
        `[component_i_sample_1, component_i_sample_2, ..., component_i_sample_n]`.

        Stacking is useful when implementing a transformation that applies the exact same
        change to every timestep in the timeseries. In such cases, the samples of each
        component can be stacked together into a single column, and the transformation can then be
        applied to each column, thereby 'vectorising' the transformation over all samples of that component;
        the `unstack_samples` method can then be used to reshape the output. For transformations that depend
        on the `time_index` or the temporal ordering of the observations, stacking should *not* be employed.

        Parameters
        ----------
        vals
            `Timeseries` or `np.ndarray` of shape `(n_timesteps, n_components, n_samples)` to be 'stacked'.

        Returns
        -------
        stacked
            `np.ndarray` of shape `(n_timesteps * n_samples, n_components)`, where the `i`th column is formed
            by concatenating all of the samples of the `i`th component in `vals`.
        """
        if isinstance(vals, TimeSeries):
            vals = vals.all_values()
        shape = vals.shape
        new_shape = (shape[0] * shape[2], shape[1])
        stacked = np.swapaxes(vals, 1, 2).reshape(new_shape)
        return stacked

    @staticmethod
    def unstack_samples(
        vals: np.ndarray,
        n_timesteps: Optional[int] = None,
        n_samples: Optional[int] = None,
        series: Optional[TimeSeries] = None,
    ) -> np.ndarray:
        """
        Reshapes the 2D array returned by `stack_samples` back into an array of shape
        `(n_timesteps, n_components, n_samples)`; this 'undoes' the reshaping of
        `stack_samples`. Either `n_components`, `n_samples`, or `series` must be specified.

        Parameters
        ----------
        vals
            `np.ndarray` of shape `(n_timesteps * n_samples, n_components)` to be 'unstacked'.
        n_timesteps
            Optionally, the number of timesteps in the array originally passed to `stack_samples`.
            Does *not* need to be provided if `series` is specified.
        n_samples
            Optionally, the number of samples in the array originally passed to `stack_samples`.
            Does *not* need to be provided if `series` is specified.
        series
            Optionally, the `TimeSeries` object used to create `vals`; `n_samples` is inferred
            from this.

        Returns
        -------
        unstacked
            `np.ndarray` of shape `(n_timesteps, n_components, n_samples)`.

        """
        if series is not None:
            n_samples = series.n_samples
        else:
            if all(x is None for x in [n_timesteps, n_samples]):
                raise_log(
                    ValueError(
                        "Must specify either `n_timesteps`, `n_samples`, or `series`."
                    ),
                    logger=logger,
                )
        n_components = vals.shape[-1]
        if n_timesteps is not None:
            reshaped_vals = vals.reshape(n_timesteps, -1, n_components)
        else:
            reshaped_vals = vals.reshape(-1, n_samples, n_components)
        unstacked = np.swapaxes(reshaped_vals, 1, 2)
        return unstacked

    @property
    def name(self):
        """Name of the data transformer."""
        return self._name

    def __str__(self):
        return self._name

    def __repr__(self):
        return self.__str__()
