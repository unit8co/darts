"""
Invertible Data Transformer Base Class
--------------------------------------
"""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
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

        All the deriving classes have to implement the static methods :func:`ts_transform()` and
        :func:`ts_inverse_transform()`. For information on how to implement the :func:`ts_transform` method,
        please refer to the :class:`.BaseDataTransformer` documentation.

        The :func:`ts_inverse_transform()` method should be implemented in a virtually identical way to the
        :func:`ts_transform()` method: it should accept a ``TimeSeries`` as a first argument, and a dictionary
        of fixed parameters (as well as fitted parameters if the transformation also inherits from
        `FittableDataTransformer`) as a second argument. Additionally, :func:`ts_inverse_transform()` should
        also accept `*args` and `**kwargs` if additional positional/keyword arguments are expected to be
        passed. The only difference between :func:`ts_inverse_transform()` and :func:`ts_transform()`
        is that the former should 'undo' the transformation made to a `TimeSeries` by the latter. Please
        refer to the :func:`ts_inverse_transform()` documentation for further information.

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
            Optionally, whether to automatically apply any provided `component_mask`s to the
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

        Example
        --------
        >>> from darts.dataprocessing.transformers import InvertibleDataTransformer
        >>> from darts.utils.timeseries_generation import linear_timeseries
        >>>
        >>> class SimpleTransform(InvertibleDataTransformer):
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
        >>>         @staticmethod
        >>>         def ts_inverse_transform(series, params, **kwargs):
        >>>             a = params['fixed']['_a']
        >>>             b = kwargs.pop('b')
        >>>             return (series - b) / a
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
        >>> transform = SimpleTransform(a=2)
        >>> series = transform.transform(series, b=3)
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
        >>> series = transform.inverse_transform(series, b=3)
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
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=mask_components,
        )

    @classmethod
    @component_masking
    def _ts_inverse_transform(cls, *args, **kwargs):
        """Applies component masking to `ts_inverse_transform`."""
        return cls.ts_inverse_transform(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any]
    ) -> TimeSeries:
        """The function that will be applied to each series when :func:`inverse_transform` is called.

        The function must take as first argument a ``TimeSeries`` object and, as a second argument, a
        dictionary containing the fixed and/or fitted parameters of the transformation; this function
        should then return an inverse transformed ``TimeSeries`` object (i.e. `ts_inverse_transform` should
        'undo' the transformation performed by `ts_transform`).

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
        to `ts_inverse_transform`; hence, `ts_inverse_transform` should also accept `*args` and/or `**kwargs` if
        positional/keyword arguments are passed to `transform`. Note that if the `mask_components` attribute of
        `InvertibleDataTransformer` is set to `False`, then the `component_mask` provided to `transform` will be passed
        as an additional keyword argument to `ts_inverse_transform`.

        The `BaseDataTransformer` class, from which `InvertibleDataTransformer` inherits, includes some helper methods
        which may prove useful when implementing a `ts_inverse_transform` function:
            1. The `apply_component_mask` and `unapply_component_mask` methods, which apply and 'unapply'
            `component_mask`s to a `TimeSeries` respectively; these methods are automatically called in `transform` if
            the `mask_component` attribute of `InvertibleDataTransformer` is set to `True`, but you may want to manually
            call them if you set `mask_components` to `False` and wish to manually specify how `component_mask`s are
            applied to a `TimeSeries`.
            2. The `stack_samples` method, which stacks all the samples in a `TimeSeries` along
            the component axis, so that the `TimeSeries` goes from shape `(n_timesteps, n_components, n_samples)` to
            shape `(n_timesteps, n_components * n_samples)`. This stacking is useful if a pointwise inverse transform
            is being implemented (i.e. transforming the value at time `t` depends only on the value of the series at
            that time `t`). Once transformed, the stacked `TimeSeries` can be 'unstacked' using the `unstack_samples`
            method.

        This method is not implemented in the base class and must be implemented in the deriving classes.

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
            Any additional keyword arguments provided to `inverse_transform`.
        kwargs
            Any additional keyword arguments provided to `inverse_transform`. Note that if the `mask_component`
            attribute of `InvertibleDataTransformer` is set to `False`, then `component_mask` will
            be passed as a keyword argument.

        Notes
        -----
        This method is designed to be a static method instead of instance methods to allow an efficient
        parallelisation also when the scaler instance is storing a non-negligible amount of data. Using instance
        methods would imply copying the instance's data through multiple processes, which can easily introduce a
        bottleneck and nullify parallelisation benefits.
        """
        pass

    def inverse_transform(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]],
        *args,
        component_mask: Optional[np.array] = None,
        series_idx: Optional[Union[int, Sequence[int]]] = None,
        **kwargs,
    ) -> Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]]:
        """Inverse transforms a (sequence of) series by calling the user-implemented `ts_inverse_transform` method.

        In case a sequence or list of lists is passed as input data, this function takes care of parallelising the
        transformation of multiple series in the sequence at the same time. Additionally,
        if the `mask_components` attribute was set to `True` when instantiating `InvertibleDataTransformer`,
        then any provided `component_mask`s will be automatically applied to each input `TimeSeries`;
        please refer to 'Notes' for further details on component masking.

        Any additionally specified `*args` and `**kwargs` are automatically passed to `ts_inverse_transform`.

        Parameters
        ----------
        series
            The series to inverse-transform.
            If a single `TimeSeries`, returns a single series.
            If a sequence of `TimeSeries`, returns a list of series. The series should be in the same order as the
            sequence used to fit the transformer.
            If a list of lists of `TimeSeries`, returns a list of lists of series. This can for example be the output
            of `ForecastingModel.historical_forecasts()` when using multiple series. Each inner list should contain
            `TimeSeries` related to the same series. The order of inner lists should be the same as the sequence used
            to fit the transformer.
        args
            Additional positional arguments for the :func:`ts_inverse_transform()` method
        component_mask : Optional[np.ndarray] = None
            Optionally, a 1-D boolean np.ndarray of length ``series.n_components`` that specifies
            which components of the underlying `series` the inverse transform should consider.
        series_idx
            Optionally, the index(es) of each series corresponding to their positions within the series used to fit
            the transformer (to retrieve the appropriate transformer parameters).
        kwargs
            Additional keyword arguments for the :func:`ts_inverse_transform()` method

        Returns
        -------
        Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]
            Inverse transformed data.

        Notes
        -----
        If the `mask_components` attribute was set to `True` when instantiating `InvertibleDataTransformer`,
        then any provided `component_mask`s will be automatically applied to each `TimeSeries` input to
        transform; `component_mask`s are simply boolean arrays of shape `(series.n_components,)` that
        specify which components of each `series` should be transformed using `ts_inverse_transform` and which
        components should not. If `component_mask[i]` is `True`, then the `i`th component of each
        `series` will be transformed by `ts_inverse_transform`. Conversely, if `component_mask[i]` is `False`,
        the `i`th component will be removed from each `series` before being passed to `ts_inverse_transform`;
        after transforming this masked series, the untransformed `i`th component will be 'added back'
        to the output. Note that automatic `component_mask`ing can only be performed if the `ts_inverse_transform`
        does *not* change the number of timesteps in each series; if this were to happen, then the transformed
        and untransformed components are unable to be concatenated back together along the component axis.

        If `mask_components` was set to `False` when instantiating `InvertibleDataTransformer`, then any provided
        `component_masks` will be passed as a keyword argument `ts_inverse_transform`; the user can then manually
        specify how the `component_mask` should be applied to each series.
        """
        if hasattr(self, "_fit_called") and not self._fit_called:
            raise_log(
                ValueError("fit() must have been called before inverse_transform()"),
                logger=logger,
            )

        desc = f"Inverse ({self._name})"

        # Take note of original input for unmasking purposes:
        called_with_single_series = False
        called_with_sequence_series = False
        series_specified = series_idx is not None
        if isinstance(series, TimeSeries):
            data = [series]
            if series_specified:
                transformer_selector = self._process_series_idx(series_idx)
            else:
                transformer_selector = [0]
            called_with_single_series = True
        elif isinstance(series[0], TimeSeries):  # Sequence[TimeSeries]
            data = series
            if series_specified:
                transformer_selector = self._process_series_idx(series_idx)
            else:
                transformer_selector = range(len(series))
            called_with_sequence_series = True
        else:  # Sequence[Sequence[TimeSeries]]
            data = []
            transformer_selector = []
            if series_specified:
                iterator_ = zip(self._process_series_idx(series_idx), series)
            else:
                iterator_ = enumerate(series)
            for idx, series_list in iterator_:
                data.extend(series_list)
                transformer_selector += [idx] * len(series_list)

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
            total=len(transformer_selector),
        )

        # apply & unapply component masking to the transform method
        kwargs["mask_components"] = self._mask_components
        kwargs["mask_components_apply_only"] = False
        kwargs["component_mask"] = component_mask

        transformed_data = _parallel_apply(
            input_iterator,
            self._ts_inverse_transform,
            self._n_jobs,
            args,
            kwargs,
        )

        if called_with_single_series:
            return transformed_data[0]
        elif called_with_sequence_series:
            return transformed_data
        else:
            cum_len = np.cumsum([0] + [len(s_) for s_ in series])
            return [
                transformed_data[cum_len[i] : cum_len[i + 1]]
                for i in range(len(cum_len) - 1)
            ]
