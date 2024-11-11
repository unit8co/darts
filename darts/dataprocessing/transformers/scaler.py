"""
Scaler
------
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class Scaler(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        scaler=None,
        name="Scaler",
        global_fit: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Generic wrapper class for using scalers on time series.

        The underlying `scaler` has to implement the ``fit()``, ``transform()`` and
        ``inverse_transform()`` methods (typically from scikit-learn).

        When the scaler is applied on multivariate series, the scaling is done per-component.
        When the series are stochastic, the scaling is done across all samples (for each given component).
        The transformation is applied independently for each dimension (component) of the time series,
        effectively merging all samples of a component in order to compute the transform.

        Notes
        -----
        The scaler will not scale the series' static covariates. This has to be done either before constructing the
        series, or later on by extracting the covariates, transforming the values and then reapplying them to the
        series. For this, see TimeSeries properties `TimeSeries.static_covariates` and method
        `TimeSeries.with_static_covariates()`

        Parameters
        ----------
        scaler
            The scaler to transform the data with. It must provide ``fit()``,
            ``transform()`` and ``inverse_transform()`` methods.
            Default: :class:`sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this will scale all
            the values of a time series between 0 and 1.
        name
            A specific name for the scaler
        global_fit
            Optionally, whether all `TimeSeries` passed to the `fit()` method should be used to fit
            a *single* set of parameters, or if a different set of parameters should be independently fitted
            to each provided `TimeSeries`. If `True`, then a `Sequence[TimeSeries]` is passed to `ts_fit`
            and a single set of parameters is fitted using all provided `TimeSeries`. If `False`, then
            each `TimeSeries` is individually passed to `ts_fit`, and a different set of fitted parameters
            if yielded for each of these fitting operations. See `FittableDataTransformer` documentation for
            further details.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Optionally, whether to print operations progress

        Notes
        -----
        In case the :class:`Scaler` is applied to multiple ``TimeSeries`` objects, a deep-copy of the
        chosen scaler will be instantiated, fitted, and stored, for each ``TimeSeries``.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> from darts.dataprocessing.transformers import Scaler
        >>> series = AirPassengersDataset().load()
        >>> scaler = MinMaxScaler(feature_range=(-1, 1))
        >>> transformer = Scaler(scaler)
        >>> series_transformed = transformer.fit_transform(series)
        >>> print(min(series_transformed.values()))
        [-1.]
        >>> print(max(series_transformed.values()))
        [2.]
        """

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))

        if (
            not callable(getattr(scaler, "fit", None))
            or not callable(getattr(scaler, "transform", None))
            or not callable(getattr(scaler, "inverse_transform", None))
        ):
            raise_log(
                ValueError(
                    "The provided transformer object must have fit(), transform() and inverse_transform() methods"
                ),
                logger,
            )
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self.transformer = scaler
        super().__init__(
            name=name, n_jobs=n_jobs, verbose=verbose, global_fit=global_fit
        )

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        transformer = params["fitted"]

        tr_out = transformer.transform(Scaler.stack_samples(series))

        transformed_vals = Scaler.unstack_samples(tr_out, series=series)

        return series.with_values(transformed_vals)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any], *args, **kwargs
    ) -> TimeSeries:
        transformer = params["fitted"]

        tr_out = transformer.inverse_transform(Scaler.stack_samples(series))
        inv_transformed_vals = Scaler.unstack_samples(tr_out, series=series)

        return series.with_values(inv_transformed_vals)

    @staticmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        params: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        transformer = deepcopy(params["fixed"]["transformer"])
        # If `global_fit` is `True`, then `series` will be ` Sequence[TimeSeries]`;
        # otherwise, `series` is a single `TimeSeries`:
        if isinstance(series, TimeSeries):
            series = [series]
        vals = np.concatenate([Scaler.stack_samples(ts) for ts in series], axis=0)
        scaler = transformer.fit(vals)
        return scaler
