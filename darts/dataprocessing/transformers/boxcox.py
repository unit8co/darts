"""
Box-Cox Transformer
-------------------
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax

from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_if
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class BoxCox(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        name: str = "BoxCox",
        lmbda: Optional[
            Union[float, Sequence[float], Sequence[Sequence[float]]]
        ] = None,
        optim_method: Literal["mle", "pearsonr"] = "mle",
        global_fit: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Box-Cox data transformer.

        See [1]_ for more information about Box-Cox transforms.

        The transformation is applied independently for each dimension (component) of the time series.
        For stochastic series, it is done jointly over all samples, effectively merging all samples of
        a component in order to compute the transform.

        Notes
        -----
        The scaler will not scale the series' static covariates. This has to be done either before constructing the
        series, or later on by extracting the covariates, transforming the values and then reapplying them to the
        series. For this, see TimeSeries properties `TimeSeries.static_covariates` and method
        `TimeSeries.with_static_covariates()`

        Parameters
        ----------
        name
            A specific name for the transformer
        lmbda
            The parameter :math:`\\lambda` of the Box-Cox transform. If a single float is given, the same
            :math:`\\lambda` value will be used for all dimensions of the series, for all the series.
            If a sequence is given, there is one value per component in the series. If a sequence of sequence
            is given, there is one value per component for all series.
            If `None` given, will automatically find an optimal value of :math:`\\lambda` (for each dimension
            of the time series, for each time series) using :func:`scipy.stats.boxcox_normmax`
            with ``method=optim_method``.
        optim_method
            Specifies which method to use to find an optimal value for the lmbda parameter.
            Either 'mle' or 'pearsonr'. Ignored if `lmbda` is not `None`.
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
            passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Whether to print operations progress

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import BoxCox
        >>> series = AirPassengersDataset().load()
        >>> transformer = BoxCox(lmbda=0.2)
        >>> series_transformed = transformer.fit_transform(series)
        >>> print(series_transformed.head())
        <TimeSeries (DataArray) (Month: 5, component: 1, sample: 1)>
        array([[[7.84735157]],
            [[7.98214351]],
            [[8.2765364 ]],
            [[8.21563229]],
            [[8.04749318]]])
        Coordinates:
        * Month      (Month) datetime64[ns] 1949-01-01 1949-02-01 ... 1949-05-01
        * component  (component) object '#Passengers'
        Dimensions without coordinates: sample

        References
        ----------
        .. [1] https://otexts.com/fpp2/transformations.html#mathematical-transformations
        """
        raise_if(
            not isinstance(optim_method, str)
            or optim_method not in ["mle", "pearsonr"],
            "optim_method parameter must be either 'mle' or 'pearsonr'",
            logger,
        )

        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self._lmbda = lmbda
        self._optim_method = optim_method
        if isinstance(lmbda, Sequence) and isinstance(lmbda[0], Sequence):
            parallel_params = ("_lmbda",)
        else:
            parallel_params = False

        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            parallel_params=parallel_params,
            mask_components=True,
            global_fit=global_fit,
        )

    @staticmethod
    def ts_fit(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        params: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> Union[Sequence[float], pd.Series]:
        lmbda, method = params["fixed"]["_lmbda"], params["fixed"]["_optim_method"]
        # If `global_fit` is `True`, then `series` will be ` Sequence[TimeSeries]`;
        # otherwise, `series` is a single `TimeSeries`:
        if isinstance(series, TimeSeries):
            series = [series]
        if lmbda is None:
            # Compute optimal lmbda for each dimension of the time series. In this case, the return type is
            # an ndarray and not a Sequence
            vals = np.concatenate([BoxCox.stack_samples(ts) for ts in series], axis=0)
            lmbda = np.apply_along_axis(boxcox_normmax, axis=0, arr=vals, method=method)

        elif isinstance(lmbda, Sequence):
            raise_if(
                len(lmbda) != series[0].width,
                "lmbda should have one value per dimension (ie. column or variable) of the time series",
                logger,
            )
        else:
            # Replicate lmbda to match dimensions of the time series
            lmbda = [lmbda] * series[0].width

        return lmbda

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        lmbda = params["fitted"]

        vals = BoxCox.stack_samples(series)
        transformed_vals = np.stack(
            [boxcox(vals[:, i], lmbda=lmbda[i]) for i in range(series.width)], axis=1
        )
        transformed_vals = BoxCox.unstack_samples(transformed_vals, series=series)
        return series.with_values(transformed_vals)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        lmbda = params["fitted"]

        vals = BoxCox.stack_samples(series)
        inv_transformed_vals = np.stack(
            [inv_boxcox(vals[:, i], lmbda[i]) for i in range(series.width)], axis=1
        )
        inv_transformed_vals = BoxCox.unstack_samples(
            inv_transformed_vals, series=series
        )
        return series.with_values(inv_transformed_vals)
