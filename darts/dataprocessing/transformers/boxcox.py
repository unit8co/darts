"""
Box-Cox Transformer
-------------------
"""

from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax

from darts.logging import get_logger, raise_if
from darts.timeseries import TimeSeries

from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer

logger = get_logger(__name__)


class BoxCox(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        name: str = "BoxCox",
        lmbda: Optional[
            Union[float, Sequence[float], Sequence[Sequence[float]]]
        ] = None,
        optim_method="mle",
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

        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)

        raise_if(
            not isinstance(optim_method, str)
            or optim_method not in ["mle", "pearsonr"],
            "optim_method parameter must be either 'mle' or 'pearsonr'",
            logger,
        )

        self._lmbda = lmbda
        self._optim_method = optim_method

    def _fit_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Optional[Union[Sequence[float], float]]]]:

        if isinstance(self._lmbda, Sequence) and isinstance(self._lmbda[0], Sequence):
            # CASE 0: Sequence[Sequence[float]]
            raise_if(
                len(self._lmbda) != len(series),
                "with multiple time series the number of lmbdas sequences must equal the number of time \
                        series",
                logger,
            )
            return zip(series, self._lmbda)
        else:
            # CASE 1: Sequence[float], float, None. Replicating the same value for each TS
            lmbda_gen = (self._lmbda for _ in range(len(series)))
            return zip(series, lmbda_gen)

    def _transform_iterator(self, series: Sequence[TimeSeries]) -> Iterator[Tuple]:
        return zip(series, self._fitted_params)

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple]:
        return zip(series, self._fitted_params)

    @staticmethod
    def ts_fit(
        series: TimeSeries,
        lmbda: Optional[Union[float, Sequence[float]]],
        method,
        *args,
        **kwargs
    ) -> Union[Sequence[float], pd.core.series.Series]:
        component_mask = kwargs.get("component_mask", None)

        if lmbda is None:
            # Compute optimal lmbda for each dimension of the time series. In this case, the return type is
            # an ndarray and not a Sequence
            vals = BoxCox._reshape_in(series, component_mask=component_mask)
            lmbda = np.apply_along_axis(boxcox_normmax, axis=0, arr=vals, method=method)

        elif isinstance(lmbda, Sequence):
            raise_if(
                len(lmbda) != series.width,
                "lmbda should have one value per dimension (ie. column or variable) of the time series",
                logger,
            )
        else:
            # Replicate lmbda to match dimensions of the time series
            lmbda = [lmbda] * series.width

        return lmbda

    @staticmethod
    def ts_transform(
        series: TimeSeries,
        lmbda: Union[Sequence[float], pd.core.series.Series],
        **kwargs
    ) -> TimeSeries:
        component_mask = kwargs.get("component_mask", None)

        vals = BoxCox._reshape_in(series, component_mask=component_mask)
        series_width = series.width if component_mask is None else sum(component_mask)
        transformed_vals = np.stack(
            [boxcox(vals[:, i], lmbda=lmbda[i]) for i in range(series_width)], axis=1
        )
        return series.with_values(
            BoxCox._reshape_out(series, transformed_vals, component_mask=component_mask)
        )

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries,
        lmbda: Union[Sequence[float], pd.core.series.Series],
        **kwargs
    ) -> TimeSeries:
        component_mask = kwargs.get("component_mask", None)

        vals = BoxCox._reshape_in(series, component_mask=component_mask)
        inv_transformed_vals = np.stack(
            [inv_boxcox(vals[:, i], lmbda[i]) for i in range(series.width)], axis=1
        )
        return series.with_values(
            BoxCox._reshape_out(
                series, inv_transformed_vals, component_mask=component_mask
            )
        )

    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], **kwargs
    ) -> "FittableDataTransformer":
        # adding lmbda and optim_method params
        return super().fit(series, method=self._optim_method, **kwargs)
