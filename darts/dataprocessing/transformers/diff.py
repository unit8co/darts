"""
Differencing Transformer
------------------------
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.logging import get_logger, raise_log
from darts.timeseries import concatenate

logger = get_logger(__name__)


class Diff(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        lags: int | Sequence[int] = 1,
        dropna: bool = True,
        name: str = "Diff",
        n_jobs: int = 1,
        verbose: bool = False,
        columns: str | list[str] | None = None,
    ):
        r"""Differencing data transformer.

        Differencing is typically applied to a time series to make it stationary; see [1]_ for further details.
        The transformation is applied independently over each dimension (component) and sample of the time series.

        Notes
        -----
        `Diff` sequentially applies a series of :math:`m`-lagged differencing operations (i.e.
        :math:`y\prime_t = y_t - y_{t-m}`) to a time series; please refer to [2]_ for further details about lagged
        differencing. The :math:`m` value to use for each differencing operation is specified by the `lags` parameter;
        for example, setting `lags = [1, 12]` first applies 1-lagged differencing to the time series, and then
        12-lagged differencing to the 1-lagged differenced series.

        Each element in `lags` represents a single first-order differencing operation, so the 'total order' of the
        differencing equals `len(lags)`. To specify second-order differencing then (i.e. the equivalent of
        `series.diff(n=2)`), one should specify `lags = [1,1]` (i.e. two sequential 1-lagged, first-order
        differences); see [3]_ for further details on second-order differencing.

        Upon computing each :math:`m`-lagged difference, the first :math:`m` values of the time series are 'lost',
        since differences cannot be computed for these values. The length of a `series` transformed by
        `Diff(lags=lags)` will therefore be `series.n_timesteps - sum(lags)`.

        Parameters
        ----------
        name
            A specific name for the transformer
        lags
            Specifies the lag values to be used for each first-order differencing operation (i.e. the :math:`m`
            value in :math:`y'_t = y_t - y_{t-m}`). If a single int is provided, only one differencing
            operation is performed with this specified lag value. If a sequence of ints is provided, multiple
            differencing operations are sequentially performed using each value in `lags`, one after the other.
            For example, specifying `lags = [2, 3]` will effectively compute
            `series.diff(n=1, periods=2).diff(n=1, periods=3)`.
        dropna
            Optionally, specifies if values which can't be differenced (i.e. at the start of the series) should be
            dropped. Note that if `dropna = True`, then a `component_mask` cannot be specified, since the undifferenced
            components will be of a different length to the differenced ones.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        verbose
            Whether to print operations progress
        columns
            Optionally, a string or list of strings specifying the names of the components (columns)
            to transform. If specified, only these components will be transformed, and the remaining
            components will be kept untouched. For more information refer to the `BaseDataTransformer`
            documentation.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import Diff
        >>> series = AirPassengersDataset().load()
        >>> first_order_diff = Diff(lags=1, dropna=True).fit_transform(series)
        >>> print(first_order_diff.values()[:5])
        [[ 6.]
         [14.]
         [-3.]
         [-8.]
         [14.]]
        >>> second_order_diff = Diff(lags=[1, 2], dropna=False).fit_transform(series)
        >>> print(second_order_diff.values()[:5])
        [[ nan]
         [ nan]
         [ nan]
         [ -9.]
         [-22.]]

        References
        ----------
        .. [1] https://otexts.com/fpp2/stationarity.html
        .. [2] https://otexts.com/fpp2/stationarity.html#seasonal-differencing
        .. [3] https://otexts.com/fpp2/stationarity.html#second-order-differencing

        """

        if not isinstance(lags, Sequence):
            lags = (lags,)
        # Define fixed params (i.e. attributes defined before calling `super().__init__`):
        self._lags = lags
        self._dropna = dropna
        # Don't automatically apply `component_mask` - need to throw error when `dropna = True`
        # and `component_mask` is specified:
        super().__init__(
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
            mask_components=False,
            columns=columns,
        )

    @staticmethod
    def ts_fit(series: TimeSeries, params: Mapping[str, Any], **kwargs) -> Any:
        lags, _ = params["fixed"]["_lags"], params["fixed"]["_dropna"]
        lags_sum = sum(lags)
        if series.n_timesteps <= lags_sum:
            raise_log(
                ValueError(
                    f"Series requires at least {lags_sum + 1} timesteps "
                    f"to difference with lags {lags}; series only has "
                    f"{series.n_timesteps} timesteps."
                ),
                logger,
            )
        component_mask = kwargs.get("component_mask")

        # store start values for later undifferencing (inverse transform)
        # `start_vals` store all columns, also the ones that are not diffed
        start_vals = series.all_values(copy=True)[:lags_sum]
        # `diffed` focus on columns that are actually diffed; first `lags_sum` values of time series will be 'lost'
        diffed = Diff.apply_component_mask(series, component_mask, return_ts=False)[
            :lags_sum
        ]
        cutoff = 0
        component_mask_ = component_mask if component_mask is not None else slice(None)
        for lag in lags:
            # Store first `lag` values of current differencing step:
            start_vals[cutoff:, component_mask_, :] = diffed
            diffed = diffed[lag:, :, :] - diffed[:-lag, :, :]
            cutoff += lag
        return start_vals, component_mask, series.start_time(), series.freq

    @staticmethod
    def ts_transform(
        series: TimeSeries, params: Mapping[str, Any], **kwargs
    ) -> TimeSeries:
        lags, dropna = params["fixed"]["_lags"], params["fixed"]["_dropna"]
        component_mask = kwargs.get("component_mask")
        diffed = Diff.apply_component_mask(series, component_mask, return_ts=True)
        for lag in lags:
            diffed = diffed.diff(n=1, periods=lag, dropna=dropna)
        # `series` needs same `n_timesteps` as `diffed` for `unapply_component_mask`
        if dropna:
            series = series.drop_before(sum(lags) - 1)
        return Diff.unapply_component_mask(series, diffed, component_mask)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries,
        params: Mapping[str, Any],
        **kwargs,
    ) -> TimeSeries:
        lags, dropna = params["fixed"]["_lags"], params["fixed"]["_dropna"]
        start_vals, fit_component_mask, start_time, freq = params["fitted"]
        insample: TimeSeries | None = kwargs.pop("insample", None)

        if series.freq != freq:
            raise_log(
                ValueError(
                    f"Series is of frequency {series.freq}, but "
                    f"transform was fitted to data of frequency {freq}."
                ),
                logger,
            )
        # Start dates 'missing' from differenced series if dropna = True, so need to shift forward:
        expected_start = start_time + sum(lags) * series.freq if dropna else start_time

        component_mask = kwargs.get("component_mask")
        if np.any(fit_component_mask != component_mask):
            raise_log(
                ValueError(
                    "Provided `component_mask` does not match "
                    "`component_mask` specified when `fit` was called."
                ),
                logger,
            )

        if insample is not None:
            # `insample` must be the direct output of fit_transform() on the training series,
            # i.e. the transformed series in the same space as `series` (the forecast).
            # We prepend the relevant suffix of `insample` to `series` so that the combined
            # series starts at `expected_start`, satisfying the standard inverse-transform
            # entry condition, then slice back to only the forecast window.
            if insample.freq != freq:
                raise_log(
                    ValueError(
                        f"insample is of frequency {insample.freq}, but "
                        f"transform was fitted to data of frequency {freq}."
                    ),
                    logger,
                )
            if insample.start_time() > expected_start:
                raise_log(
                    ValueError(
                        f"Expected insample to begin at or before time {expected_start}; "
                        f"instead, it begins at {insample.start_time()}. "
                        "insample must be the direct output of fit_transform() "
                        "applied to the training series."
                    ),
                    logger,
                )
            elif insample.start_time() < expected_start:
                # Trim to expected_start — this can happen during rolling-window retrain
                # when insample covers history prior to the current fit window.
                insample = insample.drop_before(expected_start, keep_point=True)
            if insample.n_components != series.n_components:
                raise_log(
                    ValueError(
                        f"Expected insample to have {series.n_components} components; "
                        f"instead, it has {insample.n_components}."
                    ),
                    logger,
                )
            if insample.n_samples != series.n_samples:
                raise_log(
                    ValueError(
                        f"Expected insample to have {series.n_samples} samples; "
                        f"instead, it has {insample.n_samples}."
                    ),
                    logger,
                )
            forecast_start = series.start_time()
            # Keep only the part of insample that strictly precedes the forecast.
            # When insample already ends before the forecast (the common case), use it
            # as-is to avoid an out-of-bounds error from drop_after().
            if insample.end_time() < forecast_start:
                suffix = insample
            else:
                suffix = insample.drop_after(forecast_start)
            if suffix.n_timesteps == 0:
                raise_log(
                    ValueError(
                        "insample must contain at least one timestep strictly "
                        f"before the forecast start {forecast_start}; "
                        f"insample ends at {insample.end_time()}."
                    ),
                    logger,
                )
            # combined starts at expected_start, satisfying the standard entry condition.
            combined = concatenate([suffix, series], axis=0)
            # kwargs no longer contains 'insample' (already popped), so this won't recurse.
            result = Diff.ts_inverse_transform(combined, params, **kwargs)
            # Return only the forecast portion of the undifferenced result.
            return result.drop_before(forecast_start, keep_point=True)

        if series.start_time() != expected_start:
            raise_log(
                ValueError(
                    f"Expected series to begin at time {expected_start}; "
                    f"instead, it begins at time {series.start_time()}."
                ),
                logger,
            )
        if dropna:
            start_shape = (sum(lags), series.n_components, series.n_samples)
            start_fill_vals = np.full(start_shape, fill_value=np.nan)

            # fill start values with components that were not diffed
            if component_mask is not None:
                start_fill_vals[:, ~component_mask, :] = start_vals[
                    :, ~component_mask, :
                ]
            series = series.prepend_values(start_fill_vals)

        # from this point, only look at columns that were actually diffed
        vals = Diff.apply_component_mask(series, component_mask, return_ts=False).copy()
        if component_mask is not None:
            start_vals = start_vals[:, component_mask, :]

        if vals.shape[1] != start_vals.shape[1]:
            raise_log(
                ValueError(
                    f"Expected series to have {start_vals.shape[1]} components; "
                    f"instead, it has {vals.shape[1]}."
                ),
                logger,
            )
        if vals.shape[2] != start_vals.shape[2]:
            raise_log(
                ValueError(
                    f"Expected series to have {start_vals.shape[2]} samples; "
                    f"instead, it has {vals.shape[2]}."
                ),
                logger,
            )
        cutoff = sum(lags)
        for lag in reversed(lags):
            cutoff -= lag
            to_undiff = vals[cutoff:, :, :]
            to_undiff[:lag, :, :] = start_vals[cutoff : cutoff + lag, :, :]
            for i in range(lag):
                to_undiff[i::lag, :, :] = np.cumsum(to_undiff[i::lag, :, :], axis=0)
            vals[cutoff:, :, :] = to_undiff
        vals = Diff.unapply_component_mask(series, vals, component_mask)
        return TimeSeries(
            times=series.time_index,
            values=vals,
            components=series.components,
            copy=False,
            **series._attrs,
        )
