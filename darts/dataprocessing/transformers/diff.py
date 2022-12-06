from typing import Any, Iterator, Sequence, Tuple, Union

import numpy as np

from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import TimeSeries

from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer

logger = get_logger(__name__)


class Diff(FittableDataTransformer, InvertibleDataTransformer):
    def __init__(
        self,
        lags: Union[int, Sequence[int]] = 1,
        dropna: bool = True,
        name: str = "Diff",
        n_jobs: int = 1,
        verbose: bool = False,
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

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.dataprocessing.transformers import Diff
        >>> series = AirPassengersDataset().load()
        >>> first_order_diff = Diff(lags=1, dropna=True).fit_transform(series)
        >>> print(first_order_diff.head())
        <TimeSeries (DataArray) (Month: 5, component: 1, sample: 1)>
        array([[[ 6.]],
            [[14.]],
            [[-3.]],
            [[-8.]],
            [[14.]]])
        Coordinates:
        * Month      (Month) datetime64[ns] 1949-02-01 1949-03-01 ... 1949-06-01
        * component  (component) object '#Passengers'
        >>> second_order_diff = Diff(lags=[1, 2], dropna=False).fit_transform(series)
        >>> print(second_order_diff.head())
        <TimeSeries (DataArray) (Month: 5, component: 1, sample: 1)>
        array([[[ nan]],
            [[ nan]],
            [[ nan]],
            [[ -9.]],
            [[-22.]]])
        Coordinates:
        * Month      (Month) datetime64[ns] 1949-01-01 1949-02-01 ... 1949-05-01
        * component  (component) object '#Passengers'
        References
        ----------
        .. [1] https://otexts.com/fpp2/stationarity.html
        .. [2] https://otexts.com/fpp2/stationarity.html#seasonal-differencing
        .. [3] https://otexts.com/fpp2/stationarity.html#second-order-differencing

        """
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        if not isinstance(lags, Sequence):
            lags = (lags,)
        self._lags = lags
        self._dropna = dropna

    def _fit_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Sequence[int], int]]:
        return ((ts, self._lags, self._dropna) for ts in series)

    @staticmethod
    def ts_fit(series: TimeSeries, lags: Sequence[int], dropna, **kwargs) -> Any:
        lags_sum = sum(lags)
        raise_if(
            series.n_timesteps <= lags_sum,
            (
                f"Series requires at least {lags_sum + 1} timesteps "
                f"to difference with lags {lags}; series only has "
                f"{series.n_timesteps} timesteps."
            ),
        )
        component_mask = Diff._get_component_mask(kwargs, dropna)
        vals = Diff._reshape_in(series, component_mask, flatten=False)
        # First `lags_sum` values of time series will be 'lost' due to differencing;
        # need to remember these values to 'undifference':
        start_vals = vals[:lags_sum, :, :]
        diffed = start_vals
        cutoff = 0
        for lag in lags:
            # Store first `lag` values of current differencing step:
            start_vals[cutoff:, :, :] = diffed
            diffed = diffed[lag:, :, :] - diffed[:-lag, :, :]
            cutoff += lag
        return start_vals, component_mask, series.start_time(), series.freq

    def _transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, Sequence[int], bool]]:
        return ((subseries, self._lags, self._dropna) for subseries in series)

    @staticmethod
    def ts_transform(
        series: TimeSeries, lags: Sequence[int], dropna: bool, **kwargs
    ) -> TimeSeries:
        component_mask = Diff._get_component_mask(kwargs, dropna)
        diffed = series.copy()
        if component_mask is not None:
            diffed = diffed.drop_columns(series.columns[~component_mask])
        for lag in lags:
            diffed = diffed.diff(n=1, periods=lag, dropna=dropna)
        if component_mask is not None:
            # Add back masked (i.e. undifferenced) components:
            diffed = Diff._reshape_out(
                series, diffed.all_values(), component_mask, flatten=False
            )
            diffed = series.with_values(diffed)
        return diffed

    def _inverse_transform_iterator(
        self, series: Sequence[TimeSeries]
    ) -> Iterator[Tuple[TimeSeries, float]]:
        lags = (self._lags for _ in range(len(series)))
        dropna = (self._dropna for _ in range(len(series)))
        return zip(series, lags, dropna, self._fitted_params)

    @staticmethod
    def ts_inverse_transform(
        series: TimeSeries,
        lags: Sequence[int],
        dropna: bool,
        fitted_params: Tuple[np.ndarray, np.ndarray, int, int],
        **kwargs,
    ) -> TimeSeries:
        start_vals, fit_component_mask, start_time, freq = fitted_params
        raise_if_not(
            series.freq == freq,
            (
                f"Series is of frequency {series.freq}, but "
                f"transform was fitted to data of frequency {freq}."
            ),
        )
        # Start dates 'missing' from differenced series if dropna = True, so need to shift forward:
        expected_start = start_time + sum(lags) * series.freq if dropna else start_time
        raise_if_not(
            series.start_time() == expected_start,
            (
                f"Expected series to begin at time {expected_start}; "
                f"instead, it begins at time {series.start_time()}."
            ),
        )
        component_mask = Diff._get_component_mask(kwargs, dropna)
        raise_if_not(
            np.all(fit_component_mask == component_mask),
            (
                "Provided `component_mask` does not match "
                "`component_mask` specified when `fit` was called."
            ),
        )
        if dropna:
            nan_shape = (sum(lags), series.n_components, series.n_samples)
            nan_vals = np.full(nan_shape, fill_value=np.nan)
            series = series.prepend_values(nan_vals)
        vals = Diff._reshape_in(series, component_mask, flatten=False)
        raise_if_not(
            vals.shape[1] == start_vals.shape[1],
            (
                f"Expected series to have {start_vals.shape[1]} components; "
                f"instead, it has {vals.shape[1]}."
            ),
        )
        raise_if_not(
            vals.shape[2] == start_vals.shape[2],
            (
                f"Expected series to have {start_vals.shape[2]} samples; "
                f"instead, it has {vals.shape[2]}."
            ),
        )
        cutoff = sum(lags)
        for lag in reversed(lags):
            cutoff -= lag
            to_undiff = vals[cutoff:, :, :]
            to_undiff[:lag, :, :] = start_vals[cutoff : cutoff + lag, :, :]
            for i in range(lag):
                to_undiff[i::lag, :, :] = np.cumsum(to_undiff[i::lag, :, :], axis=0)
            vals[cutoff:, :, :] = to_undiff
        vals = Diff._reshape_out(series, vals, component_mask, flatten=False)
        return series.with_values(vals)

    @staticmethod
    def _get_component_mask(kwargs, dropna):
        component_mask = kwargs.get("component_mask", None)
        raise_if(
            dropna and (component_mask is not None),
            "Cannot specify `component_mask` with `dropna = True`.",
        )
        return component_mask
