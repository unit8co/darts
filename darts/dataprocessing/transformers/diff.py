from typing import Any, Iterator, Sequence, Tuple, Union

import numpy as np
from more_itertools import always_iterable

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
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose)
        self._lags = tuple(always_iterable(lags))
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
        start_vals = vals[:lags_sum, :, :]
        diffed = start_vals
        cutoff = 0
        for lag in lags:
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
        fitted_params: tuple[np.ndarray, np.ndarray, int, int],
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
