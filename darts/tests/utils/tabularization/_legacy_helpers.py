"""Test-only wrappers for the pre-config tabularization signatures.

The public API is config-first. These helpers keep the existing generated test suites readable while
they are migrated incrementally.
"""

from itertools import chain
from math import inf

from darts import TimeSeries
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import (
    ForecastConfig,
    create_lagged_component_names,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.tabularization import _get_feature_times


def _check_lags(lags, lags_past_covariates, lags_future_covariates):
    all_lags = [lags, lags_past_covariates, lags_future_covariates]
    suffixes = ["", "_past_covariates", "_future_covariates"]
    lags_is_none = []
    for i, (suffix, lags_i) in enumerate(zip(suffixes, all_lags)):
        lags_is_none.append(lags_i is None)
        if lags_i is None:
            continue
        is_target_or_past = i < 2
        max_lag = -1 if is_target_or_past else inf
        if isinstance(lags_i, dict):
            lags_i = list(set(chain(*lags_i.values())))
        if any((lag > max_lag or not isinstance(lag, int)) for lag in lags_i):
            raise ValueError(
                f"`lags{suffix}` must be a `Sequence` or `Dict` containing only `int` "
                f"values less than {max_lag + 1}."
            )
    if all(lags_is_none):
        raise ValueError(
            "Must specify at least one of: `lags`, `lags_past_covariates`, "
            "`lags_future_covariates`."
        )


def _config(
    *,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length=1,
    output_chunk_shift=0,
    multi_models=True,
    check_inputs=True,
):
    if check_inputs:
        if not isinstance(output_chunk_length, int) or output_chunk_length < 1:
            raise ValueError("`output_chunk_length` must be a positive `int`.")
        _check_lags(lags, lags_past_covariates, lags_future_covariates)
    # Build with no shift and then set the field: the legacy helpers applied
    # ``output_chunk_shift`` to the time index, not to the future lags.
    config = ForecastConfig(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        output_chunk_shift=0,
        multi_models=multi_models,
    )
    config.output_chunk_shift = output_chunk_shift
    return config


def _create_lagged_training_data(
    target_series: TimeSeriesLike,
    output_chunk_length: int,
    output_chunk_shift: int,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    multi_models: bool = True,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
    stride: int = 1,
    sample_weight: TimeSeries | str | None = None,
):
    config = _config(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        output_chunk_shift=output_chunk_shift,
        multi_models=multi_models,
        check_inputs=check_inputs,
    )
    return create_lagged_training_data(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        concatenate=concatenate,
        sample_weight=sample_weight,
        stride=stride,
    )


def _create_lagged_prediction_data(
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
    stride: int = 1,
    show_warnings: bool = True,
):
    config = _config(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        check_inputs=check_inputs,
    )
    return create_lagged_prediction_data(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        concatenate=concatenate,
        stride=stride,
        show_warnings=show_warnings,
    )


def _create_lagged_component_names(
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length: int = 1,
    concatenate: bool = True,
    use_static_covariates: bool = False,
):
    for lags_i in (lags, lags_past_covariates, lags_future_covariates):
        if isinstance(lags_i, dict) and "default_lags" in lags_i:
            raise ValueError(
                "All the lags must be explicitly defined, 'default_lags' is not allowed in "
                "the lags dictionary."
            )
    config = _config(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
    )
    return create_lagged_component_names(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        concatenate=concatenate,
        use_static_covariates=use_static_covariates,
    )


def _get_feature_times_legacy(
    target_series: TimeSeries | None = None,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length: int = 1,
    output_chunk_shift: int = 0,
    is_training: bool = True,
    return_min_and_max_lags: bool = False,
    check_inputs: bool = True,
    show_warnings: bool = True,
):
    if is_training and target_series is None:
        raise ValueError("Must specify `target_series` when `is_training = True`.")
    config = _config(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        output_chunk_shift=output_chunk_shift,
        check_inputs=check_inputs,
    )
    return _get_feature_times(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        is_training=is_training,
        return_min_and_max_lags=return_min_and_max_lags,
        check_inputs=check_inputs,
        show_warnings=show_warnings,
    )
