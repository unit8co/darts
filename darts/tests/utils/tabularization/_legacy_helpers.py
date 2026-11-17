"""Test-only wrappers for the pre-config tabularization signatures.

The public API takes a `ForecastConfig`. These helpers let the existing generated test suites keep passing
the lags and output chunk arguments directly: they build the `ForecastConfig` (which validates them) and call
the public functions.

The legacy helpers did not apply `output_chunk_shift` to the future lags (the model shifted them before calling
the helpers), while `ForecastConfig` applies it, like the model. The wrappers subtract the shift from the future
lags, so that the suites keep their expectations. `test_public_api.py` covers the shifted future lags of the
public helpers.
"""

from darts import TimeSeries
from darts.typing import TimeSeriesLike
from darts.utils.data.tabularization import (
    ForecastConfig,
    create_lagged_component_names,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.tabularization import _get_feature_times


def _unshift_future_lags(lags_future_covariates, output_chunk_shift):
    """Undo the `output_chunk_shift` that `ForecastConfig` applies to the future lags."""
    if not output_chunk_shift or lags_future_covariates is None:
        return lags_future_covariates

    def unshift(lags):
        if (
            isinstance(lags, tuple)
            and len(lags) == 2
            and all(isinstance(lag, int) for lag in lags)
        ):
            lags = list(range(-lags[0], lags[1]))
        if isinstance(lags, list) and all(isinstance(lag, int) for lag in lags):
            return [lag - output_chunk_shift for lag in lags]
        return lags

    if isinstance(lags_future_covariates, dict):
        return {comp: unshift(lags) for comp, lags in lags_future_covariates.items()}
    return unshift(lags_future_covariates)


def _config(
    *,
    lags=None,
    lags_past_covariates=None,
    lags_future_covariates=None,
    output_chunk_length=1,
    output_chunk_shift=0,
    multi_models=True,
):
    return ForecastConfig(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=_unshift_future_lags(
            lags_future_covariates, output_chunk_shift
        ),
        output_chunk_length=output_chunk_length,
        output_chunk_shift=output_chunk_shift,
        multi_models=multi_models,
    )


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
    )
    return _get_feature_times(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        is_training=is_training,
        return_min_and_max_lags=return_min_and_max_lags,
        show_warnings=show_warnings,
    )
