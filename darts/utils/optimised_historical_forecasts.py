from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.timeseries_generation import generate_index

logger = get_logger(__name__)


def _get_historical_forecast_boundaries(
    model: ForecastingModel,
    series: TimeSeries,
    series_idx: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    start: Optional[Union[pd.Timestamp, float, int]],
    forecast_horizon: int,
    overlap_end: bool,
    freq: pd.DateOffset,
    show_warnings: bool = True,
) -> Tuple[Any, ...]:
    """
    Based on the boundaries of the forecastable time index, generates the boundaries of each covariates using the lags.

    For TimeSeries with a RangeIndex, the boundaries are converted to absolute indexes to slice the array appropriately
    when start > 0.

    When applicable, move the start boundaries to the value provided by the user.
    """
    # obtain forecastable indexes boundaries, as values from the time index
    historical_forecasts_time_index = model._get_historical_forecastable_time_index(
        series,
        past_covariates,
        future_covariates,
        is_training=False,
        reduce_to_bounds=True,
    )

    if historical_forecasts_time_index is None:
        raise_log(
            ValueError(
                "Cannot build a single input for prediction with the provided model, "
                f"`series` and `*_covariates` at series index: {series_idx}. The minimum "
                "prediction input time index requirements were not met. "
                "Please check the time index of `series` and `*_covariates`."
            )
        )
        return ()

    # shift the end of the forecastable index based on `overlap_end`` and `forecast_horizon``
    last_valid_pred_time = model._get_last_prediction_time(
        series,
        forecast_horizon,
        overlap_end,
    )

    historical_forecasts_time_index = (
        historical_forecasts_time_index[0],
        min(historical_forecasts_time_index[1], last_valid_pred_time),
    )

    # when applicable, shift the start of the forecastable index based on `start`
    if start is not None:
        start_time_ = series.get_timestamp_at_point(start)
        # ignore user-defined `start`
        if (
            not historical_forecasts_time_index[0]
            <= start_time_
            <= historical_forecasts_time_index[-1]
        ):
            if show_warnings:
                model._historical_forecasts_start_warnings(
                    idx=series_idx,
                    start=start,
                    start_time_=start_time_,
                    historical_forecasts_time_index=historical_forecasts_time_index,
                )
        else:
            historical_forecasts_time_index = (
                max(historical_forecasts_time_index[0], start_time_),
                historical_forecasts_time_index[1],
            )

    # re-adjust the slicing indexes to account for the lags
    (
        min_target_lag,
        _,
        min_past_cov_lag,
        _,
        min_future_cov_lag,
        max_future_cov_lag,
    ) = model.extreme_lags

    # target lags are <= 0
    hist_fct_tgt_start, hist_fct_tgt_end = historical_forecasts_time_index
    if min_target_lag is not None:
        hist_fct_tgt_start += min_target_lag * freq
    hist_fct_tgt_end -= 1 * freq
    # past lags are <= 0
    hist_fct_pc_start, hist_fct_pc_end = historical_forecasts_time_index
    if min_past_cov_lag is not None:
        hist_fct_pc_start += min_past_cov_lag * freq
    hist_fct_pc_end = hist_fct_tgt_end
    # future lags can be anything
    hist_fct_fc_start, hist_fct_fc_end = historical_forecasts_time_index
    if min_future_cov_lag is not None and min_future_cov_lag < 0:
        hist_fct_fc_start += min_future_cov_lag * freq
    if max_future_cov_lag is not None and max_future_cov_lag > 0:
        hist_fct_fc_end += max_future_cov_lag * freq

    # convert relative integer index to absolute, make end bound inclusive
    if series.has_range_index:
        hist_fct_tgt_start = series.get_index_at_point(hist_fct_tgt_start)
        hist_fct_tgt_end = series.get_index_at_point(hist_fct_tgt_end) + 1
        hist_fct_pc_start = series.get_index_at_point(hist_fct_pc_start)
        hist_fct_pc_end = series.get_index_at_point(hist_fct_pc_end) + 1
        hist_fct_fc_start = series.get_index_at_point(hist_fct_fc_start)
        hist_fct_fc_end = series.get_index_at_point(hist_fct_fc_end) + 1

    return (
        historical_forecasts_time_index[0],
        historical_forecasts_time_index[1],
        hist_fct_tgt_start,
        hist_fct_tgt_end,
        hist_fct_pc_start,
        hist_fct_pc_end,
        hist_fct_fc_start,
        hist_fct_fc_end,
    )


def _optimised_historical_forecasts_regression_last_points_only(
    model,
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]] = None,
    future_covariates: Optional[Sequence[TimeSeries]] = None,
    num_samples: int = 1,
    start: Optional[Union[pd.Timestamp, float, int]] = None,
    forecast_horizon: int = 1,
    stride: int = 1,
    overlap_end: bool = False,
    show_warnings: bool = True,
) -> Union[
    TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
]:
    """Optimized historical forecasts for RegressionModel with last_points_only = True


    if multi_models is True, needs to shift the start so that the "last point" is on the first forcatable time index

    """
    forecasts_list = []
    for idx, series_ in enumerate(series):
        past_covariates_ = past_covariates[idx] if past_covariates is not None else None
        future_covariates_ = (
            future_covariates[idx] if future_covariates is not None else None
        )
        freq = series_.freq

        # obtain forecastable indexes boundaries, adjust target & covariates boundaries accordingly
        (
            hist_fct_start,
            hist_fct_end,
            hist_fct_tgt_start,
            hist_fct_tgt_end,
            hist_fct_pc_start,
            hist_fct_pc_end,
            hist_fct_fc_start,
            hist_fct_fc_end,
        ) = _get_historical_forecast_boundaries(
            model=model,
            series=series_,
            series_idx=idx,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            start=start,
            forecast_horizon=forecast_horizon,
            overlap_end=overlap_end,
            freq=freq,
            show_warnings=show_warnings,
        )

        # Additional shift, to account for the model output_chunk_length
        if model.output_chunk_length != forecast_horizon and not model.multi_models:
            # used to convert the shift into the appropriate unit
            unit = freq if series_.has_datetime_index else 1

            shift = model.output_chunk_length - forecast_horizon

            hist_fct_tgt_start -= shift * unit
            hist_fct_pc_start -= shift * unit
            hist_fct_fc_start -= shift * unit

            hist_fct_tgt_end -= shift * unit
            hist_fct_pc_end -= shift * unit
            hist_fct_fc_end -= shift * unit

        X, times = create_lagged_prediction_data(
            target_series=series_[hist_fct_tgt_start:hist_fct_tgt_end],
            past_covariates=None
            if past_covariates_ is None
            else past_covariates_[hist_fct_pc_start:hist_fct_pc_end],
            future_covariates=None
            if future_covariates_ is None
            else future_covariates_[hist_fct_fc_start:hist_fct_fc_end],
            lags=model.lags.get("target", None),
            lags_past_covariates=model.lags.get("past", None),
            lags_future_covariates=model.lags.get("future", None),
            uses_static_covariates=model.uses_static_covariates,
            last_static_covariates_shape=model._static_covariates_shape,
            max_samples_per_ts=None,
            check_inputs=True,
            use_moving_windows=True,
            concatenate=False,
        )

        # stride can be applied directly (same for input and historical forecasts)
        X = X[0][::stride, :, 0]

        # repeat rows for probabilistic forecast
        forecast = model._predict_and_sample(
            np.repeat(X, num_samples, axis=0), num_samples
        )

        # reshape into (forecasted indexes, n_components, n_samples), components are interleaved
        forecast = forecast.reshape(X.shape[0], -1, num_samples)

        # extract the last sub-model forecast for each component
        if model.multi_models:
            forecast = forecast[
                :,
                (forecast_horizon - 1)
                * series_.n_components : (forecast_horizon)
                * series_.n_components,
                :,
            ]

        forecasts_list.append(
            TimeSeries.from_times_and_values(
                times=times[0]
                if stride == 1 and model.output_chunk_length == 1
                else generate_index(
                    start=hist_fct_start + (forecast_horizon - 1) * freq,
                    length=forecast.shape[0],
                    freq=freq * stride,
                ),
                values=forecast,
                columns=series_.columns,
                static_covariates=series_.static_covariates,
                hierarchy=series_.hierarchy,
            )
        )
    return forecasts_list if len(series) > 1 else forecasts_list[0]


def _optimised_historical_forecasts_regression_all_points(
    model,
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]] = None,
    future_covariates: Optional[Sequence[TimeSeries]] = None,
    num_samples: int = 1,
    start: Optional[Union[pd.Timestamp, float, int]] = None,
    forecast_horizon: int = 1,
    stride: int = 1,
    overlap_end: bool = False,
    show_warnings: bool = True,
) -> Union[
    TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
]:
    """Optimized historical forecasts for RegressionModel with last_points_only = False"""
    forecasts_list = []
    for idx, series_ in enumerate(series):
        past_covariates_ = past_covariates[idx] if past_covariates is not None else None
        future_covariates_ = (
            future_covariates[idx] if future_covariates is not None else None
        )
        freq = series_.freq

        # obtain forecastable indexes boundaries, adjust target & covariates boundaries accordingly
        (
            hist_fct_start,
            hist_fct_end,
            hist_fct_tgt_start,
            hist_fct_tgt_end,
            hist_fct_pc_start,
            hist_fct_pc_end,
            hist_fct_fc_start,
            hist_fct_fc_end,
        ) = _get_historical_forecast_boundaries(
            model=model,
            series=series_,
            series_idx=idx,
            past_covariates=past_covariates_,
            future_covariates=future_covariates_,
            start=start,
            forecast_horizon=forecast_horizon,
            overlap_end=overlap_end,
            freq=freq,
            show_warnings=show_warnings,
        )

        # Additional shift, to account for the model output_chunk_length
        shift_start = 0
        shift_end = 0
        if model.output_chunk_length > 1:
            # used to convert the shift into the appropriate unit
            unit = freq if series_.has_datetime_index else 1

            if not model.multi_models:
                shift_start = model.output_chunk_length - 1

                hist_fct_tgt_start -= shift_start * unit
                hist_fct_pc_start -= shift_start * unit
                hist_fct_fc_start -= shift_start * unit

            if model.output_chunk_length == forecast_horizon:
                shift_end = model.output_chunk_length - 1

                hist_fct_tgt_end += shift_end * unit
                hist_fct_pc_end += shift_end * unit
                hist_fct_fc_end += shift_end * unit

        X, _ = create_lagged_prediction_data(
            target_series=series_[hist_fct_tgt_start:hist_fct_tgt_end],
            past_covariates=None
            if past_covariates_ is None
            else past_covariates_[hist_fct_pc_start:hist_fct_pc_end],
            future_covariates=None
            if future_covariates_ is None
            else future_covariates_[hist_fct_fc_start:hist_fct_fc_end],
            lags=model.lags.get("target", None),
            lags_past_covariates=model.lags.get("past", None),
            lags_future_covariates=model.lags.get("future", None),
            uses_static_covariates=model.uses_static_covariates,
            last_static_covariates_shape=model._static_covariates_shape,
            max_samples_per_ts=None,
            check_inputs=True,
            use_moving_windows=True,
            concatenate=False,
        )

        # stride must be applied post-hoc to avoid missing values
        X = X[0][:, :, 0]

        # repeat rows for probabilistic forecast
        forecast = model._predict_and_sample(
            np.repeat(X, num_samples, axis=0), num_samples
        )

        require_auto_regression: bool = forecast_horizon > model.output_chunk_length

        # reshape and stride the forecast into (forecastable_index, forecast_horizon, n_components, num_samples)
        if model.multi_models:
            if require_auto_regression:
                raise_log(ValueError("Not supported"))
            else:
                # components are interleaved
                forecast = forecast.reshape(
                    X.shape[0],
                    model.output_chunk_length,
                    series_.n_components,
                    num_samples,
                )

                if (
                    forecast_horizon == model.output_chunk_length
                    and forecast_horizon > 1
                ):
                    forecast = forecast[:-shift_end:stride, :forecast_horizon]
                else:
                    forecast = forecast[::stride, :forecast_horizon]
        else:
            if require_auto_regression:
                raise_log(ValueError("Not supported"))
            else:
                # components are interleaved
                forecast = forecast.reshape(X.shape[0], -1, num_samples)

                forecast = sliding_window_view(
                    forecast, (forecast_horizon, series_.n_components, num_samples)
                )

                if forecast_horizon != model.output_chunk_length:
                    forecast = forecast[
                        : -shift_start + forecast_horizon - 1 : stride,
                        0,
                        0,
                        :forecast_horizon,
                        :,
                        :,
                    ]
                elif forecast_horizon > 1:
                    forecast = forecast[
                        : -forecast_horizon + 1 : stride, 0, 0, :forecast_horizon, :, :
                    ]
                else:
                    forecast = forecast[::stride, 0, 0, :forecast_horizon, :, :]

        # TODO: check if faster to create in the loop
        new_times = generate_index(
            start=hist_fct_start,
            length=forecast_horizon * stride * forecast.shape[0],
            freq=freq,
        )

        forecasts_ = []
        for idx_ftc, step_fct in enumerate(
            range(0, forecast.shape[0] * stride, stride)
        ):
            forecasts_.append(
                TimeSeries.from_times_and_values(
                    times=new_times[step_fct : step_fct + forecast_horizon],
                    values=forecast[idx_ftc],
                    columns=series_.columns,
                    static_covariates=series_.static_covariates,
                    hierarchy=series_.hierarchy,
                )
            )

        forecasts_list.append(forecasts_)
    return forecasts_list if len(series) > 1 else forecasts_list[0]
