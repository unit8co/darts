from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darts.logging import get_logger
from darts.timeseries import TimeSeries
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.historical_forecasts.utils import _get_historical_forecast_boundaries
from darts.utils.timeseries_generation import generate_index

logger = get_logger(__name__)


def _optimized_historical_forecasts_regression_last_points_only(
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
    predict_likelihood_parameters: bool = False,
) -> Union[
    TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
]:
    """
    Optimized historical forecasts for RegressionModel with last_points_only = True

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.
    """
    forecasts_list = []
    for idx, series_ in enumerate(series):
        past_covariates_ = past_covariates[idx] if past_covariates is not None else None
        future_covariates_ = (
            future_covariates[idx] if future_covariates is not None else None
        )
        freq = series_.freq
        forecast_components = (
            model._likelihood_components_names(series_)
            if predict_likelihood_parameters
            else series_.columns
        )

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
            target_series=None
            if len(model.lags.get("target", [])) == 0
            else series_[hist_fct_tgt_start:hist_fct_tgt_end],
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
            x=np.repeat(X, num_samples, axis=0),
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )
        # forecast has shape ((forecastable_index_length-1)*num_samples, k, n_component)
        # where k = output_chunk length if multi_models, 1 otherwise

        # reshape into (forecasted indexes, n_components, n_samples), components are interleaved
        forecast = forecast.reshape(X.shape[0], -1, num_samples)

        # extract the last sub-model forecast for each component
        if model.multi_models:
            forecast = forecast[
                :,
                (forecast_horizon - 1)
                * len(forecast_components) : (forecast_horizon)
                * len(forecast_components),
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
                    name=series_.time_index.name,
                ),
                values=forecast,
                columns=forecast_components,
                static_covariates=series_.static_covariates,
                hierarchy=series_.hierarchy,
            )
        )
    return forecasts_list if len(series) > 1 else forecasts_list[0]


def _optimized_historical_forecasts_regression_all_points(
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
    predict_likelihood_parameters: bool = False,
) -> Union[
    TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
]:
    """
    Optimized historical forecasts for RegressionModel with last_points_only = False.

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.
    """
    forecasts_list = []
    for idx, series_ in enumerate(series):
        past_covariates_ = past_covariates[idx] if past_covariates is not None else None
        future_covariates_ = (
            future_covariates[idx] if future_covariates is not None else None
        )
        freq = series_.freq
        forecast_components = (
            model._likelihood_components_names(series_)
            if predict_likelihood_parameters
            else series_.columns
        )

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

            # last lagged inputs are removed as the last prediction of length output_chunk_length will include them
            if model.output_chunk_length == forecast_horizon:
                shift_end = model.output_chunk_length - 1

                hist_fct_tgt_end += shift_end * unit
                hist_fct_pc_end += shift_end * unit
                hist_fct_fc_end += shift_end * unit

        X, _ = create_lagged_prediction_data(
            target_series=None
            if len(model.lags.get("target", [])) == 0
            else series_[hist_fct_tgt_start:hist_fct_tgt_end],
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
            x=np.repeat(X, num_samples, axis=0),
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )

        # reshape and stride the forecast into (forecastable_index, forecast_horizon, n_components, num_samples)
        if model.multi_models:
            # forecast has shape ((forecastable_index_length-1)*num_samples, output_chunk_length, n_component)
            # and the components are interleaved
            forecast = forecast.reshape(
                X.shape[0],
                model.output_chunk_length,
                len(forecast_components),
                num_samples,
            )

            if forecast_horizon == model.output_chunk_length and forecast_horizon > 1:
                forecast = forecast[:-shift_end:stride]
            # only keep the prediction of the first forecast_horizon sub-models
            else:
                forecast = forecast[::stride, :forecast_horizon]
        else:
            # forecast has shape ((forecastable_index_length-1)*num_samples, 1, n_component)
            # and the components are interleaved
            forecast = forecast.reshape(X.shape[0], -1, num_samples)

            # forecasts depend on lagged data only, output_chunk_length is reconstitued by applying a sliding window
            forecast = sliding_window_view(
                forecast, (forecast_horizon, len(forecast_components), num_samples)
            )

            # apply stride, remove the last windows, slice output_chunk_length to keep forecast_horizon values
            if forecast_horizon != model.output_chunk_length:
                forecast = forecast[
                    : -shift_start + forecast_horizon - 1 : stride,
                    0,
                    0,
                    :forecast_horizon,
                    :,
                    :,
                ]
            # apply stride, remove the last windows
            elif forecast_horizon > 1:
                forecast = forecast[: -forecast_horizon + 1 : stride, 0, 0, :, :, :]
            # apply stride
            else:
                forecast = forecast[::stride, 0, 0, :, :, :]

        # TODO: check if faster to create in the loop
        new_times = generate_index(
            start=hist_fct_start,
            length=forecast_horizon * stride * forecast.shape[0],
            freq=freq,
            name=series_.time_index.name,
        )

        forecasts_ = []
        for idx_ftc, step_fct in enumerate(
            range(0, forecast.shape[0] * stride, stride)
        ):
            forecasts_.append(
                TimeSeries.from_times_and_values(
                    times=new_times[step_fct : step_fct + forecast_horizon],
                    values=forecast[idx_ftc],
                    columns=forecast_components,
                    static_covariates=series_.static_covariates,
                    hierarchy=series_.hierarchy,
                )
            )

        forecasts_list.append(forecasts_)
    return forecasts_list if len(series) > 1 else forecasts_list[0]
