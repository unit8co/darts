from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries
from darts.utils import _build_tqdm_iterator
from darts.utils.data.tabularization import (
    _extract_lagged_vals_from_windows,
    _get_lagged_indices,
    create_lagged_prediction_data,
    strided_moving_window,
)
from darts.utils.historical_forecasts.utils import (
    _get_historical_forecast_boundaries,
)
from darts.utils.utils import generate_index

logger = get_logger(__name__)


def _optimized_historical_forecasts_last_points_only(
    model,
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]] = None,
    future_covariates: Optional[Sequence[TimeSeries]] = None,
    num_samples: int = 1,
    start: Optional[Union[pd.Timestamp, float, int]] = None,
    start_format: Literal["position", "value"] = "value",
    forecast_horizon: int = 1,
    stride: int = 1,
    overlap_end: bool = False,
    show_warnings: bool = True,
    verbose: bool = False,
    predict_likelihood_parameters: bool = False,
    **kwargs,
) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for RegressionModel with last_points_only = True

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.

    The data_transformers are applied in historical_forecasts (input and predictions)
    """
    forecasts_list = []
    iterator = _build_tqdm_iterator(
        series, verbose, total=len(series), desc="historical forecasts"
    )
    for idx, series_ in enumerate(iterator):
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
            start_format=start_format,
            forecast_horizon=forecast_horizon,
            overlap_end=overlap_end,
            stride=stride,
            freq=freq,
            show_warnings=show_warnings,
        )

        # Additional shift, to account for the model output_chunk_length
        if model.output_chunk_length != forecast_horizon and not model.multi_models:
            # used to convert the shift into the appropriate unit
            unit = freq if series_.has_datetime_index else 1

            shift = model.output_chunk_length - forecast_horizon

            hist_fct_tgt_start -= shift * unit
            hist_fct_pc_start -= (
                shift * unit if hist_fct_pc_start is not None else hist_fct_pc_start
            )
            hist_fct_fc_start -= (
                shift * unit if hist_fct_fc_start is not None else hist_fct_fc_start
            )

            hist_fct_tgt_end -= shift * unit
            hist_fct_pc_end -= (
                shift * unit if hist_fct_pc_end is not None else hist_fct_pc_end
            )
            hist_fct_fc_end -= (
                shift * unit if hist_fct_fc_end is not None else hist_fct_fc_end
            )

        X, times = create_lagged_prediction_data(
            target_series=(
                None
                if model._get_lags("target") is None
                and not model.uses_static_covariates
                else series_[hist_fct_tgt_start:hist_fct_tgt_end]
            ),
            past_covariates=(
                None
                if past_covariates_ is None
                else past_covariates_[hist_fct_pc_start:hist_fct_pc_end]
            ),
            future_covariates=(
                None
                if future_covariates_ is None
                else future_covariates_[hist_fct_fc_start:hist_fct_fc_end]
            ),
            lags=model._get_lags("target"),
            lags_past_covariates=model._get_lags("past"),
            lags_future_covariates=model._get_lags("future"),
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
            **kwargs,
        )
        # forecast has shape ((forecastable_index_length-1)*num_samples, k, n_component)
        # where k = output_chunk length if multi_models, 1 otherwise
        # reshape into (forecasted indexes, output_chunk_length, n_components, n_samples)
        forecast = np.moveaxis(
            forecast.reshape(
                X.shape[0],
                num_samples,
                model.output_chunk_length if model.multi_models else 1,
                -1,
            ),
            1,
            -1,
        )

        # iteratively extend the historical forecasts
        if forecast_horizon > model.output_chunk_length:
            forecast = _optimized_hf_autoregression(
                model=model,
                X=X,
                times=times[0],
                stride=stride,
                forecast=forecast,
                series=series_,
                past_covariates=past_covariates_,
                future_covariates=future_covariates_,
                forecast_horizon=forecast_horizon,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                tgt_hf_idx=(hist_fct_tgt_start, hist_fct_tgt_end),
                pc_hf_idx=(hist_fct_pc_start, hist_fct_pc_end),
                fc_hf_idx=(hist_fct_fc_start, hist_fct_fc_end),
                **kwargs,
            )

        # extract the last sub-model forecast for each component
        if model.multi_models:
            forecast = forecast[:, forecast_horizon - 1]
        else:
            forecast = forecast[:, 0]

        forecasts_list.append(
            TimeSeries.from_times_and_values(
                times=(
                    times[0]
                    if stride == 1 and model.output_chunk_length == 1
                    else generate_index(
                        start=hist_fct_start
                        + (forecast_horizon + model.output_chunk_shift - 1) * freq,
                        length=forecast.shape[0],
                        freq=freq * stride,
                        name=series_.time_index.name,
                    )
                ),
                values=forecast,
                columns=forecast_components,
                static_covariates=series_.static_covariates,
                hierarchy=series_.hierarchy,
            )
        )
    return forecasts_list


def _optimized_historical_forecasts_all_points(
    model,
    series: Sequence[TimeSeries],
    past_covariates: Optional[Sequence[TimeSeries]] = None,
    future_covariates: Optional[Sequence[TimeSeries]] = None,
    num_samples: int = 1,
    start: Optional[Union[pd.Timestamp, float, int]] = None,
    start_format: Literal["position", "value"] = "value",
    forecast_horizon: int = 1,
    stride: int = 1,
    overlap_end: bool = False,
    show_warnings: bool = True,
    verbose: bool = False,
    predict_likelihood_parameters: bool = False,
    **kwargs,
) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for RegressionModel with last_points_only = False.

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.
    """
    forecasts_list = []
    iterator = _build_tqdm_iterator(
        series, verbose, total=len(series), desc="historical forecasts"
    )
    for idx, series_ in enumerate(iterator):
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
            start_format=start_format,
            forecast_horizon=forecast_horizon,
            overlap_end=overlap_end,
            stride=stride,
            freq=freq,
            show_warnings=show_warnings,
        )

        # Additional shift, to account for the model output_chunk_length
        shift_start = 0
        if model.output_chunk_length > 1:
            # used to convert the shift into the appropriate unit
            unit = freq if series_.has_datetime_index else 1

            if not model.multi_models:
                shift_start = model.output_chunk_length - 1

                hist_fct_tgt_start -= shift_start * unit
                hist_fct_pc_start -= (
                    shift_start * unit
                    if hist_fct_pc_start is not None
                    else hist_fct_pc_start
                )
                hist_fct_fc_start -= (
                    shift_start * unit
                    if hist_fct_fc_start is not None
                    else hist_fct_fc_start
                )

        X, times = create_lagged_prediction_data(
            target_series=(
                None
                if model._get_lags("target") is None
                and not model.uses_static_covariates
                else series_[hist_fct_tgt_start:hist_fct_tgt_end]
            ),
            past_covariates=(
                None
                if past_covariates_ is None
                else past_covariates_[hist_fct_pc_start:hist_fct_pc_end]
            ),
            future_covariates=(
                None
                if future_covariates_ is None
                else future_covariates_[hist_fct_fc_start:hist_fct_fc_end]
            ),
            lags=model._get_lags("target"),
            lags_past_covariates=model._get_lags("past"),
            lags_future_covariates=model._get_lags("future"),
            uses_static_covariates=model.uses_static_covariates,
            last_static_covariates_shape=model._static_covariates_shape,
            max_samples_per_ts=None,
            check_inputs=True,
            use_moving_windows=True,
            concatenate=False,
            show_warnings=False,
        )

        # stride must be applied post-hoc to avoid missing values
        X = X[0][:, :, 0]

        # repeat rows for probabilistic forecast
        forecast = model._predict_and_sample(
            x=np.repeat(X, num_samples, axis=0),
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
            **kwargs,
        )
        # forecast has shape ((forecastable_index_length-1)*num_samples, k, n_component)
        # where k = output_chunk length if multi_models, 1 otherwise
        # reshape into (forecasted indexes, output_chunk_length, n_components, n_samples)
        forecast = np.moveaxis(
            forecast.reshape(
                X.shape[0],
                num_samples,
                model.output_chunk_length if model.multi_models else 1,
                -1,
            ),
            1,
            -1,
        )

        if model.multi_models:
            forecast = forecast[::stride, :forecast_horizon]
        else:
            # entire forecast horizon is given by multiple (previous) forecasts -> apply sliding window
            forecast = sliding_window_view(
                forecast[:, 0],
                (forecast_horizon, len(forecast_components), num_samples),
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
            # apply stride
            else:
                forecast = forecast[::stride, 0, 0, :, :, :]

        # iteratively extend the historical forecasts
        if forecast_horizon > model.output_chunk_length:
            forecast = _optimized_hf_autoregression(
                model=model,
                X=X,
                times=times[0],
                stride=stride,
                forecast=forecast,
                series=series_,
                past_covariates=past_covariates_,
                future_covariates=future_covariates_,
                forecast_horizon=forecast_horizon,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                tgt_hf_idx=(hist_fct_tgt_start, hist_fct_tgt_end),
                pc_hf_idx=(hist_fct_pc_start, hist_fct_pc_end),
                fc_hf_idx=(hist_fct_fc_start, hist_fct_fc_end),
                **kwargs,
            )

        # TODO: check if faster to create in the loop
        new_times = generate_index(
            start=hist_fct_start + model.output_chunk_shift * series_.freq,
            length=forecast_horizon + (forecast.shape[0] - 1) * stride,
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
    return forecasts_list


def _optimized_hf_autoregression(
    model,
    X: np.array,
    times: pd.Index,
    stride: int,
    forecast: TimeSeries,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    forecast_horizon: int,
    num_samples: int,
    predict_likelihood_parameters: bool,
    tgt_hf_idx: tuple,
    pc_hf_idx: tuple,
    fc_hf_idx: tuple,
    **kwargs,
):
    """perform autoregression on all the horizon in parallel.

    the parent function is responsible for extracting the relevant timestamps (last_point_only)
    """
    if predict_likelihood_parameters:
        raise_log(
            ValueError(
                "`forecast_horizon > output_chunk_length` is not supported when `predict_likelihood_parameters=True`."
            ),
            logger,
        )
        # forecast_ = model._predict_and_sample(
        #    x=X,
        #    num_samples=1,
        #    predict_likelihood_parameters=False,
        #    **kwargs,
        # )

    forecast_ = forecast
    # indexes used to control tabularization boundaries
    hist_fct_tgt_start, hist_fct_tgt_end = tgt_hf_idx
    hist_fct_pc_start, hist_fct_pc_end = pc_hf_idx
    hist_fct_fc_start, hist_fct_fc_end = fc_hf_idx
    unit = series.freq if series.has_datetime_index else 1
    # information necessary to run tabularization
    lags_extract, lags_order = _get_lagged_indices(
        model._get_lags("target"),
        model._get_lags("past"),
        model._get_lags("future"),
    )
    min_lag = model._get_lags("target")[0]
    max_lag = model._get_lags("target")[-1]

    # TODO: pre-allocate a np.ndarray large enough to contain all the iterations?
    # if there is no gap in the lags and their are contiguous to the target, X can be reused
    if max_lag - min_lag == len(model._get_lags("target")) and max_lag == -1:
        windows = X
    # use sliding window to extract all the steps between min_lag and forecasted step to be able
    # (no gaps) to dynamically add the forecasts from the model for autoregression
    else:
        # find the index of the first historical forecast window
        hist_fct_tgt_start_idx = (
            np.searchsorted(series.time_index, hist_fct_tgt_start) - 1
        )
        # keeping the whole window from t - max_lag to t - 1
        window_full_len = -min_lag
        first_window_start_idx = hist_fct_tgt_start_idx - max_lag
        first_window_end_idx = first_window_start_idx + window_full_len
        # starting over from the series as the lags might contain lags (cannot use X directly)
        # TODO: the last window might need to be handled separately as the sliding window cannot be extracted
        vals = series.all_values(copy=False)[
            first_window_start_idx : first_window_end_idx + len(times) - 1, :, :
        ]
        # apply the moving window
        windows = strided_moving_window(
            x=vals,
            window_len=window_full_len,
            stride=stride,
            axis=0,
            check_inputs=False,
        )

    # iteratively move the auto-regression window across all the historical forecasts
    win_start_idx = 0
    while forecast.shape[1] < forecast_horizon:
        # add the forecast at the end of each window
        windows = np.concatenate([windows, np.swapaxes(forecast_, 1, 3)], axis=3)
        # window from which the lagged values will actually be extracted
        win_start_idx += model.output_chunk_length
        win_end_idx = win_start_idx + max_lag - min_lag + 1
        # extract the lagged values
        lagged_vals = _extract_lagged_vals_from_windows(
            windows[:, :, :, win_start_idx:win_end_idx],
            lags_to_extract=lags_extract[0],
            lags_shift=0,
        )
        # reorder the lagged values
        X_ = lagged_vals[:, lags_order[0]]

        # shift the boundaries of the covariates
        hist_fct_pc_start += model.output_chunk_length * unit
        hist_fct_pc_end += model.output_chunk_length * unit
        hist_fct_fc_start += model.output_chunk_length * unit
        hist_fct_fc_end += model.output_chunk_length * unit

        # target series is passed just for the static covariates (lags=None)
        if model.uses_static_covariates or past_covariates or future_covariates:
            X_covs, _ = create_lagged_prediction_data(
                target_series=(None if not model.uses_static_covariates else series),
                past_covariates=(
                    None
                    if past_covariates is None
                    else past_covariates[hist_fct_pc_start:hist_fct_pc_end]
                ),
                future_covariates=(
                    None
                    if future_covariates is None
                    else future_covariates[hist_fct_fc_start:hist_fct_fc_end]
                ),
                lags=None,
                lags_past_covariates=model._get_lags("past"),
                lags_future_covariates=model._get_lags("future"),
                uses_static_covariates=model.uses_static_covariates,
                last_static_covariates_shape=model._static_covariates_shape,
                max_samples_per_ts=None,
                check_inputs=False,
                use_moving_windows=True,
                concatenate=False,
            )
            # TODO: directly create the strided dataset once supported
            # apply stride
            X_covs = X_covs[0][::stride]
            # concatenate target with the strided covariates
            X_ = np.concatenate([X_, X_covs], axis=1)

        # repeat the features dimension to match forecasts (probabilistic)
        if series.is_deterministic and num_samples > 1:
            # TODO: make sure the reshaping of the forecasts is correct
            X_ = np.repeat(X_, num_samples, axis=0)

        # generate the forecasts
        forecast_ = model._predict_and_sample(
            x=X_[:, :, 0],
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
            **kwargs,
        )
        forecast_ = np.moveaxis(
            forecast_.reshape(
                X.shape[0],
                num_samples,
                model.output_chunk_length if model.multi_models else 1,
                -1,
            ),
            1,
            -1,
        )
        forecast = np.concatenate([forecast, forecast_], axis=1)
    # retain only the first forecast_horizon steps
    return forecast[:, :forecast_horizon]
