"""
Optimized Historical Forecasts for SKLearnModel
-----------------------------------------------
"""

from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.utils import _build_tqdm_iterator
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.historical_forecasts.utils import (
    _get_historical_forecast_boundaries,
    _process_predict_start_points_bounds,
)
from darts.utils.utils import generate_index

logger = get_logger(__name__)


def _optimized_historical_forecasts_regression(
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
    random_state: Optional[int] = None,
    predict_kwargs: Optional[dict[str, Any]] = None,
    last_points_only: bool = False,
) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for SKLearnModel with last_points_only = False.

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.

    The data_transformers are applied in historical_forecasts (input and predictions)
    """

    # # Calling super().predict() to perform sanity checks
    # # e.g. is autoregression allowed, are parameters coherent, etc.
    # super().predict(
    #     n=forecast_horizon,
    #     series=series,
    #     past_covariates=past_covariates,
    #     future_covariates=future_covariates,
    #     num_samples=num_samples,
    #     verbose=verbose,
    #     predict_likelihood_parameters=predict_likelihood_parameters,
    #     show_warnings=show_warnings,
    # )

    predict_kwargs = predict_kwargs or {}
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
            model.likelihood.component_names(series=series_)
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

        # Compute left and right bounds to know how many forecasts will be produced
        left_bound = (
            len(series_)
            if hist_fct_start > series_.end_time()
            else series_.get_index_at_point(hist_fct_start)
        )
        right_bound = (
            len(series_)
            if hist_fct_end > series_.end_time()
            else series_.get_index_at_point(hist_fct_end)
        )

        bounds_array, _ = _process_predict_start_points_bounds(
            series=[series_],
            bounds=np.array([[left_bound, right_bound]]),
            stride=stride,
        )
        left_bound, right_bound = bounds_array[0].astype(int)

        n_forecasts = (
            (right_bound - left_bound) // stride + 1 if right_bound >= left_bound else 0
        )

        # In case of multi-models, we predict all steps at once so autoregression is done
        # output_chunk_length by output_chunk_length. Without multi-models,
        # a single model predicts one step at a time and we shift
        # the input sequence by one time step after each prediction.
        if model.multi_models:
            step = model.output_chunk_length
        else:
            step = 1

        # If forecast_horizon <= step, we can predict all steps in one go
        # so X will be of shape (n_forecasts, n_lags, n_components)
        # Otherwise, X will be of shape (n_forecasts, n_lags, n_components, n_prediction_iterations)
        # where n_prediction_iterations = ceil(forecast_horizon / step)
        X, _ = create_lagged_prediction_data(
            target_series=(
                None
                if model._get_lags("target") is None
                and not model.uses_static_covariates
                else series_[hist_fct_tgt_start:hist_fct_tgt_end]
            ),
            past_covariates=(
                None
                if past_covariates_ is None
                else past_covariates_[hist_fct_pc_start:]
            ),
            future_covariates=(
                None
                if future_covariates_ is None
                else future_covariates_[hist_fct_fc_start:]
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
            forecast_horizon=forecast_horizon,
            step=step,
        )

        # stride must be applied post-hoc to avoid missing values
        X = X[0][:, :, 0]
        predictions = []

        # In case of not doing autoregression, n_prediction_iterations is 1,
        # so internally the last dimension has been collapsed.
        # Add a dummy dimension so the for loop works as expected
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        # Trick to fix with stride, because NaN are in X at the end of the series, but
        # with stride, we might not get NaN in current_X, so we manually set the relevant
        # columns to NaN so that they get filled with previous predictions
        if not model.multi_models and "target" in model.lags:
            offset_idx = 1
            end_idx = len(model.lags["target"])
            for i in range(model.output_chunk_length, X.shape[-1]):
                start_idx = end_idx - offset_idx
                X[:, start_idx:end_idx, i] = np.nan
                offset_idx += 1

        last_step_shift = 0
        t_pred = 0
        start_idx = 0

        forecasts = []
        forecast = None
        for pred_idx in range(X.shape[-1]):
            # in case of autoregressive forecast `(t_pred > 0)` and if `n` is not a round multiple of `step`,
            # we have to step back `step` from `n` in the last iteration
            if 0 < forecast_horizon - t_pred < step and t_pred > 0:
                last_step_shift = t_pred - (forecast_horizon - step)
                t_pred = forecast_horizon - step

            # Select the appropriate slice of X to process and cut according to stride
            # X can have more samples than needed, so cut to n_forecasts
            # current_X = X[::stride, :, pred_idx]
            current_X = X[::, :, pred_idx]
            if not model.multi_models:
                current_X = current_X[::stride][:n_forecasts, ...]

            current_X = np.repeat(current_X, num_samples, axis=0)

            # When predictions exist, we are either doing autoregression, or predicting multiple steps with
            # multi_models=False
            # We are also interested in filling current_X with previous predictions when we use target
            # (so when model.lags contains "target")
            # if predictions and "target" in model.lags:

            # Fill history is when we are doing autoregression
            fill_history = t_pred + 1 > model.output_chunk_length

            if np.isnan(current_X).any():
                # When we have NaN in current_X, we need to fill them
                _, col = np.where(np.isnan(current_X))
                col = np.unique(col)

                if fill_history:
                    if model.multi_models:
                        # Flatten the forecasts as a 2D array for easier indexing
                        # Instead of having (n_series * n_samples, output_chunk_length, n_components),
                        # we have (n_series * n_samples, output_chunk_length * n_components)
                        # Because X is built the same way, so the indexing will be coherent
                        flattened_forecasts = forecasts.reshape(forecasts.shape[0], -1)
                        # Find which prediction steps we need to fill in current_X. It's the last len(col) steps
                        # but if last_step_shift > 0, we need to shift accordingly
                        slice_start = flattened_forecasts.shape[1] - (
                            col.size + last_step_shift
                        )
                        slice_start = 0 if slice_start < 0 else slice_start
                        slice_end = slice_start + len(col)
                        slice_end = (
                            flattened_forecasts.shape[1]
                            if slice_end > flattened_forecasts.shape[1]
                            else slice_end
                        )

                        # Fill the current X columns with the corresponding forecasts
                        current_X[:, col] = flattened_forecasts[
                            :,
                            slice_start:slice_end,
                        ].reshape(current_X[:, col].shape)
                    else:
                        # When not using multi_models, we predict one step at a time so we only need to
                        # shift the start_idx one by one each prediction iteration
                        flattened_forecasts = forecasts.reshape(forecasts.shape[0], -1)
                        current_X[:, col] = flattened_forecasts[
                            :,
                            start_idx : start_idx + col.size,
                        ].reshape(current_X[:, col].shape)
                    if col.size == len(model.lags["target"]) and forecast is not None:
                        start_idx += forecast.shape[-1]
                else:
                    current_X[:, col] = forecasts.reshape(forecasts.shape[0], -1)[
                        :, : col.size
                    ].reshape(current_X[:, col].shape)

            # repeat rows for probabilistic forecast
            forecast = model._predict(
                x=current_X,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                **predict_kwargs,
            )
            predictions.append(forecast[:, last_step_shift:, ...])
            forecasts = np.concatenate(predictions, axis=1)
            t_pred += step
        forecast = np.concatenate(predictions, axis=1)

        # Cut up to forecast_horizon because we might have predicted to much in the last
        # iteration (if step does not divide forecast_horizon)
        forecast = forecast[:, :forecast_horizon, :]

        # bring into correct shape: (n_forecasts, n_components, n_samples)
        forecast = np.moveaxis(
            forecast.reshape(
                -1,
                num_samples,
                forecast_horizon,
                len(forecast_components),
            ),
            1,
            -1,
        )
        # For multi-models, we need to apply the stride now, as we predicted all steps at once
        # For non-multi-models, we already applied the stride when building X
        if model.multi_models:
            forecast = forecast[::stride][:n_forecasts]
        else:
            forecast = forecast[:n_forecasts]

        # Construct TimeSeries objects
        # Depending on last_points_only, either return only the last points of each forecast
        # or all forecasts as a list of TimeSeries
        if last_points_only:
            new_times = generate_index(
                start=hist_fct_start
                + (forecast_horizon + model.output_chunk_shift - 1) * freq,
                length=forecast.shape[0],
                freq=freq * stride,
                name=series_._time_index.name,
            )
            forecasts_ = TimeSeries(
                times=new_times,
                values=forecast[:, -1],
                components=forecast_components,
                static_covariates=series_.static_covariates,
                hierarchy=series_.hierarchy,
                metadata=series_.metadata,
                copy=False,
            )
        else:
            forecasts_ = []

            new_times = generate_index(
                start=hist_fct_start + model.output_chunk_shift * series_.freq,
                length=forecast_horizon + (forecast.shape[0] - 1) * stride,
                freq=freq,
                name=series_._time_index.name,
            )
            for idx_ftc, step_fct in enumerate(
                range(0, forecast.shape[0] * stride, stride)
            ):
                ts = TimeSeries(
                    times=new_times[step_fct : step_fct + forecast_horizon],
                    values=forecast[idx_ftc],
                    components=forecast_components,
                    static_covariates=series_.static_covariates,
                    hierarchy=series_.hierarchy,
                    metadata=series_.metadata,
                    copy=False,
                )
                forecasts_.append(ts)

        forecasts_list.append(forecasts_)
    return forecasts_list
