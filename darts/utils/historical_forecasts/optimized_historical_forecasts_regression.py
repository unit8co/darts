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
from darts.utils.ts_utils import get_single_series
from darts.utils.utils import generate_index

logger = get_logger(__name__)

_NP_2_OR_ABOVE = int(np.__version__.split(".")[0]) >= 2
_STABLE_SORT_KWARGS = {"stable": True} if _NP_2_OR_ABOVE else {"kind": "stable"}


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
    random_state: Optional[int] = None,
    predict_kwargs: Optional[dict[str, Any]] = None,
) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for SKLearnModel with last_points_only = True

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.

    The data_transformers are applied in historical_forecasts (input and predictions)
    """
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
        forecast = model._predict(
            x=np.repeat(X, num_samples, axis=0),
            num_samples=num_samples,
            predict_likelihood_parameters=predict_likelihood_parameters,
            random_state=random_state,
            **predict_kwargs,
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

        # extract the last sub-model forecast for each component
        if model.multi_models:
            forecast = forecast[:, forecast_horizon - 1]
        else:
            forecast = forecast[:, 0]

        if (
            stride == 1
            and model.output_chunk_length == 1
            and model.output_chunk_shift == 0
        ):
            times = times[0]
        else:
            times = generate_index(
                start=hist_fct_start
                + (forecast_horizon + model.output_chunk_shift - 1) * freq,
                length=forecast.shape[0],
                freq=freq * stride,
                name=series_._time_index.name,
            )

        forecasts_list.append(
            TimeSeries(
                times=times,
                values=forecast,
                components=forecast_components,
                static_covariates=series_.static_covariates,
                hierarchy=series_.hierarchy,
                metadata=series_.metadata,
                copy=False,
            )
        )
    return forecasts_list


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
    # # TODO: make this better
    # target_lag_tag = "_target_lag"
    # lagged_target_names = [
    #     name for name in model.lagged_feature_names if target_lag_tag in name
    # ]
    # target_lags_order = [
    #     int(name.split(target_lag_tag)[-1]) for name in lagged_target_names
    # ]

    # prepare index to reorder features by lags across components
    if "target" in model.component_lags:
        component_lags = [
            comp_lags for comp_lags in model.component_lags["target"].values()
        ]
    else:
        component_lags = [model.lags["target"]] * get_single_series(series).n_components
    component_lags_reordered = np.concatenate(component_lags).argsort(
        **_STABLE_SORT_KWARGS
    )
    _ = component_lags_reordered
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

        # latest possible prediction start is one time step after end of target series
        if hist_fct_start > series_.end_time():
            left_bound = len(series_)
        else:
            left_bound = series_.get_index_at_point(hist_fct_start)

        if hist_fct_end > series_.end_time():
            right_bound = len(series_)
        else:
            right_bound = series_.get_index_at_point(hist_fct_end)

        bounds_array, cum_lengths = _process_predict_start_points_bounds(
            series=[series_],
            bounds=np.array([[left_bound, right_bound]]),
            stride=stride,
        )
        n_forecasts = cum_lengths[0]

        # TODO: update comment and explain
        if (
            forecast_horizon <= model.output_chunk_length + model.output_chunk_length
        ) or (forecast_horizon % model.output_chunk_length == 0):
            is_simple_forecast = True
        else:
            is_simple_forecast = False

        # In case of multi-models, we predict all steps at once so autoregression is done
        # output_chunk_length by output_chunk_length. Without multi-models,
        # a single model predicts one step at a time and we shift
        # the input sequence by one time step after each prediction.
        if model.multi_models and is_simple_forecast:
            step = model.output_chunk_length
        else:
            step = 1

        output_step = model.output_chunk_length if model.multi_models else 1

        # If forecast_horizon <= step, we can predict all steps in one go
        # so X will be of shape (n_forecasts, n_lags, n_samples = 1)
        # Otherwise, X will be of shape (n_forecasts, n_lags, n_samples = 1, n_prediction_iterations)
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
            forecast_horizon=forecast_horizon,
            step=step,
        )

        X = X[0][:, :, 0]

        # # stride can be applied directly without auto-regression, or when forecast horizon is a
        # # round-multiple of output chunk length
        # if is_simple_forecast:
        #     X = X[::stride]
        X = X[::stride]

        predictions = []

        # Without autoregression, n_prediction_iterations is 1,
        # so internally the last dimension has been collapsed.
        # Add a dummy dimension so the for loop works as expected
        if X.ndim == 2:
            X = X[:, :, np.newaxis]

        # Trick to fix with stride, because NaN are in X at the end of the series, but
        # with stride, we might not get NaN in current_X, so we manually set the relevant
        # columns to NaN so that they get filled with previous predictions
        # TODO: does not work yet with multivariate target
        if not model.multi_models and "target" in model.lags:
            offset_idx = 1
            end_idx = len(model.lags["target"])
            for i in range(model.output_chunk_length, X.shape[-1]):
                start_idx = end_idx - offset_idx
                X[:, start_idx:end_idx, i] = np.nan
                offset_idx += 1

        lags_output = np.array([i for i in range(model.output_chunk_length)])

        # last_step_shift = 0
        # t_pred = 0
        t_pred_new = output_step
        # start_idx = 0

        # forecasts = []
        # forecast = None
        for pred_idx in range(X.shape[-1]):
            # Select the X for the current forecast iteration
            current_X = X[:, :, pred_idx]
            # if not model.multi_models:
            #     # TODO: why can't we take stride directly?
            #     # with multi-models we can directly take the stridden examples
            #     current_X = current_X[::stride]

            current_X = np.repeat(current_X, num_samples, axis=0)

            # When predictions exist, we are either doing autoregression, or predicting multiple steps with
            # multi_models=False
            # We are also interested in filling current_X with previous predictions when we use target
            # (so when model.lags contains "target")
            # if predictions and "target" in model.lags:

            # # Fill history is when we are doing autoregression
            # fill_history = t_pred + 1 > model.output_chunk_length
            #
            # if fill_history:
            #     lags_output = np.array([[i for i in range(model.output_chunk_length)]] * series_.n_components)
            #     lags_output_adjust = lags_output - step
            #     for comp_idx, (comp_lags, output_lags) in enumerate(zip(component_lags, lags_output_adjust)):
            #         for lag_idx, lag in enumerate(comp_lags):
            #             if lag in output_lags:
            #
            # # lags are identical for multiple series: pre-compute lagged features and reordered lagged features
            # if np.isnan(current_X).any():
            #     # When we have NaN in current_X, we need to fill them
            #     _, col = np.where(np.isnan(current_X))
            #     col = np.unique(col)
            #
            #     if fill_history:
            #         if model.multi_models:
            #             # Flatten the forecasts as a 2D array for easier indexing
            #             # Instead of having (n_series * n_samples, output_chunk_length, n_components),
            #             # we have (n_series * n_samples, output_chunk_length * n_components)
            #             # Because X is built the same way, so the indexing will be coherent
            #             flattened_forecasts = forecasts.reshape(forecasts.shape[0], -1)
            #             # Find which prediction steps we need to fill in current_X. It's the last len(col) steps
            #             # but if last_step_shift > 0, we need to shift accordingly
            #             slice_start = flattened_forecasts.shape[1] - (
            #                 col.size + last_step_shift
            #             )
            #             slice_start = 0 if slice_start < 0 else slice_start
            #             slice_end = slice_start + len(col)
            #             slice_end = (
            #                 flattened_forecasts.shape[1]
            #                 if slice_end > flattened_forecasts.shape[1]
            #                 else slice_end
            #             )
            #
            #             # Fill the current X columns with the corresponding forecasts
            #             current_X[:, col] = flattened_forecasts[
            #                 :,
            #                 slice_start:slice_end,
            #             ].reshape(current_X[:, col].shape)
            #         else:
            #             # When not using multi_models, we predict one step at a time so we only need to
            #             # shift the start_idx one by one each prediction iteration
            #             flattened_forecasts = forecasts.reshape(forecasts.shape[0], -1)
            #             current_X[:, col] = flattened_forecasts[
            #                 :,
            #                 start_idx : start_idx + col.size,
            #             ].reshape(current_X[:, col].shape)
            #         if col.size == len(model.lags["target"]) and forecast is not None:
            #             start_idx += forecast.shape[-1]
            #     else:
            #         current_X[:, col] = forecasts.reshape(forecasts.shape[0], -1)[
            #             :, : col.size
            #         ].reshape(current_X[:, col].shape)

            # repeat rows for probabilistic forecast
            forecast = model._predict(
                x=current_X,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                **predict_kwargs,
            )

            # update history of future X in case of auto-regression
            # TODO, make it work for
            #  - auto-reg with forecast_horizon % ocl > 0
            #  - multi_models=False
            for auto_reg_idx in range(1, X.shape[-1] - pred_idx):
                lags_output_adjust = lags_output - (step * auto_reg_idx)
                next_X = X[:, :, pred_idx + auto_reg_idx]

                update_x_indices = []
                take_y_indices = []
                counter = 0
                for comp_idx, comp_lags in enumerate(component_lags):
                    for lag_idx, lag in enumerate(comp_lags):
                        y_pos = np.argwhere(lags_output_adjust == lag)
                        if len(y_pos) > 0:
                            update_x_indices.append(component_lags_reordered[counter])
                            take_y_indices.append(
                                comp_idx + y_pos[0, 0] * series_.n_components
                            )
                        counter += 1
                next_X[:, update_x_indices] = forecast.reshape(forecast.shape[0], -1)[
                    :, take_y_indices
                ]

            if t_pred_new % output_step == 0 or t_pred_new + step == forecast_horizon:
                predictions.append(forecast)
            # elif pred_idx + 1 == X.shape[-1]:
            #     last_step_shift = t_pred - (forecast_horizon - step)
            #     d = 1

            # # in case of autoregressive forecast `(t_pred > 0)` and if `n` is not a round multiple of `step`,
            # # we have to step back `step` from `n` in the last iteration
            # if 0 < forecast_horizon - t_pred < step and t_pred > 0:
            #     last_step_shift = t_pred - (forecast_horizon - step)
            #     t_pred = forecast_horizon - step
            t_pred_new += step

            # forecast = forecast[:, last_step_shift:, ...]
            # predictions.append(forecast)
            # t_pred += step
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
        if not is_simple_forecast:
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
