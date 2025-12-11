"""
Optimized Historical Forecasts for SKLearnModel
-----------------------------------------------
"""

from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darts import TimeSeries
from darts.logging import get_logger
from darts.utils import _build_tqdm_iterator
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.historical_forecasts.utils import _get_historical_forecast_boundaries
from darts.utils.ts_utils import get_single_series
from darts.utils.utils import generate_index

logger = get_logger(__name__)

_NP_2_OR_ABOVE = int(np.__version__.split(".")[0]) >= 2
_STABLE_SORT_KWARGS = {"stable": True} if _NP_2_OR_ABOVE else {"kind": "stable"}


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
    multi_models = model.multi_models
    output_chunk_length = model.output_chunk_length
    output_chunk_shift = model.output_chunk_shift

    # get target lags and X positions for auto-regression
    if model._get_lags("target") is not None:
        if "target" in model.component_lags:
            target_lags = model.component_lags["target"].values()
        else:
            target_lags = [model.lags["target"]] * get_single_series(
                series
            ).n_components

        # map which target component lag belongs to which position in X
        counter = 0
        target_lag_positions = [[] for _ in range(len(target_lags))]
        for lag in range(min(model.lags["target"]), max(model.lags["target"]) + 1):
            for comp_idx, comp_lags in enumerate(target_lags):
                if lag in comp_lags:
                    target_lag_positions[comp_idx].append(counter)
                    counter += 1
    else:
        target_lags, target_lag_positions = [], []

    # get the output lags
    if multi_models:
        lags_output = np.array([i for i in range(output_chunk_length)])
    else:
        lags_output = np.array([output_chunk_length - 1])
    lags_output += output_chunk_shift

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

        is_auto_regression = forecast_horizon > output_chunk_length + output_chunk_shift

        # a simple forecast
        if (not is_auto_regression) or (forecast_horizon % output_chunk_length == 0):
            is_simple_forecast = True
        else:
            is_simple_forecast = False

        # TODO: update docs
        # In case of multi-models, we predict all steps at once so autoregression is done
        # output_chunk_length by output_chunk_length. Without multi-models,
        # a single model predicts one step at a time and we shift
        # the input sequence by one time step after each prediction.
        if multi_models and is_simple_forecast:
            step = output_chunk_length
        else:
            # without multi-models: we must predict each output time with step=1
            # or  multi-models: we must predict each output time with step=1
            step = 1

        output_step = output_chunk_length if multi_models else 1

        # If forecast_horizon <= step, we can predict all steps in one go
        # so X will be of shape (n_forecasts, n_lags, n_samples = 1)
        # Otherwise, X will be of shape (n_forecasts, n_lags, n_samples = 1, n_prediction_iterations)
        # where n_prediction_iterations = ceil(forecast_horizon / step)
        X, _ = create_lagged_prediction_data(
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
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
            multi_models=multi_models,
            forecast_horizon=forecast_horizon,
            step=step,
        )

        # -> (n_forecasts, n_lags, n_prediction_iterations)
        X = X[0][:, :, 0]
        if X.ndim == 2:
            # without autoregression, the last dimension was collapsed; bring back dimension
            X = X[:, :, np.newaxis]

        # TODO: update comment
        # stride can be applied directly without auto-regression, or when forecast horizon is a
        if multi_models or is_auto_regression:
            X = X[::stride]

        # -> (n_forecasts * n_samples, n_lags, n_prediction_iterations)
        X = np.repeat(X, num_samples, axis=0)

        predictions = []
        t_pred = output_chunk_length if multi_models else 1
        for pred_idx in range(X.shape[-1]):
            # Select the X for the current forecast iteration
            current_X = X[:, :, pred_idx]

            # repeat rows for probabilistic forecast
            forecast = model._predict(
                x=current_X,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                **predict_kwargs,
            )

            # in case of auto-regression: update history of future Xs with current forecast
            for auto_reg_idx in range(1, X.shape[-1] - pred_idx):
                # get the future iteration's lagged features
                next_X = X[:, :, pred_idx + auto_reg_idx]

                # determine which lags the forecasts correspond to in the future iteration
                lags_output_adjust = lags_output - (step * auto_reg_idx)

                # find matches between forecasted components and component-specific lags of the future iteration
                take_y_indices = []
                update_x_indices = []
                for comp_idx, comp_lags in enumerate(target_lags):
                    for lag_idx, lag in enumerate(comp_lags):
                        y_pos = np.argwhere(lags_output_adjust == lag)
                        if len(y_pos) > 0:
                            update_x_indices.append(
                                target_lag_positions[comp_idx][lag_idx]
                            )
                            take_y_indices.append(
                                comp_idx + y_pos[0, 0] * series_.n_components
                            )

                # update future X with current matched predictions
                next_X[:, update_x_indices] = forecast.reshape(forecast.shape[0], -1)[
                    :, take_y_indices
                ]

            # forecast has shape ((forecastable_index_length-1)*num_samples, k, n_component)
            # where k = output_chunk length if multi_models, 1 otherwise
            # reshape into (forecasted indexes, output_chunk_length, n_components, n_samples)
            forecast = np.moveaxis(
                forecast.reshape(
                    X.shape[0] // num_samples,
                    num_samples,
                    output_chunk_length if multi_models else 1,
                    -1,
                ),
                1,
                -1,
            )

            if not multi_models and not is_auto_regression:
                # entire forecast horizon is given by multiple (previous) forecasts -> apply sliding window
                forecast = sliding_window_view(
                    forecast[:, 0],
                    (forecast_horizon, len(forecast_components), num_samples),
                )

                # apply stride, remove the last windows, slice output_chunk_length to keep forecast_horizon values
                if forecast_horizon != output_chunk_length:
                    # Additional shift, to account for the model output_chunk_length
                    # used to convert the shift into the appropriate unit
                    forecast = forecast[
                        : forecast_horizon - output_chunk_length : stride,
                        0,
                        0,
                        :forecast_horizon,
                        :,
                        :,
                    ]
                # apply stride
                else:
                    forecast = forecast[::stride, 0, 0, :, :, :]

            if t_pred % output_step == 0 or t_pred > forecast_horizon:
                predictions.append(forecast[:, :forecast_horizon])
            elif t_pred + output_chunk_shift == forecast_horizon:
                take_last_n = (
                    forecast_horizon - output_chunk_shift
                ) % output_chunk_length
                predictions.append(forecast[:, -take_last_n:])

            t_pred += step

        forecast = np.concatenate(predictions, axis=1)

        # depending on last_points_only, either return only the last points of each forecast
        # or all forecasts as a list of TimeSeries
        if last_points_only:
            new_times = generate_index(
                start=hist_fct_start
                + (forecast_horizon + output_chunk_shift - 1) * freq,
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
                start=hist_fct_start + output_chunk_shift * series_.freq,
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
