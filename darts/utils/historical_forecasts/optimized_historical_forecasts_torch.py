"""
Optimized Historical Forecasts for `TorchForecastingModel`
----------------------------------------------------------
"""

import inspect
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.utils.historical_forecasts.utils import (
    _get_historical_forecast_boundaries,
    _process_predict_start_points_bounds,
)
from darts.utils.timeseries_generation import _build_forecast_series_from_schema

logger = get_logger(__name__)


def _optimized_historical_forecasts(
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
    last_points_only: bool = True,
    show_warnings: bool = True,
    verbose: bool = False,
    predict_likelihood_parameters: bool = False,
    random_state: Optional[int] = None,
    predict_kwargs: Optional[dict[str, Any]] = None,
) -> Union[Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for TorchForecastingModels

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.

    The data_transformers are applied in historical_forecasts (input and predictions)
    """
    predict_kwargs = predict_kwargs or {}
    if "verbose" not in predict_kwargs:
        predict_kwargs["verbose"] = verbose
    bounds = []
    for idx, series_ in enumerate(series):
        past_covariates_ = past_covariates[idx] if past_covariates is not None else None
        future_covariates_ = (
            future_covariates[idx] if future_covariates is not None else None
        )
        # obtain forecastable indexes boundaries, adjust target & covariates boundaries accordingly
        (
            hist_fct_start,
            hist_fct_end,
            _,
            _,
            _,
            _,
            _,
            _,
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
            freq=series_.freq,
            show_warnings=show_warnings,
        )
        left_bound = series_.get_index_at_point(hist_fct_start)
        # latest possible prediction start is one time step after end of target series
        if hist_fct_end > series_.end_time():
            right_bound = len(series_)
        else:
            right_bound = series_.get_index_at_point(hist_fct_end)
        bounds.append((left_bound, right_bound))

    bounds, cum_lengths = _process_predict_start_points_bounds(
        series=series,
        bounds=np.array(bounds),
        stride=stride,
    )

    # TODO: is there a better way to call the super().predict() from TorchForecastingModel, without having to
    #  import it? (avoid circular imports)
    tfm_cls = [
        cls
        for cls in model.__class__.__mro__
        if cls.__name__ == "TorchForecastingModel"
    ][0]
    super_predict_params = inspect.signature(super(tfm_cls, model).predict).parameters
    super(tfm_cls, model).predict(
        n=forecast_horizon,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        num_samples=num_samples,
        predict_likelihood_parameters=predict_likelihood_parameters,
        show_warnings=show_warnings,
        **{k: v for k, v in predict_kwargs.items() if k in super_predict_params},
    )

    dataset = model._build_inference_dataset(
        n=forecast_horizon,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        stride=stride,
        bounds=bounds,
    )

    # to avoid having to generate `TimeSeries` twice when `last_points_only=True`, we only
    # return the values in that case
    model_out = model.predict_from_dataset(
        n=forecast_horizon,
        dataset=dataset,
        num_samples=num_samples,
        predict_likelihood_parameters=predict_likelihood_parameters,
        values_only=last_points_only,
        random_state=random_state,
        **predict_kwargs,
    )

    # torch model returns output in the order of the historical forecasts: we reorder per time series
    forecasts_list = []
    likelihood_component_names_fn = (
        model.likelihood.component_names if predict_likelihood_parameters else None
    )
    for series_idx in range(len(series)):
        pred_idx_start = 0 if not series_idx else cum_lengths[series_idx - 1]
        pred_idx_end = cum_lengths[series_idx]

        if last_points_only:
            # model output is tuple of (np.ndarray of predictions, series schemas, pred start times)
            preds = model_out[0][pred_idx_start:pred_idx_end]
            schema = model_out[1][pred_idx_start]
            pred_start = model_out[2][pred_idx_start]

            # predictions come with the entire horizon: we extract last values
            preds = preds[:, -1]
            pred_start += (forecast_horizon - 1) * schema["time_freq"]

            # adjust frequency with stride
            schema["time_freq"] *= stride

            # predictions come with the entire horizon: we extract last values
            preds = _build_forecast_series_from_schema(
                values=preds,
                schema=schema,
                pred_start=pred_start,
                predict_likelihood_parameters=predict_likelihood_parameters,
                likelihood_component_names_fn=likelihood_component_names_fn,
                copy=False,
            )
        else:
            # model output is already a sequence of forecasted `TimeSeries`
            preds = model_out[pred_idx_start:pred_idx_end]
        forecasts_list.append(preds)
    return forecasts_list
