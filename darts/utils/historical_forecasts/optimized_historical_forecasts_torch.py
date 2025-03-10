import inspect
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from darts.logging import get_logger
from darts.timeseries import TimeSeries
from darts.utils.historical_forecasts.utils import (
    _get_historical_forecast_boundaries,
    _process_predict_start_points_bounds,
)
from darts.utils.utils import generate_index

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
    **kwargs,
) -> Union[Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
    """
    Optimized historical forecasts for TorchForecastingModels

    Rely on _check_optimizable_historical_forecasts() to check that the assumptions are verified.

    The data_transformers are applied in historical_forecasts (input and predictions)
    """
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
        forecast_horizon,
        series,
        past_covariates,
        future_covariates,
        num_samples=num_samples,
        predict_likelihood_parameters=predict_likelihood_parameters,
        show_warnings=show_warnings,
        **{k: v for k, v in kwargs.items() if k in super_predict_params},
    )

    dataset = model._build_inference_dataset(
        target=series,
        n=forecast_horizon,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        stride=stride,
        bounds=bounds,
    )

    predictions = model.predict_from_dataset(
        forecast_horizon,
        dataset,
        verbose=verbose,
        num_samples=num_samples,
        predict_likelihood_parameters=predict_likelihood_parameters,
        **kwargs,
    )

    # torch models return list of time series in order of historical forecasts: we reorder per time series
    forecasts_list = []
    for series_idx in range(len(series)):
        pred_idx_start = 0 if not series_idx else cum_lengths[series_idx - 1]
        pred_idx_end = cum_lengths[series_idx]
        preds = predictions[pred_idx_start:pred_idx_end]
        if last_points_only:
            # torch predictions come with the entire horizon: we extract last values
            preds = TimeSeries.from_times_and_values(
                times=generate_index(
                    start=preds[0].end_time(),
                    length=len(preds),
                    freq=preds[0].freq * stride,
                ),
                values=np.concatenate(
                    [p.all_values(copy=False)[-1:, :, :] for p in preds], axis=0
                ),
                columns=preds[0].columns,
                static_covariates=preds[0].static_covariates,
                hierarchy=preds[0].hierarchy,
                metadata=preds[0].metadata,
            )
        forecasts_list.append(preds)
    return forecasts_list
