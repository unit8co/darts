from typing import Optional

import numpy as np

from darts import TimeSeries


def prepare_onnx_inputs(
    model,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Helper function to slice and concatenate the input features.

    In order to remove the dependency on the `model` argument, it can be decomposed into
    the following arguments (and simplified depending on the characteristics of the model used):
      - model_icl
      - model_ocl
      - model_uses_past_covs
      - model_uses_future_covs
      - model_uses_static_covs
    """
    past_feats, future_feats, static_feats = None, None, None
    # get input & output windows
    past_start = series.end_time() - (model.input_chunk_length - 1) * series.freq
    past_end = series.end_time()
    future_start = past_end + 1 * series.freq
    future_end = past_end + model.output_chunk_length * series.freq
    # extract all historic and future features from target, past and future covariates
    past_feats = series[past_start:past_end].values()
    if past_covariates and model.uses_past_covariates:
        # extract past covariates
        past_feats = np.concatenate(
            [past_feats, past_covariates[past_start:past_end].values()], axis=1
        )
    if future_covariates and model.uses_future_covariates:
        # extract past part of future covariates
        past_feats = np.concatenate(
            [past_feats, future_covariates[past_start:past_end].values()], axis=1
        )
        # extract future part of future covariates
        future_feats = future_covariates[future_start:future_end].values()
    # add batch dimension -> (batch, n time steps, n components)
    past_feats = np.expand_dims(past_feats, axis=0).astype(series.dtype)
    future_feats = np.expand_dims(future_feats, axis=0).astype(series.dtype)
    # extract static covariates
    if series.has_static_covariates and model.uses_static_covariates:
        static_feats = np.expand_dims(series.static_covariates_values(), axis=0).astype(
            series.dtype
        )
    return past_feats, future_feats, static_feats
