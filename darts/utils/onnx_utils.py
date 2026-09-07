"""
ONNX Utils
----------
"""

import numpy as np

from darts import TimeSeries


def prepare_onnx_inputs(
    model,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
) -> dict[str, np.ndarray]:
    """Helper function to slice input features for ONNX inference.

    Returns a dictionary mapping ONNX input names to numpy arrays with a leading batch dimension.
    Only includes inputs that the model uses.
    """
    # get input & output windows
    past_start = series.end_time() - (model.input_chunk_length - 1) * series.freq
    past_end = series.end_time()
    future_start = past_end + 1 * series.freq
    future_end = past_end + model.output_chunk_length * series.freq

    inputs: dict[str, np.ndarray] = {}
    inputs["past_target"] = np.expand_dims(
        series[past_start:past_end].values(), axis=0
    ).astype(series.dtype)

    if past_covariates is not None and model.uses_past_covariates:
        inputs["past_cov"] = np.expand_dims(
            past_covariates[past_start:past_end].values(), axis=0
        ).astype(series.dtype)

    if future_covariates is not None and model.uses_future_covariates:
        inputs["historic_future_cov"] = np.expand_dims(
            future_covariates[past_start:past_end].values(), axis=0
        ).astype(series.dtype)
        inputs["future_cov"] = np.expand_dims(
            future_covariates[future_start:future_end].values(), axis=0
        ).astype(series.dtype)

    if series.has_static_covariates and model.uses_static_covariates:
        inputs["static_cov"] = np.expand_dims(
            series.static_covariates_values(), axis=0
        ).astype(series.dtype)

    return inputs
