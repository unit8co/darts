"""
ONNX utilities.

ONNX inference utilities for Darts torch forecasting models are torch-free (ONNX Runtime + NumPy).
"""

from darts.utils.onnx.inference import (
    OnnxModelSpec,
    extract_point_forecast,
    prepare_onnx_inputs,
    run_onnx_prediction,
)

__all__ = [
    "OnnxModelSpec",
    "extract_point_forecast",
    "prepare_onnx_inputs",
    "run_onnx_prediction",
]
