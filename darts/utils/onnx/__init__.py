"""
ONNX Utilities
--------------

ONNX inference utilities for Darts torch forecasting models are torch-free (ONNX Runtime + NumPy).
"""

from darts.utils.onnx.inference import (
    OnnxModelSpec,
    prepare_onnx_inputs,
    run_onnx_prediction,
)

__all__ = [
    "OnnxModelSpec",
    "prepare_onnx_inputs",
    "run_onnx_prediction",
]
