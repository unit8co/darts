"""
ONNX Utilities
--------------

ONNX inference utilities for Darts torch forecasting models are torch-free (ONNX Runtime + NumPy).
"""

from darts.utils.onnx.inference import prepare_onnx_inputs, run_onnx_prediction

__all__ = [
    "prepare_onnx_inputs",
    "run_onnx_prediction",
]
