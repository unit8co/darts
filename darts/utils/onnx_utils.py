"""
ONNX Utils
----------
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from darts import TimeSeries
from darts.utils.data.torch_datasets.utils import (
    ModuleStage,
    PLModuleInput,
    _flatten_state,
    _unflatten_state,
)

_MODULE_INPUT_TENSOR_FIELDS: tuple[str, ...] = (
    "past_target",
    "past_covariates",
    "historic_future_covariates",
    "future_covariates",
    "static_covariates",
)


class _ONNXExportWrapper(nn.Module):
    """Pack named tensors into :class:`PLModuleInput` and unpack :class:`PLModuleOutput`.

    Used so every Darts torch module (including recurrent models) exports through
    the same ONNX graph: named feature tensors in, ``prediction`` and optional
    flattened ``state`` tensors out.
    """

    def __init__(
        self,
        pl_module: nn.Module,
        input_names: Sequence[str],
        state_names: Sequence[str],
        state_spec,
    ):
        super().__init__()
        self.pl_module = pl_module
        self.input_names = list(input_names)
        self.state_names = list(state_names)
        self.state_spec = state_spec

    def forward(self, *tensors: torch.Tensor):
        values = dict(zip(self.input_names + self.state_names, tensors))
        state_tensors = [values.pop(name) for name in self.state_names]
        state = (
            _unflatten_state(state_tensors, self.state_spec)
            if self.state_names
            else None
        )
        x_in = PLModuleInput(**values, state=state, stage=ModuleStage.PREDICT)
        out = self.pl_module(x_in)
        if not self.state_names:
            return out.prediction
        state_out, _ = _flatten_state(out.state)
        return (out.prediction, *state_out)


def prepare_onnx_export(
    pl_module: nn.Module, input_sample: PLModuleInput
) -> tuple[_ONNXExportWrapper, tuple[torch.Tensor, ...], list[str], list[str]]:
    """Build a generic ONNX wrapper and example tensors from a module input.

    Returns ``(wrapper, example_inputs, input_names, output_names)``.
    """
    feature_names: list[str] = []
    feature_tensors: list[torch.Tensor] = []
    for name in _MODULE_INPUT_TENSOR_FIELDS:
        tensor = getattr(input_sample, name)
        if tensor is not None:
            feature_names.append(name)
            feature_tensors.append(tensor)

    pl_module.eval()
    with torch.no_grad():
        dummy_out = pl_module(input_sample)

    state_tensors, state_spec = _flatten_state(dummy_out.state)
    state_names = [f"state_{i}" for i in range(len(state_tensors))]
    # use zeros as the initial carry (equivalent to ``state=None`` for RNNs)
    example_state = [torch.zeros_like(t) for t in state_tensors]

    wrapper = _ONNXExportWrapper(pl_module, feature_names, state_names, state_spec)
    example_inputs = tuple(feature_tensors + example_state)
    output_names = ["prediction", *state_names]
    return wrapper, example_inputs, feature_names + state_names, output_names


def prepare_onnx_inputs(
    model,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
) -> dict[str, np.ndarray]:
    """Helper function to slice input features for ONNX inference.

    Returns a dictionary mapping ONNX input names to numpy arrays with a leading batch dimension.
    Only includes feature inputs that the model uses. Recurrent ``state_*`` inputs must be
    provided separately (typically zeros for the first step).
    """
    # get input & output windows
    past_start = series.end_time() - (model.input_chunk_length - 1) * series.freq
    past_end = series.end_time()
    future_start = past_end + 1 * series.freq
    future_end = past_end + model.output_chunk_length * series.freq

    inputs: dict[str, np.ndarray] = dict()
    inputs["past_target"] = np.expand_dims(
        series[past_start:past_end].values(), axis=0
    ).astype(series.dtype)

    if past_covariates is not None and model.uses_past_covariates:
        inputs["past_covariates"] = np.expand_dims(
            past_covariates[past_start:past_end].values(), axis=0
        ).astype(series.dtype)

    if future_covariates is not None and model.uses_future_covariates:
        inputs["historic_future_covariates"] = np.expand_dims(
            future_covariates[past_start:past_end].values(), axis=0
        ).astype(series.dtype)
        inputs["future_covariates"] = np.expand_dims(
            future_covariates[future_start:future_end].values(), axis=0
        ).astype(series.dtype)

    if series.has_static_covariates and model.uses_static_covariates:
        inputs["static_covariates"] = np.expand_dims(
            series.static_covariates_values(), axis=0
        ).astype(series.dtype)

    return inputs
