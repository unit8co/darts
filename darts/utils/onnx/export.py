"""
Torch-dependent ONNX export utilities.

:func:`prepare_onnx_export` wraps a Lightning module so ``torch.onnx.export``
sees named feature tensors and a ``prediction`` output. Recurrent state is
flattened into ``state_in_*`` / ``state_out_*``. Inference after export does
not need this module; use :mod:`darts.utils.onnx.inference`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from darts.logging import raise_log
from darts.utils.data.torch_datasets.utils import ModuleStage, PLModuleInput
from darts.utils.onnx.inference import OnnxModelSpec

_MODULE_INPUT_TENSOR_FIELDS: tuple[str, ...] = (
    "past_target",
    "past_covariates",
    "historic_future_covariates",
    "future_covariates",
    "static_covariates",
)


@dataclass
class OnnxExportBundle:
    """Artifacts produced for ``torch.onnx.export``.

    Parameters
    ----------
    wrapper
        Module that packs named tensors into :class:`PLModuleInput`.
    example_inputs
        Dummy tensors matching ``input_names`` (features, then zero state).
    input_names
        Graph input names passed to ``torch.onnx.export``.
    output_names
        Graph output names (``prediction`` plus ``state_out_*``).
    spec
        Companion metadata written as ``*.onnx.spec.json``.
    dynamic_axes
        Optional dynamic-axis map; ``None`` selects dynamo export.
    """

    wrapper: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    spec: OnnxModelSpec
    dynamic_axes: dict[str, dict[int, str]] | None = None


def flatten_module_state(state: Any) -> tuple[list[torch.Tensor], Any]:
    """Flatten a nested tensor state into a list and a rebuild spec.

    Parameters
    ----------
    state
        ``None``, a tensor, or a nested tuple/list of tensors (e.g. LSTM
        ``(h, c)``).

    Returns
    -------
    tuple[list[torch.Tensor], Any]
        Flat tensors and an opaque spec for :func:`unflatten_module_state`.

    Raises
    ------
    TypeError
        If ``state`` contains an unsupported type.
    """
    if state is None:
        return [], None
    if isinstance(state, torch.Tensor):
        return [state], "tensor"
    if isinstance(state, tuple | list):
        tensors: list[torch.Tensor] = []
        child_specs = []
        for item in state:
            item_tensors, item_spec = flatten_module_state(item)
            tensors.extend(item_tensors)
            child_specs.append(item_spec)
        return tensors, (type(state).__name__, child_specs)
    raise_log(
        ValueError(
            f"Unsupported module state type `{type(state).__name__}`; expected a "
            "tensor or nested tuple/list of tensors."
        )
    )


def unflatten_module_state(tensors: Sequence[torch.Tensor], spec: Any) -> Any:
    """Rebuild a nested tensor state from a flat list and spec.

    Parameters
    ----------
    tensors
        Flat tensors in the order produced by :func:`flatten_module_state`.
    spec
        Rebuild spec from :func:`flatten_module_state`.

    Returns
    -------
    Any
        Nested state matching the original structure, or ``None``.
    """
    if spec is None:
        return None
    return _unflatten_module_state_from(list(tensors), spec)


def _unflatten_module_state_from(tensors: list[torch.Tensor], spec: Any) -> Any:
    """Pop tensors from ``tensors`` according to ``spec``."""
    if spec == "tensor":
        return tensors.pop(0)
    kind, child_specs = spec
    children = [_unflatten_module_state_from(tensors, child) for child in child_specs]
    return tuple(children) if kind == "tuple" else children


def _present_features(
    input_sample: PLModuleInput,
) -> tuple[list[str], list[torch.Tensor]]:
    """Return ``(names, tensors)`` for non-``None`` feature slots on ``input_sample``."""
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    for name in _MODULE_INPUT_TENSOR_FIELDS:
        tensor = getattr(input_sample, name)
        if tensor is not None:
            names.append(name)
            tensors.append(tensor)
    return names, tensors


class _ONNXExportWrapper(nn.Module):
    """Pack named tensors into :class:`PLModuleInput` and unpack module output.

    Same graph for every model: feature tensors in, ``prediction`` out. If the
    module returns recurrent state, flattened ``state_in_*`` / ``state_out_*``
    tensors are added. First-step state is zeros (same as ``hx=None``).

    ``pl_module.forward`` already applies reversible instance norm (via
    ``io_processor``) and returns raw likelihood parameters when a likelihood
    is set.
    """

    def __init__(
        self,
        pl_module: nn.Module,
        feature_names: Sequence[str],
        state_spec: Any,
    ):
        super().__init__()
        self.pl_module = pl_module
        self.feature_names = list(feature_names)
        self.state_spec = state_spec

    def forward(self, *tensors: torch.Tensor):
        """Map flat ONNX inputs to ``pl_module`` and flatten its output."""
        n_features = len(self.feature_names)
        values = dict(zip(self.feature_names, tensors[:n_features]))
        state = (
            unflatten_module_state(tensors[n_features:], self.state_spec)
            if self.state_spec is not None
            else None
        )
        x_in = PLModuleInput(**values, state=state, stage=ModuleStage.PREDICT)
        out = self.pl_module(x_in)
        if self.state_spec is None:
            return out.prediction
        state_out, _ = flatten_module_state(out.state)
        return out.prediction, *state_out


def prepare_onnx_export(
    pl_module: nn.Module,
    input_sample: PLModuleInput,
    *,
    input_chunk_length: int,
    output_chunk_length: int,
    output_chunk_shift: int,
    uses_past_covariates: bool,
    uses_future_covariates: bool,
    uses_static_covariates: bool,
    likelihood_parameter_names: list[str],
) -> OnnxExportBundle:
    """Build the ONNX wrapper and example tensors from a module input.

    Runs ``pl_module`` once to discover recurrent state, then wraps it so
    ``torch.onnx.export`` sees named feature tensors and a ``prediction``
    output (plus flattened ``state_*`` I/O if needed).

    Reversible instance norm and raw likelihood parameters are part of
    ``pl_module.forward`` and are exported as-is.

    Parameters
    ----------
    pl_module
        Fitted Lightning forecasting module (set to eval inside this helper).
    input_sample
        Example :class:`PLModuleInput` used to trace the graph.
    input_chunk_length
        Target history length.
    output_chunk_length
        Steps produced per graph call.
    output_chunk_shift
        Steps that inference start is shifted into the future.
    uses_past_covariates
        Whether the fitted model uses past covariates.
    uses_future_covariates
        Whether the fitted model uses future covariates.
    uses_static_covariates
        Whether the fitted model uses static covariates.
    likelihood_parameter_names
        The likelihood parameter names if the model was trained with a
        likelihood.

    Returns
    -------
    OnnxExportBundle
        Wrapper, dummy inputs, names, and :class:`OnnxModelSpec`.
    """
    feature_names, feature_tensors = _present_features(input_sample)

    pl_module.eval()
    with torch.no_grad():
        dummy_out = pl_module(input_sample)
    state_tensors, state_spec = flatten_module_state(dummy_out.state)
    state_in_names = [f"state_in_{i}" for i in range(len(state_tensors))]
    state_out_names = [f"state_out_{i}" for i in range(len(state_tensors))]

    wrapper = _ONNXExportWrapper(pl_module, feature_names, state_spec)
    example_inputs = list(feature_tensors)
    input_names = list(feature_names)
    output_names = ["prediction"]
    if state_tensors:
        example_inputs.extend(torch.zeros_like(t) for t in state_tensors)
        input_names.extend(state_in_names)
        output_names.extend(state_out_names)

    spec = OnnxModelSpec(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        output_chunk_shift=output_chunk_shift,
        uses_past_covariates=uses_past_covariates,
        uses_future_covariates=uses_future_covariates,
        uses_static_covariates=uses_static_covariates,
        likelihood_parameter_names=likelihood_parameter_names,
        feature_input_names=feature_names,
        input_names=input_names,
        output_names=output_names,
        state_input_names=state_in_names,
        state_output_names=state_out_names,
        state_input_shapes=[list(t.shape) for t in state_tensors] or None,
    )
    return OnnxExportBundle(
        wrapper=wrapper,
        example_inputs=tuple(example_inputs),
        input_names=input_names,
        output_names=output_names,
        spec=spec,
    )
