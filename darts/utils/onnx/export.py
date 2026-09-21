"""
Torch-dependent ONNX export utilities.

:func:`_prepare_onnx_export` wraps a Lightning module so ``torch.onnx.export``
sees named feature tensors and a ``prediction`` output. Recurrent state is
flattened into ``state_in_*`` / ``state_out_*``. Inference after export does
not need this module; use :mod:`darts.utils.onnx.inference`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch
import torch.nn as nn
from onnx import helper

from darts.logging import raise_log
from darts.utils.data.torch_datasets.utils import ModuleStage, PLModuleInput
from darts.utils.onnx.inference import ONNX_SPEC_METADATA_KEY, OnnxModelSpec

_MODULE_INPUT_TENSOR_FIELDS: tuple[str, ...] = (
    "past_target",
    "past_covariates",
    "historic_future_covariates",
    "future_covariates",
    "static_covariates",
)

if TYPE_CHECKING:
    from darts.models.forecasting.pl_forecasting_module import PLForecastingModule


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
        Forecasting metadata embedded in the exported ``.onnx`` file.
    dynamic_axes
        Optional dynamic-axis map; ``None`` selects dynamo export.
    """

    wrapper: nn.Module
    example_inputs: tuple[torch.Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    spec: OnnxModelSpec
    dynamic_axes: dict[str, dict[int, str]] | None = None


def to_onnx(
    pl_module: PLForecastingModule,
    path: str,
    **kwargs,
):
    """Hello

    Parameters
    ----------
    pl_module
        The ``PLForecastingModule`` to export.
    path
        Path under which to save the model at its current state.
    **kwargs
        Additional kwargs for PyTorch's :func:`torch.onnx.export` method (except ``args`` and
        the export destination). For more information, read the `official documentation
        <https://pytorch.org/docs/master/onnx.html#torch.onnx.export>`__.
    """
    # TODO: all models are currently exported with a fixed `batch_size=1`; in the future
    #  we could allow one of:
    #  - setting a custom (fixed) batch size (this results in dynamo=True)
    #  - handle dynamic batch sizes (dynamic axes; this results in dynamo=False)
    input_sample = _onnx_dummy_input(pl_module)
    bundle = pl_module._onnx_wrapper(input_sample)
    _save_onnx_export(bundle, path, **kwargs)


def _save_onnx_export(
    bundle: OnnxExportBundle,
    path: str | Path,
    **export_kwargs: Any,
) -> None:
    """Export ``bundle`` to ONNX and embed its spec in model metadata."""
    bundle.wrapper.eval()
    use_dynamo = bundle.dynamic_axes is None
    kwargs: dict[str, Any] = {
        "model": bundle.wrapper,
        "args": bundle.example_inputs,
        "input_names": bundle.input_names,
        "output_names": bundle.output_names,
        "external_data": False,
        "dynamo": use_dynamo,
        **export_kwargs,
    }
    if bundle.dynamic_axes is not None:
        kwargs["dynamic_axes"] = bundle.dynamic_axes

    if use_dynamo:
        onnx_program = torch.onnx.export(f=None, **kwargs)
        model_proto = onnx_program.model_proto
    else:
        torch.onnx.export(f=path, **kwargs)
        model_proto = onnx.load(path)

    # attach model specs and store
    helper.set_model_props(
        model_proto,
        {
            ONNX_SPEC_METADATA_KEY: json.dumps(
                asdict(bundle.spec), separators=(",", ":")
            )
        },
    )
    onnx.save(model_proto, path)


def _flatten_module_state(state: Any) -> tuple[list[torch.Tensor], Any]:
    """Flatten a nested tensor state into a list and a rebuild spec.

    Parameters
    ----------
    state
        ``None``, a tensor, or a nested tuple/list of tensors (e.g. LSTM
        ``(h, c)``).

    Returns
    -------
    tuple[list[torch.Tensor], Any]
        Flat tensors and an opaque spec for :func:`_unflatten_module_state`.

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
            item_tensors, item_spec = _flatten_module_state(item)
            tensors.extend(item_tensors)
            child_specs.append(item_spec)
        return tensors, (type(state).__name__, child_specs)
    raise_log(
        ValueError(
            f"Unsupported module state type `{type(state).__name__}`; expected a "
            "tensor or nested tuple/list of tensors."
        )
    )


def _unflatten_module_state(tensors: Sequence[torch.Tensor], spec: Any) -> Any:
    """Rebuild a nested tensor state from a flat list and spec.

    Parameters
    ----------
    tensors
        Flat tensors in the order produced by :func:`_flatten_module_state`.
    spec
        Rebuild spec from :func:`_flatten_module_state`.

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
            _unflatten_module_state(tensors[n_features:], self.state_spec)
            if self.state_spec is not None
            else None
        )
        x_in = PLModuleInput(**values, state=state, stage=ModuleStage.PREDICT)
        out = self.pl_module(x_in)
        if self.state_spec is None:
            return out.prediction
        state_out, _ = _flatten_module_state(out.state)
        return out.prediction, *state_out


def _prepare_onnx_export(
    pl_module: PLForecastingModule,
    input_sample: PLModuleInput,
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

    Returns
    -------
    OnnxExportBundle
        Wrapper, dummy inputs, names, and :class:`OnnxModelSpec`.
    """
    feature_names, feature_tensors = _present_features(input_sample)
    input_sample = _onnx_dummy_input(pl_module)

    input_chunk_length = pl_module.input_chunk_length
    output_chunk_length = pl_module.output_chunk_length or 0
    output_chunk_shift = pl_module.output_chunk_shift
    uses_past_covariates = input_sample.past_covariates is not None
    uses_future_covariates = (
        input_sample.historic_future_covariates is not None
        or input_sample.future_covariates is not None
    )
    uses_static_covariates = input_sample.static_covariates is not None
    likelihood = pl_module.likelihood
    likelihood_parameter_names = (
        likelihood.parameter_names if likelihood is not None else None
    )

    pl_module.eval()
    with torch.no_grad():
        dummy_out = pl_module(input_sample)
    state_tensors, state_spec = _flatten_module_state(dummy_out.state)
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


def _onnx_dummy_input(pl_module: PLForecastingModule) -> PLModuleInput:
    """Random example batch (size=1) used to trace the ONNX graph."""

    def _randomize(shape, time_dim: int | None = None) -> torch.Tensor | None:
        if not shape:
            return None
        if time_dim is not None:
            shape = (time_dim, shape[-1])
        return torch.rand((1,) + shape, dtype=pl_module.dtype)

    sample_shapes = pl_module.train_sample_shape
    return PLModuleInput(
        past_target=_randomize(
            sample_shapes["past_target"],
            time_dim=pl_module.input_chunk_length,
        ),
        past_covariates=_randomize(
            sample_shapes.get("past_covariates"),
            time_dim=pl_module.input_chunk_length,
        ),
        historic_future_covariates=_randomize(
            sample_shapes.get("historic_future_covariates"),
            time_dim=pl_module.input_chunk_length,
        ),
        future_covariates=_randomize(
            sample_shapes.get("future_covariates"),
            time_dim=pl_module.output_chunk_length,
        ),
        static_covariates=_randomize(
            sample_shapes.get("static_covariates"),
        ),
        # future_target is excluded: ONNX export traces the inference path only
        future_target=None,
        stage=ModuleStage.PREDICT,
    )
