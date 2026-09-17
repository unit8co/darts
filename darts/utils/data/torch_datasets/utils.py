"""
Dataset Utils
-------------
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Self, TypeAlias, overload

import numpy as np
import pandas as pd
import torch
from torch.utils._pytree import register_pytree_node

from darts.logging import raise_log

# Feature slots used for model-init snapshots, dim inference, and checkpoint shapes.
# Order matches the historical 6-element ``train_sample_shape`` list.
FEATURE_FIELDS: tuple[str, ...] = (
    "past_target",
    "past_covariates",
    "historic_future_covariates",
    "future_covariates",
    "static_covariates",
    "future_target",
)


class ModuleStage(str, Enum):
    """Loop role for a :class:`PLModuleInput`.

    Set at the batch-to-module boundary. ``forward()`` should branch on this
    instead of ``self.trainer``. Keep ``nn.Module.training`` for dropout /
    BatchNorm.

    - ``TRAIN``: ``training_step``
    - ``VALIDATE``: ``validation_step`` and Lightning sanity checking
    - ``PREDICT``: ``predict_step``, ONNX export, and raw ``forward()`` calls

    Defaults to ``PREDICT`` so exported and standalone calls take the inference
    path.
    """

    TRAIN = "train"
    VALIDATE = "val"
    PREDICT = "pred"


@dataclass(slots=True)
class _Replacable:
    """Mixin providing a cheap field-update that reuses unchanged tensor references."""

    def replace(self, **changes) -> Self:
        return replace(self, **changes)


@dataclass(slots=True)
class TorchSample(_Replacable):
    def feature_arrays(self) -> dict[str, np.ndarray | None]:
        """Feature slots used for model init (excludes ``sample_weight``)."""
        return {name: getattr(self, name) for name in FEATURE_FIELDS}

    def sample_shapes(self):
        return {
            name: arr.shape if arr is not None else None
            for name, arr in self.feature_arrays().items()
        }

    def component_dims(self) -> dict[str, int | None]:
        """Per-feature component dimension, or ``None`` if the slot is unused."""
        return {
            name: arr.shape[1] if arr is not None else None
            for name, arr in self.feature_arrays().items()
        }

    def iter_arrays(self):
        """Yield non-``None`` numpy arrays (for dtype checks)."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                yield val


@dataclass(slots=True)
class TorchTrainingSample(TorchSample):
    """Single training sample from a :class:`~darts.utils.data.TorchTrainingDataset`.

    Also stored on the forecasting model for dimension inference and covariate-use flags.
    Construct by field name and omit unused optional slots.
    """

    past_target: np.ndarray | None = None
    past_covariates: np.ndarray | None = None
    historic_future_covariates: np.ndarray | None = None
    future_covariates: np.ndarray | None = None
    static_covariates: np.ndarray | None = None
    future_target: np.ndarray | None = None
    sample_weight: np.ndarray | None = None


@dataclass(slots=True)
class TorchInferenceSample(TorchSample):
    """Single inference sample from a :class:`~darts.utils.data.TorchInferenceDataset`.

    Construct by field name and omit unused optional slots.
    """

    past_target: np.ndarray | None = None
    past_covariates: np.ndarray | None = None
    future_past_covariates: np.ndarray | None = None
    historic_future_covariates: np.ndarray | None = None
    future_covariates: np.ndarray | None = None
    static_covariates: np.ndarray | None = None
    series_schema: dict[str, Any] | None = None
    pred_time: pd.Timestamp | int | None = None


@dataclass(slots=True)
class PLModuleInput(_Replacable):
    """Model-facing batch of independent tensors.

    Concatenation is optional and model-specific; use the helpers below when needed.
    ``state`` carries recurrent / cached values from a previous ``forward`` (hidden
    state, KV cache, ...). ``stage`` is the Lightning loop role (train / validate /
    predict); it is a Python value, not a graph tensor.
    """

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None = None
    historic_future_covariates: torch.Tensor | None = None
    future_covariates: torch.Tensor | None = None
    static_covariates: torch.Tensor | None = None
    future_target: torch.Tensor | None = None
    state: Any = None
    stage: ModuleStage = ModuleStage.PREDICT

    def concatenate_past_features(self) -> torch.Tensor:
        """Concatenate past-window tensors along the component dimension."""
        return torch.cat(
            [
                tensor
                for tensor in (
                    self.past_target,
                    self.past_covariates,
                    self.historic_future_covariates,
                )
                if tensor is not None
            ],
            dim=2,
        )

    def concatenate_future_along_time(self) -> torch.Tensor:
        """Concatenate historic and future covariates along the time dimension."""
        return torch.cat(
            [
                tensor
                for tensor in (self.historic_future_covariates, self.future_covariates)
                if tensor is not None
            ],
            dim=1,
        )


@dataclass(slots=True)
class PLModuleOutput(_Replacable):
    """Model-facing ``forward`` output.

    ``prediction`` is the forecast / likelihood-parameter tensor of shape
    ``(batch, time, components, nr_params)``. ``state`` is carried to the next
    ``forward`` as ``PLModuleInput.state``.
    """

    prediction: torch.Tensor
    state: Any = None


@dataclass(slots=True)
class TorchTrainingBatch(_Replacable):
    """Collated training batch, including sample weight for the loss."""

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None = None
    historic_future_covariates: torch.Tensor | None = None
    future_covariates: torch.Tensor | None = None
    static_covariates: torch.Tensor | None = None
    sample_weight: torch.Tensor | None = None
    future_target: torch.Tensor | None = None

    def to_module_input(self, stage: ModuleStage = ModuleStage.TRAIN) -> PLModuleInput:
        """Share tensor references into a model-facing batch (no copies)."""
        return PLModuleInput(
            past_target=self.past_target,
            past_covariates=self.past_covariates,
            historic_future_covariates=self.historic_future_covariates,
            future_covariates=self.future_covariates,
            static_covariates=self.static_covariates,
            future_target=self.future_target,
            stage=stage,
        )


@dataclass(slots=True)
class TorchInferenceBatch(_Replacable):
    """Collated inference batch, including series schema and prediction start times."""

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None = None
    future_past_covariates: torch.Tensor | None = None
    historic_future_covariates: torch.Tensor | None = None
    future_covariates: torch.Tensor | None = None
    static_covariates: torch.Tensor | None = None
    series_schema: Sequence[dict[str, Any]] | None = None
    pred_time: Sequence[pd.Timestamp] | Sequence[int] | None = None

    def tile_tensors(self, batch_sample_size: int) -> Self:
        """Tile tensor fields for multi-sample (probabilistic) prediction."""

        def _tile(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return (
                tensor.tile((batch_sample_size, 1, 1)) if tensor is not None else None
            )

        return self.replace(
            past_target=_tile(self.past_target),
            past_covariates=_tile(self.past_covariates),
            future_past_covariates=_tile(self.future_past_covariates),
            historic_future_covariates=_tile(self.historic_future_covariates),
            future_covariates=_tile(self.future_covariates),
            static_covariates=_tile(self.static_covariates),
        )


def _flatten_dataclass(obj):
    flds = fields(obj)
    return [getattr(obj, f.name) for f in flds], [f.name for f in flds]


# ``stage`` is a Python loop flag, not a tracing leaf; keep it in pytree context.
_PL_MODULE_INPUT_TENSOR_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(PLModuleInput) if f.name != "stage"
)


def _flatten_pl_module_input(obj: PLModuleInput):
    return (
        [getattr(obj, name) for name in _PL_MODULE_INPUT_TENSOR_FIELDS],
        obj.stage,
    )


def _unflatten_pl_module_input(values, context):
    return PLModuleInput(
        **dict(zip(_PL_MODULE_INPUT_TENSOR_FIELDS, values)),
        stage=context,
    )


def _unflatten_pl_module_output(values, context):
    return PLModuleOutput(**dict(zip(context, values)))


register_pytree_node(
    PLModuleInput,
    flatten_fn=_flatten_pl_module_input,
    unflatten_fn=_unflatten_pl_module_input,
)
register_pytree_node(
    PLModuleOutput,
    flatten_fn=_flatten_dataclass,
    unflatten_fn=_unflatten_pl_module_output,
)


def _coerce_training_sample(sample: TorchTrainingSample | tuple) -> TorchTrainingSample:
    """Normalize a stored ``train_sample`` after loading a saved model."""
    if isinstance(sample, TorchTrainingSample):
        return sample
    if isinstance(sample, tuple):
        # legacy tuple, try to convert it to a TorchTrainingSample
        if len(sample) != 6:
            raise_log(
                ValueError(
                    "Legacy `train_sample` tuple must have 6 elements; "
                    f"got {len(sample)}."
                ),
            )
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = sample
        return TorchTrainingSample(
            past_target=past_target,
            past_covariates=past_covariates,
            historic_future_covariates=historic_future_covariates,
            future_covariates=future_covariates,
            static_covariates=static_covariates,
            future_target=future_target,
        )
    raise_log(
        ValueError(
            f"Unsupported `train_sample` type {type(sample).__name__}; "
            "expected `TorchTrainingSample` or legacy tuple."
        ),
    )


def _as_training_sample(sample: Any) -> TorchTrainingSample:
    """Validate that a dataset sample is a :class:`TorchTrainingSample`."""
    if isinstance(sample, TorchTrainingSample):
        return sample
    raise_log(
        ValueError(
            "Training dataset `__getitem__` must return a `TorchTrainingSample` "
            f"(see `TorchTrainingDataset`); received {type(sample).__name__}."
        ),
    )


def _as_inference_sample(sample: Any) -> TorchInferenceSample:
    """Validate that a dataset sample is a :class:`TorchInferenceSample`."""
    if isinstance(sample, TorchInferenceSample):
        return sample
    raise_log(
        ValueError(
            "Inference dataset `__getitem__` must return a `TorchInferenceSample` "
            f"(see `TorchInferenceDataset`); received {type(sample).__name__}."
        ),
    )


def _stack_field(samples: Sequence[Any], name: str):
    """Stack one named field across samples into a batched tensor or list."""
    first = getattr(samples[0], name)
    if isinstance(first, np.ndarray):
        return torch.from_numpy(np.stack([getattr(s, name) for s in samples], axis=0))
    if first is None:
        return None
    return [getattr(s, name) for s in samples]


@overload
def _collate_samples(
    batch: Sequence[Any],
    batch_cls: type[TorchTrainingBatch],
) -> TorchTrainingBatch: ...


@overload
def _collate_samples(
    batch: Sequence[Any],
    batch_cls: type[TorchInferenceBatch],
) -> TorchInferenceBatch: ...


def _collate_samples(
    batch: Sequence[Any],
    batch_cls: type[TorchTrainingBatch] | type[TorchInferenceBatch],
):
    """Name-based collate from sample dataclasses into a batch dataclass."""
    return batch_cls(**{f.name: _stack_field(batch, f.name) for f in fields(batch_cls)})


def _batch_collate_fn_train(batch: list[TorchTrainingSample]) -> TorchTrainingBatch:
    """Stack training samples into a named tensor batch."""
    return _collate_samples(batch, TorchTrainingBatch)


def _batch_collate_fn_predict(batch: list[TorchInferenceSample]) -> TorchInferenceBatch:
    """Stack inference samples into a named tensor batch."""
    return _collate_samples(batch, TorchInferenceBatch)


def _normalize_train_sample_shape(
    train_sample_shape: dict[str, tuple | None] | Sequence[tuple | None] | None,
) -> dict[str, tuple | None] | None:
    """Coerce ``train_sample_shape`` to a field-name dict.

    Accepts the current dict format or the historical 6-element list/tuple of shapes.
    """
    if train_sample_shape is None:
        return None
    if isinstance(train_sample_shape, dict):
        return train_sample_shape
    return {
        name: train_sample_shape[i] if i < len(train_sample_shape) else None
        for i, name in enumerate(FEATURE_FIELDS)
    }


def _train_sample_from_shapes(
    train_sample_shape: dict[str, tuple | None] | Sequence[tuple | None],
    dtype: np.dtype,
) -> TorchTrainingSample:
    """Build a mock :class:`TorchTrainingSample` from stored shapes (checkpoint load)."""
    shape_by_name = _normalize_train_sample_shape(train_sample_shape)
    if shape_by_name is None:
        raise_log(ValueError("`train_sample_shape` must not be `None`."))
    return TorchTrainingSample(**{
        name: np.zeros(shape, dtype=dtype) if shape else None
        for name, shape in shape_by_name.items()
    })


# variable input chunk length
InputChunkLength: TypeAlias = int | tuple[int, int]


def _parse_input_chunk_length(input_chunk_length: InputChunkLength) -> tuple[int, int]:
    """Parse ``input_chunk_length`` into ``(min, max)`` bounds.

    An ``int`` denotes a fixed input window. A ``(min, max)`` tuple enables
    variable-length inputs with left-padding up to ``max``.
    """
    if not isinstance(input_chunk_length, int) and (
        not isinstance(input_chunk_length, tuple) or len(input_chunk_length) != 2
    ):
        raise_log(
            ValueError(
                "`input_chunk_length` must be an integer or a `(min_length, max_length)` tuple of integers."
            ),
        )
    if isinstance(input_chunk_length, int):
        if input_chunk_length < 1:
            raise_log(
                ValueError("`input_chunk_length` must be >= 1."),
            )
        return input_chunk_length, input_chunk_length

    min_icl, max_icl = int(input_chunk_length[0]), int(input_chunk_length[1])
    if not 1 <= min_icl <= max_icl:
        raise_log(
            ValueError(
                "`input_chunk_length` tuple `(min_length, max_length)` must satisfy `1 <= min_length <= max_length`."
            ),
        )
    return min_icl, max_icl
