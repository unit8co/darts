"""
Dataset Utils
-------------
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, TypeAlias

import numpy as np
import pandas as pd
import torch

from darts.logging import raise_log

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class TorchTrainingSample(NamedTuple):
    """Model-init snapshot of feature arrays (no sample weight, no inference metadata).

    Built from a training or inference dataset sample via ``to_training_sample()``.
    Stored on the forecasting model for dimension inference and covariate-use flags.
    """

    past_target: np.ndarray | None
    past_covariates: np.ndarray | None
    historic_future_covariates: np.ndarray | None
    future_covariates: np.ndarray | None
    static_covariates: np.ndarray | None
    future_target: np.ndarray | None

    def component_dims(self) -> list[int | None]:
        """Per-slot component dimension, or ``None`` if the slot is unused."""
        return [s.shape[1] if s is not None else None for s in self]


class TorchTrainingDatasetOutput(NamedTuple):
    """Single training sample from a :class:`~darts.utils.data.TorchTrainingDataset`.

    Field order is the public dataset contract. Custom datasets may still return a
    plain 7-tuple with the same layout.
    """

    past_target: np.ndarray | None
    past_covariates: np.ndarray | None
    historic_future_covariates: np.ndarray | None
    future_covariates: np.ndarray | None
    static_covariates: np.ndarray | None
    sample_weight: np.ndarray | None
    future_target: np.ndarray

    def to_training_sample(self) -> "TorchTrainingSample":
        """Model-init snapshot: same feature slots, without sample weight."""
        return TorchTrainingSample(
            past_target=self.past_target,
            past_covariates=self.past_covariates,
            historic_future_covariates=self.historic_future_covariates,
            future_covariates=self.future_covariates,
            static_covariates=self.static_covariates,
            future_target=self.future_target,
        )


class TorchInferenceDatasetOutput(NamedTuple):
    """Single inference sample from a :class:`~darts.utils.data.TorchInferenceDataset`.

    Field order is the public dataset contract. Custom datasets may still return a
    plain 8-tuple with the same layout.
    """

    past_target: np.ndarray | None
    past_covariates: np.ndarray | None
    future_past_covariates: np.ndarray | None
    historic_future_covariates: np.ndarray | None
    future_covariates: np.ndarray | None
    static_covariates: np.ndarray | None
    series_schema: dict[str, Any]
    pred_time: pd.Timestamp | int

    def to_training_sample(self) -> TorchTrainingSample:
        """Model-relevant feature slots; ``future_target`` is ``None`` at inference."""
        return TorchTrainingSample(
            past_target=self.past_target,
            past_covariates=self.past_covariates,
            historic_future_covariates=self.historic_future_covariates,
            future_covariates=self.future_covariates,
            static_covariates=self.static_covariates,
            future_target=None,
        )


@dataclass(slots=True)
class _ReplacableBatch:
    """Mixin providing a cheap field-update that reuses unchanged tensor references."""

    def replace(self, **changes) -> Self:
        return replace(self, **changes)


@dataclass(slots=True)
class PLModuleInput(_ReplacableBatch):
    """Model-facing batch of independent tensors.

    Concatenation is optional and model-specific; use the helpers below when needed.
    """

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None = None
    historic_future_covariates: torch.Tensor | None = None
    future_covariates: torch.Tensor | None = None
    static_covariates: torch.Tensor | None = None
    future_target: torch.Tensor | None = None

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
class TorchTrainingBatch(_ReplacableBatch):
    """Collated training batch, including sample weight for the loss."""

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None
    historic_future_covariates: torch.Tensor | None
    future_covariates: torch.Tensor | None
    static_covariates: torch.Tensor | None
    sample_weight: torch.Tensor | None
    future_target: torch.Tensor

    def to_module_input(self, *, include_future_target: bool = True) -> PLModuleInput:
        """Share tensor references into a model-facing batch (no copies)."""
        return PLModuleInput(
            past_target=self.past_target,
            past_covariates=self.past_covariates,
            historic_future_covariates=self.historic_future_covariates,
            future_covariates=self.future_covariates,
            static_covariates=self.static_covariates,
            future_target=self.future_target if include_future_target else None,
        )


@dataclass(slots=True)
class TorchInferenceBatch(_ReplacableBatch):
    """Collated inference batch, including series schema and prediction start times."""

    past_target: torch.Tensor
    past_covariates: torch.Tensor | None
    future_past_covariates: torch.Tensor | None
    historic_future_covariates: torch.Tensor | None
    future_covariates: torch.Tensor | None
    static_covariates: torch.Tensor | None
    series_schema: Sequence[dict[str, Any]]
    pred_time: Sequence[pd.Timestamp] | Sequence[int]

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


def _to_training_output(sample: Sequence[Any]) -> TorchTrainingDatasetOutput:
    """Coerce a custom-dataset tuple into :class:`TorchTrainingDatasetOutput`."""
    if isinstance(sample, TorchTrainingDatasetOutput):
        return sample
    try:
        return TorchTrainingDatasetOutput(*sample)
    except TypeError:
        raise_log(
            ValueError(
                f"Training dataset `__getitem__` must return a 7-element sample "
                f"or `TorchTrainingDatasetOutput` (see `TorchTrainingDataset`); "
                f"received {len(sample)} elements."
            ),
        )


def _to_inference_output(sample: Sequence[Any]) -> TorchInferenceDatasetOutput:
    """Coerce a custom-dataset tuple into :class:`TorchInferenceDatasetOutput`."""
    if isinstance(sample, TorchInferenceDatasetOutput):
        return sample
    try:
        return TorchInferenceDatasetOutput(*sample)
    except TypeError:
        raise_log(
            ValueError(
                f"Inference dataset `__getitem__` must return an 8-element sample "
                f"or `TorchInferenceDatasetOutput` (see `TorchInferenceDataset`); "
                f"received {len(sample)} elements."
            ),
        )


def _to_training_sample(sample: Sequence[Any]) -> TorchTrainingSample:
    """Coerce a dataset sample or stored snapshot into :class:`TorchTrainingSample`.

    Accepts a ``TorchTrainingSample``, a 6-tuple with the same layout, a training
    dataset output (7-tuple / NamedTuple), or an inference dataset output
    (8-tuple / NamedTuple). Other lengths raise ``TypeError``.
    """
    if isinstance(sample, TorchTrainingSample):
        return sample

    try:
        return TorchTrainingSample(*sample)
    except TypeError:
        raise_log(
            ValueError(
                f"Expected a 6-element training sample or a `TorchTrainingSample`; "
                f"received {len(sample)} elements."
            )
        )


def _stack_batch_samples(
    batch: list[TorchTrainingDatasetOutput | TorchInferenceDatasetOutput],
) -> list[torch.Tensor | None | dict[str, Any] | pd.Timestamp | int]:
    """Stack dataset samples into tensor (or other type) batch."""
    aggregated = []
    first_sample = batch[0]
    for i in range(len(first_sample)):
        elem = first_sample[i]
        if isinstance(elem, np.ndarray):
            aggregated.append(
                torch.from_numpy(np.stack([sample[i] for sample in batch], axis=0))
            )
        elif elem is None:
            aggregated.append(None)
        else:
            aggregated.append([sample[i] for sample in batch])
    return aggregated


def _batch_collate_fn_train(
    batch: list[TorchTrainingDatasetOutput],
) -> TorchTrainingBatch:
    """Stack dataset samples into a named tensor batch for training.

    Positional mapping from the documented dataset field order to the dataclass batches
    consumed by the modules. Custom datasets may return plain tuples with the same layout.
    """
    return TorchTrainingBatch(*_stack_batch_samples(batch))


def _batch_collate_fn_predict(
    batch: list[TorchInferenceDatasetOutput],
) -> TorchInferenceBatch:
    """Stack dataset samples into a named tensor batch for inference.

    Positional mapping from the documented dataset field order to the dataclass batches
    consumed by the modules. Custom datasets may return plain tuples with the same layout.
    """
    return TorchInferenceBatch(*_stack_batch_samples(batch))


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
