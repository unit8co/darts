"""
Dataset Utils
-------------
"""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import torch

from darts.logging import raise_log

# `TorchTrainingDataset` output
# (past target, past cov, historic future cov, future cov, static cov, sample weight, future target)
TorchTrainingDatasetOutput = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]
# `TorchTrainingDataset` output converted to batch with `torch.Tensor`
TorchTrainingBatch = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]
# training sample has no sample weight
# (past target, past cov, historic future cov, future cov, static cov, future target)
TorchTrainingSample = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]


# `TorchInferenceDataset` output
# (past target, past cov, future past cov, historic future cov, future cov, static cov, target series schema, pred time)
TorchInferenceDatasetOutput = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
    pd.Timestamp | int,
]
# `TorchInferenceDataset` output converted to batch with `torch.Tensor`
TorchInferenceBatch = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    Sequence[dict[str, Any]],
    Sequence[pd.Timestamp] | Sequence[int],
]

# The useful batch features are
# (past target, past cov, historic future cov, future cov, static cov, future target)
TorchBatch = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

# the final module input is a tuple of three tensors where the past features concatenated
# (past features (past target + past cov + historic future cov), future cov, static cov, future target)
PLModuleInput = tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]

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
