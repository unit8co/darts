"""
Dataset Utils
-------------
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

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
# (past target, past cov, historic future cov, future cov, static cov)
TorchBatch = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

# the final module input is a tuple of three tensors where the past features concatenated
# (past features (past target + past cov + historic future cov), future cov, static cov)
PLModuleInput = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
