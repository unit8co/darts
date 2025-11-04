"""
Dataset Utils
-------------
"""

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch

# `TorchTrainingDataset` output
# (past target, past cov, historic future cov, future cov, static cov, sample weight, future target)
TorchTrainingDatasetOutput = tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
]
# `TorchTrainingDataset` output converted to batch with `torch.Tensor`
TorchTrainingBatch = tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]
# training sample has no sample weight
# (past target, past cov, historic future cov, future cov, static cov, future target)
TorchTrainingSample = tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
]


# `TorchInferenceDataset` output
# (past target, past cov, future past cov, historic future cov, future cov, static cov, target series schema, pred time)
TorchInferenceDatasetOutput = tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict[str, Any],
    Union[pd.Timestamp, int],
]
# `TorchInferenceDataset` output converted to batch with `torch.Tensor`
TorchInferenceBatch = tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Sequence[dict[str, Any]],
    Union[Sequence[pd.Timestamp], Sequence[int]],
]

# The useful batch features are
# (past target, past cov, historic future cov, future cov, static cov)
TorchBatch = tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]

# the final module input is a tuple of three tensors where the past features concatenated
# (past features (past target + past cov + historic future cov), future cov, static cov)
PLModuleInput = tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
