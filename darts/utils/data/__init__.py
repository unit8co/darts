"""
TimeSeries Datasets
-------------------

Datasets and utilities for preparing time series data for training and inference with Darts models,
including torch-based datasets and tabularization methods for SKLearn-like models.
"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.utils.data.torch_datasets.inference_dataset import (
        SequentialTorchInferenceDataset as SequentialTorchInferenceDataset,
    )
    from darts.utils.data.torch_datasets.inference_dataset import (
        TorchInferenceDataset as TorchInferenceDataset,
    )
    from darts.utils.data.torch_datasets.training_dataset import (
        HorizonBasedTorchTrainingDataset as HorizonBasedTorchTrainingDataset,
    )
    from darts.utils.data.torch_datasets.training_dataset import (
        SequentialTorchTrainingDataset as SequentialTorchTrainingDataset,
    )
    from darts.utils.data.torch_datasets.training_dataset import (
        ShiftedTorchTrainingDataset as ShiftedTorchTrainingDataset,
    )
    from darts.utils.data.torch_datasets.training_dataset import (
        TorchTrainingDataset as TorchTrainingDataset,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "SequentialTorchInferenceDataset": (
        "darts.utils.data.torch_datasets.inference_dataset",
        "(Py)Torch",
    ),
    "TorchInferenceDataset": (
        "darts.utils.data.torch_datasets.inference_dataset",
        "(Py)Torch",
    ),
    "HorizonBasedTorchTrainingDataset": (
        "darts.utils.data.torch_datasets.training_dataset",
        "(Py)Torch",
    ),
    "SequentialTorchTrainingDataset": (
        "darts.utils.data.torch_datasets.training_dataset",
        "(Py)Torch",
    ),
    "ShiftedTorchTrainingDataset": (
        "darts.utils.data.torch_datasets.training_dataset",
        "(Py)Torch",
    ),
    "TorchTrainingDataset": (
        "darts.utils.data.torch_datasets.training_dataset",
        "(Py)Torch",
    ),
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
