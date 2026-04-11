"""
TimeSeries Datasets
-------------------

Datasets and utilities for preparing time series data for training and inference with Darts models,
including torch-based datasets and tabularization methods for SKLearn-like models.
"""

import importlib

from darts.utils.utils import NotImportedModule

_TORCH_DATASET_NAMES = {
    "SequentialTorchInferenceDataset": "darts.utils.data.torch_datasets.inference_dataset",
    "TorchInferenceDataset": "darts.utils.data.torch_datasets.inference_dataset",
    "HorizonBasedTorchTrainingDataset": "darts.utils.data.torch_datasets.training_dataset",
    "SequentialTorchTrainingDataset": "darts.utils.data.torch_datasets.training_dataset",
    "ShiftedTorchTrainingDataset": "darts.utils.data.torch_datasets.training_dataset",
    "TorchTrainingDataset": "darts.utils.data.torch_datasets.training_dataset",
}

__all__ = list(_TORCH_DATASET_NAMES.keys())


def __getattr__(name: str):
    if name in _TORCH_DATASET_NAMES:
        try:
            mod = importlib.import_module(_TORCH_DATASET_NAMES[name])
            return getattr(mod, name)
        except ImportError:
            return NotImportedModule(module_name="(Py)Torch", warn=False)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
