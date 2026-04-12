"""
TimeSeries Datasets
-------------------

Datasets and utilities for preparing time series data for training and inference with Darts models,
including torch-based datasets and tabularization methods for SKLearn-like models.
"""

from darts.utils._lazy import setup_lazy_imports

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
