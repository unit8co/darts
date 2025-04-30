"""
TimeSeries Datasets
-------------------
"""

try:
    # inference datasets
    from darts.utils.data.torch_datasets.inference_dataset import (
        SequentialTorchInferenceDataset,
        TorchInferenceDataset,
    )

    # training datasets
    from darts.utils.data.torch_datasets.training_dataset import (
        HorizonBasedTorchTrainingDataset,
        SequentialTorchTrainingDataset,
        ShiftedTorchTrainingDataset,
        TorchTrainingDataset,
    )
except ImportError:  # Torch is not available
    from darts.utils.utils import NotImportedModule

    TorchTrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    ShiftedTorchTrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    SequentialTorchTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    HorizonBasedTorchTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )

    TorchInferenceDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    SequentialTorchInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )

__all__ = [
    "HorizonBasedTorchTrainingDataset",
    "TorchInferenceDataset",
    "SequentialTorchTrainingDataset",
    "TorchTrainingDataset",
    "ShiftedTorchTrainingDataset",
    "SequentialTorchInferenceDataset",
]
