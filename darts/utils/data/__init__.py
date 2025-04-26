"""
TimeSeries Datasets
-------------------
"""

try:
    # inference datasets
    from darts.utils.data.inference_dataset import (
        InferenceDataset,
        SequentialInferenceDataset,
    )

    # training datasets
    from darts.utils.data.training_dataset import (
        HorizonBasedTrainingDataset,
        SequentialTrainingDataset,
        ShiftedTrainingDataset,
        TrainingDataset,
    )
    from darts.utils.data.utils import ModuleInput, TrainingSample
except ImportError:  # Torch is not available
    from darts.utils.utils import NotImportedModule

    TrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    ShiftedTrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    SequentialTrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    HorizonBasedTrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)

    InferenceDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    SequentialInferenceDataset = NotImportedModule(module_name="(Py)Torch", warn=False)

    TrainingSample = NotImportedModule(module_name="(Py)Torch", warn=False)
    ModuleInput = NotImportedModule(module_name="(Py)Torch", warn=False)


__all__ = [
    "HorizonBasedTrainingDataset",
    "InferenceDataset",
    "SequentialTrainingDataset",
    "TrainingDataset",
    "ShiftedTrainingDataset",
    "SequentialInferenceDataset",
    "TrainingSample",
    "ModuleInput",
]
