"""
TimeSeries Datasets
-------------------
"""

try:
    # Base classes for training datasets:
    # Implementation (horizon-based)
    from darts.utils.data.horizon_based_dataset import HorizonBasedDataset

    # Base class and implementations for inference datasets:
    from darts.utils.data.inference_dataset import (
        DualCovariatesInferenceDataset,
        FutureCovariatesInferenceDataset,
        InferenceDataset,
        MixedCovariatesInferenceDataset,
        PastCovariatesInferenceDataset,
        SplitCovariatesInferenceDataset,
    )

    # Implementations (sequential)
    from darts.utils.data.sequential_dataset import (
        DualCovariatesSequentialDataset,
        FutureCovariatesSequentialDataset,
        MixedCovariatesSequentialDataset,
        PastCovariatesSequentialDataset,
        SplitCovariatesSequentialDataset,
    )

    # Implementations (shifted)
    from darts.utils.data.shifted_dataset import (
        DualCovariatesShiftedDataset,
        FutureCovariatesShiftedDataset,
        MixedCovariatesShiftedDataset,
        PastCovariatesShiftedDataset,
        SplitCovariatesShiftedDataset,
    )
    from darts.utils.data.training_dataset import (
        DualCovariatesTrainingDataset,
        FutureCovariatesTrainingDataset,
        MixedCovariatesTrainingDataset,
        PastCovariatesTrainingDataset,
        SplitCovariatesTrainingDataset,
        TrainingDataset,
    )
except ImportError:  # Torch is not available
    from darts.models.utils import NotImportedModule

    HorizonBasedDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    DualCovariatesInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    FutureCovariatesInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    InferenceDataset = NotImportedModule(module_name="(Py)Torch", warn=False)
    MixedCovariatesInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    PastCovariatesInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    SplitCovariatesInferenceDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    DualCovariatesSequentialDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    FutureCovariatesSequentialDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    MixedCovariatesSequentialDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    PastCovariatesSequentialDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    SplitCovariatesSequentialDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    DualCovariatesShiftedDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    FutureCovariatesShiftedDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    MixedCovariatesShiftedDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    PastCovariatesShiftedDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    SplitCovariatesShiftedDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    DualCovariatesTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    FutureCovariatesTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    MixedCovariatesTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    PastCovariatesTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    SplitCovariatesTrainingDataset = NotImportedModule(
        module_name="(Py)Torch", warn=False
    )
    TrainingDataset = NotImportedModule(module_name="(Py)Torch", warn=False)

__all__ = [
    "HorizonBasedDataset",
    "DualCovariatesInferenceDataset",
    "FutureCovariatesInferenceDataset",
    "InferenceDataset",
    "MixedCovariatesInferenceDataset",
    "PastCovariatesInferenceDataset",
    "SplitCovariatesInferenceDataset",
    "DualCovariatesSequentialDataset",
    "FutureCovariatesSequentialDataset",
    "MixedCovariatesSequentialDataset",
    "PastCovariatesSequentialDataset",
    "SplitCovariatesSequentialDataset",
    "DualCovariatesShiftedDataset",
    "FutureCovariatesShiftedDataset",
    "MixedCovariatesShiftedDataset",
    "PastCovariatesShiftedDataset",
    "SplitCovariatesShiftedDataset",
    "DualCovariatesTrainingDataset",
    "FutureCovariatesTrainingDataset",
    "MixedCovariatesTrainingDataset",
    "PastCovariatesTrainingDataset",
    "SplitCovariatesTrainingDataset",
    "TrainingDataset",
]
