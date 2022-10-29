"""
TimeSeries Datasets
-------------------
"""

try:
    # Base classes for training datasets:
    # Implementation (horizon-based)
    from .horizon_based_dataset import HorizonBasedDataset

    # Base class and implementations for inference datasets:
    from .inference_dataset import (
        DualCovariatesInferenceDataset,
        FutureCovariatesInferenceDataset,
        InferenceDataset,
        MixedCovariatesInferenceDataset,
        PastCovariatesInferenceDataset,
        SplitCovariatesInferenceDataset,
    )

    # Implementations (sequential)
    from .sequential_dataset import (
        DualCovariatesSequentialDataset,
        FutureCovariatesSequentialDataset,
        MixedCovariatesSequentialDataset,
        PastCovariatesSequentialDataset,
        SplitCovariatesSequentialDataset,
    )

    # Implementations (shifted)
    from .shifted_dataset import (
        DualCovariatesShiftedDataset,
        FutureCovariatesShiftedDataset,
        MixedCovariatesShiftedDataset,
        PastCovariatesShiftedDataset,
        SplitCovariatesShiftedDataset,
    )
    from .training_dataset import (
        DualCovariatesTrainingDataset,
        FutureCovariatesTrainingDataset,
        MixedCovariatesTrainingDataset,
        PastCovariatesTrainingDataset,
        SplitCovariatesTrainingDataset,
        TrainingDataset,
    )

except ImportError:
    # Torch is not available
    pass
