"""
TimeSeries Datasets
-------------------
"""

try:
    # Base classes for training datasets:
    from .training_dataset import (TrainingDataset, PastCovariatesTrainingDataset,
                                   FutureCovariatesTrainingDataset, DualCovariatesTrainingDataset,
                                   MixedCovariatesTrainingDataset, SplitCovariatesTrainingDataset)

    # Base class and implementations for inference datasets:
    from .inference_dataset import (InferenceDataset, PastCovariatesInferenceDataset,
                                    FutureCovariatesInferenceDataset, DualCovariatesInferenceDataset,
                                    MixedCovariatesInferenceDataset, SplitCovariatesInferenceDataset)

    # Implementations (sequential)
    from .sequential_dataset import (PastCovariatesSequentialDataset, FutureCovariatesSequentialDataset,
                                     DualCovariatesSequentialDataset, MixedCovariatesSequentialDataset,
                                     SplitCovariatesSequentialDataset)

    # Implementations (shifted)
    from .shifted_dataset import (PastCovariatesShiftedDataset, FutureCovariatesShiftedDataset,
                                  DualCovariatesShiftedDataset, MixedCovariatesShiftedDataset,
                                  SplitCovariatesShiftedDataset)

    # Implementation (horizon-based)
    from .horizon_based_dataset import HorizonBasedDataset

except ImportError:
    # Torch is not available
    pass
