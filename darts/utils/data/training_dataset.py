"""
Training Datasets Base Classes
------------------------------
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import numpy as np

from typing import Tuple, Optional
from ...logging import get_logger

logger = get_logger(__name__)


class TrainingDataset(ABC, Dataset):
    def __init__(self):
        """
        Super-class for all training datasets for torch models in Darts. These include

        * "PastCovariates" datasets (for PastCovariatesTorchModel): containing (past_target,
                                                                                past_covariates,
                                                                                future_target)
        * "FutureCovariates" datasets (for FutureCovariatesTorchModel): containing (past_target,
                                                                                    future_covariates,
                                                                                    future_target)
        * "DualCovariates" datasets (for DualCovariatesTorchModel): containing (past_target,
                                                                                historic_future_covariates,
                                                                                future_covariates,
                                                                                future_target)
        * "MixedCovariates" datasets (for MixedCovariatesTorchModel): containing (past_target,
                                                                                  past_covariates,
                                                                                  historic_future_covariates,
                                                                                  future_covariates,
                                                                                  future_target)
        * "SplitCovariates" datasets (for SplitCovariatesTorchModel): containing (past_target,
                                                                                  past_covariates,
                                                                                  future_covariates,
                                                                                  future_target)

        The covariates are optional and can be `None`.

        This is meant to be used for training (or validation), all data except `future_target` represents model
        inputs (`future_target` is the output the model are trained to predict).

        Darts `TorchForecastingModel`s can be fit from instances of `TrainingDataset` of the right type using the
        `fit_from_dataset()` method.

        `TrainingDataset` inherits torch `Dataset`; meaning that the implementations have to
        provide the `__getitem__()` method.

        It contains `np.ndarray` (and not `TimeSeries`), because training requires the values only,
        and so we can get big performance gains when slicing by returning only numpy views of the data
        underlying the `TimeSeries`.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class PastCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a PastCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, past_covariate, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        pass


class FutureCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a FutureCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, future_covariate, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        pass


class DualCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a DualCovariatesTorchModel training dataset. It contains 4-tuples of
        `(past_target, historic_future_covariates, future_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        pass


class MixedCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a MixedCovariatesTorchModel training dataset. It contains 5-tuples of
        `(past_target, past_covariates, historic_future_covariates, future_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                             Optional[np.ndarray], np.ndarray]:
        pass


class SplitCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a SplitCovariatesTorchModel training dataset. It contains 4-tuples of
        `(past_target, past_covariates, future_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        pass
