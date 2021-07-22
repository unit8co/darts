"""
TimeSeries Dataset Base Classes
-------------------------------
"""

from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset

from typing import Sequence, Tuple, Optional
from ...logging import get_logger, raise_if_not
from ...timeseries import TimeSeries

logger = get_logger(__name__)


class TrainingDataset(ABC, Sequence):
    def __init__(self):
        """
        Super-class for all training datasets in Darts. These include

        * "Past Covariates" datasets (for PastCovariatesModel): containing (past_target, future_target,
                                                                            past_covariates)
        * "Future Covariates" datasets (for FutureCovariatesModel): containing (past_target, future_target,
                                                                                future_covariates)
        * "Mixed Covariates" datasets (for MixedCovariatesModel): containing (past_target, future_target,
                                                                              past_covariates, future_covariates)

        The covariates are optional and can be `None`.

        This is meant to be used for training (or validation), all data except `future_target` represents model
        inputs and `output_target` represent model outputs.

        Darts `GlobalForecastingModel`s can be fit from instances of `TrainingDataset` using the
        `fit_from_dataset()` method.

        `TrainingDataset` inherits from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        It contains `np.ndarray` (and not `TimeSeries`), because training requires the values only,
        and so we can get big performance gains when slicing by returning only numpy views of the data.
        """
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        The data returned by this method will vary for types A, B and C models.
        """
        pass

    @abstractmethod
    def to_torch_dataset(self) -> Dataset:
        """
        Each dataset knows how to concatenate the past/future targets with past/future covariates
        into tensors for training
        """
        pass


class PastCovariatesTrainingDataset(TrainingDataset):
    def __init__(self):
        """
        Abstract class for a PastCovariatesModel training dataset. It contains 3-tuples of
        `(past_target, future_target, past_covariate)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass


class FutureCovariatesTrainingDataset(TrainingDataset):
    def __init__(self):
        """
        Abstract class for a FutureCovariatesModel training dataset. It contains 3-tuples of
        `(past_target, future_target, future_covariate)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pass


class MixedCovariatesTrainingDataset(TrainingDataset):
    def __init__(self):
        """
        Abstract class for a MixedCovariatesModel training dataset. It contains 4s-tuples of
        `(past_target, future_target, past_covariate, future_covariate)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        pass


def _get_matching_index(ts_target: TimeSeries,
                        ts_covariate: TimeSeries,
                        idx: int):
    """
    Given two overlapping series `ts_target` and `ts_covariate` and an index point of `ts_target`, returns the matching
    index point in `ts_covariate`, based on the ending times of the two series.
    The indexes are starting from the end of the series.

    This function is used to jointly slice target and covariate series in datasets. It supports both datetime and
    integer indexed series.

    Note: this function does not check if the matching index value is in `ts_covariate` or not.
    """
    raise_if_not(ts_target.freq == ts_covariate.freq,
                 'The dataset contains some target/covariates series pair that have incompatible '
                 'time axes (not the same "freq") and thus cannot be matched')

    # compute the number of steps the covariates are in advance w.r.t. the target:
    time_diff = int((ts_covariate.end_time() - ts_target.end_time()) / ts_target.freq)
    return idx + time_diff
