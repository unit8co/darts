"""
TimeSeries Dataset Base Classes
-------------------------------
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import Tensor
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
    def torch_tensors(self, idx: int):
        """
        Returns the i-th (input, output) training sample.
        Note that some datasets may return several inputs (past_{target,covariates} + future_covariates)
        """
        pass

    def to_torch_dataset(self) -> Dataset:
        return TorchTrainingDataset(self)


def _cat_with_optional(tsr1: Tensor, tsr2: Optional[Tensor]):
    if tsr2 is None:
        return tsr1
    else:
        return torch.cat([tsr1, tsr2], dim=1)


class TorchTrainingDataset(Dataset):
    def __init__(self, ds: TrainingDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.ds.torch_tensors(idx)


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

    def torch_tensors(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns the i-th (input, output) training sample.
        Here "input" is the concatenation of past_target with past_covariates and "output" is future_target.
        """
        item = self[idx]
        past_tgt, future_tgt, past_cov = item
        past_tgt, future_tgt = torch.from_numpy(past_tgt).float(), torch.from_numpy(future_tgt).float()
        past_cov = torch.from_numpy(past_cov).float() if past_cov is not None else None
        return _cat_with_optional(past_tgt, past_cov), future_tgt


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

    def torch_tensors(self, idx: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Returns the i-th (past_target, future_target (output), future_covariate) training sample.
        """
        item = self[idx]
        past_tgt, future_tgt, future_cov = item
        past_tgt, future_tgt = torch.from_numpy(past_tgt).float(), torch.from_numpy(future_tgt).float()
        future_cov = torch.from_numpy(future_cov).float() if future_cov is not None else None
        return past_tgt, future_tgt, future_cov


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

    def torch_tensors(self, idx: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Returns the i-th (past_input, future_target (output), future_covariate) training sample.
        Here "past_input" is the concatenation of past_target with some possible past_covariates
        """
        item = self[idx]
        past_tgt, future_tgt, past_cov, future_cov = item
        past_tgt, future_tgt = torch.from_numpy(past_tgt).float(), torch.from_numpy(future_tgt).float()
        past_cov = torch.from_numpy(past_cov).float() if past_cov is not None else None
        future_cov = torch.from_numpy(future_cov).float() if future_cov is not None else None
        return _cat_with_optional(past_tgt, past_cov), future_tgt, future_cov


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
