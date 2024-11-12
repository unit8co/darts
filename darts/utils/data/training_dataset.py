"""
Training Datasets Base Classes
------------------------------
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.utils import CovariateType

logger = get_logger(__name__)
SampleIndexType = tuple[
    int, int, int, int, Optional[int], Optional[int], Optional[int], Optional[int]
]


class TrainingDataset(ABC, Dataset):
    def __init__(self):
        """
        Super-class for all training datasets for torch models in Darts. These include

        * "PastCovariates" datasets (for PastCovariatesTorchModel): containing (past_target,
                                                                                past_covariates,
                                                                                static_covariates,
                                                                                future_target)
        * "FutureCovariates" datasets (for FutureCovariatesTorchModel): containing (past_target,
                                                                                    future_covariates,
                                                                                    static_covariates,
                                                                                    future_target)
        * "DualCovariates" datasets (for DualCovariatesTorchModel): containing (past_target,
                                                                                historic_future_covariates,
                                                                                future_covariates,
                                                                                static_covariates,
                                                                                future_target)
        * "MixedCovariates" datasets (for MixedCovariatesTorchModel): containing (past_target,
                                                                                  past_covariates,
                                                                                  historic_future_covariates,
                                                                                  future_covariates,
                                                                                  static_covariates,
                                                                                  future_target)
        * "SplitCovariates" datasets (for SplitCovariatesTorchModel): containing (past_target,
                                                                                  past_covariates,
                                                                                  future_covariates,
                                                                                  static_covariates,
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

        self._index_memory: dict = {}

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    def _memory_indexer(
        self,
        target_idx: int,
        target_series: TimeSeries,
        shift: int,
        input_chunk_length: int,
        output_chunk_length: int,
        end_of_output_idx: int,
        covariate_series: Optional[TimeSeries] = None,
        covariate_type: CovariateType = CovariateType.NONE,
        sample_weight_series: Optional[TimeSeries] = None,
    ) -> SampleIndexType:
        """Returns the (start, end) indices for past target, future target and covariates (sub sets) of the current
        sample `i` from `target_idx`.

        Works for all TimeSeries index types: pd.DatetimeIndex, pd.RangeIndex (and the deprecated Int64Index)

        When `target_idx` is observed for the first time, it stores the position of the sample `0` within the full
        target time series and the (start, end) indices of all sub sets.
        This allows to calculate the sub set indices for all future samples `i` by simply adjusting for the difference
        between the positions of sample `i` and sample `0`.

        Parameters
        ----------
        target_idx
            index of the current target TimeSeries.
        target_series
            current target TimeSeries.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future output series.
        end_of_output_idx
            the index where the output chunk of the current sample ends in `target_series`.
        covariate_series
            current covariate TimeSeries.
        covariate_type
            the type of covariate to extract. Instance of `CovariateType`: One of (`CovariateType.PAST`,
            `CovariateType.FUTURE`, `CovariateType.NONE`).
        sample_weight_series
            current sample weight TimeSeries.
        """

        covariate_start, covariate_end = None, None
        sample_weight_start, sample_weight_end = None, None

        # the first time target_idx is observed
        if target_idx not in self._index_memory:
            start_of_output_idx = end_of_output_idx - output_chunk_length
            start_of_input_idx = start_of_output_idx - shift

            # select forecast point and target period, using the previously computed indexes
            future_start, future_end = (
                start_of_output_idx,
                start_of_output_idx + output_chunk_length,
            )

            # select input period; look at the `input_chunk_length` points after start of input
            past_start, past_end = (
                start_of_input_idx,
                start_of_input_idx + input_chunk_length,
            )

            if covariate_type is not CovariateType.NONE:
                # not CovariateType.FUTURE -> both CovariateType.PAST and CovariateType.HISTORIC_FUTURE
                start = (
                    future_start
                    if covariate_type is CovariateType.FUTURE
                    else past_start
                )
                end = future_end if covariate_type is CovariateType.FUTURE else past_end

                # we need to be careful with getting ranges and indexes:
                # to get entire range, full_range = ts[:len(ts)]; to get last index: last_idx = ts[len(ts) - 1]
                # extract actual index value (respects datetime- and integer-based indexes; also from non-zero
                # start)
                target_time_index = target_series._time_index
                covariate_time_index = covariate_series._time_index
                start_time = target_time_index[start]
                end_time = target_time_index[end - 1]

                if (
                    start_time not in covariate_time_index
                    or end_time not in covariate_time_index
                ):
                    raise_log(
                        ValueError(
                            f"Missing covariates; could not find {covariate_type.value} covariates in index "
                            f"value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                # extract the index position (index) from index value
                covariate_start = covariate_time_index.get_loc(start_time)
                covariate_end = covariate_time_index.get_loc(end_time) + 1

            # sample weight
            if sample_weight_series is not None:
                # extract the index position (index) from index value
                target_time_index = target_series._time_index
                sample_weight_time_index = sample_weight_series._time_index

                start_time = target_time_index[future_start]
                end_time = target_time_index[future_end - 1]

                if (
                    start_time not in sample_weight_time_index
                    or end_time not in sample_weight_time_index
                ):
                    raise_log(
                        ValueError(
                            f"Missing sample weights; could not find sample weights in index "
                            f"value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                sample_weight_start = sample_weight_time_index.get_loc(start_time)
                sample_weight_end = sample_weight_time_index.get_loc(end_time) + 1

            # store position of initial sample and all relevant sub set indices
            self._index_memory[target_idx] = {
                "end_of_output_idx": end_of_output_idx,
                "past_target": (past_start, past_end),
                "future_target": (future_start, future_end),
                "covariate": (covariate_start, covariate_end),
                "sample_weight": (sample_weight_start, sample_weight_end),
            }
        else:
            # load position of initial sample and its sub set indices
            end_of_output_idx_last = self._index_memory[target_idx]["end_of_output_idx"]
            past_start, past_end = self._index_memory[target_idx]["past_target"]
            future_start, future_end = self._index_memory[target_idx]["future_target"]
            covariate_start, covariate_end = self._index_memory[target_idx]["covariate"]
            sample_weight_start, sample_weight_end = self._index_memory[target_idx][
                "sample_weight"
            ]

            # evaluate how much the new sample needs to be shifted, and shift all indexes
            idx_shift = end_of_output_idx - end_of_output_idx_last
            past_start += idx_shift
            past_end += idx_shift
            future_start += idx_shift
            future_end += idx_shift
            covariate_start = (
                covariate_start + idx_shift if covariate_start is not None else None
            )
            covariate_end = (
                covariate_end + idx_shift if covariate_end is not None else None
            )
            sample_weight_start = (
                sample_weight_start + idx_shift
                if sample_weight_start is not None
                else None
            )
            sample_weight_end = (
                sample_weight_end + idx_shift if sample_weight_end is not None else None
            )

        return (
            past_start,
            past_end,
            future_start,
            future_end,
            covariate_start,
            covariate_end,
            sample_weight_start,
            sample_weight_end,
        )


class PastCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a PastCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, past_covariate, static_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        pass


class FutureCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a FutureCovariatesTorchModel training dataset. It contains 3-tuples of
        `(past_target, future_covariate, static_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        pass


class DualCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a DualCovariatesTorchModel training dataset. It contains 4-tuples of
        `(past_target, historic_future_covariates, future_covariates, static_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        pass


class MixedCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a MixedCovariatesTorchModel training dataset. It contains 5-tuples of
        `(past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
        future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        pass


class SplitCovariatesTrainingDataset(TrainingDataset, ABC):
    def __init__(self):
        """
        Abstract class for a SplitCovariatesTorchModel training dataset. It contains 4-tuples of
        `(past_target, past_covariates, future_covariates, static_covariates, future_target)` `np.ndarray`.
        The covariates are optional and can be `None`.
        """
        super().__init__()

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        pass
