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
from darts.utils.data.utils import FeatureType

logger = get_logger(__name__)

_SampleIndexType = dict[FeatureType, tuple[Optional[int], Optional[int]]]

DatasetOutputType = tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
]

_SERIES_TYPES = [
    FeatureType.PAST_TARGET,
    FeatureType.FUTURE_TARGET,
    FeatureType.PAST_COVARIATES,
    FeatureType.HISTORIC_FUTURE_COVARIATES,
    FeatureType.FUTURE_COVARIATES,
    FeatureType.SAMPLE_WEIGHT,
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
    def __getitem__(self, idx: int) -> DatasetOutputType:
        pass

    def _memory_indexer(
        self,
        series_idx: int,
        series: TimeSeries,
        shift: int,
        input_chunk_length: int,
        output_chunk_length: int,
        end_of_output_idx: int,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        sample_weight: Optional[TimeSeries] = None,
    ) -> _SampleIndexType:
        """Returns the (start, end) indices for past target, future target and covariates (sub sets) of the current
        sample `i` from `series_idx`.

        Works for all TimeSeries index types: pd.DatetimeIndex, pd.RangeIndex (and the deprecated Int64Index)

        When `series_idx` is observed for the first time, it stores the position of the sample `0` within the full
        target time series and the (start, end) indices of all sub sets.
        This allows to calculate the sub set indices for all future samples `i` by simply adjusting for the difference
        between the positions of sample `i` and sample `0`.

        Parameters
        ----------
        series_idx
            index of the current target TimeSeries.
        series
            current target TimeSeries.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future output series.
        end_of_output_idx
            the index where the output chunk of the current sample ends in `series`.
        past_covariates
            current `past_covariates` TimeSeries.
        future_covariates
            current `future_covariates` TimeSeries.
        sample_weight
            current sample weight TimeSeries.
        """
        # store the start and end index (positions) for series and covariates
        idx_bounds = {}

        # the first time series_idx is observed
        if series_idx not in self._index_memory:
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
            idx_bounds[FeatureType.PAST_TARGET] = (past_start, past_end)
            idx_bounds[FeatureType.FUTURE_TARGET] = (future_start, future_end)

            for cov, cov_type, start, end in zip(
                [past_covariates, future_covariates, future_covariates],
                [
                    FeatureType.PAST_COVARIATES,
                    FeatureType.HISTORIC_FUTURE_COVARIATES,
                    FeatureType.FUTURE_COVARIATES,
                ],
                [past_start, past_start, future_start],
                [past_end, past_end, future_end],
            ):
                if cov is None:
                    idx_bounds[cov_type] = (None, None)
                    continue

                # we need to be careful with getting ranges and indexes:
                # to get entire range, full_range = ts[:len(ts)]; to get last index: last_idx = ts[len(ts) - 1]
                # extract actual index value (respects datetime- and integer-based indexes; also from non-zero
                # start)
                series_times = series._time_index
                cov_times = cov._time_index
                start_time = series_times[start]
                end_time = series_times[end - 1]

                if start_time not in cov_times or end_time not in cov_times:
                    raise_log(
                        ValueError(
                            f"Missing covariates; could not find `{cov_type.value}` in index "
                            f"value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                # extract the index position (index) from index value
                cov_start = cov_times.get_loc(start_time)
                cov_end = cov_times.get_loc(end_time) + 1
                idx_bounds[cov_type] = (cov_start, cov_end)

            # sample weight
            if sample_weight is not None:
                # extract the index position (index) from index value
                series_time_index = series._time_index
                sample_weight_time_index = sample_weight._time_index

                start_time = series_time_index[future_start]
                end_time = series_time_index[future_end - 1]

                if (
                    start_time not in sample_weight_time_index
                    or end_time not in sample_weight_time_index
                ):
                    raise_log(
                        ValueError(
                            f"Invalid `{FeatureType.SAMPLE_WEIGHT.value}`; could not find "
                            f"sample weights in index value range: {start_time} - {end_time}."
                        ),
                        logger=logger,
                    )

                sample_weight_start = sample_weight_time_index.get_loc(start_time)
                sample_weight_end = sample_weight_time_index.get_loc(end_time) + 1
                idx_bounds[FeatureType.SAMPLE_WEIGHT] = (
                    sample_weight_start,
                    sample_weight_end,
                )
            else:
                idx_bounds[FeatureType.SAMPLE_WEIGHT] = (None, None)

            # store position of initial sample and all relevant sub set indices
            self._index_memory[series_idx] = {
                "end_of_output_idx": end_of_output_idx,
                **idx_bounds,
            }
        else:
            # load position of initial sample and its sub set indices
            end_of_output_idx_last = self._index_memory[series_idx]["end_of_output_idx"]
            # evaluate how much the new sample needs to be shifted, and shift all indexes
            idx_shift = end_of_output_idx - end_of_output_idx_last

            for series_type in _SERIES_TYPES:
                start, end = self._index_memory[series_idx][series_type]
                if start is not None:
                    start += idx_shift
                if end is not None:
                    end += idx_shift

                idx_bounds[series_type] = (start, end)

        return idx_bounds
