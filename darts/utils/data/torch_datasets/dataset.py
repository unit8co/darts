"""
Dataset Base Class
------------------
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from torch.utils.data import Dataset

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.torch_datasets.utils import (
    TorchInferenceDatasetOutput,
    TorchTrainingDatasetOutput,
)
from darts.utils.data.utils import (
    _SERIES_TYPES,
    FeatureType,
)

logger = get_logger(__name__)

_SampleIndexType = dict[FeatureType, tuple[Optional[int], Optional[int]]]


class TorchDataset(ABC, Dataset):
    def __init__(self):
        """
        Abstract class for all datasets that can be used with Darts' `TorchForecastingModel`.

        Provides an efficient method to compute the feature index range to be extracted for any sample.
        """

        self._index_memory: dict = {}

    @abstractmethod
    def __len__(self) -> int:
        """The total number of samples that can be extracted."""

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Union[TorchTrainingDatasetOutput, TorchInferenceDatasetOutput]:
        """Returns a sample drawn from this dataset."""

    def _memory_indexer(
        self,
        series_idx: int,
        series: TimeSeries,
        shift: int,
        input_chunk_length: int,
        output_chunk_length: int,
        end_of_output_idx: int,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        sample_weight: Optional[TimeSeries],
        n: Optional[int],
    ) -> _SampleIndexType:
        """Returns dict with feature names and (start, end) index ranges.

        The features are (past target, future target, past cov, future past cov, historic future cov,
        future cov, sample weight)

        """
        # store the start and end index (positions) for series and covariates
        idx_bounds = {}

        # the first time series_idx is observed
        if series_idx not in self._index_memory:
            start_of_output_idx = end_of_output_idx - output_chunk_length
            start_of_input_idx = start_of_output_idx - shift

            # either training or inference
            is_training = n is None
            n = n or output_chunk_length

            # select forecast point and target period, using the previously computed indexes
            future_start, future_end = (
                start_of_output_idx,
                start_of_output_idx + max(n, output_chunk_length),
            )

            # select input period; look at the `input_chunk_length` points after start of input
            past_start, past_end = (
                start_of_input_idx,
                start_of_input_idx + input_chunk_length,
            )

            future_past_start, future_past_end = (
                past_end,
                past_end + max(0, n - output_chunk_length),
            )

            idx_bounds[FeatureType.PAST_TARGET] = (past_start, past_end)
            idx_bounds[FeatureType.FUTURE_TARGET] = (future_start, future_end)

            series_times = series._time_index

            # get start (inclusive) and end (exclusive) indices for all external features and sample weight
            for feat, feat_type, start, end in zip(
                [
                    past_covariates,
                    past_covariates,
                    future_covariates,
                    future_covariates,
                    sample_weight,
                ],
                [
                    FeatureType.PAST_COVARIATES,
                    FeatureType.FUTURE_PAST_COVARIATES,
                    FeatureType.HISTORIC_FUTURE_COVARIATES,
                    FeatureType.FUTURE_COVARIATES,
                    FeatureType.SAMPLE_WEIGHT,
                ],
                [past_start, future_past_start, past_start, future_start, future_start],
                [past_end, future_past_end, past_end, future_end, future_end],
            ):
                if feat is None or start == end:
                    idx_bounds[feat_type] = (None, None)
                    continue

                main_feat_type = feat_type
                if feat_type == FeatureType.HISTORIC_FUTURE_COVARIATES:
                    main_feat_type = FeatureType.FUTURE_COVARIATES
                elif feat_type == FeatureType.FUTURE_PAST_COVARIATES:
                    main_feat_type = FeatureType.PAST_COVARIATES

                if feat.freq != series.freq:
                    raise_log(
                        ValueError(
                            f"The `{main_feat_type.value}` frequency `{feat.freq}` does not match the target "
                            f"`series` frequency `{series.freq}` (at series sequence idx `{series_idx}`)."
                        )
                    )

                # we need to be careful with getting ranges and indexes:
                # to get entire range, `full_range = ts[:len(ts)]`; to get last index: `last_idx = ts[len(ts) - 1]`
                # extract actual index value (respects datetime- and integer-based indexes; also from non-zero start)
                feat_times = feat._time_index

                # `starts` represents start index (e.g. `ts[start]`)
                if start < len(series):
                    start_time = series_times[start]
                else:
                    start_time = (
                        series_times[-1] + ((start + 1) - len(series)) * feat.freq
                    )

                # `end - 1` represents the end index (e.g. `ts[end - 1]`)
                if end - 1 < len(series):
                    end_time = series_times[end - 1]
                else:
                    end_time = series_times[-1] + (end - len(series)) * feat.freq

                if start_time not in feat_times or end_time not in feat_times:
                    if is_training:
                        raise_log(
                            ValueError(
                                f"Invalid `{main_feat_type.value}`; could not find values in index "
                                f"range: {start_time} - {end_time}."
                            ),
                            logger=logger,
                        )

                    # inference: more verbose error including forecast horizon;
                    # either feature starts too late or ends too early
                    if feat.start_time() > start_time:
                        raise_log(
                            ValueError(
                                f"For the given forecasting case, the provided `{main_feat_type.value}` at "
                                f"series sequence index `{series_idx}` do not extend far enough into the past. The "
                                f"`{main_feat_type.value}` must start at or before time step `{start_time}`, "
                                f"whereas now the start is at time step `{feat.start_time()}`."
                            ),
                            logger=logger,
                        )
                    else:
                        forecast_info = (
                            "n > output_chunk_length"
                            if n > output_chunk_length
                            else "n <= output_chunk_length"
                        )
                        raise_log(
                            ValueError(
                                f"For the given forecasting horizon `n={n}`, the provided `{main_feat_type.value}` at "
                                f"series sequence index `{series_idx}` do not extend far enough into the future. As "
                                f"`{forecast_info}` the `{main_feat_type.value}` must end at or after time step "
                                f"`{end_time}`, whereas now the end is at time step `{feat.end_time()}`."
                            ),
                            logger=logger,
                        )

                # extract the index position (index) from the start index
                feat_start = feat_times.get_loc(start_time)
                # extract the exclusive range end position (idx + 1) from the end index
                feat_end = feat_times.get_loc(end_time) + 1
                idx_bounds[feat_type] = (feat_start, feat_end)

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

                idx_bounds[series_type] = (
                    start + idx_shift if start is not None else start,
                    end + idx_shift if end is not None else end,
                )

        return idx_bounds
