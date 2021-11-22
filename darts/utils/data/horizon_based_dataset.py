"""
Horizon-Based Training Dataset
------------------------------
"""

import os
from typing import Union, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from ...logging import raise_if_not, get_logger
from ...timeseries import TimeSeries
from .training_dataset import PastCovariatesTrainingDataset
from .utils import _get_matching_index

logger = get_logger(__name__)


SampleIndexType = Tuple[int, int, int, int, int, int]


class HorizonBasedDataset(PastCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 output_chunk_length: int = 12,
                 lh: Tuple[int, int] = (1, 3),
                 lookback: int = 3) -> None:
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_target) arrays,
        in a way inspired by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        The "past" series have length `lookback * output_chunk_length`, and the "future" series has length
        `output_chunk_length`.

        Given the horizon `output_chunk_length` of a model, this dataset will compute some "past/future"
        splits as follows:
        First a "forecast point" is selected in the the range of the last
        `(min_lh * output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.
        The "future" then consists in the following `output_chunk_length` points, and the "past" will be the preceding
        `lookback * output_chunk_length` points.

        All the series in the provided sequence must be long enough; i.e. have length at least
        `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be at least 1
        (to have targets of length exactly `1 * output_chunk_length`).
        The target and covariates time series are sliced together using their time indexes for alignment.

        The sampling is uniform both over the number of time series and the number of samples per series;
        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N
        time series in the sequence.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        output_chunk_length
            The length of the "output" series emitted by the model
        lh
            A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.
            For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points
            before the end of the series. It is required that `min_lh >= 1`.
        lookback:
            A integer interval for the length of the input in the emitted input and output splits, expressed as a
            multiple of `output_chunk_length`. For instance, `lookback=3` will emit "inputs" of lengths `3 * output_chunk_length`.
        """
        super().__init__()

        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        self.output_chunk_length = output_chunk_length
        self.min_lh, self.max_lh = lh
        self.lookback = lookback

        # Checks
        raise_if_not(self.max_lh >= self.min_lh >= 1,
                     'The lh parameter should be an int tuple (min_lh, max_lh), '
                     'with 1 <= min_lh <= max_lh')
        raise_if_not(covariates is None or len(self.target_series) == len(self.covariates),
                     'The provided sequence of target series must have the same length as '
                     'the provided sequence of covariate series.')

        self.nr_samples_per_ts = (self.max_lh - self.min_lh) * self.output_chunk_length
        self.total_nr_samples = len(self.target_series) * self.nr_samples_per_ts

        self._index_memory = {}

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_nr_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        # determine the index of the time series.
        ts_idx = idx // self.nr_samples_per_ts

        # select the time series
        ts_target = self.target_series[ts_idx]
        target_vals = ts_target.values(copy=False)

        raise_if_not(len(target_vals) >= (self.lookback + self.max_lh) * self.output_chunk_length,
                     'The dataset contains some input/target series that are shorter than '
                     '`(lookback + max_lh) * H` ({}-th)'.format(ts_idx))

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.nr_samples_per_ts)
        lh_idx = idx - (ts_idx * self.nr_samples_per_ts)

        cov_type = None if self.covariates is None else self.PAST_COV_TYPE

        # The time series index of our forecasting point (indexed from the end of the series):
        end_of_output_idx = len(ts_target) - ((self.min_lh - 1) * self.output_chunk_length + lh_idx)

        # optionally, load covariates
        ts_covariate = self.covariates[ts_idx] if self.covariates is not None else None

        shift = self.lookback * self.output_chunk_length
        input_chunk_length = shift

        # get all indices for the current sample
        past_start, past_end, future_start, future_end, cov_start, cov_end = \
            self._memory_indexer(ts_idx,
                                 end_of_output_idx,
                                 shift,
                                 input_chunk_length,
                                 ts_target,
                                 ts_covariate,
                                 cov_type)

        # extract sample target
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]

        # optionally, extract sample covariates
        covariate = None
        if self.covariates is not None:
            raise_if_not(cov_end <= len(ts_covariate),
                         f"The dataset contains 'past' covariates that don't extend far enough into the future. "
                         f"({idx}-th sample)")

            covariate = ts_covariate.values(copy=False)[cov_start:cov_end]

            raise_if_not(len(covariate) == len(past_target),
                         f"The dataset contains 'past' covariates whose time axis doesn't allow to obtain the "
                         f"input (or output) chunk relative to the target series.")

        return past_target, covariate, future_target

    def _memory_indexer(self,
                        ts_idx: int,
                        end_of_output_idx: int,
                        shift: int,
                        input_chunk_length: int,
                        ts_target: TimeSeries,
                        ts_covariate: TimeSeries,
                        cov_type: Optional[bool] = None) -> SampleIndexType:
        """Returns the (start, end) indices for past target, future target and covariates (sub sets) of the current
        sample `i` from `ts_idx`.

        Works for all TimeSeries index types: pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex

        When `ts_idx` is observed for the first time, it stores the position of the sample `0` within the full target
        time series and the (start, end) indices of all sub sets.
        This allows to calculate the sub set indices for all future samples `i` by simply adjusting for the difference
        between the positions of sample `i` and sample `0`.

        Parameters
        ----------
        ts_idx
            index of the current target TimeSeries.
        ts_target
            current target TimeSeries.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        input_chunk_length
            The length of the emitted past series.
        end_of_output_idx
            the index where the output chunk of the current sample ends in `ts_target`.
        ts_target
            current target TimeSeries.
        ts_covariate
            current covariate TimeSeries.
        cov_type:
            the type of covariate to extract: One of (None, 'past', 'future').
        """

        cov_start, cov_end = None, None

        # the first time ts_idx is observed
        if ts_idx not in self._index_memory:
            start_of_output_idx = end_of_output_idx - self.output_chunk_length
            start_of_input_idx = start_of_output_idx - shift

            # select forecast point and target period, using the previously computed indexes
            future_start, future_end = start_of_output_idx, start_of_output_idx + self.output_chunk_length

            # select input period; look at the `input_chunk_length` points after start of input
            past_start, past_end = start_of_input_idx, start_of_input_idx + input_chunk_length

            if self.covariates is not None:
                start = future_start if cov_type == self.FUTURE_COV_TYPE else past_start
                end = future_end if cov_type == self.FUTURE_COV_TYPE else past_end

                # we need to be careful with getting ranges and indexes:
                # to get entire range, full_range = ts[:len(ts)]; to get last index: last_idx = ts[len(ts) - 1]

                # extract actual index value (respects datetime- and integer-based indexes; also from non-zero start)
                start_time = ts_target.time_index[start]
                end_time = ts_target.time_index[end - 1]

                raise_if_not(start_time in ts_covariate.time_index and end_time in ts_covariate.time_index,
                             f'Missing covariates; could not find {cov_type} covariates in index value range: '
                             f'{start_time} - {end_time}.')

                # extract the index position (index) from index value
                cov_start = ts_covariate.time_index.get_loc(start_time)
                cov_end = ts_covariate.time_index.get_loc(end_time) + 1

            # store position of initial sample and all relevant sub set indices
            self._index_memory[ts_idx] = {
                'end_of_output_idx': end_of_output_idx,
                'past_target': (past_start, past_end),
                'future_target': (future_start, future_end),
                'covariate': (cov_start, cov_end),
            }
        else:
            # load position of initial sample and its sub set indices
            end_of_output_idx_last = self._index_memory[ts_idx]['end_of_output_idx']
            past_start, past_end = self._index_memory[ts_idx]['past_target']
            future_start, future_end = self._index_memory[ts_idx]['future_target']
            cov_start, cov_end = self._index_memory[ts_idx]['covariate']

            # evaluate how much the new sample needs to be shifted, and shift all indexes
            idx_shift = end_of_output_idx - end_of_output_idx_last
            past_start += idx_shift
            past_end += idx_shift
            future_start += idx_shift
            future_end += idx_shift
            cov_start = cov_start + idx_shift if cov_start is not None else None
            cov_end = cov_end + idx_shift if cov_end is not None else None

        return past_start, past_end, future_start, future_end, cov_start, cov_end
