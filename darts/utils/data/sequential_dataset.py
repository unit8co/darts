"""
Sequential Training Dataset
---------------------------
"""

from typing import Union, Sequence, Optional, Tuple
import numpy as np

from ...timeseries import TimeSeries
from .timeseries_dataset import TrainingDataset

from ..utils import raise_if_not


class SequentialDataset(TrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (input, output, input_covariates) arrays, where "input" and
        "input_covariates" have length `input_chunk_length`, and "output" has length `output_chunk_length`.

        The target and covariates series are sliced together, and therefore must have the same length.
        In addition, each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        The recommended use of this class is to either build it from a list of `TimeSeries` (if all your series fit
        in memory), or implement your own `Sequence` of time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
        input_chunk_length
            The length of the emitted input series.
        output_chunk_length
            The length of the emitted output series.
        max_samples_per_ts
            This is an upper bound on the number of (input, output, input_covariates) tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().__init__()

        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        raise_if_not(covariates is None or len(self.target_series) == len(self.covariates),
                     'The provided sequence of target series must have the same length as '
                     'the provided sequence of covariate series.')

        self.input_chunk_length, self.output_chunk_length = input_chunk_length, output_chunk_length
        self.max_samples_per_ts = max_samples_per_ts

        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = max(len(ts) for ts in self.target_series) - \
                                      self.output_chunk_length - self.input_chunk_length + 1

        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_target = self.target_series[ts_idx].values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(ts_target) - self.input_chunk_length - self.output_chunk_length + 1

        raise_if_not(n_samples_in_ts >= 1,
                     'The dataset contains some time series that are too short to contain '
                     '`input_chunk_length + `output_chunk_length` ({}-th series)'.format(ts_idx))

        # Determine the index of the forecasting point.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        lh_idx = (idx - (ts_idx * self.max_samples_per_ts)) % n_samples_in_ts

        # The time series index of our forecasting point (indexed from the end of the series):
        forecast_point_idx = self.output_chunk_length + lh_idx

        # select input and outputs, using the previously computed indexes
        input_target = ts_target[-(forecast_point_idx + self.input_chunk_length):-forecast_point_idx]
        if forecast_point_idx == self.output_chunk_length:
            # we need this case because "-0" is not supported as an indexing bound
            output_target = ts_target[-forecast_point_idx:]
        else:
            output_target = ts_target[-forecast_point_idx:-forecast_point_idx + self.output_chunk_length]

        # optionally also produce the input covariate
        input_covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx].values(copy=False)

            raise_if_not(len(ts_covariate) == len(ts_target),
                         'The dataset contains some target/covariate series '
                         'pair that are not the same size ({}-th)'.format(ts_idx))

            input_covariate = ts_covariate[-(forecast_point_idx + self.input_chunk_length):-forecast_point_idx]

        return input_target, output_target, input_covariate
