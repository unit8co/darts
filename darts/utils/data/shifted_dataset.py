"""
Shifted Training Dataset
------------------------
"""

from typing import Union, Sequence, Optional, Tuple
import numpy as np

from ...timeseries import TimeSeries
from .timeseries_dataset import TrainingDataset
from ..utils import raise_if_not


class ShiftedDataset(TrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 shift_covariates: bool = False,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (input, output, input_covariates) arrays, which all have length
        `length`.
        The "output" is the "input" target shifted by `shift` time steps forward. So if an emitted "input"
        (and "input_covariates") goes from position `i` to `i+length`, the emitted output will go from position
        `i+shift` to `i+shift+length`.

        The target and covariates series are sliced together, and therefore must have the same length.
        In addition, each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
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
        covariates
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in this list must be at least as long as their corresponding target series and
            must have the same starting point.
        length
            The length of the emitted input and output series.
        shift
            The number of time steps by which to shift the output relative to the input.
        shift_covariates
            Whether or not to shift the covariates forward the same way as the target.
            Block models require this parameter to be set to `False` In the case of recurrent
            model, this parameter should be set to `True`.
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

        self.length, self.shift, self.shift_covariates = length, shift, shift_covariates
        self.max_samples_per_ts = max_samples_per_ts

        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = max(len(ts) for ts in self.target_series) - \
                                      self.length - self.shift + 1

        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        raise_if_not(min(len(ts) for ts in self.target_series) - self.length - self.shift + 1 > 0,
                     "Every target series needs to be at least `length + shift` long")

        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_target = self.target_series[ts_idx].values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(ts_target) - self.length - self.shift + 1

        raise_if_not(n_samples_in_ts >= 1,
                     'The dataset contains some time series that are too short to contain '
                     '`input_chunk_length + `output_chunk_length` ({}-th series)'.format(ts_idx))

        # Determine the index of the end of the output, starting from the end.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (idx - (ts_idx * self.max_samples_per_ts)) % n_samples_in_ts

        # select forecast point and target period, using the previously computed indexes
        if end_of_output_idx == 0:
            # we need this case because "-0" is not supported as an indexing bound
            output_series = ts_target[-self.length:]
        else:
            output_series = ts_target[-(self.length + end_of_output_idx):-end_of_output_idx]

        # select input period; look at the `input_chunk_length` points before the forecast point
        input_series = ts_target[-(self.length + end_of_output_idx + self.shift):-(end_of_output_idx + self.shift)]

        # optionally also produce the input covariate
        input_covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx].values(copy=False)[:len(ts_target)]

            raise_if_not(len(ts_covariate) == len(ts_target),
                         'The dataset contains some target/covariate series '
                         'pair that are not the same size ({}-th)'.format(ts_idx))

            if self.shift_covariates:
                if end_of_output_idx == 0:
                    # we need this case because "-0" is not supported as an indexing bound
                    input_covariate = ts_covariate[-self.length:]
                else:
                    input_covariate = ts_covariate[-(self.length + end_of_output_idx):-end_of_output_idx]
            else:
                input_covariate = ts_covariate[-(self.length + end_of_output_idx + self.shift):
                                               -(end_of_output_idx + self.shift)]

        return input_series, output_series, input_covariate
