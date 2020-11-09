from typing import Union, Sequence, Optional
from ...timeseries import TimeSeries
from .timeseries_dataset import TimeSeriesDataset
from ..utils import raise_if_not


class ShiftedDataset(TimeSeriesDataset):
    def __init__(self,
                 input_series: Union[TimeSeries, Sequence[TimeSeries]],
                 target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 seq_length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (input, target) series, both of length `seq_length`.
        The "target" series is in advance of `shift` time steps compared to the "input" series.

        The input and target time series are sliced together, and therefore must have the same time axes.
        In addition, each series must be long enough to contain at least one (input, target) pair; i.e., each
        series must have length at least `seq_length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Some particular slices may
        be sampled more often than others if they belong to shorter time series.

        The recommended use of this class is to either build it from a list of `TimeSeries` (if all your series fit
        in memory), or implement your own `Sequence` of time series (i.e., re-implement `__len__()` and `__getitem__()`)
        and give such an instance as argument to this class.

        Parameters
        ----------
        input_series
            One or a sequence of `TimeSeries` containing the input dimensions.
        target_series:
            Optionally, one or a sequence of `TimeSeries` containing the target dimensions. If this parameter is not
            set, the dataset will use `input_series` instead. If it is set, the provided sequence must have
            the same length as that of `input_series`. In addition, all the target series must have the same time axis
            as the corresponding input series.
            All the emitted target series start after the end of the emitted input series.
        seq_length
            The length of the emitted input and target series.
        shift
            The number of time steps separating input and target; target is in advance of input by `shift` steps.
        max_samples_per_ts
            This is an upper bound on the number of (input, target) tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and ensure proper sampling.
            If `None`, it will read all of the individual time series in advance to know their sizes,
            which might be expensive on big datasets.
            If some series turn out to have length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().__init__()

        self.input_series = [input_series] if isinstance(input_series, TimeSeries) else input_series
        if target_series is None:
            self.target_series = self.input_series
        else:
            self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series

        raise_if_not(len(self.input_series) == len(self.target_series),
                     'The provided sequence of target series must have the same length as '
                     'the provided sequence of input series.')

        self.seq_length, self.shift = seq_length, shift
        self.max_samples_per_ts = max_samples_per_ts

        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = max(len(ts) for ts in self.input_series) - \
                                      self.seq_length - self.shift + 1

        self.ideal_nr_samples = len(self.input_series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx):
        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_input = self.input_series[ts_idx]

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(ts_input) - self.seq_length - self.shift + 1

        raise_if_not(n_samples_in_ts >= 1,
                     'The dataset contains some time series that are too short to contain '
                     '`input_length + `target_length` ({}-th series)'.format(ts_idx))

        # Determine the index of the end of the target, starting from the end.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_target_idx = (idx - (ts_idx * len(self.input_series))) % n_samples_in_ts

        # read the target time series
        ts_target = self.target_series[ts_idx]

        # TODO: check full time index
        raise_if_not(len(ts_input) == len(ts_target),
                     'The dataset contains some input/target series pair that are not the same size ({}-th)'.format(
                         ts_idx
                     ))

        # select forecast point and target period, using the previously computed indexes
        if end_of_target_idx == 0:
            # we need this case because "-0" is not supported as an indexing bound
            target_series = ts_target[-self.seq_length:]
        else:
            target_series = ts_target[-(self.seq_length + end_of_target_idx):-end_of_target_idx]

        # select input period; look at the `input_length` points before the forecast point
        input_series = ts_input[-(self.seq_length + end_of_target_idx + self.shift):-(end_of_target_idx + self.shift)]

        return input_series, target_series
