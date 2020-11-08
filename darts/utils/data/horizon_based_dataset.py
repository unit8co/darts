from typing import Union, Optional, Sequence, Tuple
from ...logging import raise_if_not, get_logger
from ...timeseries import TimeSeries
from .timeseries_dataset import TimeSeriesDataset

logger = get_logger(__name__)


class HorizonBasedTrainDataset(TimeSeriesDataset):
    def __init__(self,
                 horizon: int,
                 input_series: Union[TimeSeries, Sequence[TimeSeries]],
                 target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 lh: Tuple[int] = (1, 3),
                 lookback: int = 3) -> None:
        """
        This dataset computes (input, target) splits. in a way inspired by the N-BEATS
        way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        Given the horizon `H` of a model, this dataset will compute some (input, target) splits as follows.
        First a "forecast point" is selected in the the range of the last `(min_lh * H, max_lh * H)`
        points before the end of the time series. The target then consists in the following `H` points, and the data
        will be the preceding `lookback * H` points.

        All the series in the provided sequence must be long enough; i.e. have length at least
        `(lookback + max_lh) * H`, and `min_lh` must be at least 1 (to have targets of length exactly `1 * H`).
        The input and target time series are sliced together, and therefore must have the same time axes.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform both over the number of time series and the number of samples per series;
        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N
        time series in the sequence.

        The recommended use of this class is to either build it from a list of `TimeSeries` (if all your series fit
        in memory), or implement your own `Sequence` of time series (i.e., re-implement `__len__()` and `__getitem__()`)
        and give such an instance as argument to this class.

        Parameters
        ----------
        horizon:
            The horizon `H` of the underlying model being trained. The emitted targets will have length `H`.
        input_series:
            One or a sequence of `TimeSeries` containing the input dimensions.
        target_series:
            Optionally, one or a sequence of `TimeSeries` containing the target dimensions. If this parameter is not
            set, the dataset will use `input_series` instead. If it is set, the provided sequence must have
            the same length as that of `input_series`. In addition, all the target series must have the time axis as
            the corresponding input series.
            All the emitted target series start after the end of the emitted input series.
        lh
            A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.
            For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points
            before the end of the series. It is required that `min_lh >= 1`.
        lookback:
            A integer interval for the length of the input in the emitted (input, target) splits, expressed as a
            multiple of the horizon `H`. For instance, `lookback=3` will emit "inputs" of lengths `3H`.
        """
        super().__init__()

        self.input_series = [input_series] if isinstance(input_series, TimeSeries) else input_series
        if target_series is None:
            self.target_series = self.input_series
        else:
            self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.H = horizon
        self.min_lh, self.max_lh = lh
        self.lookback = lookback

        # Checks
        raise_if_not(self.max_lh >= self.min_lh >= 1,
                     'The lh parameter should be an int tuple (min_lh, max_lh), '
                     'with 1 <= min_lh <= max_lh')
        raise_if_not(len(self.input_series) == len(self.target_series),
                     'The provided sequence of target series must have the same length as '
                     'the provided sequence of input series.')

        self.nr_samples_per_ts = (self.max_lh - self.min_lh) * self.H
        self.total_nr_samples = len(self.input_series) * self.nr_samples_per_ts

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_nr_samples

    def __getitem__(self, idx: int) -> Tuple[TimeSeries, TimeSeries]:
        # determine the index of the time series.
        ts_idx = idx // self.nr_samples_per_ts

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.nr_samples_per_ts)
        lh_idx = idx - (ts_idx * len(self.input_series))

        # The time series index of our forecasting point (indexed from the end of the series):
        forecast_point_idx = self.min_lh * self.H + lh_idx

        # Sanity check... TODO: remove
        assert lh_idx < (self.max_lh - self.min_lh) * self.H, 'bug in the Lh indexing'

        # select the time series
        ts_input = self.input_series[ts_idx]
        ts_target = self.target_series[ts_idx]

        # TODO: check full time index
        raise_if_not(len(ts_input) == len(ts_target),
                     'The dataset contains some input/target series pair that are not the same size ({}-th)'.format(
                         ts_idx
                     ))
        raise_if_not(len(ts_input) >= (self.lookback + self.max_lh) * self.H,
                     'The dataset contains some input/target series that are shorter than '
                     '`(lookback + max_lh) * H` ({}-th)'.format(ts_idx))

        # select forecast point and target period, using the previously computed indexes
        if forecast_point_idx == self.H:
            # we need this case because "-0" is not supported as an indexing bound
            target_series = ts_target[-forecast_point_idx:]
        else:
            target_series = ts_target[-forecast_point_idx:-forecast_point_idx+self.H]

        # select input period; look at the `lookback * H` points before the forecast point
        input_series = ts_input[-(forecast_point_idx + self.lookback * self.H):-forecast_point_idx]

        return input_series, target_series
