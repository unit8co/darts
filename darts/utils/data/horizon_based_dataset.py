"""
Horizon-Based Training Dataset
------------------------------
"""

from typing import Union, Optional, Sequence, Tuple
import numpy as np

from ...logging import raise_if_not, get_logger
from ...timeseries import TimeSeries
from .timeseries_dataset import TrainingDataset

logger = get_logger(__name__)


class HorizonBasedDataset(TrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 output_chunk_length: int = 12,
                 lh: Tuple[int, int] = (1, 3),
                 lookback: int = 3) -> None:
        """
        A time series dataset containing tuples of (input, output, input_covariates) arrays, in a way inspired
        by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        The "input" and "input_covariates" have length `lookback * output_chunk_length`, and the "output" has length
        `output_chunk_length`.

        Given the horizon `output_chunk_length` of a model, this dataset will compute some (input, target) splits as follows
        (the logic for "input_covariates" is the same as for "input"):
        First a "forecast point" is selected in the the range of the last
        `(min_lh * output_chunk_length, max_lh * output_chunk_length)` points before the end of the time series.
        The target then consists in the following `output_chunk_length` points, and the data will be the preceding
        `lookback * output_chunk_length` points.

        All the series in the provided sequence must be long enough; i.e. have length at least
        `(lookback + max_lh) * output_chunk_length`, and `min_lh` must be at least 1
        (to have targets of length exactly `1 * output_chunk_length`).
        The target and covariates time series are sliced together, and therefore must have the same length.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform both over the number of time series and the number of samples per series;
        i.e. the i-th sample of this dataset has 1/(N*M) chance of coming from any of the M samples in any of the N
        time series in the sequence.

        The recommended use of this class is to either build it from a list of `TimeSeries` (if all your series fit
        in memory), or implement your own `Sequence` of time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`.
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

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_nr_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # determine the index of the time series.
        ts_idx = idx // self.nr_samples_per_ts

        # determine the index lh_idx of the forecasting point (the last point of the input series, before the target)
        # lh_idx should be in [0, self.nr_samples_per_ts)
        lh_idx = idx - (ts_idx * self.nr_samples_per_ts)

        # The time series index of our forecasting point (indexed from the end of the series):
        forecast_point_idx = self.min_lh * self.output_chunk_length + lh_idx

        # Sanity check... TODO: remove
        assert lh_idx < (self.max_lh - self.min_lh) * self.output_chunk_length, 'bug in the Lh indexing'

        # select the time series
        ts_target = self.target_series[ts_idx].values(copy=False)

        raise_if_not(len(ts_target) >= (self.lookback + self.max_lh) * self.output_chunk_length,
                     'The dataset contains some input/target series that are shorter than '
                     '`(lookback + max_lh) * H` ({}-th)'.format(ts_idx))

        # select forecast point and target period, using the previously computed indexes
        if forecast_point_idx == self.output_chunk_length:
            # we need this case because "-0" is not supported as an indexing bound
            output_series = ts_target[-forecast_point_idx:]
        else:
            output_series = ts_target[-forecast_point_idx:-forecast_point_idx+self.output_chunk_length]

        # select input period; look at the `lookback * output_series` points before the forecast point
        input_series = ts_target[-(forecast_point_idx + self.lookback * self.output_chunk_length):-forecast_point_idx]

        # optionally also produce the input covariate
        input_covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx].values(copy=False)

            raise_if_not(len(ts_covariate) == len(ts_target),
                         'The dataset contains some target/covariate series '
                         'pair that are not the same size ({}-th)'.format(ts_idx))

            input_covariate = ts_covariate[-(forecast_point_idx + self.lookback * self.output_chunk_length):-forecast_point_idx]

        return input_series, output_series, input_covariate
