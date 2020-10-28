from torch.utils.data import Dataset
from typing import Union, Optional, Sequence, Tuple
from ...logging import raise_if_not, raise_if, get_logger

from ...timeseries import TimeSeries


class TimeSeriesDataset(Dataset):
    def __init__(self,
                 series: Union[TimeSeries, Sequence[TimeSeries]],
                 horizon: int,
                 lh: Tuple[int] = (1, 3),
                 lookback: Tuple[int] = (2, 7)) -> None:
        """
        A time series dataset, which can be built from a sequence of time series.
        This dataset will compute some (data, target) splits, and is meant to be
        used to train PyTorch-based models.
        It is inspired by the N-BEATS way of training on the M4 dataset: https://arxiv.org/abs/1905.10437.

        Given the horizon `H` of a model, this dataset will compute some (data, target) splits as follows.
        First a "forecast point" is selected in the the range of the last `(min_lh, max_lh)`
        points before the end of the time series. The target is then the following (up to) `H` points, and the data
        will be the preceding `min_lb` to `max_lb` points.

        All the series in the provided sequence must be long enough; i.e. have length at least `max_lh + min_lb`,
        otherwise an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e. the i-th sample of this dataset has 1/N chance
        of coming from any of the N time series in the sequence, and so depending on the parameters,
        the same (data, target) sample may be sampled more often if it belongs to shorter time series.
        Note: all (data, target) samples will be sampled with the same frequency only if all the time series in
        the provided sequence have length at least `(max_lb + max_lh) * H`.

        The recommended use of this class is to either build it from a list of TimeSeries (if all your series fit
        in memory), or implement your own Sequence of time series (i.e., re-implement `__len__()` and `__getitem__()`)
        and give such an instance as argument to this class.

        :param series: one or a sequence of TimeSeries
        :param horizon: The horizon `H` of the underlying model being trained. The emitted targets
                        will have length (up to) `H`.
        :param lh: A `(min_lh, max_lh)` interval for the forecast point, starting from the end of the series.
                   For example, `(1, 3)` will select forecast points uniformly between `1*H` and `3*H` points
                   before the end of the series.
                   If `min_lh < 1`, the "targets" in the emitted (data, target) splits might be shorter than
                   the horizon `H`. Use this only with models that are capable of masking appropriately the series
                   during training.
        :param lookback: A `(min_lb, max_lb)` interval for the length of the "data" in the emitted (data, target) splits.
                         For instance, `(2, 7)` will emit "data" of lengths between `2H` and `7H`, depending
                         on the maximum size allowed by the series' length and the selection of the forecast point.
        """

        self.series = [series] if isinstance(series, TimeSeries) else series
        self.H = horizon
        self.min_lh, self.max_lh = lh
        self.min_lb, self.max_lb = lookback

        # Checks
        raise_if_not(self.min_lh <= self.max_lh,
                     'The lh parameter should be an int tuple (min_lh, max_lh), with min_lh <= max_lh')
        raise_if_not(self.min_lb <= self.max_lh,
                     'The lookback parameter should be an int tuple (min_lb, max_lb), with min_lb <= max_lb')

        self.ideal_nr_samples = len(self.series) * (self.max_lh - self.min_lh)*self.H * (self.max_lb - self.min_lb)

    def __len__(self):
        """
        Returns the (ideal) number of possible (data, target) splits.
        The actual number of possible splits might be lower, if some time series are shorter than
         `(max_lb + max_lh) * H`.
        """
        return self.ideal_nr_samples

    def __getitem__(self, idx):
        pass