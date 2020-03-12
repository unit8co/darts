from ..timeseries import TimeSeries
import numpy as np
from sklearn.base import TransformerMixin
import torch

from typing import List


class TimeSeriesDataset1D(torch.utils.data.Dataset):
    def __init__(self, series: List[TimeSeries],  # or a timeseries containing all data?
                 train_window: int = 1, label_window: int = 1,
                 full: bool = False, scaler: TransformerMixin = None):
        """
        Construct a dataset for pytorch.
        Will split the different time series in mini-sequences using sliding-window method.

        :param series: A List of independant TimeSeries to be included in the dataset.
        :param train_window: The sequence length of the mini-sequences.
        :param label_window: The sequence length of the target sequence, starting at the end of the train sequence.
        :param full: If True, the target sequence is the train sequence with a lag of 1. `label_window` is ignored.
        :param scaler: A (fitted) scaler from scikit-learn (optional).
        """
        if type(series) is TimeSeries:
            series = [series]
        self.series = [ts.values() for ts in series]
        self.series = np.stack(self.series)
        self.nbr_series, self.len_series = self.series.shape
        self.scaler = scaler
        self._fit_called = False
        # self.series = torch.from_numpy(self.series).float()  # not possible to cast in advance
        self.tw = train_window
        if self.tw is None:
            self.tw = len(series[0]) - 1
        self.lw = label_window
        self.full = full
        if full:
            self.lw = self.tw - 1
        assert self.tw > 0, "The input sequence length must be non null. It is {}".format(self.tw)
        assert self.lw > 0, "The output sequence length must be non null. It is {}".format(self.lw)

    def fit_scaler(self, scaler: TransformerMixin):
        """
        Use a scaler from scikit-learn, fit it on the dataset data, and transform the data.

        :param scaler: A scaler from scikit-learn.
        :return: The scaler fitted.
        """
        if self._fit_called:
            self.inverse_transform()
        self.scaler = scaler.fit(self.series.reshape(-1, 1))
        self.series = self.scaler.transform(self.series)
        self._fit_called = True
        return self.scaler

    def transform(self, scaler=None):
        """
        Transform the data accordingly to the fitted scaler.

        :param scaler: A fitted scaler from scikit-learn (optional)
        """
        if scaler is not None:
            self.scaler = scaler
        if self.scaler is None:
            raise AssertionError("fit_scaler must be called before transform if no scaler is given")
        self.series = self.scaler.transform(self.series)
        self._fit_called = True

    def inverse_transform(self):
        """
        Undo the transformation performed by the scaler.
        """
        if self.scaler is None:
            raise AssertionError("fit_scaler must be called before inverse_transform if no scaler is given")
        self.series = self.scaler.inverse_transform(self.series)

    def __len__(self):
        if self.full:
            return (self.len_series - self.tw) * self.nbr_series
        else:
            return (self.len_series - self.tw - self.lw + 1) * self.nbr_series

    def __getitem__(self, index):
        # todo: should we cast to torch before?
        lw = 1 if self.full else self.lw
        id_series = index // (self.len_series - self.tw - lw + 1)
        idx = index % (self.len_series - self.tw - lw + 1)
        sequence = self.series[id_series, idx:idx + self.tw]
        if self.full:
            target = self.series[id_series, idx + lw:idx + self.tw + lw]
        else:
            target = self.series[id_series, idx + self.tw:idx + self.tw + lw]
        sequence = torch.from_numpy(sequence).float()
        target = torch.from_numpy(target).float()
        return sequence.unsqueeze(1), target.unsqueeze(1)

    @staticmethod
    def _input_label_batch(series: TimeSeries, train_window: int = 1, label_window: int = 1) -> [np.ndarray,
                                                                                                 np.ndarray]:
        sequences = []
        labels = []
        length = len(series)
        for i in range(length - train_window - label_window + 1):
            sequences.append(series.values()[i:i + train_window])
            labels.append(series.values()[i + train_window:i + train_window + label_window])
        return np.array(sequences), np.array(labels)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    pd_series = pd.Series(range(100), index=pd.date_range('20130101', '20130410'))
    pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 6))
    series_ = TimeSeries(pd_series)
    scaler_ = MinMaxScaler()
    dataset = TimeSeriesDataset1D(series_, 12, 2)
    scaler_ = dataset.fit_scaler(scaler_)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=60, shuffle=True,
                                               num_workers=2, pin_memory=True, drop_last=False)
    for batch_idx, (data, target_) in enumerate(train_loader):
        print(data.size(), target_.size())
        print(data.dtype, target_.dtype)
