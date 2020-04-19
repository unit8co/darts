from ..timeseries import TimeSeries
import numpy as np
from sklearn.base import TransformerMixin
import torch

from typing import List, Union


class TimeSeriesDataset1D(torch.utils.data.Dataset):

    def __init__(self,
                 series: Union[TimeSeries, List[TimeSeries]],
                 data_length: int = 1,
                 target_length: int = 1,
                 scaler: TransformerMixin = None):
        """
        Constructs a PyTorch Dataset from a univariate TimeSeries, or from a list of univariate TimeSeries.
        The Dataset iterates a moving window over the time series. The resulting slices contain `(data, target)`,
        where `data` is a 1-D sub-sequence of length [data_length] and target is the 1-D sub-sequence of length
        [target_length] following it in the time series.

        :param series: Either a TimeSeries or a list of TimeSeries to be included in the dataset.
        :param data_length: The length of the training sub-sequences.
        :param target_length: The length of the target sub-sequences, starting at the end of the training sub-sequence.
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
        self.data_length = data_length
        if self.data_length is None:
            self.data_length = len(series[0]) - 1
        self.target_length = target_length

        assert self.data_length > 0, "The input sequence length must be non null. It is {}".format(self.data_length)
        assert self.target_length > 0, "The output sequence length must be non null. It is {}".format(self.target_length)

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
        return (self.len_series - self.data_length - self.target_length + 1) * self.nbr_series

    def __getitem__(self, index):
        # TODO: Cast to PyTorch tensors on the right device in advance
        id_series = index // (self.len_series - self.data_length - self.target_length + 1)
        id_time = index % (self.len_series - self.data_length - self.target_length + 1)
        data = self.series[id_series, id_time:id_time+self.data_length]
        target = self.series[id_series, id_time+self.data_length:id_time+self.data_length+self.target_length]
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).float()
        return data.unsqueeze(1), target.unsqueeze(1)

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
