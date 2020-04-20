from ..timeseries import TimeSeries
import numpy as np
import torch


class TimeSeriesDataset1D(torch.utils.data.Dataset):

    def __init__(self,
                 series: TimeSeries,
                 data_length: int = 1,
                 target_length: int = 1):
        """
        Constructs a PyTorch Dataset from a univariate TimeSeries.
        The Dataset iterates a moving window over the time series. The resulting slices contain `(data, target)`,
        where `data` is a 1-D sub-sequence of length [data_length] and target is the 1-D sub-sequence of length
        [target_length] following it in the time series.

        :param series: Either a TimeSeries or a list of TimeSeries to be included in the dataset.
        :param data_length: The length of the training sub-sequences.
        :param target_length: The length of the target sub-sequences, starting at the end of the training sub-sequence.
        :param scaler: A (fitted) scaler from scikit-learn (optional).
        """
        self.series_values = series.values()

        # self.series = torch.from_numpy(self.series).float()  # not possible to cast in advance
        self.len_series = len(series)
        self.data_length = len(series) - 1 if data_length is None else data_length
        self.target_length = target_length

        assert self.data_length > 0, "The input sequence length must be positive. It is {}".format(self.data_length)
        assert self.target_length > 0, "The output sequence length must be positive. It is {}".format(self.target_length)

    def __len__(self):
        return self.len_series - self.data_length - self.target_length + 1

    def __getitem__(self, index):
        # TODO: Cast to PyTorch tensors on the right device in advance
        idx = index % (self.len_series - self.data_length - self.target_length + 1)
        data = self.series_values[idx:idx+self.data_length]
        target = self.series_values[idx+self.data_length:idx+self.data_length+self.target_length]
        return torch.from_numpy(data).float().unsqueeze(1), torch.from_numpy(target).float().unsqueeze(1)

    # @staticmethod
    # def _input_label_batch(series: TimeSeries, train_window: int = 1, label_window: int = 1) -> [np.ndarray,
    #                                                                                              np.ndarray]:
    #     sequences = []
    #     labels = []
    #     length = len(series)
    #     for i in range(length - train_window - label_window + 1):
    #         sequences.append(series.values()[i:i + train_window])
    #         labels.append(series.values()[i + train_window:i + train_window + label_window])
    #     return np.array(sequences), np.array(labels)
