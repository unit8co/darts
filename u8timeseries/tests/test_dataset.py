import unittest
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import torch

from ..timeseries import TimeSeries
from ..utils import TimeSeriesDataset


class MyTestCase(unittest.TestCase):
    __test__ = True
    times = pd.date_range('20130101', '20130410')
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries(pd_series)
    scaler = MinMaxScaler()
    scaler.fit(series.values().reshape(-1, 1))
    data, label = TimeSeriesDataset._input_label_batch(series, 12, 4)

    def test_creation(self):
        # Should be a timeseries object
        with self.assertRaises(TypeError):
            dataset = TimeSeriesDataset(self.pd_series, self.scaler)
        # Should have a scaler fitted
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            dataset = TimeSeriesDataset(self.series, MinMaxScaler())
        # Cannot have train window <= 0
        with self.assertRaises(AssertionError):
            dataset = TimeSeriesDataset(self.series, self.scaler, -1)
        # Cannot have label window <= 0
        with self.assertRaises(AssertionError):
            dataset = TimeSeriesDataset(self.series, self.scaler, 1, -1)

    def test_content(self):
        # Can have no transformation
        dataset = TimeSeriesDataset(self.series, None, 12, 4)
        self.assertEqual(len(dataset), len(self.data))
        # correct types
        self.assertEqual(type(dataset[0][0]), torch.Tensor)
        self.assertEqual(dataset[0][0].dtype, torch.float32)
        # take correct elements
        self.assertEqual(dataset[5][0].numpy()[0], self.pd_series.values[5])
        self.assertEqual(dataset[5][1].numpy()[0], self.pd_series.values[5+12])
        # content same size from beginning to end
        self.assertEqual(dataset[0][1].size(), dataset[len(dataset)-1][1].size())


if __name__ == '__main__':
    unittest.main()
