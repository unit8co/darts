import unittest
import pandas as pd
import torch

from ..timeseries import TimeSeries
from ..utils import TimeSeriesDataset1D


class DatasetTestCase(unittest.TestCase):
    __test__ = True
    times = pd.date_range('20130101', '20130410')
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries(pd_series)

    def test_creation(self):
        # Cannot have train window <= 0
        with self.assertRaises(ValueError):
            dataset = TimeSeriesDataset1D(self.series, -1)
        # Cannot have label window <= 0
        with self.assertRaises(ValueError):
            dataset = TimeSeriesDataset1D(self.series, 1, -1)

    def test_content(self):
        # Can have no transformation
        dataset = TimeSeriesDataset1D(self.series, 12, 4)
        self.assertEqual(len(dataset), 100-12-4+1)
        # correct types
        self.assertEqual(type(dataset[0][0]), torch.Tensor)
        self.assertEqual(dataset[0][0].dtype, torch.float32)
        # take correct elements
        self.assertEqual(dataset[5][0].numpy()[0], self.pd_series.values[5])
        self.assertEqual(dataset[5][1].numpy()[0], self.pd_series.values[5+12])
        # content same size from beginning to end
        self.assertEqual(dataset[0][1].size(), dataset[len(dataset)-1][1].size())
