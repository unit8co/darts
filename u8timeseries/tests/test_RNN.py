import unittest
import pandas as pd
import shutil
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

from .. import RNN, RNNModel
from ..timeseries import TimeSeries


class MyTestCase(unittest.TestCase):
    __test__ = True
    times = pd.date_range('20130101', '20130410')
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries(pd_series)
    module = RNN('RNN', input_size=1, output_size=1, hidden_dim=25, n_layers=1, hidden_linear=[], dropout=0)

    def test_creation(self):
        # do not test zero and neg values for module creation, because it already present in pytorch
        with self.assertRaises(ValueError):
            # cannot choose any string
            RNNModel(model='CNN')
        # can give a custom module
        model1 = RNNModel(self.module)
        model2 = RNNModel("RNN")
        self.assertEqual(model1.model.__repr__(), model2.model.__repr__())

    def test_fit(self):
        model = RNNModel(n_epochs=2)
        with self.assertRaises(AssertionError):
            model.fit(self.series)
        model.set_optimizer()
        with self.assertRaises(AssertionError):
            model.fit(self.series)
        model.set_scheduler()
        model.fit(self.series)

        model2 = RNNModel(n_epochs=10)
        model2.set_optimizer()
        model2.set_scheduler()
        model2.load_from_checkpoint()
        model2.fit(self.series)

        shutil.rmtree('./checkpoints')


if __name__ == '__main__':
    unittest.main()
