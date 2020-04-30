import unittest
import pandas as pd
import shutil

from .. import RNNModule, RNNModel
from ..timeseries import TimeSeries


class RNNModelTestCase(unittest.TestCase):
    __test__ = True
    times = pd.date_range('20130101', '20130410')
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries(pd_series)
    module = RNNModule('RNN', input_size=1, output_length=1, hidden_dim=25,
                       num_layers=1, num_layers_out_fc=[], dropout=0)

    def test_creation(self):
        with self.assertRaises(ValueError):
            # cannot choose any string
            RNNModel(model='UnknownRNN?')
        # can give a custom module
        model1 = RNNModel(self.module)
        model2 = RNNModel("RNN")
        self.assertEqual(model1.model.__repr__(), model2.model.__repr__())

    def test_fit(self):
        # Test basic fit()
        model = RNNModel(n_epochs=2)
        model.fit(self.series)

        # Test fit-save-load cycle
        model2 = RNNModel('LSTM', n_epochs=4, model_name='unittest-model-lstm')
        model2.fit(self.series)
        model_loaded = model2.load_from_checkpoint(model_name='unittest-model-lstm', best=False)
        pred1 = model2.predict(n=6)
        pred2 = model_loaded.predict(n=6)

        # Two models with the same parameters should deterministically yield the same output
        self.assertEqual(sum(pred1.values() - pred2.values()), 0.)

        # Another random model should not
        model3 = RNNModel('RNN')
        model3.fit(self.series)
        pred3 = model3.predict(n=6)
        self.assertNotEqual(sum(pred1.values() - pred3.values()), 0.)

        shutil.rmtree('.u8ts')
