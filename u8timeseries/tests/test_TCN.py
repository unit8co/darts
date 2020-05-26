import unittest
import logging

from ..models.tcn_model import TCNModel
from ..utils import timeseries_generation as tg


class TCNModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_creation(self):
        with self.assertRaises(ValueError):
            # cannot choose a kernel size larger than the input length
            TCNModel(kernel_size=100, input_length=20)
        TCNModel()

    def test_fit(self):
        large_ts = tg.constant_timeseries(length=100, value=1000)
        small_ts = tg.constant_timeseries(length=100, value=10)

        # Test basic fit and predict
        model = TCNModel(n_epochs=20, num_layers=1)
        model.fit(large_ts[:98])
        pred = model.predict(n=2).values()[0]

        # Test whether model trained on one series is better than one trained on another
        model2 = TCNModel(n_epochs=20, num_layers=1)
        model2.fit(small_ts[:98])
        pred2 = model2.predict(n=2).values()[0]
        self.assertTrue(abs(pred2 - 10) < abs(pred - 10))
