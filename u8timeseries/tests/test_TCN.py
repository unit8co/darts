import shutil
import unittest

import pandas as pd

from ..models.tcn_model import TCNModel
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape


class TCNModelTestCase(unittest.TestCase):

    def test_creation(self):
        with self.assertRaises(ValueError):
            # cannot choose a kernel size larger than the input length
            TCNModel(kernel_size=100, input_length=50)
        model = TCNModel()
    
    def test_fit(self):
        sine_ts = tg.sine_timeseries(length=100)

        # Test basic fit
        model1 = TCNModel(n_epochs=2)
        model1.fit(sine_ts[:80])

        # Test model improvement
        model2 = TCNModel(n_epochs=4)
        model2.fit(sine_ts[:80])
        self.assertTrue(mape(model2.predict(n=20), sine_ts[80:]) <= mape(model1.predict(n=20), sine_ts[80:]))
