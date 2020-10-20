import unittest
import pandas as pd
from math import log

from darts.dataprocessing.transformers import BoxCox, Mapper
from darts.utils.timeseries_generation import sine_timeseries, linear_timeseries


class BoxCoxTestCase(unittest.TestCase):

    sine_series = sine_timeseries(length=50, value_y_offset=5, value_frequency=0.05)
    lin_series = linear_timeseries(start_value=1, end_value=10, length=50)
    multi_series = sine_series.stack(lin_series)

    def test_boxbox_lambda(self):
        boxcox = BoxCox()

        boxcox.fit(self.multi_series, 0.3)
        self.assertEqual(boxcox._lmbda, [0.3, 0.3])

        boxcox.fit(self.multi_series, [0.3, 0.4])
        self.assertEqual(boxcox._lmbda, [0.3, 0.4])

        with self.assertRaises(ValueError):
            boxcox.fit(self.multi_series, [0.2, 0.4, 0.5])

        boxcox.fit(self.multi_series, optim_method='mle')
        lmbda1 = boxcox._lmbda
        boxcox.fit(self.multi_series, optim_method='pearsonr')
        lmbda2 = boxcox._lmbda

        self.assertNotEqual(lmbda1.array, lmbda2.array)

    def test_boxcox_transform(self):
        log_mapper = Mapper(lambda x: log(x))
        boxcox = BoxCox()

        transformed1 = log_mapper.transform(self.sine_series)
        transformed2 = boxcox.fit(self.sine_series, lmbda=0).transform(self.sine_series)

        self.assertEqual(transformed1, transformed2)

    def test_boxcox_inverse(self):
        boxcox = BoxCox()
        transformed = boxcox.fit_transform(self.multi_series)
        back = boxcox.inverse_transform(transformed)
        pd.testing.assert_frame_equal(self.multi_series._df, back._df, check_exact=False)
