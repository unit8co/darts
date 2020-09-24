import unittest
import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from darts.preprocessing import ScalerWrapper
from darts.utils import timeseries_generation as tg


class TransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    series1 = tg.random_walk_timeseries(length=100) * 20 - 10.
    series2 = series1.stack(tg.random_walk_timeseries(length=100) * 20 - 100.)

    def test_scaling(self):
        transformer1 = ScalerWrapper(MinMaxScaler(feature_range=(0, 2)))
        transformer2 = ScalerWrapper(StandardScaler())

        series1_tr1 = transformer1.fit_transform(self.series1)
        series1_tr2 = transformer2.fit_transform(self.series1)

        # should comply with scaling constraints
        self.assertAlmostEqual(min(series1_tr1.values().flatten()), 0.)
        self.assertAlmostEqual(max(series1_tr1.values().flatten()), 2.)
        self.assertAlmostEqual(np.mean(series1_tr2.values().flatten()), 0.)
        self.assertAlmostEqual(np.std(series1_tr2.values().flatten()), 1.)

        # test inverse transform
        series1_recovered = transformer2.inverse_transform(series1_tr2)
        np.testing.assert_almost_equal(series1_recovered.values().flatten(), self.series1.values().flatten())
        self.assertEqual(series1_recovered.width, self.series1.width)
