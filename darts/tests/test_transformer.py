import unittest

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..preprocessing import ScalerWrapper
from ..utils import timeseries_generation as tg


class TransformerTestCase(unittest.TestCase):
    __test__ = True
    series1 = tg.random_walk_timeseries(length=100) * 20 - 10.
    series2 = series1.stack(tg.random_walk_timeseries(length=100) * 20 - 100.)

    def test_scaling(self):
        transformer1 = ScalerWrapper(MinMaxScaler(feature_range=(0, 2)))
        transformer2 = ScalerWrapper(StandardScaler())

        series1_tr1 = transformer1.fit_transform(self.series1)
        series1_tr2 = transformer2.fit_transform(self.series1)

        transformer3 = ScalerWrapper(MinMaxScaler(feature_range=(0, 2)))
        transformer4 = ScalerWrapper(StandardScaler())

        series2_tr3 = transformer3.fit_transform(self.series2)
        series2_tr4 = transformer4.fit_transform(self.series2)

        # should comply with scaling constraints
        self.assertAlmostEqual(min(series1_tr1.values().flatten()), 0.)
        self.assertAlmostEqual(max(series1_tr1.values().flatten()), 2.)
        self.assertAlmostEqual(np.mean(series1_tr2.values().flatten()), 0.)
        self.assertAlmostEqual(np.std(series1_tr2.values().flatten()), 1.)

        self.assertAlmostEqual(min(series2_tr3.values().flatten()), 0.)
        self.assertAlmostEqual(max(series2_tr3.values().flatten()), 2.)
        self.assertAlmostEqual(np.mean(series2_tr4.values().flatten()), 0.)
        self.assertAlmostEqual(np.std(series2_tr4.values().flatten()), 1.)

        # test inverse transform
        series1_recovered = transformer2.inverse_transform(series1_tr2)
        series2_recovered = transformer3.inverse_transform(series2_tr3)
        np.testing.assert_almost_equal(series1_recovered.values().flatten(), self.series1.values().flatten())
        np.testing.assert_almost_equal(series2_recovered.values().flatten(), self.series2.values().flatten())
        self.assertEqual(series1_recovered.width, self.series1.width)
        self.assertEqual(series2_recovered.width, self.series2.width)
