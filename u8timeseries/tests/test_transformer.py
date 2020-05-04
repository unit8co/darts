import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .. import Transformer
from ..utils import timeseries_generation as tg


class TransformerTestCase(unittest.TestCase):
    __test__ = True
    series = tg.random_walk_timeseries(length=100) * 20 - 10.

    def test_scaling(self):
        transformer1 = Transformer(MinMaxScaler(feature_range=(0, 2)))
        transformer2 = Transformer(StandardScaler())

        series_tr1 = transformer1.fit_transform(self.series)
        series_tr2 = transformer2.fit_transform(self.series)

        # should comply with scaling constraints
        self.assertAlmostEqual(min(series_tr1.values()), 0.)
        self.assertAlmostEqual(max(series_tr1.values()), 2.)
        self.assertAlmostEqual(np.mean(series_tr2.values()), 0.)
        self.assertAlmostEqual(np.std(series_tr2.values()), 1.)
