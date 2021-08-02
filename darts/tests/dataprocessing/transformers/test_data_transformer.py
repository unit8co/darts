import unittest
import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from darts.dataprocessing.transformers import Scaler
from darts.utils import timeseries_generation as tg


class DataTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    series1 = tg.random_walk_timeseries(length=100, column_name='series1') * 20 - 10.
    series2 = series1.stack(tg.random_walk_timeseries(length=100) * 20 - 100.)

    
    col_1 = series1.columns

    def test_scaling(self):
        self.series3 = self.series1[:1]
        transformer1 = Scaler(MinMaxScaler(feature_range=(0, 2)))
        transformer2 = Scaler(StandardScaler())

        series1_tr1 = transformer1.fit_transform(self.series1)
        series1_tr2 = transformer2.fit_transform(self.series1)
        series3_tr2 = transformer2.transform(self.series3)

        # should have the defined name above
        self.assertEqual(self.series1.columns[0], 'series1')

        # should keep columns pd.Index
        self.assertEqual(self.col_1, series1_tr1.columns)

        # should comply with scaling constraints
        self.assertAlmostEqual(min(series1_tr1.values().flatten()), 0.)
        self.assertAlmostEqual(max(series1_tr1.values().flatten()), 2.)
        self.assertAlmostEqual(np.mean(series1_tr2.values().flatten()), 0.)
        self.assertAlmostEqual(np.std(series1_tr2.values().flatten()), 1.)

        # test inverse transform
        series1_recovered = transformer2.inverse_transform(series1_tr2)
        series3_recovered = transformer2.inverse_transform(series3_tr2)
        np.testing.assert_almost_equal(series1_recovered.values().flatten(), self.series1.values().flatten())
        self.assertEqual(series1_recovered.width, self.series1.width)
        self.assertEqual(series3_recovered, series1_recovered[:1])

    def test_multi_ts_scaling(self):
        transformer1 = Scaler(MinMaxScaler(feature_range=(0, 2)))
        transformer2 = Scaler(StandardScaler())

        series_array = [self.series1, self.series2]

        series_array_tr1 = transformer1.fit_transform(series_array)
        series_array_tr2 = transformer2.fit_transform(series_array)

        for index in range(len(series_array)):
            self.assertAlmostEqual(min(series_array_tr1[index].values().flatten()), 0.)
            self.assertAlmostEqual(max(series_array_tr1[index].values().flatten()), 2.)
            self.assertAlmostEqual(np.mean(series_array_tr2[index].values().flatten()), 0.)
            self.assertAlmostEqual(np.std(series_array_tr2[index].values().flatten()), 1.)

        series_array_rec1 = transformer1.inverse_transform(series_array_tr1)
        series_array_rec2 = transformer2.inverse_transform(series_array_tr2)

        for index in range(len(series_array)):
            np.testing.assert_almost_equal(series_array_rec1[index].values().flatten(),
                                           series_array[index].values().flatten())
            np.testing.assert_almost_equal(series_array_rec2[index].values().flatten(),
                                           series_array[index].values().flatten())
