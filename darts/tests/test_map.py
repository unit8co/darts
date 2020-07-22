import unittest
import numpy as np
import pandas as pd
import logging

from ..timeseries import TimeSeries

class MapTestCase(unittest.TestCase):

    fn = np.sin

    series = TimeSeries.from_times_and_values(pd.date_range('20000101', '20000110'), np.random.randn(10,3))

    df_0 = series.pd_dataframe()
    df_2 = series.pd_dataframe()
    df_01 = series.pd_dataframe()
    df_012 = series.pd_dataframe()

    df_0[[0]] = df_0[[0]].applymap(fn)
    df_2[[2]] = df_2[[2]].applymap(fn)
    df_01[[0,1]] = df_01[[0,1]].applymap(fn)
    df_012 = df_012.applymap(fn)

    series_0 = TimeSeries(df_0, 'D')
    series_2 = TimeSeries(df_2, 'D')
    series_01 = TimeSeries(df_01, 'D')
    series_012 = TimeSeries(df_012, 'D')

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_map(self):
        self.assertEqual(series_0, series.map(fn,0))
        self.assertEqual(series_0, series.map(fn, [0]))
        self.assertEqual(series_2, series.map(fn, 2))
        self.assertEqual(series_01, series.map(fn, [0,1]))
        self.assertEqual(series_012, series.map(fn, [0,1,2]))
        self.assertEqual(series_012, series.map(fn))
