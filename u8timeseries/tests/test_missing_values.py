import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from u8timeseries.utils.missing_values import auto_fillna


class MissingValuesTestCase(unittest.TestCase):

    time = pd.date_range('20130101', '20130130')
    lin = [float(i) for i in range(len(time))]
    cub = [float(i-4)**2 for i in range(len(time))]
    series1: TimeSeries = TimeSeries.from_times_and_values(time, np.array([2.0]*len(time)))
    series2: TimeSeries = TimeSeries.from_times_and_values(time, np.array(lin))
    series3: TimeSeries = TimeSeries.from_times_and_values(time, np.array([10]*10 + lin[-20:]))
    series4: TimeSeries = TimeSeries.from_times_and_values(time, np.array(lin[:20] + [19]*10))
    series5: TimeSeries = TimeSeries.from_times_and_values(time, np.array(cub))
    series6: TimeSeries = TimeSeries.from_times_and_values(time, [0]*2 + cub[2:-2] + [-1]*2)

    def test_fill_constant(self):
        seriesA: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                    np.array([np.nan] * 5 + [2.0] * 5 + [np.nan] * 5 + [2.0] * 10 + [np.nan] * 5))

        # Check that no changes are made if there are no missing values
        self.assertEqual(self.series1, auto_fillna(self.series1))

        # Check that a constant function is filled to a constant function
        self.assertEqual(self.series1, auto_fillna(seriesA))

    def test_linear(self):
        seriesB: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                                               np.array(self.lin[:10] + [np.nan]*10 + self.lin[-10:]))

        # Check for linear interpolation part
        self.assertEqual(self.series2, auto_fillna(seriesB))

    def test_bfill(self):
        seriesC: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                                               np.array([np.nan] * 10 + self.lin[-20:]))

        # Check that auto-backfill works properly
        self.assertEqual(self.series3, auto_fillna(seriesC))
        self.assertNotEqual(self.series3, auto_fillna(seriesC, first=2))

    def test_ffil(self):
        seriesD: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                                               np.array(self.lin[:20] + [np.nan] * 10))

        self.assertEqual(self.series4, auto_fillna(seriesD))
        self.assertNotEqual(self.series4, auto_fillna(seriesD, last=20))

    def test_fill_quad(self):
        seriesE: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                                               np.array(self.cub[:10] + [np.nan]*10 + self.cub[-10:]))
        seriesF: TimeSeries = TimeSeries.from_times_and_values(self.time,
                                                               np.array([np.nan]*2 + self.cub[2:10] +
                                                                        [np.nan]*10 + self.cub[-10:-2] + [np.nan]*2))

        self.assertEqual(self.series5, round(auto_fillna(seriesE, interpolate='quadratic'), 7))
        self.assertEqual(self.series6, round(auto_fillna(seriesF, first=0, last=-1, interpolate='quadratic'), 2))
        # extrapolate values outside
        self.assertEqual(self.series5,
                         round(auto_fillna(seriesF, interpolate='quadratic', fill_value='extrapolate'), 2))

