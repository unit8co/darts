from darts.timeseries import TimeSeries
from darts.metrics import mape

import pandas as pd
import numpy as np

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg
from darts.dataprocessing import dtw
from darts.metrics import mae, dtw_mae, dtw_metric
import unittest

from matplotlib import pyplot as plt

import time

def _series_from_values(input):
    start = "20021001"
    series = TimeSeries.from_times_and_values(pd.date_range(start, freq="d", periods=len(input)), input)
    return series

class DynamicTimeWarpingTestCase(DartsBaseTestClass):
    def test_shift(self):
        input1 = [1, 1, 1, 1, 1.2, 1.4, 1.2, 1, 1, 1, 1, 1, 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1, 1]
        input2 = [1] + input1[:-1]

        expected_path = [(0,0)] + list((i-1,i) for i in range(1,len(input1))) + [(len(input1)-1, len(input2)-1)]

        series1 = _series_from_values(input1)
        series2 = _series_from_values(input2)

        exact_alignment = dtw.dtw(series1, series2, multigrid_radius=-1)

        self.assertEqual(exact_alignment.distance(), 0, "Minimum cost between two shifted series should be 0")
        self.assertTrue(np.array_equal(exact_alignment.path(), expected_path), "Incorrect path")

    def test_multi_grid(self):
        size = 2**5 # test odd sizes
        freq = 1 / size
        input1 = np.cos(np.arange(size) * 2 * np.pi * freq)
        input2 = np.sin(np.arange(size) * 2 * np.pi * freq) + 0.1 * np.random.random(size=size)

        series1 = _series_from_values(input1)
        series2 = _series_from_values(input2)

        exact_distance = dtw.dtw(series1, series2, multigrid_radius=-1).distance()
        approx_distance = dtw.dtw(series1, series2, multigrid_radius=1).distance()

        self.assertAlmostEqual(exact_distance, approx_distance, 3)

    def test_sakoe_chiba_window(self):
        length = 20
        freq = 1/length
        series1 = tg.sine_timeseries(length=length, value_frequency=freq, value_phase=0)
        series2 = tg.sine_timeseries(length=length, value_frequency=freq, value_phase=np.pi/4)

        window = 2
        alignment = dtw.dtw(series1, series2, window= dtw.SakoeChiba(window_size= 2))
        path = alignment.path()

        for i, j in path:
            self.assertGreaterEqual(window, abs(i - j))

    def test_itakura_window(self):
        n = 5
        m = 6
        slope = 1.5

        window = dtw.Itakura(max_slope=slope)
        window.init_size(n, m)

        cells = list(window)
        self.assertEqual(cells, [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (4, 3), (5, 4)])

    def test_metric(self):
        length = 20
        freq = 1 / length
        series1 = tg.sine_timeseries(length=length, value_frequency=freq, value_phase=0)
        series2 = tg.sine_timeseries(length=length, value_frequency=freq, value_phase=np.pi / 4)

        mae1 = dtw_mae(series1, series2)
        mae2 = dtw_metric(series1, series2, metric=mae)

        self.assertGreater(0.5, mae1)
        self.assertAlmostEqual(mae1, mae2)


def _dtw_exact():
    dtw.dtw(series1, series2, multigrid_radius=-1).distance()


def _dtw_multigrid():
    dtw.dtw(series1, series2, multigrid_radius=1).distance()


def _benchmark_dtw():
    size = 2 ** 5
    freq = 1 / size
    input1 = np.cos(np.arange(size) * 2 * np.pi * freq)
    input2 = np.sin(np.arange(size) * 2 * np.pi * freq) + 0.1 * np.random.random(size=size)

    global series1, series2

    series1 = _series_from_values(input1)
    series2 = _series_from_values(input2)

    align = dtw.dtw(series1, series2, multigrid_radius=2)
    align.plot()

    plt.show()
    import cProfile
    cProfile.run("_dtw_exact()", sort="tottime")
    cProfile.run("_dtw_multigrid()", sort="tottime")

if __name__ == '__main__':
    unittest.main()
