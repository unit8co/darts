import unittest

import numpy as np
import pandas as pd

from darts.dataprocessing import dtw
from darts.metrics import dtw_metric, mae, mape
from darts.tests.base_test_class import DartsBaseTestClass
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg


def _series_from_values(values):
    return TimeSeries.from_values(
        np.array(values) if isinstance(values, list) else values
    )


class DynamicTimeWarpingTestCase(DartsBaseTestClass):
    length = 20
    freq = 1 / length
    series1 = tg.sine_timeseries(
        length=length, value_frequency=freq, value_phase=0, value_y_offset=5
    )
    series2 = tg.sine_timeseries(
        length=length, value_frequency=freq, value_phase=np.pi / 4, value_y_offset=5
    )

    def test_shift(self):
        input1 = [
            1,
            1,
            1,
            1,
            1.2,
            1.4,
            1.2,
            1,
            1,
            1,
            1,
            1,
            1,
            1.2,
            1.4,
            1.6,
            1.8,
            1.6,
            1.4,
            1.2,
            1,
            1,
        ]
        input2 = [1] + input1[:-1]

        expected_path = (
            [(0, 0)]
            + list((i - 1, i) for i in range(1, len(input1)))
            + [(len(input1) - 1, len(input2) - 1)]
        )

        series1 = _series_from_values(input1)
        series2 = _series_from_values(input2)

        exact_alignment = dtw.dtw(series1, series2, multi_grid_radius=-1)

        self.assertEqual(
            exact_alignment.distance(),
            0,
            "Minimum cost between two shifted series should be 0",
        )
        self.assertTrue(
            np.array_equal(exact_alignment.path(), expected_path), "Incorrect path"
        )

    def test_multi_grid(self):
        size = 2**5 - 1  # test odd size
        freq = 1 / size
        input1 = np.cos(np.arange(size) * 2 * np.pi * freq)
        input2 = np.sin(np.arange(size) * 2 * np.pi * freq) + 0.1 * np.random.random(
            size=size
        )

        series1 = _series_from_values(input1)
        series2 = _series_from_values(input2)

        exact_distance = dtw.dtw(series1, series2, multi_grid_radius=-1).distance()
        approx_distance = dtw.dtw(series1, series2, multi_grid_radius=1).distance()

        self.assertAlmostEqual(exact_distance, approx_distance, 3)

    def test_sakoe_chiba_window(self):
        window = 2
        alignment = dtw.dtw(
            self.series1, self.series2, window=dtw.SakoeChiba(window_size=2)
        )
        path = alignment.path()

        for i, j in path:
            self.assertGreaterEqual(window, abs(i - j))

    def test_itakura_window(self):
        n = 6
        m = 5
        slope = 1.5

        window = dtw.Itakura(max_slope=slope)
        window.init_size(n, m)

        cells = list(window)
        self.assertEqual(
            cells,
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (4, 2),
                (4, 3),
                (4, 4),
                (5, 2),
                (5, 3),
                (5, 4),
                (5, 5),
                (6, 4),
                (6, 5),
            ],
        )

        sizes = [(10, 43), (543, 45), (34, 11)]

        for n, m in sizes:
            slope = m / n + 1

            series1 = tg.sine_timeseries(length=n, value_frequency=1 / n, value_phase=0)
            series2 = tg.sine_timeseries(
                length=m, value_frequency=1 / m, value_phase=np.pi / 4
            )

            dist = dtw.dtw(series1, series2, window=dtw.Itakura(slope)).mean_distance()
            self.assertGreater(1, dist)

    def test_warp(self):
        # Support different time dimension names
        xa1 = self.series1.data_array().rename({"time": "time1"})
        xa2 = self.series2.data_array().rename({"time": "time2"})

        static_covs = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])
        series1 = TimeSeries.from_xarray(xa1).with_static_covariates(static_covs)
        series2 = TimeSeries.from_xarray(xa2).with_static_covariates(static_covs)

        alignment = dtw.dtw(series1, series2)

        warped1, warped2 = alignment.warped()
        self.assertAlmostEqual(alignment.mean_distance(), mae(warped1, warped2))
        assert warped1.static_covariates.equals(series1.static_covariates)
        assert warped2.static_covariates.equals(series2.static_covariates)

        """
        See DTWAlignment.warped for why this functionality is currently disabled

        #Mutually Exclusive Option
        with self.assertRaises(ValueError):
            alignment.warped(take_dates=True, range_index=True)

        #Take_dates does not support indexing by RangeIndex
        with self.assertRaises(ValueError):
            xa3 = xa1.copy()
            xa3["time1"] = pd.RangeIndex(0, len(self.series1))

            dtw.dtw(TimeSeries.from_xarray(xa3), series2).warped(take_dates=True)


        warped1, warped2 = alignment.warped(take_dates=True)
        self.assertTrue(np.all(warped1.time_index == warped2.time_index))
        """

    def test_metric(self):
        metric1 = dtw_metric(self.series1, self.series2, metric=mae)
        metric2 = dtw_metric(self.series1, self.series2, metric=mape)

        self.assertGreater(0.5, metric1)
        self.assertGreater(5, metric2)

    def test_nans(self):
        with self.assertRaises(ValueError):
            series1 = _series_from_values([np.nan, 0, 1, 2, 3])
            series2 = _series_from_values([0, 1, 2, 3, 4])

            dtw.dtw(series1, series2)

    def test_plot(self):
        align = dtw.dtw(self.series2, self.series1)
        align.plot()
        align.plot_alignment()

    def test_multivariate(self):
        n = 2

        values1 = np.repeat(self.series1.univariate_values(), n)
        values2 = np.repeat(self.series2.univariate_values(), n)

        values1 = values1.reshape((-1, n))
        values2 = values2.reshape((-1, n))

        multi_series1 = TimeSeries.from_values(values1)
        multi_series2 = TimeSeries.from_values(values2)

        radius = 2

        alignment_uni = dtw.dtw(self.series1, self.series2, multi_grid_radius=radius)
        alignment_multi = dtw.dtw(
            multi_series1, multi_series2, multi_grid_radius=radius
        )

        self.assertTrue(np.all(alignment_uni.path() == alignment_multi.path()))


# MINI_BENCHMARK
def _dtw_exact():
    dtw.dtw(series1, series2, multi_grid_radius=-1).distance()


def _dtw_multigrid():
    dtw.dtw(series1, series2, multi_grid_radius=1).distance()


def _benchmark_dtw():
    size1 = 2**10
    size2 = 2**10
    freq1 = 1 / size1
    freq2 = 1 / size2
    input1 = np.cos(np.arange(size1) * 2 * np.pi * freq1)
    input2 = np.sin(np.arange(size2) * 2 * np.pi * freq2) + 0.1 * np.random.random(
        size=size2
    )

    global series1, series2

    series1 = _series_from_values(input1)
    series2 = _series_from_values(input2)

    import cProfile

    cProfile.run("_dtw_exact()", sort="tottime")
    cProfile.run("_dtw_multigrid()", sort="tottime")


if __name__ == "__main__":
    unittest.main()
