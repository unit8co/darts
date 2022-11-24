import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from more_itertools import always_iterable

from darts.dataprocessing.transformers import Diff
from darts.timeseries import TimeSeries
from darts.timeseries import concatenate as darts_concat
from darts.utils.timeseries_generation import linear_timeseries, sine_timeseries


class DiffTestCase(unittest.TestCase):
    sine_series = [
        5 * sine_timeseries(length=50, value_frequency=f) for f in (0.05, 0.1, 0.15)
    ]
    sine_series = darts_concat(
        sine_series,
        axis=1,
    )

    def assert_series_equal(self, series1, series2, equal_nan, to_compare=None):
        if to_compare is not None:
            series1 = series1.drop_columns(series1.columns[~to_compare])
            series2 = series2.drop_columns(series2.columns[~to_compare])
        np.testing.assert_allclose(
            series1.all_values(), series2.all_values(), atol=1e-8, equal_nan=equal_nan
        )
        self.assertTrue(series1.time_index.equals(series2.time_index))

    def test_diff_quad_series(self):
        # Cumulative summing linear series = quadratic series; test that
        # differencing quadratic series = linear series
        lin_series = linear_timeseries(start_value=1, end_value=10, length=50)
        quad_series = TimeSeries(lin_series.data_array().cumsum(axis=0))
        for dropna in (False, True):
            diff = Diff(lags=1, dropna=dropna)
            diff.fit(quad_series)
            # Differencing causes first point to be 'dropped':
            transformed_series = diff.transform(quad_series)
            expected_transform = lin_series.drop_before(0)
            if not dropna:
                expected_transform = expected_transform.prepend_values([np.nan])
            self.assert_series_equal(
                expected_transform, transformed_series, equal_nan=~dropna
            )
            self.assert_series_equal(
                quad_series,
                diff.inverse_transform(transformed_series),
                equal_nan=~dropna,
            )

    def test_diff_inverse_transform_beyond_fit_data(self):
        # Test that Diff class can 'undifference' data that extends beyond
        # what was used to fit it (e.g. like when undifferencing a forecast of a differenced timeseries)

        test_cases = [
            (1, True),
            (1, False),
            ([1, 2, 3, 2, 1], True),
            ([1, 2, 3, 2, 1], False),
        ]

        short_sine = self.sine_series.copy().drop_after(10)
        for (lags, dropna) in test_cases:
            # Fit Diff to subsection of data:
            diff = Diff(lags=lags, dropna=dropna)
            diff.fit(short_sine)
            # Difference entire time series:
            to_undiff = self.sine_series.copy()
            for lag in always_iterable(lags):
                to_undiff = to_undiff.diff(n=1, periods=lag, dropna=dropna)
            # Should be able to undifference entire series even though only fitted on start:
            self.assert_series_equal(
                self.sine_series, diff.inverse_transform(to_undiff), equal_nan=~dropna
            )

    def test_diff_multi_ts(self):
        # Don't test dropna = True AND mask is not None, since this should throw error (see):
        component_mask = np.array([True, False, True])
        test_cases = [
            (1, True, None),
            (1, False, None),
            ([1, 2, 3, 2, 1], True, None),
            ([1, 2, 3, 2, 1], False, None),
            (1, False, component_mask),
            ([1, 2, 3, 2, 1], False, component_mask),
        ]
        for (lags, dropna, mask) in test_cases:
            diff = Diff(lags=lags, dropna=dropna)
            transformed = diff.fit_transform(
                [self.sine_series, self.sine_series], component_mask=mask
            )
            if mask is not None:
                # Masked components should be undifferenced:
                self.assert_series_equal(
                    self.sine_series,
                    transformed[0],
                    equal_nan=~dropna,
                    to_compare=~mask,
                )
                self.assert_series_equal(
                    self.sine_series,
                    transformed[1],
                    equal_nan=~dropna,
                    to_compare=~mask,
                )
            # Should recover original sine_series:
            back = diff.inverse_transform(transformed, component_mask=mask)
            self.assert_series_equal(self.sine_series, back[0], equal_nan=~dropna)
            self.assert_series_equal(self.sine_series, back[1], equal_nan=~dropna)

    def test_diff_stochastic_series(self):
        test_cases = [
            (1, True),
            (1, False),
            ([1, 2, 3, 2, 1], True),
            ([1, 2, 3, 2, 1], False),
        ]

        vals = np.random.rand(10, 5, 10)
        series = TimeSeries.from_values(vals)

        for (lags, dropna) in test_cases:
            transformer = Diff(lags=lags, dropna=dropna)
            new_series = transformer.fit_transform(series)
            series_back = transformer.inverse_transform(new_series)
            self.assert_series_equal(series, series_back, equal_nan=~dropna)

    def test_diff_dropna_and_component_mask_specified(self):
        diff = Diff(lags=1, dropna=True)
        with self.assertRaises(ValueError) as e:
            diff.fit(self.sine_series, component_mask=np.array([1, 0, 1], dtype=bool))
        self.assertEqual(
            "Cannot specify `component_mask` with `dropna = True`.",
            str(e.exception),
        )

    def test_diff_series_too_short(self):
        lags = tuple(always_iterable(1000))
        diff = Diff(lags=lags)
        with self.assertRaises(ValueError) as e:
            diff.fit(self.sine_series)
        self.assertEqual(
            (
                f"Series requires at least {sum(lags) + 1} timesteps "
                f"to difference with lags {lags}; series only "
                f"has {self.sine_series.n_timesteps} timesteps."
            ),
            str(e.exception),
        )

    def test_diff_incompatible_inverse_transform_date(self):
        vals = np.random.rand(10, 5)
        series1 = TimeSeries.from_times_and_values(
            values=vals, times=pd.date_range(start="1/1/2018", freq="d", periods=10)
        )
        series2 = TimeSeries.from_times_and_values(
            values=vals, times=pd.date_range(start="1/2/2018", freq="d", periods=10)
        )
        for dropna in (False, True):
            diff = Diff(lags=1, dropna=dropna)
            diff.fit(series1)
            series2_diffed = series2.diff(n=1, periods=1, dropna=dropna)
            with self.assertRaises(ValueError) as e:
                diff.inverse_transform(series2_diffed)
            expected_start = (
                series1.start_time()
                if not dropna
                else series1.start_time() + series1.freq
            )
            self.assertEqual(
                (
                    f"Expected series to begin at time {expected_start}; "
                    f"instead, it begins at time {series2_diffed.start_time()}."
                ),
                str(e.exception),
            )

    def test_diff_incompatible_inverse_transform_freq(self):
        vals = np.random.rand(10, 5)
        series1 = TimeSeries.from_times_and_values(
            values=vals, times=pd.date_range(start="1/1/2018", freq="W", periods=10)
        )
        series2 = TimeSeries.from_times_and_values(
            values=vals, times=pd.date_range(start="1/1/2018", freq="M", periods=10)
        )
        diff = Diff(lags=1, dropna=True)
        diff.fit(series1)
        with self.assertRaises(ValueError) as e:
            diff.inverse_transform(series2.diff(n=1, periods=1, dropna=True))
        self.assertEqual(
            f"Series is of frequency {series2.freq}, but transform was fitted to data of frequency {series1.freq}.",
            str(e.exception),
        )

    def test_diff_incompatible_inverse_transform_shape(self):
        vals = np.random.rand(10, 5, 5)
        dates = pd.date_range(start="1/1/2018", freq="W", periods=10)
        series = TimeSeries.from_times_and_values(values=vals, times=dates)
        diff = Diff(lags=1, dropna=True)
        diff.fit(series)
        series_rm_comp = TimeSeries.from_times_and_values(
            values=vals[:, 1:, :], times=dates
        )
        with self.assertRaises(ValueError) as e:
            diff.inverse_transform(series_rm_comp.diff(n=1, periods=1, dropna=True))
        self.assertEqual(
            f"Expected series to have {series.n_components} components; "
            f"instead, it has {series.n_components-1}.",
            str(e.exception),
        )
        series_rm_samp = TimeSeries.from_times_and_values(
            values=vals[:, :, 1:], times=dates
        )
        with self.assertRaises(ValueError) as e:
            diff.inverse_transform(series_rm_samp.diff(n=1, periods=1, dropna=True))
        self.assertEqual(
            f"Expected series to have {series.n_samples} samples; "
            f"instead, it has {series.n_samples-1}.",
            str(e.exception),
        )

    def test_diff_multiple_calls_to_fit(self):
        """
        This test checks whether calling the scaler twice is calculating new lambdas instead of
        keeping the old ones
        """
        diff = Diff(lags=[2, 2], dropna=True)

        diff.fit(self.sine_series)
        startvals1 = deepcopy(diff._fitted_params)[0][0]

        diff.fit(self.sine_series + 1)
        startvals2 = deepcopy(diff._fitted_params)[0][0]

        self.assertFalse(np.allclose(startvals1, startvals2))
