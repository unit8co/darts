import itertools
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts import concatenate as darts_concat
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Diff
from darts.dataprocessing.transformers.scaler import Scaler
from darts.datasets import AirPassengersDataset
from darts.utils.timeseries_generation import linear_timeseries, sine_timeseries


class TestDiff:
    sine_series = [
        5 * sine_timeseries(length=50, value_frequency=f) for f in (0.05, 0.1, 0.15)
    ]
    sine_series = darts_concat(
        sine_series,
        axis=1,
    )

    def assert_series_equal(
        self,
        series1: TimeSeries,
        series2: TimeSeries,
        equal_nan: bool,
        to_compare: np.ndarray | None = None,
    ):
        """
        Helper to compare series differenced by `Diff`.

        Parameters
        ----------
            series1
                First `TimeSeries` object to compare.
            series2
                Second `TimeSeries` object to compare.
            equal_nan
                Whether to compare `NaN` values as equal (see `help(np.testing.assert_allclose)`); should set
                `equal_nan=(not dropna)` (i.e. should throw error if there are `NaN` values when user has requested
                for them to be dropped).
            to_compare
                boolean `np.ndarray` which specifies which subset of columns should be compared in `series1` and
                `series2`. Used when checking that components not specified by `component_mask` have NOT been
                differenced after `fit`ing (i.e. `to_compare = ~component_mask`). If `to_compare = None`, all
                columns in `series1` and `series2` are compared.
        """
        if to_compare is not None:
            series1 = series1.drop_columns(series1.columns[~to_compare])
            series2 = series2.drop_columns(series2.columns[~to_compare])
        np.testing.assert_allclose(
            series1.all_values(), series2.all_values(), atol=1e-8, equal_nan=equal_nan
        )
        assert series1.time_index.equals(series2.time_index)

    def test_diff_quad_series(self):
        """
        Tests that differencing a quadratic series (computed by `cumsum`ing a linear series)
        yields the original linear series.
        """
        lin_series = linear_timeseries(start_value=1, end_value=10, length=50)
        quad_series = TimeSeries(
            times=lin_series.time_index,
            values=lin_series.all_values().cumsum(axis=0),
            components=lin_series.components,
        )
        for dropna in (False, True):
            diff = Diff(lags=1, dropna=dropna)
            diff.fit(quad_series)
            # Differencing causes first point to be 'dropped':
            transformed_series = diff.transform(quad_series)
            expected_transform = lin_series.drop_before(0)
            if not dropna:
                expected_transform = expected_transform.prepend_values([np.nan])
            self.assert_series_equal(
                expected_transform, transformed_series, equal_nan=(not dropna)
            )
            self.assert_series_equal(
                series1=quad_series,
                series2=diff.inverse_transform(transformed_series),
                equal_nan=(not dropna),
            )

    def test_diff_inverse_transform_beyond_fit_data(self):
        """
        Tests that `Diff` class can 'undifference' data that extends beyond
         what was used to fit it (e.g. like when undifferencing a forecast of
         a differenced timeseries, so as to yield a forecast of the undifferenced
         time series)
        """

        # (`lag`, `dropna`) pairs:
        test_cases = [
            (1, True),
            (1, False),
            ([1, 2, 3, 2, 1], True),
            ([1, 2, 3, 2, 1], False),
        ]

        # Artifically truncate series:
        short_sine = self.sine_series.copy().drop_after(10)
        short_sine_copy = short_sine.copy()
        for lags, dropna in test_cases:
            # Fit Diff to truncated series:
            diff = Diff(lags=lags, dropna=dropna)
            diff.fit(short_sine)
            assert short_sine == short_sine_copy

            # Difference entire time series:
            to_undiff = self.sine_series.copy()
            if not isinstance(lags, Sequence):
                lags = (lags,)
            for lag in lags:
                to_undiff = to_undiff.diff(n=1, periods=lag, dropna=dropna)

            to_undiff_copy = to_undiff
            # Should be able to undifference entire series even though only fitted
            # to truncated series:
            self.assert_series_equal(
                series1=self.sine_series,
                series2=diff.inverse_transform(to_undiff),
                equal_nan=(not dropna),
            )
            assert to_undiff == to_undiff_copy

    def test_diff_multi_ts(self):
        """
        Tests that `Diff` correctly behaves when given multiple time series, both when a `component_mask`
        is provided and when one is not.
        """

        # (`lags`, `dropna`, `component_mask`) triplets; note that combination of `dropna = True``
        # AND `component_mask is not None` is not tested here, since this should throw error
        # (see `test_diff_dropna_and_component_mask_specified`):
        component_mask = np.array([True, False, True])
        test_cases = [
            (1, True, None),
            (1, False, None),
            ([1, 2, 3, 2, 1], True, None),
            ([1, 2, 3, 2, 1], False, None),
            (1, False, component_mask),
            ([1, 2, 3, 2, 1], False, component_mask),
        ]
        for lags, dropna, mask in test_cases:
            diff = Diff(lags=lags, dropna=dropna)
            transformed = diff.fit_transform(
                [self.sine_series, self.sine_series], component_mask=mask
            )
            if mask is not None:
                # Masked components should be undifferenced:
                self.assert_series_equal(
                    self.sine_series,
                    transformed[0],
                    equal_nan=(not dropna),
                    to_compare=~mask,
                )
                self.assert_series_equal(
                    self.sine_series,
                    transformed[1],
                    equal_nan=(not dropna),
                    to_compare=~mask,
                )
            # Should recover original sine_series:
            back = diff.inverse_transform(transformed, component_mask=mask)
            self.assert_series_equal(self.sine_series, back[0], equal_nan=(not dropna))
            self.assert_series_equal(self.sine_series, back[1], equal_nan=(not dropna))

    @pytest.mark.parametrize("component_mask", [None, np.array([True] * 5)])
    def test_diff_stochastic_series(self, component_mask):
        """
        Tests that `Diff` class correctly differences and then undifferences a
        random series with multiple samples.
        """
        test_cases = [
            (1, True),
            (1, False),
            ([1, 2, 3, 2, 1], True),
            ([1, 2, 3, 2, 1], False),
        ]

        vals = np.random.rand(10, 5, 10)
        series = TimeSeries.from_values(vals)
        series_copy = series.copy()

        for lags, dropna in test_cases:
            component_mask = component_mask if not dropna else None
            transformer = Diff(lags=lags, dropna=dropna)
            new_series = transformer.fit_transform(
                series, component_mask=component_mask
            )
            assert series == series_copy

            new_series_copy = new_series.copy()
            series_back = transformer.inverse_transform(
                new_series, component_mask=component_mask
            )

            # Should recover original series:
            self.assert_series_equal(series, series_back, equal_nan=(not dropna))
            assert new_series == new_series_copy

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [True, False],
            [[1], [1, 2]],
        ),
    )
    def test_diff_with_component_mask_or_columns(self, config):
        """
        Tests that `Diff` works with columns or component masks in combination with other parameters.
        """
        dropna, mask_components, lags = config

        mask = np.array([1, 0, 1], dtype=bool)

        kwargs = (
            dict(columns=self.sine_series.columns[mask])
            if not mask_components
            else dict()
        )
        tf_kwargs = dict(component_mask=mask) if mask_components else dict()
        diff = Diff(lags=lags, dropna=dropna, **kwargs)

        series_tf = diff.fit_transform(self.sine_series, **tf_kwargs)

        vals_orig, vals_tf = self.sine_series.values(), series_tf.values()
        vals_tf_slice = slice(None) if dropna else slice(sum(lags), None)

        # non-transformed columns must be equal
        np.testing.assert_array_almost_equal(
            vals_tf[vals_tf_slice, ~mask], vals_orig[sum(lags) :, ~mask]
        )

        # transformed columns must be diffed
        vals_expected = vals_orig.copy()[:, mask]
        for idx, lag in enumerate(lags):
            vals_expected = vals_expected[lag:] - vals_expected[:-lag]
        np.testing.assert_array_almost_equal(
            vals_tf[vals_tf_slice, mask], vals_expected
        )

        # inverse transformed must be equal to original values
        series_inv_tf = diff.inverse_transform(series_tf, **tf_kwargs)
        np.testing.assert_array_almost_equal(series_inv_tf.values(), vals_orig)

    def test_diff_series_too_short(self):
        """
        Tests that `Diff` throws error is length of series is less than `sum(lags)` (i.e.
        there's not enough data to fit the differencer)
        """
        lags = (1000,)
        diff = Diff(lags=lags)
        with pytest.raises(ValueError) as e:
            diff.fit(self.sine_series)
        assert (
            f"Series requires at least {sum(lags) + 1} timesteps "
            f"to difference with lags {lags}; series only "
            f"has {self.sine_series.n_timesteps} timesteps."
        ) == str(e.value)

    def test_diff_incompatible_inverse_transform_date(self):
        """
        Tests that `Diff` throws error when given series to `inverse_transform`
        which starts at a date not equal to the first data of the fitting data.
        """
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
            with pytest.raises(ValueError) as e:
                diff.inverse_transform(series2_diffed)
            expected_start = (
                series1.start_time()
                if (not dropna)
                else series1.start_time() + series1.freq
            )
            assert (
                f"Expected series to begin at time {expected_start}; "
                f"instead, it begins at time {series2_diffed.start_time()}."
            ) == str(e.value)

    def test_diff_incompatible_inverse_transform_freq(self):
        """
        Tests that `Diff` throws error when given series to `inverse_transform`
        that has different frequency than fitting data.
        """
        vals = np.random.rand(10, 5)
        series1 = TimeSeries.from_times_and_values(
            values=vals, times=pd.date_range(start="1/1/2018", freq="W", periods=10)
        )
        series2 = TimeSeries.from_times_and_values(
            values=vals,
            times=pd.date_range(start="1/1/2018", freq="ME", periods=10),
        )
        diff = Diff(lags=1, dropna=True)
        diff.fit(series1)
        with pytest.raises(ValueError) as e:
            diff.inverse_transform(series2.diff(n=1, periods=1, dropna=True))
        assert (
            f"Series is of frequency {series2.freq}, but transform was fitted to data of frequency {series1.freq}."
            == str(e.value)
        )

    def test_diff_incompatible_inverse_transform_shape(self):
        """
        Tests that `Diff` throws error when given series to `inverse_transform`
        that has different number of components or samples than the fitting data.
        """
        vals = np.random.rand(10, 5, 5)
        dates = pd.date_range(start="1/1/2018", freq="W", periods=10)
        series = TimeSeries.from_times_and_values(values=vals, times=dates)
        diff = Diff(lags=1, dropna=True)
        diff.fit(series)
        series_rm_comp = TimeSeries.from_times_and_values(
            values=vals[:, 1:, :], times=dates
        )
        with pytest.raises(ValueError) as e:
            diff.inverse_transform(series_rm_comp.diff(n=1, periods=1, dropna=True))
        assert (
            f"Expected series to have {series.n_components} components; "
            f"instead, it has {series.n_components - 1}." == str(e.value)
        )
        series_rm_samp = TimeSeries.from_times_and_values(
            values=vals[:, :, 1:], times=dates
        )
        with pytest.raises(ValueError) as e:
            diff.inverse_transform(series_rm_samp.diff(n=1, periods=1, dropna=True))
        assert (
            f"Expected series to have {series.n_samples} samples; "
            f"instead, it has {series.n_samples - 1}." == str(e.value)
        )

    def test_diff_multiple_calls_to_fit(self):
        """
        Tests that `Diff` updates `start_vals` parameter when refitted to new data.
        """
        diff = Diff(lags=[2, 2], dropna=True)

        diff.fit(self.sine_series)
        startvals1 = deepcopy(diff._fitted_params)[0][0]

        diff.fit(self.sine_series + 1)
        startvals2 = deepcopy(diff._fitted_params)[0][0]

        assert not np.allclose(startvals1, startvals2)


class TestDiffInsample:
    """Tests for `Diff.inverse_transform(..., insample=...)` using AirPassengersDataset."""

    @pytest.fixture(autouse=True)
    def _load_air_passengers(self):
        """Load AirPassengers once per test; split into train/test halves."""
        full = AirPassengersDataset().load()
        # Split at the halfway point along the time axis.
        split = len(full) // 2
        self.train = full[:split]
        self.test = full[split:]
        self.full = full
        self.train_end = self.train.end_time()

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    @staticmethod
    def _split_diff(full_diff: TimeSeries, train_end) -> tuple[TimeSeries, TimeSeries]:
        """Split a differenced series at `train_end` using the original time boundary.

        Returns ``(train_diff, test_diff)`` where:
        - ``train_diff`` covers everything up to and including ``train_end``.
        - ``test_diff`` covers everything strictly after ``train_end``.
        """
        train_diff = full_diff.drop_after(train_end, keep_point=True)
        test_diff = full_diff.drop_before(train_end)
        return train_diff, test_diff

    # ------------------------------------------------------------------
    # 1. Parity: insample path == direct full-series path (sliced to test)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "lags,dropna",
        [
            (1, True),
            (1, False),
            ([1, 2], True),
            ([1, 2], False),
            ([1, 12], True),
        ],
    )
    def test_insample_parity_with_full_series(self, lags, dropna):
        """
        Inverting the test-half diff with insample=train_half must give the same
        result as inverting the full differenced series and slicing the test window.
        """
        diff = Diff(lags=lags, dropna=dropna)
        full_diff = diff.fit_transform(self.full)

        train_diff, test_diff = self._split_diff(full_diff, self.train_end)

        # Baseline: full inverse then slice to test window.
        result_full = diff.inverse_transform(full_diff)
        expected = result_full.drop_before(self.test.start_time(), keep_point=True)

        # New insample path.
        actual = diff.inverse_transform(test_diff, insample=train_diff)

        np.testing.assert_allclose(
            expected.all_values(),
            actual.all_values(),
            atol=1e-8,
            equal_nan=True,
        )
        assert expected.time_index.equals(actual.time_index)

    # ------------------------------------------------------------------
    # 2. Forecast-only inverse: result matches the original level series
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "lags,dropna",
        [
            (1, True),
            (1, False),
            ([1, 2], True),
            ([1, 2], False),
            ([1, 12], True),
            ([1, 12], False),
        ],
    )
    def test_insample_forecast_recovers_original(self, lags, dropna):
        """
        Diff.inverse_transform(test_diff, insample=train_diff) must recover
        the original level test series exactly, without any manual concatenation.
        """
        diff = Diff(lags=lags, dropna=dropna)
        # Fit only on the training half; transform the entire series in diff-space.
        diff.fit(self.train)
        full_diff = diff.transform(self.full)

        train_diff, test_diff = self._split_diff(full_diff, self.train_end)

        recovered = diff.inverse_transform(test_diff, insample=train_diff)

        np.testing.assert_allclose(
            self.test.all_values(),
            recovered.all_values(),
            atol=1e-6,
            equal_nan=True,
        )
        assert self.test.time_index.equals(recovered.time_index)

    # ------------------------------------------------------------------
    # 3. Equivalence: insample path == old manual-concat workaround
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("lags,dropna", [(1, True), ([1, 12], True)])
    def test_insample_equivalent_to_manual_concat(self, lags, dropna):
        """
        The insample path must produce exactly the same result as the manual
        workaround: concatenate(train_diff, test_diff) then inverse_transform,
        sliced back to the test window.
        """
        diff = Diff(lags=lags, dropna=dropna)
        diff.fit(self.train)
        full_diff = diff.transform(self.full)

        train_diff, test_diff = self._split_diff(full_diff, self.train_end)

        # Old manual workaround.
        manual_concat = darts_concat([train_diff, test_diff], axis=0)
        result_manual = diff.inverse_transform(manual_concat)
        result_manual = result_manual.drop_before(
            self.test.start_time(), keep_point=True
        )

        # New insample path.
        result_insample = diff.inverse_transform(test_diff, insample=train_diff)

        np.testing.assert_allclose(
            result_manual.all_values(),
            result_insample.all_values(),
            atol=1e-8,
        )
        assert result_manual.time_index.equals(result_insample.time_index)

    # ------------------------------------------------------------------
    # 4. Component mask respected
    # ------------------------------------------------------------------
    def test_insample_with_component_mask(self):
        """
        When a component_mask is supplied only the masked components are
        inverse-differenced; unmasked components must pass through unchanged.
        """
        multi = darts_concat([AirPassengersDataset().load()] * 3, axis=1)
        mask = np.array([True, False, True])

        diff = Diff(lags=1, dropna=False)
        transformed = diff.fit_transform(multi, component_mask=mask)

        train_end = multi[: len(multi) // 2].end_time()
        train_diff, test_diff = self._split_diff(transformed, train_end)

        recovered = diff.inverse_transform(
            test_diff, component_mask=mask, insample=train_diff
        )

        np.testing.assert_allclose(
            multi[len(multi) // 2 :].all_values(),
            recovered.all_values(),
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # 5. Validation errors
    # ------------------------------------------------------------------
    def test_insample_wrong_freq_raises(self):
        """insample with a different frequency must raise ValueError."""
        diff = Diff(lags=1, dropna=True)
        full_diff = diff.fit_transform(self.full)
        _, test_diff = self._split_diff(full_diff, self.train_end)

        # Build an insample-shaped series with quarterly frequency (wrong).
        bad_insample = TimeSeries.from_times_and_values(
            times=pd.date_range(start=full_diff.start_time(), periods=10, freq="QE"),
            values=np.zeros((10, 1)),
        )
        with pytest.raises(ValueError, match="insample is of frequency"):
            diff.inverse_transform(test_diff, insample=bad_insample)

    def test_insample_wrong_start_raises(self):
        """insample that does not start at expected_start must raise ValueError."""
        diff = Diff(lags=1, dropna=True)
        train_diff = diff.fit_transform(self.train)
        full_diff = diff.transform(self.full)
        _, test_diff = self._split_diff(full_diff, self.train_end)

        # Trim one step from the front so start_time shifts by one period.
        bad_insample = train_diff.drop_before(train_diff.start_time())

        with pytest.raises(
            ValueError, match="Expected insample to begin at or before time"
        ):
            diff.inverse_transform(test_diff, insample=bad_insample)

    def test_insample_too_short_raises(self):
        """insample must contain at least one timestep strictly before the forecast.

        We trigger this by making the forecast start exactly at expected_start, so
        that drop_after(forecast_start) removes all of insample.
        """
        diff = Diff(lags=1, dropna=True)
        full_diff = diff.fit_transform(self.full)
        expected_start = full_diff.start_time()

        # Forecast that starts at expected_start — no room for suffix in insample.
        forecast_at_expected_start = TimeSeries.from_times_and_values(
            times=pd.date_range(start=expected_start, periods=5, freq=full_diff.freq),
            values=np.zeros((5, 1)),
        )
        with pytest.raises(ValueError, match="insample must contain at least one"):
            diff.inverse_transform(forecast_at_expected_start, insample=full_diff)

    # ------------------------------------------------------------------
    # 6. Pipeline: [Diff → Scaler] — insample threads through both stages
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("lags,dropna", [(1, True), ([1, 12], True)])
    def test_pipeline_insample(self, lags, dropna):
        """
        Pipeline([Diff, Scaler]).inverse_transform(test_tf, insample=train_tf)
        must recover the original level test series end-to-end.
        """
        pipeline = Pipeline([Diff(lags=lags, dropna=dropna), Scaler()])
        full_transformed = pipeline.fit_transform(self.full)

        # Use the original train/test time boundary to split the transformed output.
        train_tf, test_tf = self._split_diff(full_transformed, self.train_end)

        recovered = pipeline.inverse_transform(test_tf, insample=train_tf)

        np.testing.assert_allclose(
            self.test.all_values(),
            recovered.all_values(),
            atol=1e-4,
        )
        assert self.test.time_index.equals(recovered.time_index)

    # ------------------------------------------------------------------
    # 7. Stochastic (multi-sample) series
    # ------------------------------------------------------------------
    def test_insample_stochastic(self):
        """insample path must work correctly for stochastic (multi-sample) series."""
        rng = np.random.default_rng(42)
        n_samples = 20
        base = AirPassengersDataset().load()

        # Build a stochastic series: base level + independent noise per sample.
        base_vals = base.all_values()  # (T, 1, 1)
        # Broadcast to (T, 1, n_samples) then add per-sample noise.
        stochastic_vals = np.broadcast_to(base_vals, (len(base), 1, n_samples)).copy()
        stochastic_vals += rng.normal(0, 3.0, stochastic_vals.shape)
        stochastic = TimeSeries.from_times_and_values(
            times=base.time_index,
            values=stochastic_vals,
        )

        train_end = stochastic[: len(stochastic) // 2].end_time()

        diff = Diff(lags=1, dropna=True)
        diff.fit(stochastic[: len(stochastic) // 2])
        full_diff = diff.transform(stochastic)

        train_diff, test_diff = self._split_diff(full_diff, train_end)

        recovered = diff.inverse_transform(test_diff, insample=train_diff)

        np.testing.assert_allclose(
            stochastic[len(stochastic) // 2 :].all_values(),
            recovered.all_values(),
            atol=1e-5,
        )
        assert stochastic[len(stochastic) // 2 :].time_index.equals(
            recovered.time_index
        )
