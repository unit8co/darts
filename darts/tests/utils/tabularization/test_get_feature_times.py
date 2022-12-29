import warnings
from itertools import product
from typing import Optional

import pandas as pd

from darts import TimeSeries
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import get_feature_times
from darts.utils.timeseries_generation import linear_timeseries


class GetFeatureTimesTestCase(DartsBaseTestClass):
    """
    Tests `get_feature_times` function defined in `darts.utils.data.tabularization`.
    """

    @staticmethod
    def get_feature_times(
        series: TimeSeries,
        min_lag: Optional[int],
        max_lag: Optional[int],
        output_chunk_length: Optional[int] = None,
        is_target: bool = False,
    ):
        """
        Helper function that returns all the times within a *single series* that can be used to
        create features and labels. This means that the `series` and `*_lag` inputs must correspond
        to *either* the `target_series`, the `past_covariates`, *or*  the `future_covariates`.

        If `series` is `target_series` (i.e. `is_target = True`), then:
            - The last `output_chunk_length - 1` times are exluded, since these times do not
            have `(output_chunk_length - 1)` values ahead of them and, therefore, we can't
            create labels for these values.
            - The first `max_lag` values are excluded, since these values don't have `max_lag`
            values preceeding them, which means that we can't create features for these times.

        If `series` is either `past_covariates` or `future_covariates` (i.e. `is_target = False`),
        then the first `max_lag` times are excluded, since these values don't have `max_lag` values
        preceeding them, which means that we can't create features for these times.
        """
        times = series.time_index
        # Assume `max_lag` is specified if `min_lag` is specified:
        lags_specified = min_lag is not None
        if not is_target and lags_specified:
            times = times.union(
                [times[-1] + i * series.freq for i in range(1, min_lag + 1)]
            )
        if lags_specified:
            times = times[max_lag:]
        if is_target and (output_chunk_length > 1):
            times = times[: -output_chunk_length + 1]
        return times

    def test_feature_times_training_range_idx(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags_combos = (
            {"vals": [-1], "max": 1, "min": 1},
            {"vals": [-2, -1], "max": 2, "min": 1},
            {"vals": [-6, -4, -3], "max": 6, "min": 3},
            {"vals": [-4, -6, -3], "max": 6, "min": 3},
        )
        ocl_combos = (1, 2, 5, 10)
        for (lags, lags_past, lags_future, ocl) in product(
            lags_combos, lags_combos, lags_combos, ocl_combos
        ):
            feature_times = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
                output_chunk_length=ocl,
                is_training=True,
            )
            target_expected = self.get_feature_times(
                target, lags["min"], lags["max"], ocl, is_target=True
            )
            past_expected = self.get_feature_times(
                past, lags_past["min"], lags_past["max"]
            )
            future_expected = self.get_feature_times(
                future, lags_future["min"], lags_future["max"]
            )
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_feature_times_training_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="1d")
        past = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=25, freq="2d")
        future = linear_timeseries(start=pd.Timestamp("1/3/2000"), length=30, freq="3d")
        lags_combos = (
            {"vals": [-1], "max": 1, "min": 1},
            {"vals": [-2, -1], "max": 2, "min": 1},
            {"vals": [-6, -4, -3], "max": 6, "min": 3},
            {"vals": [-4, -6, -3], "max": 6, "min": 3},
        )
        ocl_combos = (1, 2, 5, 10)
        for (lags, lags_past, lags_future, ocl) in product(
            lags_combos, lags_combos, lags_combos, ocl_combos
        ):
            feature_times = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
                output_chunk_length=ocl,
                is_training=True,
            )
            target_expected = self.get_feature_times(
                target, lags["min"], lags["max"], ocl, is_target=True
            )
            past_expected = self.get_feature_times(
                past, lags_past["min"], lags_past["max"]
            )
            future_expected = self.get_feature_times(
                future, lags_future["min"], lags_future["max"]
            )
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_feature_times_prediction_range_idx(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags_combos = (
            {"vals": [-1], "max": 1, "min": 1},
            {"vals": [-2, -1], "max": 2, "min": 1},
            {"vals": [-6, -4, -3], "max": 6, "min": 3},
            {"vals": [-4, -6, -3], "max": 6, "min": 3},
        )
        for (lags, lags_past, lags_future) in product(
            lags_combos, lags_combos, lags_combos
        ):
            feature_times = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
                is_training=False,
            )
            target_expected = self.get_feature_times(target, lags["min"], lags["max"])
            past_expected = self.get_feature_times(
                past, lags_past["min"], lags_past["max"]
            )
            future_expected = self.get_feature_times(
                future, lags_future["min"], lags_future["max"]
            )
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_feature_times_prediction_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="1d")
        past = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=25, freq="2d")
        future = linear_timeseries(start=pd.Timestamp("1/3/2000"), length=30, freq="3d")
        lags_combos = (
            {"vals": [-1], "max": 1, "min": 1},
            {"vals": [-2, -1], "max": 2, "min": 1},
            {"vals": [-6, -4, -3], "max": 6, "min": 3},
            {"vals": [-4, -6, -3], "max": 6, "min": 3},
        )
        for (lags, lags_past, lags_future) in product(
            lags_combos, lags_combos, lags_combos
        ):
            feature_times = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
                is_training=False,
            )
            target_expected = self.get_feature_times(target, lags["min"], lags["max"])
            past_expected = self.get_feature_times(
                past, lags_past["min"], lags_past["max"]
            )
            future_expected = self.get_feature_times(
                future, lags_future["min"], lags_future["max"]
            )
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_feature_times_output_chunk_length_range_idx(self):
        """
        Tests that the last feature time for the `target_series`
        returned by `get_feature_times` corresponds to
        `output_chunk_length - 1` timesteps *before* the end of
        the target series; this is the last time point in
        `target_series` which has enough values in front of it
        to create a label. This particular test uses range time
        index series to check this behaviour.
        """
        target = linear_timeseries(start=0, length=20, freq=2)
        # Test multiple `output_chunk_length` values:
        for ocl in (1, 2, 3, 4, 5):
            feature_times = get_feature_times(
                target_series=target,
                lags=[-2, -3, -5],
                output_chunk_length=ocl,
                is_training=True,
            )
            self.assertEqual(
                feature_times[0][-1], target.end_time() - target.freq * (ocl - 1)
            )

    def test_feature_times_output_chunk_length_datetime_idx(self):
        """
        Tests that the last feature time for the `target_series`
        returned by `get_feature_times` when `is_training = True`
        corresponds to the time that is `(output_chunk_length - 1)`
        timesteps *before* the end of the target series; this is the
        last time point in `target_series` which has enough values
        in front of it to create a label. This particular test uses
        datetime time index series to check this behaviour.
        """
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="2d")
        # Test multiple `output_chunk_length` values:
        for ocl in (1, 2, 3, 4, 5):
            # `is_training = True`
            feature_times = get_feature_times(
                target_series=target,
                lags=[-2, -3, -5],
                output_chunk_length=ocl,
                is_training=True,
            )
            self.assertEqual(
                feature_times[0][-1], target.end_time() - target.freq * (ocl - 1)
            )

    def test_feature_times_lags_range_idx(self):
        """
        Tests that the first feature time for the `target_series`
        returned by `get_feature_times` corresponds to the time
        that is `max_lags` timesteps *after* the start of
        the target series; this is the first time point in
        `target_series` which has enough values in preceeding it
        to create a feature. This particular test uses range time
        index series to check this behaviour.
        """
        target = linear_timeseries(start=0, length=20, freq=2)
        # Expect same behaviour when training and predicting:
        for is_training in (False, True):
            for max_lags in (-1, -2, -3, -4, -5):
                feature_times = get_feature_times(
                    target_series=target,
                    lags=[-1, max_lags],
                    is_training=is_training,
                )
                self.assertEqual(
                    feature_times[0][0],
                    target.start_time() + target.freq * abs(max_lags),
                )

    def test_feature_times_lags_datetime_idx(self):
        """
        Tests that the first feature time for the `target_series`
        returned by `get_feature_times` corresponds to the time
        that is `max_lags` timesteps *after* the start of
        the target series; this is the first time point in
        `target_series` which has enough values in preceeding it
        to create a feature. This particular test uses datetime time
        index series to check this behaviour.
        """
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="2d")
        # Expect same behaviour when training and predicting:
        for is_training in (False, True):
            for max_lags in (-1, -2, -3, -4, -5):
                feature_times = get_feature_times(
                    target_series=target,
                    lags=[-1, max_lags],
                    is_training=is_training,
                )
                self.assertEqual(
                    feature_times[0][0],
                    target.start_time() + target.freq * abs(max_lags),
                )

    def test_feature_times_training_single_time_range_idx(self):
        """
        Tests that the first feature time for the `target_series`
        returned by `get_feature_times` corresponds to the time
        that is `max_lags` timesteps *after* the start of
        the target series; this is the first time point in
        `target_series` which has enough values in preceeding it
        to create a feature. This particular test uses datetime time
        index series to check this behaviour.
        """
        series = linear_timeseries(start=0, length=2, freq=1)
        feature_times = get_feature_times(
            target_series=series,
            output_chunk_length=1,
            lags=[-1],
            is_training=True,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], 1)
        target = linear_timeseries(start=0, length=2, freq=1)
        future = linear_timeseries(start=2, length=1, freq=2)
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            output_chunk_length=1,
            lags=[-1],
            lags_future_covariates=[-2],
            is_training=True,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], 1)
        self.assertEqual(len(feature_times[2]), 1)
        self.assertEqual(feature_times[2][0], 6)

    def test_feature_times_training_single_time_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=2, freq="d")
        feature_times = get_feature_times(
            target_series=target,
            output_chunk_length=1,
            lags=[-1],
            is_training=True,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], pd.Timestamp("1/2/2000"))
        future = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=1, freq="2d")
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            output_chunk_length=1,
            lags=[-1],
            lags_future_covariates=[-2],
            is_training=True,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], pd.Timestamp("1/2/2000"))
        self.assertEqual(len(feature_times[2]), 1)
        self.assertEqual(feature_times[2][0], pd.Timestamp("1/6/2000"))

    def test_feature_times_prediction_single_time_range_idx(self):
        series = linear_timeseries(start=0, length=1, freq=1)
        feature_times = get_feature_times(
            target_series=series,
            lags=[-1],
            is_training=False,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], 1)
        target = linear_timeseries(start=0, length=1, freq=1)
        future = linear_timeseries(start=2, length=1, freq=2)
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates=[-2],
            is_training=False,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], 1)
        self.assertEqual(len(feature_times[2]), 1)
        self.assertEqual(feature_times[2][0], 6)

    def test_feature_times_prediction_single_time_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=1, freq="d")
        feature_times = get_feature_times(
            target_series=target,
            lags=[-1],
            is_training=False,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], pd.Timestamp("1/2/2000"))
        future = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=1, freq="2d")
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates=[-2],
            is_training=False,
        )
        self.assertEqual(len(feature_times[0]), 1)
        self.assertEqual(feature_times[0][0], pd.Timestamp("1/2/2000"))
        self.assertEqual(len(feature_times[2]), 1)
        self.assertEqual(feature_times[2][0], pd.Timestamp("1/6/2000"))

    def test_feature_times_extend_beyond_time_index_datetime_idx(self):
        lags = [-4]
        past = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=1, freq="2d")
        target = linear_timeseries(start=pd.Timestamp("1/10/2000"), length=5, freq="3d")
        feature_times = get_feature_times(
            target_series=target,
            past_covariates=past,
            lags_past_covariates=lags,
            is_training=True,
        )
        self.assertTrue(feature_times[0].equals(target.time_index))
        self.assertEqual(len(feature_times[1]), 1)
        self.assertEqual(feature_times[1][0], pd.Timestamp("1/10/2000"))
        feature_times = get_feature_times(
            past_covariates=past,
            lags_past_covariates=lags,
            is_training=False,
        )
        self.assertEqual(len(feature_times[1]), 1)
        self.assertEqual(feature_times[1][0], pd.Timestamp("1/10/2000"))

    def test_feature_times_extend_beyond_time_index_range_idx(self):
        lags = [-4]
        past = linear_timeseries(start=2, length=1, freq=2)
        target = linear_timeseries(start=10, length=5, freq=3)
        feature_times = get_feature_times(
            target_series=target,
            past_covariates=past,
            lags_past_covariates=lags,
            is_training=True,
        )
        self.assertTrue(feature_times[0].equals(target.time_index))
        self.assertEqual(len(feature_times[1]), 1)
        self.assertEqual(feature_times[1][0], 10)
        feature_times = get_feature_times(
            past_covariates=past,
            lags_past_covariates=lags,
            is_training=False,
        )
        self.assertEqual(len(feature_times[1]), 1)
        self.assertEqual(feature_times[1][0], 10)

    def test_feature_times_unspecified_series(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = {"vals": [-1, -2], "max": 2, "min": 1}
        lags_past = {"vals": [-2, -5], "max": 5, "min": 2}
        lags_future = {"vals": [-3, -5], "max": 5, "min": 3}
        expected_target = self.get_feature_times(target, lags["min"], lags["max"])
        expected_past = self.get_feature_times(past, lags_past["min"], lags_past["max"])
        expected_future = self.get_feature_times(
            future, lags_future["min"], lags_future["max"]
        )

        feature_times = get_feature_times(
            target_series=target, lags=lags["vals"], is_training=False
        )
        self.assertTrue(expected_target.equals(feature_times[0]))
        self.assertEqual(feature_times[1], None)
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            past_covariates=past,
            lags_past_covariates=lags_past["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertTrue(expected_past.equals(feature_times[1]))
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertEqual(feature_times[1], None)
        self.assertTrue(expected_future.equals(feature_times[2]))
        feature_times = get_feature_times(
            target_series=target,
            past_covariates=past,
            lags=lags["vals"],
            lags_past_covariates=lags_past["vals"],
            is_training=False,
        )
        self.assertTrue(expected_target.equals(feature_times[0]))
        self.assertTrue(expected_past.equals(feature_times[1]))
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=lags["vals"],
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertTrue(expected_target.equals(feature_times[0]))
        self.assertEqual(feature_times[1], None)
        self.assertTrue(expected_future.equals(feature_times[2]))
        feature_times = get_feature_times(
            past_covariates=past,
            future_covariates=future,
            lags_past_covariates=lags_past["vals"],
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertTrue(expected_past.equals(feature_times[1]))
        self.assertTrue(expected_future.equals(feature_times[2]))

    def test_feature_times_unspecified_lag_or_series_warning(self):
        """
        Tests that `
        """
        # Define some arbitrary input values:
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = [-1, -2]
        lags_past = [-2, -5]
        lags_future = [-3, -5]
        # Specify `future_covariates` but not `lags_future_covariates` when `is_training = False`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                is_training=False,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `lags_future_covariates` but not `future_covariates` when `is_training = False`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                past_covariates=past,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `future_covariates` but not `lags_future_covariates` and
        # `target_series` but not lags, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                output_chunk_length=1,
                is_training=True,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `target_series` but not lags, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                is_training=True,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `past_covariates` but not `lags_past_covariates`, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                lags=lags,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            self.assertEqual(len(w), 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`past_covariates` was specified without accompanying "
                    "`lags_past_covariates` and, thus, will be ignored."
                ),
            )
            self.assertEqual(
                str(w[1].message),
                (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `past_covariates` but not `lags_past_covariates`, and `target_series`
        # but not `lags` when `is_training = False`:
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                is_training=True,
            )
            self.assertEqual(len(w), 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertEqual(
                str(w[0].message),
                (
                    "`past_covariates` was specified without accompanying "
                    "`lags_past_covariates` and, thus, will be ignored."
                ),
            )
            self.assertEqual(
                str(w[1].message),
                (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                ),
            )
        # Specify `target_series` but not `lags` when `is_training = False`:
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
        # Specify `target_series` but not `lags` when `is_training = True`;
        # this should *not* throw a warning:
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=True,
            )
            self.assertEqual(len(w), 0)

    def test_feature_times_unspecified_training_inputs_error(self):
        """
        Tests that `get_feature_times` throws correct error when
        `target_series` and/or `output_chunk_length` hasn't been
        specified when `is_training = True`.
        """
        output_chunk_length = 1
        # Don't specify `target_series`:
        with self.assertRaises(ValueError) as e:
            get_feature_times(output_chunk_length=output_chunk_length, is_training=True)
        self.assertEqual(
            ("Must specify `target_series` when `is_training = True`."),
            str(e.exception),
        )
        # Don't specify neither `target_series` nor `output_chunk_length`
        with self.assertRaises(ValueError) as e:
            get_feature_times(is_training=True)
        self.assertEqual(
            ("Must specify `target_series` when `is_training = True`."),
            str(e.exception),
        )

    def test_feature_times_no_lags_specified_error(self):
        """
        Tests that `get_feature_times` throws correct error
        when no lags have been specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=target, is_training=False)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_feature_times_series_too_short_error(self):
        """
        Tests that `get_feature_times` throws correct error
        when provided series are too short for specified
        lag and/or `output_chunk_length` values.
        """
        series = linear_timeseries(start=1, length=2, freq=1)
        # `target_series` too short when predicting:
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[-20, -1], is_training=False)
        self.assertEqual(
            (
                "`target_series` must have at least `-min(lags) + max(lags) + 1` = 20 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
        # `target_series` too short when training:
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                target_series=series,
                lags=[-20],
                output_chunk_length=5,
                is_training=True,
            )
        self.assertEqual(
            (
                "`target_series` must have at least `-min(lags) + output_chunk_length` = 25 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
        # `past_covariates` too short when training:
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                target_series=series,
                past_covariates=series,
                lags_past_covariates=[-20, -1],
                output_chunk_length=1,
                is_training=True,
            )
        self.assertEqual(
            (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 20 timesteps; "
                "instead, it only has 2."
            ),
            str(e.exception),
        )

    def test_feature_times_invalid_lag_values_error(self):
        """
        Tests that `get_feature_times` throws correct error
        when provided with invalid lag values (i.e. not less than
        0 if `lags`, or not less than 1 if `lags_past_covariates` or
        `lags_future_covariates`).
        """
        series = linear_timeseries(start=1, length=2, freq=1)
        # `lags` not <= -1:
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[0], is_training=False)
        self.assertEqual(
            ("`lags` must be a `Sequence` containing only `int` values less than 0."),
            str(e.exception),
        )
        # `lags_past_covariates` not <= 0:
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                past_covariates=series, lags_past_covariates=[1], is_training=False
            )
        self.assertEqual(
            (
                "`lags_past_covariates` must be a `Sequence` containing only `int` values less than 1."
            ),
            str(e.exception),
        )
        # `lags_future_covariates` not <= 0:
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                future_covariates=series, lags_future_covariates=[1], is_training=False
            )
        self.assertEqual(
            (
                "`lags_future_covariates` must be a `Sequence` containing only `int` values less than 1."
            ),
            str(e.exception),
        )
