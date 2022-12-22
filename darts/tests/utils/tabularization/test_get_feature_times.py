import warnings
from itertools import product

import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import get_feature_times
from darts.utils.timeseries_generation import linear_timeseries


class GetFeatureTimesTestCase(DartsBaseTestClass):
    """
    Tests `get_feature_time` function defined in `darts.utils.data.tabularization`.
    """

    @staticmethod
    def get_feature_times(
        series, min_lag, max_lag, output_chunk_length=None, is_label=False
    ):
        times = series.time_index
        if not is_label and (min_lag is not None):
            times = times.union(
                [times[-1] + i * series.freq for i in range(1, min_lag + 1)]
            )
        if min_lag is not None:
            times = times[max_lag:]
        if is_label and (output_chunk_length > 1):
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
                target, lags["min"], lags["max"], ocl, is_label=True
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
                target, lags["min"], lags["max"], ocl, is_label=True
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

    def test_feature_times_output_chunk_length_end_range_idx(self):
        target = linear_timeseries(start=0, length=20, freq=2)
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

    def test_feature_times_output_chunk_length_end_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="2d")
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

    def test_feature_times_lags_start_range_idx(self):
        target = linear_timeseries(start=0, length=20, freq=2)
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

    def test_feature_times_lags_start_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="2d")
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
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                )
            )
        # Specify `lags_future_covariates` but not `future_covariates` when `is_training = False`
        with warnings.catch_warnings(record=True) as w:
            _ = get_feature_times(
                past_covariates=past,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                )
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
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                )
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
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                )
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
            self.assertTrue(len(w) == 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`past_covariates` was specified without accompanying "
                    "`lags_past_covariates` and, thus, will be ignored."
                )
            )
            self.assertTrue(
                str(w[1].message)
                == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                )
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
            self.assertTrue(len(w) == 2)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(issubclass(w[1].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`past_covariates` was specified without accompanying "
                    "`lags_past_covariates` and, thus, will be ignored."
                )
            )
            self.assertTrue(
                str(w[1].message)
                == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                )
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
            self.assertTrue(len(w) == 1)

    def test_feature_times_unspecified_training_inputs_error(self):
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
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=target, is_training=False)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_feature_times_series_too_short_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[-20, -1], is_training=False)
        self.assertEqual(
            (
                "`target_series` must have at least `-min(lags) + max(lags) + 1` = 20 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
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

    def test_feature_times_non_negative_lags_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[0], is_training=False)
        self.assertEqual(
            ("`lags` must be a `Sequence` containing only negative `int` values."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                past_covariates=series, lags_past_covariates=[0], is_training=False
            )
        self.assertEqual(
            (
                "`lags_past_covariates` must be a `Sequence` containing only negative `int` values."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                future_covariates=series, lags_future_covariates=[0], is_training=False
            )
        self.assertEqual(
            (
                "`lags_future_covariates` must be a `Sequence` containing only negative `int` values."
            ),
            str(e.exception),
        )
