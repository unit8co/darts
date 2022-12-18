import warnings
from itertools import product

import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import get_feature_times
from darts.utils.timeseries_generation import linear_timeseries


class GetFeatureTimesTestCase(DartsBaseTestClass):
    def test_training_feature_times_range_idx(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags_combos = (
            {"vals": [-1], "max": 1},
            {"vals": [-2, -1], "max": 2},
            {"vals": [-6, -4, -3], "max": 6},
            {"vals": [-4, -6, -3], "max": 6},
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
            target_expected = (
                target.time_index[lags["max"] : (-ocl + 1)]
                if ocl > 1
                else target.time_index[lags["max"] :]
            )
            past_expected = past.time_index[lags_past["max"] :]
            future_expected = future.time_index[lags_future["max"] :]
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_training_feature_times_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="1d")
        past = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=25, freq="2d")
        future = linear_timeseries(start=pd.Timestamp("1/3/2000"), length=30, freq="3d")
        lags_combos = (
            {"vals": [-1], "max": 1},
            {"vals": [-2, -1], "max": 2},
            {"vals": [-6, -4, -3], "max": 6},
            {"vals": [-4, -6, -3], "max": 6},
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
            target_expected = (
                target.time_index[lags["max"] : (-ocl + 1)]
                if ocl > 1
                else target.time_index[lags["max"] :]
            )
            past_expected = past.time_index[lags_past["max"] :]
            future_expected = future.time_index[lags_future["max"] :]
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_prediction_feature_times_range_idx(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags_combos = (
            {"vals": [-1], "max": 1},
            {"vals": [-2, -1], "max": 2},
            {"vals": [-6, -4, -3], "max": 6},
            {"vals": [-4, -6, -3], "max": 6},
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
            target_expected = target.time_index[lags["max"] :]
            past_expected = past.time_index[lags_past["max"] :]
            future_expected = future.time_index[lags_future["max"] :]
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_prediction_feature_times_datetime_idx(self):
        target = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=20, freq="1d")
        past = linear_timeseries(start=pd.Timestamp("1/2/2000"), length=25, freq="2d")
        future = linear_timeseries(start=pd.Timestamp("1/3/2000"), length=30, freq="3d")
        lags_combos = (
            {"vals": [-1], "max": 1},
            {"vals": [-2, -1], "max": 2},
            {"vals": [-6, -4, -3], "max": 6},
            {"vals": [-4, -6, -3], "max": 6},
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
            target_expected = target.time_index[lags["max"] :]
            past_expected = past.time_index[lags_past["max"] :]
            future_expected = future.time_index[lags_future["max"] :]
            self.assertTrue(target_expected.equals(feature_times[0]))
            self.assertTrue(past_expected.equals(feature_times[1]))
            self.assertTrue(future_expected.equals(feature_times[2]))

    def test_unspecified_series(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = {"vals": [-1, -2], "max": 2}
        lags_past = {"vals": [-2, -5], "max": 5}
        lags_future = {"vals": [-3, -5], "max": 5}
        feature_times = get_feature_times(
            target_series=target, lags=lags["vals"], is_training=False
        )
        self.assertTrue(target.time_index[lags["max"] :].equals(feature_times[0]))
        self.assertEqual(feature_times[1], None)
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            past_covariates=past,
            lags_past_covariates=lags_past["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertTrue(past.time_index[lags_past["max"] :].equals(feature_times[1]))
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertEqual(feature_times[1], None)
        self.assertTrue(
            future.time_index[lags_future["max"] :].equals(feature_times[2])
        )
        feature_times = get_feature_times(
            target_series=target,
            past_covariates=past,
            lags=lags["vals"],
            lags_past_covariates=lags_past["vals"],
            is_training=False,
        )
        self.assertTrue(target.time_index[lags["max"] :].equals(feature_times[0]))
        self.assertTrue(past.time_index[lags_past["max"] :].equals(feature_times[1]))
        self.assertEqual(feature_times[2], None)
        feature_times = get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=lags["vals"],
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertTrue(target.time_index[lags["max"] :].equals(feature_times[0]))
        self.assertEqual(feature_times[1], None)
        self.assertTrue(
            future.time_index[lags_future["max"] :].equals(feature_times[2])
        )
        feature_times = get_feature_times(
            past_covariates=past,
            future_covariates=future,
            lags_past_covariates=lags_past["vals"],
            lags_future_covariates=lags_future["vals"],
            is_training=False,
        )
        self.assertEqual(feature_times[0], None)
        self.assertTrue(past.time_index[lags_past["max"] :].equals(feature_times[1]))
        self.assertTrue(
            future.time_index[lags_future["max"] :].equals(feature_times[2])
        )

    def test_unspecified_lag_or_series_warning(self):
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

    def test_unspecified_training_inputs_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        output_chunk_length = 1
        # Don't specify `target_series`:
        with self.assertRaises(ValueError) as e:
            get_feature_times(output_chunk_length=output_chunk_length, is_training=True)
        self.assertEqual(
            (
                "Must specify `target_series` and `output_chunk_length` "
                "when `is_training = True`."
            ),
            str(e.exception),
        )
        # Don't specify `output_chunk_length`:
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=target, is_training=True)
        self.assertEqual(
            (
                "Must specify `target_series` and `output_chunk_length` "
                "when `is_training = True`."
            ),
            str(e.exception),
        )
        # Don't specify neither `target_series` nor `output_chunk_length`
        with self.assertRaises(ValueError) as e:
            get_feature_times(is_training=True)
        self.assertEqual(
            (
                "Must specify `target_series` and `output_chunk_length` "
                "when `is_training = True`."
            ),
            str(e.exception),
        )

    def test_no_lags_specified_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=target, is_training=False)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_series_too_short_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[-20], is_training=False)
        self.assertEqual(
            (
                "`target_series` must have at least `-min(lags) + 1` = 21 "
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
                "`target_series` must have at least "
                "`-min(lags) + 1 + output_chunk_length` = 26 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                target_series=series,
                past_covariates=series,
                lags_past_covariates=[-20],
                output_chunk_length=1,
                is_training=True,
            )
        self.assertEqual(
            (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + 1` = 21 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )

    def test_non_negative_lags_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            get_feature_times(target_series=series, lags=[0], is_training=False)
        self.assertEqual(
            ("`lags` must contain only negative values."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                past_covariates=series, lags_past_covariates=[0], is_training=False
            )
        self.assertEqual(
            ("`lags_past_covariates` must contain only negative values."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            get_feature_times(
                future_covariates=series, lags_future_covariates=[0], is_training=False
            )
        self.assertEqual(
            ("`lags_future_covariates` must contain only negative values."),
            str(e.exception),
        )
