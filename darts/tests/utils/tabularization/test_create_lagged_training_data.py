import warnings
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import create_lagged_training_data
from darts.utils.timeseries_generation import linear_timeseries


class CreateLaggedTrainingDataTestCase(DartsBaseTestClass):

    lag_combos = (
        {"vals": None, "max": None},
        {"vals": [-1, -3], "max": 3},
        {"vals": [-3, -1], "max": 3},
    )
    output_chunk_length_combos = (1, 3)
    multi_models_combos = (False, True)
    max_samples_per_ts_combos = (1, 2, None)

    @staticmethod
    def get_feature_times(
        target: TimeSeries,
        past: TimeSeries,
        future: TimeSeries,
        lags_max: Optional[int],
        lags_past_max: Optional[int],
        lags_future_max: Optional[int],
        output_chunk_length: int,
        max_samples_per_ts: Optional[int],
    ) -> pd.Index:
        times = target.time_index
        if output_chunk_length > 1:
            times = times.intersection(target.time_index[: -output_chunk_length + 1])
        if lags_max is not None:
            times = times.intersection(target.time_index[lags_max:])
        if lags_past_max is not None:
            times = times.intersection(past.time_index[lags_past_max:])
        if lags_future_max is not None:
            times = times.intersection(future.time_index[lags_future_max:])
        if max_samples_per_ts is not None:
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_features(
        series: TimeSeries, feature_times: pd.Index, lags: Optional[Sequence[int]]
    ) -> np.array:
        if lags is None:
            features = None
        else:
            array_vals = series.all_values(copy=False)
            features = []
            for time in feature_times:
                feature_i = []
                for lag in lags:
                    lag_time = time + lag * series.freq
                    idx = np.searchsorted(series.time_index, lag_time)
                    feature_i.append(array_vals[idx, :, 0].reshape(-1))
                feature_i = np.concatenate(feature_i, axis=0)
                features.append(feature_i)
            features = np.stack(features, axis=0)
        return features

    @staticmethod
    def get_labels(
        target: TimeSeries,
        feature_times: pd.Index,
        output_chunk_length: int,
        multi_models: bool,
    ):
        array_vals = target.all_values(copy=False)
        label_iter = (
            list(range(output_chunk_length))
            if multi_models
            else [output_chunk_length - 1]
        )
        labels = []
        for time in feature_times:
            label_i = []
            for i in label_iter:
                label_time = time + i * target.freq
                idx = np.searchsorted(target.time_index, label_time)
                label_i.append(array_vals[idx, :, 0].reshape(-1))
            label_i = np.concatenate(label_i, axis=0)
            labels.append(label_i)
        labels = np.stack(labels, axis=0)
        return labels

    def test_equal_freq_range_index(self):
        target = linear_timeseries(start_value=0, end_value=10, start=2, end=20, freq=2)
        past = linear_timeseries(start_value=10, end_value=20, start=4, end=23, freq=2)
        future = linear_timeseries(
            start_value=20, end_value=30, start=6, end=26, freq=2
        )
        param_combos = product(
            self.lag_combos,
            self.lag_combos,
            self.lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
        )
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
        ) in param_combos:
            if all(
                x is None
                for x in [lags["vals"], lags_past["vals"], lags_future["vals"]]
            ):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags["max"],
                lags_past["max"],
                lags_future["max"],
                output_chunk_length,
                max_samples_per_ts,
            )
            target_features = self.get_features(target, feature_times, lags["vals"])
            past_features = self.get_features(past, feature_times, lags_past["vals"])
            future_features = self.get_features(
                future, feature_times, lags_future["vals"]
            )
            to_concat = [
                x
                for x in (target_features, past_features, future_features)
                if x is not None
            ]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feature_times, output_chunk_length, multi_models
            )
            # `create_lagged_training_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y, times = create_lagged_training_data(
                    target,
                    output_chunk_length,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    multi_models=multi_models,
                    max_samples_per_ts=max_samples_per_ts,
                )
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_equal_freq_datetime_index(self):
        target = linear_timeseries(
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/16/2000"),
            freq="2d",
        )
        past = linear_timeseries(
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            end=pd.Timestamp("1/18/2000"),
            freq="2d",
        )
        future = linear_timeseries(
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="2d",
        )
        param_combos = product(
            self.lag_combos,
            self.lag_combos,
            self.lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
        )
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
        ) in param_combos:
            if all(
                x is None
                for x in [lags["vals"], lags_past["vals"], lags_future["vals"]]
            ):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags["max"],
                lags_past["max"],
                lags_future["max"],
                output_chunk_length,
                max_samples_per_ts,
            )
            target_features = self.get_features(target, feature_times, lags["vals"])
            past_features = self.get_features(past, feature_times, lags_past["vals"])
            future_features = self.get_features(
                future, feature_times, lags_future["vals"]
            )
            to_concat = [
                x
                for x in (target_features, past_features, future_features)
                if x is not None
            ]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feature_times, output_chunk_length, multi_models
            )
            # `create_lagged_training_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y, times = create_lagged_training_data(
                    target,
                    output_chunk_length,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    multi_models=multi_models,
                    max_samples_per_ts=max_samples_per_ts,
                )
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_unequal_freq_range_index(self):
        target = linear_timeseries(start_value=0, end_value=10, start=1, end=20, freq=1)
        past = linear_timeseries(start_value=10, end_value=20, start=2, end=23, freq=2)
        future = linear_timeseries(
            start_value=20, end_value=30, start=3, end=26, freq=3
        )
        param_combos = product(
            self.lag_combos,
            self.lag_combos,
            self.lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
        )
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
        ) in param_combos:
            if all(
                x is None
                for x in [lags["vals"], lags_past["vals"], lags_future["vals"]]
            ):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags["max"],
                lags_past["max"],
                lags_future["max"],
                output_chunk_length,
                max_samples_per_ts,
            )
            target_features = self.get_features(target, feature_times, lags["vals"])
            past_features = self.get_features(past, feature_times, lags_past["vals"])
            future_features = self.get_features(
                future, feature_times, lags_future["vals"]
            )
            to_concat = [
                x
                for x in (target_features, past_features, future_features)
                if x is not None
            ]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feature_times, output_chunk_length, multi_models
            )
            # `create_lagged_training_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y, times = create_lagged_training_data(
                    target,
                    output_chunk_length,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    multi_models=multi_models,
                    max_samples_per_ts=max_samples_per_ts,
                )
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_unequal_freq_datetime_index(self):
        target = linear_timeseries(
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="d",
        )
        past = linear_timeseries(
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/23/2000"),
            freq="2d",
        )
        future = linear_timeseries(
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/3/2000"),
            end=pd.Timestamp("1/26/2000"),
            freq="3d",
        )
        param_combos = product(
            self.lag_combos,
            self.lag_combos,
            self.lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
        )
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
        ) in param_combos:
            if all(
                x is None
                for x in [lags["vals"], lags_past["vals"], lags_future["vals"]]
            ):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags["max"],
                lags_past["max"],
                lags_future["max"],
                output_chunk_length,
                max_samples_per_ts,
            )
            target_features = self.get_features(target, feature_times, lags["vals"])
            past_features = self.get_features(past, feature_times, lags_past["vals"])
            future_features = self.get_features(
                future, feature_times, lags_future["vals"]
            )
            to_concat = [
                x
                for x in (target_features, past_features, future_features)
                if x is not None
            ]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feature_times, output_chunk_length, multi_models
            )
            # `create_lagged_training_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, y, times = create_lagged_training_data(
                    target,
                    output_chunk_length,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    multi_models=multi_models,
                    max_samples_per_ts=max_samples_per_ts,
                )
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_no_shared_times_error(self):
        series_1 = linear_timeseries(start=0, length=4, freq=1)
        series_2 = linear_timeseries(start=10, length=4, freq=1)
        series_3 = linear_timeseries(start=10, length=4, freq=2)
        lags = [-1]
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series_1,
                output_chunk_length=1,
                lags=lags,
                past_covariates=series_2,
                lags_past_covariates=lags,
            )
        self.assertEqual(
            "Specified series do not share any common times for which features can be created.",
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series_1,
                output_chunk_length=1,
                lags=lags,
                past_covariates=series_3,
                lags_past_covariates=lags,
            )
        self.assertEqual(
            "Specified series do not share any common times for which features can be created.",
            str(e.exception),
        )

    def test_no_specified_series_lags_pairs(self):
        series_1 = linear_timeseries(start=1, length=10, freq=1)
        series_2 = linear_timeseries(start=1, length=10, freq=2)
        lags = [-1]
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series_1, output_chunk_length=1, lags_past_covariates=lags
            )
        self.assertEqual(
            "Must specify at least one series-lags pair.",
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series_1,
                output_chunk_length=1,
                lags_future_covariates=lags,
                past_covariates=series_2,
            )
        self.assertEqual(
            "Must specify at least one series-lags pair.",
            str(e.exception),
        )

    def test_invalid_output_chunk_length_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=target, output_chunk_length=0, lags=lags
            )
        self.assertEqual(
            "`output_chunk_length` must be a positive `int`.",
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=target, output_chunk_length=1.1, lags=lags
            )
        self.assertEqual(
            "`output_chunk_length` must be a positive `int`.",
            str(e.exception),
        )

    def test_no_lags_specified_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(target_series=target, output_chunk_length=1)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_series_too_short_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=5,
                lags=[-20],
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
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                past_covariates=series,
                lags_past_covariates=[-20],
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
            create_lagged_training_data(
                target_series=series, output_chunk_length=1, lags=[0]
            )
        self.assertEqual(
            ("`lags` must be a `Sequence` containing only negative `int` values."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                past_covariates=series,
                lags_past_covariates=[0],
            )
        self.assertEqual(
            (
                "`lags_past_covariates` must be a `Sequence` containing only negative `int` values."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                future_covariates=series,
                lags_future_covariates=[0],
            )
        self.assertEqual(
            (
                "`lags_future_covariates` must be a `Sequence` containing only negative `int` values."
            ),
            str(e.exception),
        )

    def test_unspecified_lag_or_series_warning(self):
        series = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                lags=lags,
                future_covariates=series,
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
            _ = create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                lags=lags,
                lags_future_covariates=lags,
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
        # `past_covariates` but not `lags_past_covariates`, and `target_series`
        # but not `lags`:
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_training_data(
                target_series=series,
                lags=lags,
                output_chunk_length=1,
                past_covariates=series,
                lags_future_covariates=lags,
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
