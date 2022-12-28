import warnings
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts import concatenate as darts_concatenate
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import create_lagged_training_data
from darts.utils.timeseries_generation import linear_timeseries


class CreateLaggedTrainingDataTestCase(DartsBaseTestClass):

    """
    Tests `create_lagged_training_data` function defined in `darts.utils.data.tabularization`.
    """

    target_lag_combos = (
        {"vals": None, "max": None, "min": None},
        {"vals": [-1, -3], "max": 3, "min": 1},
        {"vals": [-1, -3], "max": 3, "min": 1},
    )
    covariates_lag_combos = (*target_lag_combos, {"vals": [0, -2], "max": 2, "min": 0})
    output_chunk_length_combos = (1, 3)
    multi_models_combos = (False, True)
    max_samples_per_ts_combos = (1, 2, None)

    @staticmethod
    def create_multivariate_timeseries(n_components, **linear_timeseries_kwargs):
        start_value = linear_timeseries_kwargs.pop("start_value", 0)
        end_value = linear_timeseries_kwargs.pop("end_value", 1)
        timeseries = []
        for i in range(n_components):
            timeseries_i = linear_timeseries(
                start_value=start_value + i,
                end_value=end_value + i,
                **linear_timeseries_kwargs
            )
            timeseries.append(timeseries_i)
        return darts_concatenate(timeseries, axis=1)

    @staticmethod
    def get_feature_times_single_series(
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

    @staticmethod
    def get_feature_times(
        target,
        past,
        future,
        lags,
        lags_past,
        lags_future,
        output_chunk_length,
        max_samples_per_ts,
    ):
        times = CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
            target, lags["min"], lags["max"], output_chunk_length, is_label=True
        )
        if lags_past["vals"] is not None:
            past_times = (
                CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
                    past, lags_past["min"], lags_past["max"]
                )
            )
            times = times.intersection(past_times)
        if lags_future["vals"] is not None:
            future_times = (
                CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
                    future, lags_future["min"], lags_future["max"]
                )
            )
            times = times.intersection(future_times)
        if (max_samples_per_ts is not None) and (len(times) > max_samples_per_ts):
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_features(
        series: TimeSeries, feats_times: pd.Index, lags: Optional[Sequence[int]]
    ) -> np.array:
        if lags is None:
            features = None
        else:
            time_index = series.time_index
            if time_index[-1] < feats_times[-1]:
                is_range_idx = isinstance(time_index[0], int)
                if is_range_idx:
                    idx_to_add = pd.RangeIndex(
                        start=time_index[-1], stop=feats_times[-1] + 1, step=series.freq
                    )
                else:
                    idx_to_add = pd.date_range(
                        start=time_index[-1], end=feats_times[-1], freq=series.freq
                    )
                time_index = time_index.union(idx_to_add)
            array_vals = series.all_values(copy=False)[:, :, 0]
            features = []
            for time in feats_times:
                idx = np.searchsorted(time_index, time)
                feature_i = []
                for lag in lags:
                    idx_to_get = idx + lag
                    feature_i.append(array_vals[idx_to_get, :].reshape(-1))
                feature_i = np.concatenate(feature_i, axis=0)
                features.append(feature_i)
            features = np.stack(features, axis=0)
        return features

    @staticmethod
    def get_labels(
        target: TimeSeries,
        feats_times: pd.Index,
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
        for time in feats_times:
            label_i = []
            for i in label_iter:
                label_time = time + i * target.freq
                idx = np.searchsorted(target.time_index, label_time)
                label_i.append(array_vals[idx, :, 0].reshape(-1))
            label_i = np.concatenate(label_i, axis=0)
            labels.append(label_i)
        labels = np.stack(labels, axis=0)
        return labels

    def test_lagged_training_data_equal_freq_range_index(self):
        target = self.create_multivariate_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, length=8, freq=2
        )
        past = self.create_multivariate_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, length=9, freq=2
        )
        future = self.create_multivariate_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, length=10, freq=2
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
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
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
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
                    use_moving_windows=True,
                )
            feats_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
                output_chunk_length,
                max_samples_per_ts,
            )
            target_feats = self.get_features(target, feats_times, lags["vals"])
            past_feats = self.get_features(past, feats_times, lags_past["vals"])
            future_feats = self.get_features(future, feats_times, lags_future["vals"])
            all_feats = (target_feats, past_feats, future_feats)
            to_concat = [x for x in all_feats if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feats_times, output_chunk_length, multi_models
            )
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_equal_freq_datetime_index(self):
        target = self.create_multivariate_timeseries(
            n_components=2,
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/2/2000"),
            length=8,
            freq="2d",
        )
        past = self.create_multivariate_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            length=9,
            freq="2d",
        )
        future = self.create_multivariate_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            length=10,
            freq="2d",
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
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
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
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
                    use_moving_windows=True,
                )
            feats_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
                output_chunk_length,
                max_samples_per_ts,
            )
            target_feats = self.get_features(target, feats_times, lags["vals"])
            past_feats = self.get_features(past, feats_times, lags_past["vals"])
            future_feats = self.get_features(future, feats_times, lags_future["vals"])
            all_feats = (target_feats, past_feats, future_feats)
            to_concat = [x for x in all_feats if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feats_times, output_chunk_length, multi_models
            )
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_unequal_freq_range_index(self):
        target = self.create_multivariate_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1
        )
        past = self.create_multivariate_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2
        )
        future = self.create_multivariate_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
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
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
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
                    use_moving_windows=False,
                )
            feats_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
                output_chunk_length,
                max_samples_per_ts,
            )
            target_feats = self.get_features(target, feats_times, lags["vals"])
            past_feats = self.get_features(past, feats_times, lags_past["vals"])
            future_feats = self.get_features(future, feats_times, lags_future["vals"])
            all_feats = (target_feats, past_feats, future_feats)
            to_concat = [x for x in all_feats if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feats_times, output_chunk_length, multi_models
            )
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_unequal_freq_datetime_index(self):
        target = self.create_multivariate_timeseries(
            n_components=2,
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/1/2000"),
            length=20,
            freq="d",
        )
        past = self.create_multivariate_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/2/2000"),
            length=10,
            freq="2d",
        )
        future = self.create_multivariate_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/3/2000"),
            length=7,
            freq="3d",
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
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
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
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
                    use_moving_windows=False,
                )
            feats_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
                output_chunk_length,
                max_samples_per_ts,
            )
            target_feats = self.get_features(target, feats_times, lags["vals"])
            past_feats = self.get_features(past, feats_times, lags_past["vals"])
            future_feats = self.get_features(future, feats_times, lags_future["vals"])
            all_feats = (target_feats, past_feats, future_feats)
            to_concat = [x for x in all_feats if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.get_labels(
                target, feats_times, output_chunk_length, multi_models
            )
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_extend_past_and_future_covariates_range_idx(self):
        target = linear_timeseries(start=0, end=10, start_value=1, end_value=2)
        lags = [-1]
        past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
        lags_past = [-2]
        future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        lags_future = [-4]
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-2, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        expected_y = target.all_values(copy=False)[-1, :, 0]
        for use_moving_windows in (False, True):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=1,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time())
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))

    def test_lagged_training_data_extend_past_and_future_covariates_datetime_idx(self):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/11/2000"),
            start_value=1,
            end_value=2,
        )
        lags = [-1]
        past = linear_timeseries(
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/9/2000"),
            start_value=2,
            end_value=3,
        )
        lags_past = [-2]
        future = linear_timeseries(
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/7/2000"),
            start_value=3,
            end_value=4,
        )
        lags_future = [-4]
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-2, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        expected_y = target.all_values(copy=False)[-1, :, 0]
        for use_moving_windows in (False, True):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=1,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time())
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))

    def test_lagged_training_data_single_point_range_idx(self):
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                lags=[-1],
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.zeros((1, 1, 1)), X))
            self.assertTrue(np.allclose(np.ones((1, 1, 1)), y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], 1)

    def test_lagged_training_data_single_point_datetime_idx(self):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=2, start_value=0, end_value=1
        )
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                lags=[-1],
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.zeros((1, 1, 1)), X))
            self.assertTrue(np.allclose(np.ones((1, 1, 1)), y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], pd.Timestamp("1/2/2000"))

    def test_lagged_training_data_zero_lags_range_idx(self):
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        past = linear_timeseries(start=1, length=1, start_value=1, end_value=2)
        future = linear_timeseries(start=1, length=1, start_value=2, end_value=3)
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=[-1],
                lags_past_covariates=[0],
                lags_future_covariates=[0],
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1), X))
            self.assertTrue(np.allclose(np.ones((1, 1, 1)), y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], 1)

    def test_lagged_training_data_zero_lags_datetime_idx(self):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=2, start_value=0, end_value=1
        )
        past = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=1, end_value=2
        )
        future = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=2, end_value=3
        )
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=[-1],
                lags_past_covariates=[0],
                lags_future_covariates=[0],
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1), X))
            self.assertTrue(np.allclose(np.ones((1, 1, 1)), y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], pd.Timestamp("1/2/2000"))

    def test_lagged_training_data_no_shared_times_error(self):
        series_1 = linear_timeseries(start=0, length=4, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=1)
        series_3 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=2)
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

    def test_lagged_training_data_no_specified_series_lags_pairs_error(self):
        series_1 = linear_timeseries(start=1, length=10, freq=1)
        series_2 = linear_timeseries(start=1, length=10, freq=2)
        lags = [-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=series_1,
                    output_chunk_length=1,
                    lags_past_covariates=lags,
                )
        self.assertEqual(
            "Must specify at least one series-lags pair.",
            str(e.exception),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    def test_lagged_training_data_invalid_output_chunk_length_error(self):
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

    def test_lagged_training_data_no_lags_specified_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(target_series=target, output_chunk_length=1)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_lagged_training_data_series_too_short_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=5,
                lags=[-20, -10],
            )
        self.assertEqual(
            (
                "`target_series` must have at least "
                "`-min(lags) + output_chunk_length` = 25 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                past_covariates=series,
                lags_past_covariates=[-5, -3],
            )
        self.assertEqual(
            (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 3 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )

    def test_lagged_training_data_invalid_lag_values_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series, output_chunk_length=1, lags=[0]
            )
        self.assertEqual(
            ("`lags` must be a `Sequence` containing only `int` values less than 0."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                past_covariates=series,
                lags_past_covariates=[1],
            )
        self.assertEqual(
            (
                "`lags_past_covariates` must be a `Sequence` containing only `int` values less than 1."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                future_covariates=series,
                lags_future_covariates=[1],
            )
        self.assertEqual(
            (
                "`lags_future_covariates` must be a `Sequence` containing only `int` values less than 1."
            ),
            str(e.exception),
        )

    def test_lagged_training_data_unspecified_lag_or_series_warning(self):
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
        # Specify `target_series` but not `lags` - this should *not*
        # throw a warning:
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                past_covariates=series,
                lags_past_covariates=lags,
            )
            self.assertTrue(len(w) == 0)
