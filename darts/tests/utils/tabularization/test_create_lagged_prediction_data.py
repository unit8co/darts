import warnings
from itertools import product
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts import concatenate as darts_concatenate
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.timeseries_generation import linear_timeseries


class CreateLaggedPredictionDataTestCase(DartsBaseTestClass):

    """
    Tests `create_lagged_prediction_data` function defined in `darts.utils.data.tabularization`.
    """

    target_lag_combos = (
        {"vals": None, "max": None, "min": None},
        {"vals": [-1, -3], "max": 3, "min": 1},
        {"vals": [-3, -1], "max": 3, "min": 1},
    )
    covariates_lag_combos = (*target_lag_combos, {"vals": [0, -2], "max": 2, "min": 0})
    max_samples_per_ts_combos = (1, 2, None)

    @staticmethod
    def create_multivariate_timeseries(n_components, **linear_timeseries_kwargs):
        """
        Helper function called by tests that creates a multi-component `TimeSeries`
        by concatenating a series of `n_components` nearly-identical `linear_timeseries`
        together; the meaning of the `start_value`, `end_value`, `start`, and `end` parameters
        is the same as in `linear_timeseries`.

        To distinguish each component from  one another, the values of the `i`th component in the
        returned `TimeSeries` equals the values of the `0`th component plus `i`.

        To create a `TimeSeries` with a `pd.DatetimeIndex`, `start`, `end` and `freq` should all
        be `str`s; alternatively to create a `TimeSeries` with a `pd.RangeIndex`, `start`, `end`,
        and `freq` should all be `int`s.
        """
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
    def get_feature_times(
        target: TimeSeries,
        past: TimeSeries,
        future: TimeSeries,
        lags: Optional[int],
        lags_past: Optional[int],
        lags_future: Optional[int],
        max_samples_per_ts: Optional[int],
    ) -> pd.Index:
        """
        Helper function called by tests that returns all of the time index values for
        which prediction features can be constructed. If `max_samples_per_ts` is
        specified, only the most recent `max_samples_per_ts` times are taken.

        By noting that the time `t` *can* be used to create prediction features (unlike
        when creating training data, where the time `t` is used as a label), prediction f
        eatures cannot be created for times which have fewer that `lags_max_i - 1` values
        preceeding them.

        Hence, is the intersection of `series_i.time_index[max_lag_i-1:]` for those `series_i`
        in `(target_series, past_covariates, future_covariates)` which are not `None`, and where
        `max_lag_i` is the largest sized lag specified for the particular series.
        """
        times = None
        all_series = [target, past, future]
        all_lags = [lags, lags_past, lags_future]
        for series_i, lags_i in zip(all_series, all_lags):
            if lags_i["vals"] is not None:
                times_i = series_i.time_index
                times_i = times_i.union(
                    [
                        times_i[-1] + i * series_i.freq
                        for i in range(1, lags_i["min"] + 1)
                    ]
                )
                times_i = times_i[lags_i["max"] :]
                if times is None:
                    times = times_i
                else:
                    times = times.intersection(times_i)
        if max_samples_per_ts is not None:
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_features(
        series: TimeSeries, feats_times: pd.Index, lags: Optional[Sequence[int]]
    ) -> np.array:
        """
        Helper function called by tests that assembles the `X` prediction features matrix
        for a *single series* (i.e. *just* for `target_series`, `past_covariates`, or
        `future_covariates`).

        This is done by iterating over each time point in `feature_times` and then over each lag values
        in `lags`; for each time-lag combination, the value of the series that is `(lag + 1)` index
        positions away from time `t` is extracted. Since the time `t` itself is treated as a feature
        when creating lagged prediction data (as opposed to treating time `t` as a label, as is the case
        when creating lagged training data),

        Note that the . since the time `t` can
        be used to

        Unlike, no 'vectorization tricks' are involved here, which makes it much slower than whats implemented
        in `create_lagged_prediction_data`. The key point here is that this implementation

        """
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
                idx = np.searchsorted(series.time_index, time)
                feature_i = []
                for lag in lags:
                    idx_to_get = idx + lag
                    feature_i.append(array_vals[idx_to_get, :].reshape(-1))
                feature_i = np.concatenate(feature_i, axis=0)
                features.append(feature_i)
            features = np.stack(features, axis=0)
        return features

    def test_lagged_prediction_data_equal_freq_range_index(self):
        target = self.create_multivariate_timeseries(
            start_value=0, end_value=10, start=2, end=20, freq=2, n_components=2
        )
        past = self.create_multivariate_timeseries(
            start_value=10, end_value=20, start=4, end=23, freq=2, n_components=3
        )
        future = self.create_multivariate_timeseries(
            start_value=20, end_value=30, start=6, end=26, freq=2, n_components=4
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
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
            # `create_lagged_prediction_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, times = create_lagged_prediction_data(
                    target_series=target,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    max_samples_per_ts=max_samples_per_ts,
                    use_moving_windows=True,
                )
            self.assertEqual(X.shape[0], len(feature_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_lagged_prediction_data_equal_freq_datetime_index(self):
        target = self.create_multivariate_timeseries(
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/16/2000"),
            freq="2d",
            n_components=2,
        )
        past = self.create_multivariate_timeseries(
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            end=pd.Timestamp("1/18/2000"),
            freq="2d",
            n_components=3,
        )
        future = self.create_multivariate_timeseries(
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="2d",
            n_components=4,
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
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
            # `create_lagged_prediction_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, times = create_lagged_prediction_data(
                    target_series=target,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    max_samples_per_ts=max_samples_per_ts,
                    use_moving_windows=True,
                )
            self.assertEqual(X.shape[0], len(feature_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_lagged_prediction_data_unequal_freq_range_index(self):
        target = self.create_multivariate_timeseries(
            start_value=0, end_value=10, start=2, end=20, freq=1, n_components=2
        )
        past = self.create_multivariate_timeseries(
            start_value=10, end_value=20, start=4, end=23, freq=2, n_components=3
        )
        future = self.create_multivariate_timeseries(
            start_value=20, end_value=30, start=6, end=26, freq=3, n_components=4
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
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
            # `create_lagged_prediction_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, times = create_lagged_prediction_data(
                    target_series=target,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    max_samples_per_ts=max_samples_per_ts,
                    use_moving_windows=False,
                )
            self.assertEqual(X.shape[0], len(feature_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_lagged_prediction_data_unequal_freq_datetime_index(self):
        target = self.create_multivariate_timeseries(
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="d",
            n_components=2,
        )
        past = self.create_multivariate_timeseries(
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/23/2000"),
            freq="2d",
            n_components=3,
        )
        future = self.create_multivariate_timeseries(
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/3/2000"),
            end=pd.Timestamp("1/26/2000"),
            freq="3d",
            n_components=4,
        )
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            feature_times = self.get_feature_times(
                target,
                past,
                future,
                lags,
                lags_past,
                lags_future,
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
            # `create_lagged_prediction_data` throws warning when a series is specified, but
            # the corresponding lag is not - silence this warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, times = create_lagged_prediction_data(
                    target_series=target,
                    past_covariates=past,
                    future_covariates=future,
                    lags=lags["vals"],
                    lags_past_covariates=lags_past["vals"],
                    lags_future_covariates=lags_future["vals"],
                    max_samples_per_ts=max_samples_per_ts,
                    use_moving_windows=False,
                )
            self.assertEqual(X.shape[0], len(feature_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feature_times.equals(times))

    def test_lagged_prediction_data_extend_past_and_future_covariates_range_idx(self):
        target = linear_timeseries(start=0, end=9, start_value=1, end_value=2)
        lags = [-1]
        past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
        lags_past = [-2]
        future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        lags_future = [-4]
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-1, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=1,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time() + target.freq)
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))

    def test_lagged_prediction_data_extend_past_and_future_covariates_datetime_idx(
        self,
    ):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"),
            end=pd.Timestamp("1/10/2000"),
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
                target.all_values(copy=False)[-1, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=1,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time() + target.freq)
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))

    def test_lagged_prediction_data_single_point_range_idx(self):
        target = linear_timeseries(start=0, length=1, start_value=0, end_value=1)
        lag = 5
        for use_moving_windows in (True, True):
            X, times = create_lagged_prediction_data(
                target, lags=[-lag], use_moving_windows=use_moving_windows
            )
            self.assertTrue(np.allclose(np.zeros((1, 1, 1)), X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time() + lag * target.freq)

    def test_lagged_prediction_data_single_point_datetime_idx(self):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=1, start_value=0, end_value=1
        )
        lag = 5
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target, lags=[-lag], use_moving_windows=use_moving_windows
            )
            self.assertTrue(np.allclose(np.zeros((1, 1, 1)), X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time() + lag * target.freq)

    def test_lagged_prediction_data_zero_lags_range_idx(self):
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        past = linear_timeseries(start=1, length=1, start_value=1, end_value=2)
        future = linear_timeseries(start=1, length=1, start_value=2, end_value=3)
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=[-1],
                lags_past_covariates=[0],
                lags_future_covariates=[0],
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1), X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], 1)

    def test_lagged_prediction_data_zero_lags_datetime_idx(self):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=2, start_value=0, end_value=1
        )
        past = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=1, end_value=2
        )
        future = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=2, end_value=3
        )
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=[-1],
                lags_past_covariates=[0],
                lags_future_covariates=[0],
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1), X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], pd.Timestamp("1/2/2000"))

    def test_lagged_prediction_data_no_shared_times_error(self):
        series_1 = linear_timeseries(start=0, length=4, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=1)
        series_3 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=2)
        lags = [-1]
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(
                target_series=series_1,
                lags=lags,
                past_covariates=series_2,
                lags_past_covariates=lags,
            )
        self.assertEqual(
            "Specified series do not share any common times for which features can be created.",
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(
                target_series=series_1,
                lags=lags,
                past_covariates=series_3,
                lags_past_covariates=lags,
            )
        self.assertEqual(
            "Specified series do not share any common times for which features can be created.",
            str(e.exception),
        )

    def test_lagged_prediction_data_no_specified_series_lags_pairs_error(self):
        series = linear_timeseries(start=1, length=10, freq=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series,
                    lags_future_covariates=[-1],
                    past_covariates=series,
                )
        self.assertEqual(
            "Must specify at least one series-lags pair.",
            str(e.exception),
        )

    def test_lagged_prediction_data_no_lags_specified_error(self):
        target = linear_timeseries(start=1, length=20, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(target_series=target)
        self.assertEqual(
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
            str(e.exception),
        )

    def test_lagged_prediction_data_series_too_short_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(
                target_series=series,
                lags=[-20, -1],
            )
        self.assertEqual(
            (
                "`target_series` must have at least "
                "`-min(lags) + max(lags) + 1` = 20 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(
                past_covariates=series,
                lags_past_covariates=[-20, -1],
            )
        self.assertEqual(
            (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 20 "
                "timesteps; instead, it only has 2."
            ),
            str(e.exception),
        )

    def test_lagged_prediction_data_invalid_lag_values_error(self):
        series = linear_timeseries(start=1, length=2, freq=1)
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(target_series=series, lags=[0])
        self.assertEqual(
            ("`lags` must be a `Sequence` containing only `int` values less than 0."),
            str(e.exception),
        )
        with self.assertRaises(ValueError) as e:
            create_lagged_prediction_data(
                target_series=series,
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
            create_lagged_prediction_data(
                target_series=series,
                future_covariates=series,
                lags_future_covariates=[1],
            )
        self.assertEqual(
            (
                "`lags_future_covariates` must be a `Sequence` containing only `int` values less than 1."
            ),
            str(e.exception),
        )

    def test_lagged_prediction_data_unspecified_lag_or_series_warning(self):
        series = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_prediction_data(
                target_series=series,
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
            _ = create_lagged_prediction_data(
                target_series=series,
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
            _ = create_lagged_prediction_data(
                target_series=series,
                lags=lags,
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
        # Specify `target_series` but not `lags` - this *should* throw
        # a warning when creating prediction data:
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_prediction_data(
                target_series=series,
                past_covariates=series,
                lags_past_covariates=lags,
            )
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`target_series` was specified without accompanying "
                    "`lags` and, thus, will be ignored."
                )
            )
        # Specify `target_series` but not `lags` - this *should* throw
        # a warning when creating prediction data:
        with warnings.catch_warnings(record=True) as w:
            _ = create_lagged_prediction_data(
                lags=lags,
                past_covariates=series,
                lags_past_covariates=lags,
            )
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue(
                str(w[0].message)
                == (
                    "`lags` was specified without accompanying "
                    "`target_series` and, thus, will be ignored."
                )
            )
