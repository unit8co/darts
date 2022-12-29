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
    Tests `create_lagged_prediction_data` function defined in `darts.utils.data.tabularization`. There
    are broadly two 'groups' of tests defined in this module:
        1. 'Generated Test Cases': these test that `create_lagged_prediction_data` produces the same outputs
        as a simplified implementation of the 'time intersection' feature generation method (see
        `darts.utils.data.tabularization` for further details). For these tests, the 'correct answer' is not
        directly specified; instead, it is generated from a set of input parameters using the set of simplified
        functions. The rationale behind this approach is that it allows for many different combinations of input
        values to be effortlessly tested. The drawback of this, however, is that the correctness of these tests
        assumes that the simplified functions have been implemented correctly - if this isn't the case, then these
        tests are not to be trusted. In saying this, these simplified functions are significantly easier to
        understand and debug than the `create_lagged_prediction_data` function they're helping to test.
        2. 'Specified Test Cases': these test that `create_lagged_prediction_data` returns an exactly specified
        output; these specified outputs are *not* 'generated' by another function. Although these 'specified'
        test cases tend to be simpler and less extensive than the 'generated' test cases, their correctness does
        not assume the correct implementation of any other function.
    """

    #
    #   Generated Cases Tests
    #

    # Input parameter combinations used to generate test cases:
    max_samples_per_ts_combos = (1, 2, None)
    # Lag dicts store `'vals'` passed to functions, in addition to max and min
    # lag values; latter two are used to generate 'correct answer' to each test;
    # `None` lag values indicate that series should not be used as feature:
    target_lag_combos = (
        {"vals": None, "max": None, "min": None},
        {"vals": [-1, -3], "max": 3, "min": 1},
        {"vals": [-3, -1], "max": 3, "min": 1},
    )
    # `past_covariates` and `future_covariates` lags can include `0`:
    covariates_lag_combos = (*target_lag_combos, {"vals": [0, -2], "max": 2, "min": 0})

    @staticmethod
    def create_multivariate_linear_timeseries(
        n_components: int, **kwargs
    ) -> TimeSeries:
        """
        Helper function that creates a `linear_timeseries` with a specified number of
        components. To help distinguish each component from one another, `i` is added on
        to each value of the `i`th component. Any additional key word arguments are passed
        to `linear_timeseries` (`start_value`, `end_value`, `start`, `end`, `length`, etc).
        """
        timeseries = []
        for i in range(n_components):
            # Values of each component is 1 larger than the last:
            timeseries_i = linear_timeseries(**kwargs) + i
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
        Helper function that returns the times shared by all of the specified series that can be used
        to create prediction features. This is performed by excluding those time points in each series that
        have fewer than `max_lag` timepoint preceeding them, where `max_lag` is the largest lag value requested
        for that series, since features cannot be constructed for these times. A series is considered to be
        'specified' if its corresponding lag (e.g. `lags` for `target`, or `lags_future` for `future`) is not
        `None`. If requested, the last `max_samples_per_ts` times are taken.
        """
        # `times` is initialized inside loop:
        times = None
        all_series = [target, past, future]
        all_lags = [lags, lags_past, lags_future]
        for series_i, lags_i in zip(all_series, all_lags):
            # If `lags` not specified, ignore series:
            if lags_i["vals"] is not None:
                times_i = series_i.time_index
                # To create lagged features for time `t`, we only need to have series values
                # from times `t - max_lag` to `t - min_lag`; hence, we need to include
                # next `min_lag` times after the final time as potential feature times:
                times_i = times_i.union(
                    [
                        times_i[-1] + i * series_i.freq
                        for i in range(1, lags_i["min"] + 1)
                    ]
                )
                # Exclude first `max_lag` times:
                times_i = times_i[lags_i["max"] :]
                # Initialise `times` if this is first specified series:
                if times is None:
                    times = times_i
                # Intersect `times` with `times_i` if `times` already initialized:
                else:
                    times = times.intersection(times_i)
        # If requested, take only the latest `max_samples_per_ts` samples:
        if max_samples_per_ts is not None:
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def construct_X_block(
        series: TimeSeries, feature_times: pd.Index, lags: Optional[Sequence[int]]
    ) -> np.array:
        """
        Helper function that creates the lagged features 'block' of a specific
        `series` (i.e. either `target_series`, `past_covariates`, or `future_covariates`);
        the feature matrix `X` is formed by concatenating the blocks of all of the specified
        series along the components axis. Please refer to the `create_lagged_features` docstring
        for further  details about the structure of the `X` feature matrix.

        If `lags` is `None`, then `None` will be returned in lieu of an array.

        The returned `X_block` is constructed by looping over each time in `feature_times`,
        finding the index position of that time in the series, and then for each lag value in
        `lags`, offset this index position by a particular lag value; this offset index is then
        used to extract all of the components at a single lagged time.

        Unlike the implementation found in `darts.utils.data.tabularization`, this function doesn't
        use any 'vectorisation' tricks, which makes it slower to run, but more easily interpretable.
        """
        if lags is None:
            X_block = None
        else:
            series_times = series.time_index
            # Some of the times in `feature_times` may occur after the end of `series_times`
            # (see `get_feature_times_single_series` for further details). In such cases,
            # we need to extend `series_times` up until the latest time in `feature_times`:
            if series_times[-1] < feature_times[-1]:
                is_range_idx = isinstance(series_times[0], int)
                if is_range_idx:
                    # `+ 1` since `stop` index is exclusive:
                    series_times = pd.RangeIndex(
                        start=series_times[0],
                        stop=feature_times[-1] + 1,
                        step=series.freq,
                    )
                else:
                    series_times = pd.date_range(
                        start=series_times[0], end=feature_times[-1], freq=series.freq
                    )
            array_vals = series.all_values(copy=False)[:, :, 0]
            X_block = []
            for time in feature_times:
                # Find position of time within series:
                time_idx = np.searchsorted(series_times, time)
                X_row = []
                for lag in lags:
                    # Offet by particular lag value:
                    idx_to_get = time_idx + lag
                    # Extract all components at this lagged time:
                    X_row.append(array_vals[idx_to_get, :].reshape(-1))
                # Concatenate together all lagged values into a single row:
                X_row = np.concatenate(X_row, axis=0)
                X_block.append(X_row)
            # Concatenate all rows (i.e. observations) together into a block:
            X_block = np.stack(X_block, axis=0)
        return X_block

    def test_lagged_prediction_data_equal_freq_range_index(self):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with range time indices of equal
        frequencies. Since all of the timeseries are of the same frequency,
        the implementation of the 'moving window' method is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, and different values, but
        # they're all of the same frequency:
        target = self.create_multivariate_linear_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, end=20, freq=1
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, end=23, freq=2
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, end=26, freq=3
        )
        # Conduct test for each input parameter combo:
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags["vals"] else None,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
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
                max_samples_per_ts,
            )
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_prediction_data_equal_freq_datetime_index(self):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with datetime time indices of equal
        frequencies. Since all of the timeseries are of the same frequency,
        the implementation of the 'moving window' method is being tested here.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, and different values, but
        # they're all of the same frequency:
        target = self.create_multivariate_linear_timeseries(
            n_components=2,
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/16/2000"),
            freq="2d",
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            end=pd.Timestamp("1/18/2000"),
            freq="2d",
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="2d",
        )
        # Conduct test for each input parameter combo:
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags["vals"] else None,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
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
                max_samples_per_ts,
            )
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_prediction_data_unequal_freq_range_index(self):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with range time indices of unequal
        frequencies. Since all of the timeseries are *not* of the same frequency,
        the implementation of the 'time intersection' method is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, different values, and of
        # different frequencies:
        target = self.create_multivariate_linear_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, end=20, freq=1
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, end=23, freq=2
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, end=26, freq=3
        )
        # Conduct test for each input parameter combo:
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags["vals"] else None,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
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
                max_samples_per_ts,
            )
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_prediction_data_unequal_freq_datetime_index(self):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with datetime time indices of unequal
        frequencies. Since all of the timeseries are *not* of the same frequency,
        the implementation of the 'time intersection' method is being tested here.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, different values, and of
        # different frequencies:
        target = self.create_multivariate_linear_timeseries(
            n_components=2,
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/2/2000"),
            end=pd.Timestamp("1/16/2000"),
            freq="2d",
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            end=pd.Timestamp("1/18/2000"),
            freq="2d",
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            end=pd.Timestamp("1/20/2000"),
            freq="2d",
        )
        # Conduct test for each input parameter combo:
        param_combos = product(
            self.target_lag_combos,
            self.covariates_lag_combos,
            self.covariates_lag_combos,
            self.max_samples_per_ts_combos,
        )
        for (lags, lags_past, lags_future, max_samples_per_ts) in param_combos:
            all_lags = (lags["vals"], lags_past["vals"], lags_future["vals"])
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags["vals"] else None,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
                lags=lags["vals"],
                lags_past_covariates=lags_past["vals"],
                lags_future_covariates=lags_future["vals"],
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
                max_samples_per_ts,
            )
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    #
    #   Specified Cases Tests
    #

    def test_lagged_prediction_data_extend_past_and_future_covariates_range_idx(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles case where features
        can be created for a time that is *not* contained in `target_series`, `past_covariates`
        and/or `future_covariates`. This particular test checks this behaviour by using
        range index timeseries.
        """
        # Define series and lags such that a prediction feature can be generated
        # for time `target.end_time() + target.freq`, but so that this time isn't contained
        # in none of the define series. Even so, the specified lags are sufficiently
        # large here so that the series timeseries don't need to have values going up
        # to `target.end_time() + target.freq`:
        target = linear_timeseries(start=0, end=9, start_value=1, end_value=2)
        lags = [-1]
        past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
        lags_past = [-2]
        future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        lags_future = [-4]
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from the very last values of each series:
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-1, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time() + target.freq)
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))

    def test_lagged_prediction_data_extend_past_and_future_covariates_datetime_idx(
        self,
    ):
        """
        Tests that `create_lagged_prediction_data` correctly handles case where features
        can be created for a time that is *not* contained in `target_series`, `past_covariates`
        and/or `future_covariates`. This particular test checks this behaviour by using
        datetime index timeseries.
        """
        # Define series and lags such that a prediction feature can be generated
        # for time `target.end_time() + target.freq`, but so that this time isn't contained
        # in none of the define series. Even so, the specified lags are sufficiently
        # large here so that the series timeseries don't need to have values going up
        # to `target.end_time() + target.freq`:
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
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from the very last values of each series:
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-1, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time() + target.freq)
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))

    def test_lagged_prediction_data_single_point_range_idx(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles case
        where only one possible training point can be generated.  This
        particular test checks this behaviour by using range index timeseries.
        """
        # Can only create feature using first value of target (i.e. `0`):
        target = linear_timeseries(start=0, length=1, start_value=0, end_value=1)
        expected_X = np.zeros((1, 1, 1))
        # Prediction time extend beyond end of series:
        lag = 5
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target, lags=[-lag], use_moving_windows=use_moving_windows
            )
            self.assertTrue(np.allclose(expected_X, X))
            # Should only have one sample, generated for
            # `t = target.end_time() + lag * target.freq`:
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time() + lag * target.freq)

    def test_lagged_prediction_data_single_point_datetime_idx(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles case
        where only one possible training point can be generated.  This
        particular test checks this behaviour by using datetime index timeseries.
        """
        # Can only create feature using first value of target (i.e. `0`):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=1, start_value=0, end_value=1
        )
        expected_X = np.zeros((1, 1, 1))
        # Prediction time extend beyond end of series:
        lag = 5
        for use_moving_windows in (False, True):
            X, times = create_lagged_prediction_data(
                target, lags=[-lag], use_moving_windows=use_moving_windows
            )
            self.assertTrue(np.allclose(expected_X, X))
            # Should only have one sample, generated for
            # `t = target.end_time() + lag * target.freq`:
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time() + lag * target.freq)

    def test_lagged_prediction_data_zero_lags_range_idx(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles case when
        `0` is included in `lags_past_covariates` and `lags_future_covariates`
        (i.e. when we're using the values of `past_covariates` and
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point). This particular test checks this behaviour by using
        range index timeseries.
        """
        # Define `past` and `future` so their only value occurs at the same time as
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the values of `past` and `future` at the same time as the label (i.e. a lag
        # of `0` away from the only feature time):
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        past = linear_timeseries(start=1, length=1, start_value=1, end_value=2)
        future = linear_timeseries(start=1, length=1, start_value=2, end_value=3)
        # X comprises of first value of `target` (i.e. 0) and only values in `past` and `future`:
        expected_X = np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1)
        # Check correctness for 'moving windows' and 'time intersection' methods:
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
            self.assertTrue(np.allclose(expected_X, X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_prediction_data_zero_lags_datetime_idx(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles case when
        `0` is included in `lags_past_covariates` and `lags_future_covariates`
        (i.e. when we're using the values of `past_covariates` and
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point). This particular test checks this behaviour by using
        datetime index timeseries.
        """
        # Define `past` and `future` so their only value occurs at the same time as
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the values of `past` and `future` at the same time as the label (i.e. a lag
        # of `0` away from the only feature time):
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=2, start_value=0, end_value=1
        )
        past = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=1, end_value=2
        )
        future = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=1, start_value=2, end_value=3
        )
        # X comprises of first value of `target` (i.e. 0) and only values in `past` and `future`:
        expected_X = np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1)
        # Check correctness for 'moving windows' and 'time intersection' methods:
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
            self.assertTrue(np.allclose(expected_X, X))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_prediction_data_no_shared_times_error(self):
        """
        Tests that `create_lagged_prediction_data` throws correct error
        when the specified series do not share any times in common
        for creating features and labels.
        """
        # Have `series_1` share no overlap with `series_2`:
        series_1 = linear_timeseries(start=0, length=4, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=4, freq=1)
        lags = [-1]
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series_1,
                    lags=lags,
                    past_covariates=series_2,
                    lags_past_covariates=lags,
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                "Specified series do not share any common times for which features can be created.",
                str(e.exception),
            )
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

    def test_lagged_prediction_data_no_specified_series_lags_pairs_error(self):
        """
        Tests that `create_lagged_prediction_data` throws correct error
        when no lags-series pairs are specified.
        """
        # Define some arbitrary inputs:
        series = linear_timeseries(start=1, length=10, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Three warnings will be thrown indicating that series
            # is specified without corresponding lags or vice
            # versa - ignore these warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as e:
                    create_lagged_prediction_data(
                        target_series=series,
                        lags_future_covariates=[-1],
                        past_covariates=series,
                        use_moving_windows=use_moving_windows,
                    )
            self.assertEqual(
                "Must specify at least one series-lags pair.",
                str(e.exception),
            )

    def test_lagged_prediction_data_no_lags_specified_error(self):
        """
        Tests that `create_lagged_prediction_data` throws correct error
        when no lags are specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=target, use_moving_windows=use_moving_windows
                )
            self.assertEqual(
                "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
                str(e.exception),
            )

    def test_lagged_prediction_data_series_too_short_error(self):
        """
        Tests that `create_lagged_prediction_data` throws correct error
        when supplied series is too short to generate any
        features from using the specified `lags`.
        """
        series = linear_timeseries(start=1, length=2, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series,
                    lags=[-20, -1],
                    use_moving_windows=use_moving_windows,
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
                    use_moving_windows=use_moving_windows,
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
        """
        Tests that `create_lagged_prediction_data` throws correct
        error when invalid lag values are specified. More specifically:
            - If `lags` contains any value greater than `-1`, an error
            should be thrown (since times > `-1` are used for labels).
            - If `lags_past_covariates`/`lags_future_covariates` contains
            any value greater than `0`, an error should be thrown (i.e. we
            shouldn't be predicting the past using the future).
        """
        series = linear_timeseries(start=1, length=2, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Test invalid `lags` values:
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series,
                    lags=[0],
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                (
                    "`lags` must be a `Sequence` containing only `int` values less than 0."
                ),
                str(e.exception),
            )
            # Test invalid `lags_past_covariates` values:
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series,
                    past_covariates=series,
                    lags_past_covariates=[1],
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                (
                    "`lags_past_covariates` must be a `Sequence` containing only `int` values less than 1."
                ),
                str(e.exception),
            )
            # Test invalid `lags_future_covariates` values:
            with self.assertRaises(ValueError) as e:
                create_lagged_prediction_data(
                    target_series=series,
                    future_covariates=series,
                    lags_future_covariates=[1],
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                (
                    "`lags_future_covariates` must be a `Sequence` containing only `int` values less than 1."
                ),
                str(e.exception),
            )

    def test_lagged_prediction_data_unspecified_lag_or_series_warning(self):
        """
        Tests that `create_lagged_prediction_data` throws correct
        user warnings when a series is specified without any
        corresponding lags, or vice versa.
        """
        series = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        # Check warnings thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    lags=lags,
                    future_covariates=series,
                    use_moving_windows=use_moving_windows,
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
            # Specify `lags_future_covariates` but not `future_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    lags=lags,
                    lags_future_covariates=lags,
                    use_moving_windows=use_moving_windows,
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
            # `past_covariates` but not `lags_past_covariates`, and `target_series`
            # but not `lags`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    lags=lags,
                    past_covariates=series,
                    lags_future_covariates=lags,
                    use_moving_windows=use_moving_windows,
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
            # Specify `target_series` but not `lags` - this *should* throw
            # a warning when creating prediction data:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    use_moving_windows=use_moving_windows,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertEqual(
                    str(w[0].message),
                    (
                        "`target_series` was specified without accompanying "
                        "`lags` and, thus, will be ignored."
                    ),
                )
            # Specify `target_series` but not `lags` - this *should* throw
            # a warning when creating prediction data:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    lags=lags,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    use_moving_windows=use_moving_windows,
                )
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertEqual(
                    str(w[0].message),
                    (
                        "`lags` was specified without accompanying "
                        "`target_series` and, thus, will be ignored."
                    ),
                )
