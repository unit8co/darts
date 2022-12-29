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
    Tests the `create_lagged_training_data` function defined in `darts.utils.data.tabularization`. There are broadly
    two 'groups' of tests defined in this module:
        1. 'Generated Test Cases': these test that `create_lagged_training_data` produces the same outputs
        as a simplified implementation of the 'time intersection' feature generation method (see
        `darts.utils.data.tabularization` for further details). For these tests, the 'correct answer' is not
        directly specified; instead, it is generated from a set of input parameters using the set of simplified
        functions. The rationale behind this approach is that it allows for many different combinations of input
        values to be effortlessly tested. The drawback of this, however, is that the correctness of these tests
        assumes that the simplified functions have been implemented correctly - if this isn't the case, then these
        tests are not to be trusted. In saying this, these simplified functions are significantly easier to
        understand and debug than the `create_lagged_training_data` function they're helping to test.
        2. 'Specified Test Cases': these test that `create_lagged_training_data` returns an exactly specified
        output; these specified outputs are *not* 'generated' by another function. Although these 'specified'
        test cases tend to be simpler and less extensive than the 'generated' test cases, their correctness does
        not assume the correct implementation of any other function.
    """

    #
    #   Generated Cases Tests
    #

    # Input parameter combinations used to generate test cases:
    output_chunk_length_combos = (1, 3)
    multi_models_combos = (False, True)
    max_samples_per_ts_combos = (1, 2, None)
    # Lag dicts store `'vals'` passed to functions, in addition to max and min
    # lag values; latter two are used to generate 'correct answer' to each test;
    # `None` lag values indicate that series should not be used as feature:
    target_lag_combos = (
        {"vals": None, "max": None, "min": None},
        {"vals": [-1, -3], "max": 3, "min": 1},
        {"vals": [-1, -3], "max": 3, "min": 1},
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
        lags: Optional[Sequence[int]],
        lags_past: Optional[Sequence[int]],
        lags_future: Optional[Sequence[int]],
        output_chunk_length: Optional[int],
        max_samples_per_ts: Optional[int],
    ):
        """
        Helper function that returns the times shared by all of the specified series that can be used
        to create features and labels. This is performed by using the helper function
        `get_feature_times_single_series` (defined below) to extract the feature times of each of
        the specified series, and then intersecting these features times with one another. A series is
        considered to be 'specified' if its corresponding lag (e.g. `lags` for `target`, or `lags_future`
        for `future`) is not `None`. If requested, the last `max_samples_per_ts` times are taken.
        """
        # Get feature times for `target_series`:
        times = CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
            target, lags["min"], lags["max"], output_chunk_length, is_target=True
        )
        include_past = lags_past["vals"] is not None
        if include_past:
            # Intersect `target_series` feature times with `past_covariates` feature times:
            past_times = (
                CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
                    past, lags_past["min"], lags_past["max"]
                )
            )
            times = times.intersection(past_times)
        include_future = lags_future["vals"] is not None
        if include_future:
            # Intersect `target_series` feature times with `future_covariates` feature times:
            future_times = (
                CreateLaggedTrainingDataTestCase.get_feature_times_single_series(
                    future, lags_future["min"], lags_future["max"]
                )
            )
            times = times.intersection(future_times)
        # Take most recent `max_samples_per_ts` samples if requested:
        if (max_samples_per_ts is not None) and (len(times) > max_samples_per_ts):
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_feature_times_single_series(
        series: TimeSeries,
        min_lag: Optional[int],
        max_lag: Optional[int],
        output_chunk_length: Optional[int] = None,
        is_target: bool = False,
    ) -> pd.Index:
        """
        Helper function that returns all the times within a *single series* that can be used to
        create features and labels. This means that the `series` and `*_lag` inputs must correspond
        to *either* the `target_series`, the `past_covariates`, *or*  the `future_covariates`.

        If `series` is `target_series`, then:
            - The last `output_chunk_length - 1` times are exluded, since these times do not
            have `(output_chunk_length - 1)` values ahead of them and, therefore, we can't
            create labels for these values.
            - The first `max_lag` values are excluded, since these values don't have `max_lag`
            values preceeding them, which means that we can't create features for these times.

        If `series` is either `past_covariates` or `future_covariates`, then the first `max_lag`
        times are excluded, since these values don't have `max_lag` values preceeding them, which
        means that we can't create features for these times.

        This function is called by the `get_feature_times` helper function.
        """
        # Assume that if `min_lag` is specified, so is `max_lag`:
        lags_specified = min_lag is not None
        times = series.time_index
        # To create lagged features for time `t`, we only need to have series values
        # from times `t - max_lag` to `t - min_lag`; hence, we need to include
        # next `min_lag` times after the final time as potential feature times. We
        # don't need to do this for the target series, however, since this series must
        # include all of the feature times (these time are used as labels):
        if not is_target and lags_specified:
            times = times.union(
                [times[-1] + i * series.freq for i in range(1, min_lag + 1)]
            )
        if lags_specified:
            times = times[max_lag:]
        if is_target and (output_chunk_length > 1):
            times = times[: -output_chunk_length + 1]
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

    @staticmethod
    def create_y(
        target: TimeSeries,
        feature_times: pd.Index,
        output_chunk_length: int,
        multi_models: bool,
    ) -> np.ndarray:
        """
        Helper function that constructs the labels array `y` from the target series.
        This is done by by looping over each time in `feature_times`, finding the index
        position of that time in the target series, and then for each timestep ahead of
        this time we wish to predict, offset this time index by this timestep. This offset
        index is then used to extract all components of the target series at this time which
        is to be predicted. Please refer to the `create_lagged_features` docstring for further
        details about the structure of the `y` labels matrix.
        """
        array_vals = target.all_values(copy=False)
        y = []
        for time in feature_times:
            time_idx = np.searchsorted(target.time_index, time)
            # If `multi_models = True`, want to predict all of the values from time `t` to
            # time `t + output_chunk_lenth - 1`; if `multi_models = False`, only want to
            # predict time `t + output_chunk_length - 1`:
            timesteps_ahead = (
                range(output_chunk_length)
                if multi_models
                else (output_chunk_length - 1,)
            )
            y_row = []
            for i in timesteps_ahead:
                # Offset index of `time` by how many timesteps we're looking ahead:
                idx_to_get = time_idx + i
                # Extract all components at this time:
                y_row.append(array_vals[idx_to_get, :, 0].reshape(-1))
            # Concatenate all times to be predicted into a single row:
            y_row = np.concatenate(y_row, axis=0)
            y.append(y_row)
        # Stack all rows into a single labels matrix:
        y = np.stack(y, axis=0)
        return y

    def test_lagged_training_data_equal_freq_range_index(self):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all of the combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with range time indices of equal
        frequencies. Since all of the timeseries are of the same frequency,
        the implementation of the 'moving window' method is being tested here.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, and different values, but
        # they're all of the same frequency:
        target = self.create_multivariate_linear_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, length=8, freq=2
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, length=9, freq=2
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, length=10, freq=2
        )
        # Conduct test for each input parameter combo:
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
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
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
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(
                target, feats_times, output_chunk_length, multi_models
            )
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_equal_freq_datetime_index(self):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all of the combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
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
            length=8,
            freq="2d",
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/4/2000"),
            length=9,
            freq="2d",
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/6/2000"),
            length=10,
            freq="2d",
        )
        # Conduct test for each input parameter combo:
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
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
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
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(
                target, feats_times, output_chunk_length, multi_models
            )
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_unequal_freq_range_index(self):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all of the combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with range time indices of unequal
        frequencies. Since all of the timeseries are *not* of the same frequency,
        the implementation of the 'time intersection' method is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, different values, and different
        # frequencies:
        target = self.create_multivariate_linear_timeseries(
            n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3
        )
        # Conduct test for each input parameter combo:
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
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
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
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(
                target, feats_times, output_chunk_length, multi_models
            )
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    def test_lagged_training_data_unequal_freq_datetime_index(self):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all of the combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with datetime time indices of unequal
        frequencies. Since all of the timeseries are *not* of the same frequency,
        the implementation of the 'time intersection' method is being tested here.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, different values, and different
        # frequencies:
        target = self.create_multivariate_linear_timeseries(
            n_components=2,
            start_value=0,
            end_value=10,
            start=pd.Timestamp("1/1/2000"),
            length=20,
            freq="d",
        )
        past = self.create_multivariate_linear_timeseries(
            n_components=3,
            start_value=10,
            end_value=20,
            start=pd.Timestamp("1/2/2000"),
            length=10,
            freq="2d",
        )
        future = self.create_multivariate_linear_timeseries(
            n_components=4,
            start_value=20,
            end_value=30,
            start=pd.Timestamp("1/3/2000"),
            length=7,
            freq="3d",
        )
        # Conduct test for each input parameter combo:
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
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                past_covariates=past if lags_past["vals"] else None,
                future_covariates=future if lags_future["vals"] else None,
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
            # Construct `X` by constructing each block, then concatenate these
            # blocks together along component axis:
            X_target = self.construct_X_block(target, feats_times, lags["vals"])
            X_past = self.construct_X_block(past, feats_times, lags_past["vals"])
            X_future = self.construct_X_block(future, feats_times, lags_future["vals"])
            all_X = (X_target, X_past, X_future)
            to_concat = [x for x in all_X if x is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            expected_y = self.create_y(
                target, feats_times, output_chunk_length, multi_models
            )
            # Number of observations should match number of feature times:
            self.assertEqual(X.shape[0], len(feats_times))
            self.assertEqual(y.shape[0], len(feats_times))
            self.assertEqual(X.shape[0], len(times))
            self.assertEqual(y.shape[0], len(times))
            # Check that outputs match:
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))
            self.assertTrue(feats_times.equals(times))

    #
    #   Specified Cases Tests
    #

    def test_lagged_training_data_extend_past_and_future_covariates_range_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case where features
        and labels can be created for a time that is *not* contained in `past_covariates`
        and/or `future_covariates`. This particular test checks this behaviour by using
        range index timeseries.
        """
        # Define series and lags such that a training example can be generated
        # for time `target.end_time()`, but so that this time isn't contained in either
        # `past` or `future`. Even so, `lags_past` and `lags_future` are sufficiently
        # large here so that the past and future timeseries don't need to have
        # values going up to `target.end_time()`:
        target = linear_timeseries(start=0, end=10, start_value=1, end_value=2)
        lags = [-1]
        past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
        lags_past = [-2]
        future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        lags_future = [-4]
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from second-to-last value of `target` (i.e.
        # the value immediately prior to the label), and the very last values of
        # `past` and `future`:
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-2, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        # Label is very last value of `target`:
        expected_y = target.all_values(copy=False)[-1, :, 0]
        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        for use_moving_windows in (False, True):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time())
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))

    def test_lagged_training_data_extend_past_and_future_covariates_datetime_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case where features
        and labels can be created for a time that is *not* contained in `past_covariates`
        and/or `future_covariates`. This particular test checks this behaviour by using
        datetime index timeseries.
        """
        # Define series and lags such that a training example can be generated
        # for time `target.end_time()`, but this time isn't contained in either
        # `past` or `future`. Even so, `lags_past` and `lags_future` are sufficiently
        # large here so that the past and future timeseries don't need to have
        # values going up to `target.end_time()`:
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
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from second-to-last value of `target` (i.e.
        # the value immediately prior to the label), and the very last values of
        # `past` and `future`:
        expected_X = np.concatenate(
            [
                target.all_values(copy=False)[-2, :, 0],
                past.all_values(copy=False)[-1, :, 0],
                future.all_values(copy=False)[-1, :, 0],
            ]
        ).reshape(1, -1)
        # Label is very last value of `target`:
        expected_y = target.all_values(copy=False)[-1, :, 0]
        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        for use_moving_windows in (False, True):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length=1,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=use_moving_windows,
            )
            self.assertEqual(times[0], target.end_time())
            self.assertTrue(np.allclose(expected_X, X[:, :, 0]))
            self.assertTrue(np.allclose(expected_y, y[:, :, 0]))

    def test_lagged_training_data_single_point_range_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case
        where only one possible training point can be generated.  This
        particular test checks this behaviour by using range index timeseries.
        """
        # Can only create feature using first value of series (i.e. `0`)
        # and can only create label using last value of series (i.e. `1`)
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        output_chunk_length = 1
        lags = [-1]
        expected_X = np.zeros((1, 1, 1))
        expected_y = np.ones((1, 1, 1))
        # Test correctness for 'moving window' and for 'time intersection' methods, as well
        # as for different `multi_models` values:
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                lags=lags,
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(expected_X, X))
            self.assertTrue(np.allclose(expected_y, y))
            # Should only have one sample, generated for `t = target.end_time()`:
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_training_data_single_point_datetime_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case
        where only one possible training point can be generated. This
        particular test checks this behaviour by using datetime index timeseries.
        """
        # Can only create feature using first value of series (i.e. `0`)
        # and can only create label using last value of series (i.e. `1`)
        target = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=2, start_value=0, end_value=1
        )
        output_chunk_length = 1
        lags = [-1]
        expected_X = np.zeros((1, 1, 1))
        expected_y = np.ones((1, 1, 1))
        # Test correctness for 'moving window' and for 'time intersection' methods, as well
        # as for different `multi_models` values:
        for (use_moving_windows, multi_models) in product([False, True], [False, True]):
            X, y, times = create_lagged_training_data(
                target,
                output_chunk_length,
                lags=lags,
                multi_models=multi_models,
                use_moving_windows=use_moving_windows,
            )
            self.assertTrue(np.allclose(expected_X, X))
            self.assertTrue(np.allclose(expected_y, y))
            # Should only have one sample, generated for `t = target.end_time()`:
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_training_data_zero_lags_range_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case when
        `0` is included in `lags_past_covariates` and `lags_future_covariates`
        (i.e. when we're using the values of `past_covariates` and
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point). This particular test checks this behaviour by using
        range index timeseries.
        """
        # Define `past` and `future` so their only value occurs at the same time at
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the values of `past` and `future` at the same time as the label (i.e. a lag
        # of `0` away from the only feature time):
        target = linear_timeseries(start=0, length=2, start_value=0, end_value=1)
        past = linear_timeseries(
            start=target.end_time(), length=1, start_value=1, end_value=2
        )
        future = linear_timeseries(
            start=target.end_time(), length=1, start_value=2, end_value=3
        )
        # X comprises of first value of `target` (i.e. 0) and only values in `past` and `future`:
        expected_X = np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1)
        expected_y = np.ones((1, 1, 1))
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
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
            self.assertTrue(np.allclose(expected_X, X))
            self.assertTrue(np.allclose(expected_y, y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_training_data_zero_lags_datetime_idx(self):
        """
        Tests that `create_lagged_training_data` correctly handles case when
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
            start=target.end_time(), length=1, start_value=1, end_value=2
        )
        future = linear_timeseries(
            start=target.end_time(), length=1, start_value=2, end_value=3
        )
        # X comprises of first value of `target` (i.e. 0) and only values in `past` and `future`:
        expected_X = np.array([0.0, 1.0, 2.0]).reshape(1, 3, 1)
        expected_y = np.ones((1, 1, 1))
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
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
            self.assertTrue(np.allclose(expected_X, X))
            self.assertTrue(np.allclose(expected_y, y))
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], target.end_time())

    def test_lagged_training_data_no_shared_times_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
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
                create_lagged_training_data(
                    target_series=series_1,
                    output_chunk_length=1,
                    lags=lags,
                    past_covariates=series_2,
                    lags_past_covariates=lags,
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                "Specified series do not share any common times for which features can be created.",
                str(e.exception),
            )

    def test_lagged_training_data_no_specified_series_lags_pairs_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when no lags-series pairs are specified.
        """
        # Define some arbitrary inputs:
        series_1 = linear_timeseries(start=1, length=10, freq=1)
        series_2 = linear_timeseries(start=1, length=10, freq=2)
        lags = [-1]
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Warnings will be thrown indicating that `past_covariates`
            # is specified without `lags_past_covariates` - ignore this:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as e:
                    create_lagged_training_data(
                        target_series=series_1,
                        output_chunk_length=1,
                        lags_past_covariates=lags,
                        use_moving_windows=use_moving_windows,
                    )
            self.assertEqual(
                "Must specify at least one series-lags pair.",
                str(e.exception),
            )
            # Warnings will be thrown indicating that `past_covariates`
            # is specified without `lags_past_covariates`, and that
            # `lags_future_covariates` specified without
            # `future_covariates` - ignore both warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as e:
                    create_lagged_training_data(
                        target_series=series_1,
                        output_chunk_length=1,
                        lags_future_covariates=lags,
                        past_covariates=series_2,
                        use_moving_windows=use_moving_windows,
                    )
            self.assertEqual(
                "Must specify at least one series-lags pair.",
                str(e.exception),
            )

    def test_lagged_training_data_invalid_output_chunk_length_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when `output_chunk_length` is set to a non-`int` value (e.g. a
        `float`) or a non-positive value (e.g. `0`).
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=0,
                    lags=lags,
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                "`output_chunk_length` must be a positive `int`.",
                str(e.exception),
            )
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=1.1,
                    lags=lags,
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                "`output_chunk_length` must be a positive `int`.",
                str(e.exception),
            )

    def test_lagged_training_data_no_lags_specified_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when no lags are specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=1,
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
                str(e.exception),
            )

    def test_lagged_training_data_series_too_short_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when supplied `target_series` is too short to generate any
        features/labels from using the specified `lags` and
        `output_chunk_length` values, and when supplied
        `past_covariates`/`future_covariates` is too short to generate
        any features from using the specified
        `lags_past_covariates`/`lags_future_covariates`.
        """
        # `lags` and `output_chunk_length` too large test:
        series = linear_timeseries(start=1, length=2, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=5,
                    lags=[-20, -10],
                    use_moving_windows=use_moving_windows,
                )
            self.assertEqual(
                (
                    "`target_series` must have at least "
                    "`-min(lags) + output_chunk_length` = 25 "
                    "timesteps; instead, it only has 2."
                ),
                str(e.exception),
            )
            # `lags_past_covariates` too large test:
            with self.assertRaises(ValueError) as e:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    past_covariates=series,
                    lags_past_covariates=[-5, -3],
                    use_moving_windows=use_moving_windows,
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
        """
        Tests that `create_lagged_training_data` throws correct
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
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
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
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
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
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
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

    def test_lagged_training_data_unspecified_lag_or_series_warning(self):
        """
        Tests that `create_lagged_training_data` throws correct
        user warnings when a series is specified without any
        corresponding lags, or vice versa. The only exception
        to this is that a warning shouldn't be thrown if
        `target_series` is specified without any `lags`, since
        the `target_series` is still used to construct labels,
        even if its not used to create features (i.e. `target_series`
        is not ignored if `lags` is not specified).
        """
        series = linear_timeseries(start=1, length=20, freq=1)
        lags = [-1]
        # Check warnings thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Specify `future_covariates`, but not `lags_future_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
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
            # Specify `lags_future_covariates`, but not `future_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
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
            # `past_covariates` but not `lags_past_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    lags=lags,
                    output_chunk_length=1,
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
            # Specify `target_series`, but not `lags` - unlike previous tests,
            # this should *not* throw a warning:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    use_moving_windows=use_moving_windows,
                )
                self.assertEqual(len(w), 0)
