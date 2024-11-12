import itertools
import warnings
from collections.abc import Sequence
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts import concatenate as darts_concatenate
from darts.logging import get_logger, raise_if_not, raise_log
from darts.utils.data.tabularization import create_lagged_prediction_data
from darts.utils.timeseries_generation import linear_timeseries


class TestCreateLaggedPredictionData:
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
    #   Helper Functions for Generated Test Cases
    #

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
        to create features and labels. This is performed by using the helper functions
        `get_feature_times_target_or_past` and `get_feature_times_future` (both defined below) to extract
        the feature times from the target series, past covariates, and future covariates respectively, and
        then intersecting these features times with one another. A series is considered to be 'specified'
        if its corresponding lag (e.g. `lags` for `target`, or `lags_future` for `future`) is not `None`.
        If requested, the last `max_samples_per_ts` times are taken.

        This function is basically a simplified implementation of `get_feature_times` in `tabularization.py`
        that only works for `is_training = False`.
        """
        # `times` is initialized inside loop:
        times = None
        all_series = [target, past, future]
        all_lags = [lags, lags_past, lags_future]
        for i, (series_i, lags_i) in enumerate(zip(all_series, all_lags)):
            # If `lags` not specified, ignore series:
            lags_specified = lags_i is not None
            is_target_or_past = i < 2
            if lags_specified:
                if is_target_or_past:
                    times_i = (
                        TestCreateLaggedPredictionData.get_feature_times_target_or_past(
                            series_i, lags_i
                        )
                    )
                else:
                    times_i = TestCreateLaggedPredictionData.get_feature_times_future(
                        series_i, lags_i
                    )
            else:
                times_i = None
            if times_i is not None:
                # Initialise `times` if this is first specified series:
                if times is None:
                    times = times_i
                # Intersect `times` with `times_i` if `times` already initialized:
                else:
                    times = times.intersection(times_i)
        # Take most recent `max_samples_per_ts` samples if requested:
        if (max_samples_per_ts is not None) and (len(times) > max_samples_per_ts):
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_feature_times_target_or_past(
        series: TimeSeries,
        lags: Sequence[int],
    ) -> pd.Index:
        """
        Helper function called by `get_feature_times` that extracts all of the times within `target_series`
        *or* `past_covariates` that can be used to create features; because we can assume all of the values
        in `lags` are negative for `target_series` and `past_covariates`, the feature time extract process is
        the same for these series when constructing prediction data.

        More specifically, we can create features for times within `serues` that have at least `max_lag = -min(lags)`
        values preceeding them, since these preceeding values are required to construct a feature vector for that time.
        Since the first `max_lag` times do not fulfill this condition, they are exluded.

        Because we're create features for predicting here, we don't need to worry about whether we can produce
        labels corresponding each time.

        Importantly, features can be constructed for times that occur after the end of `series`: this is because:
            1. We don't need to have all the `series` values up to time `t` to construct a feature for this time;
            instead, we only need to have the values from time `t - min_lag` to `t - max_lag`, where
            `min_lag = -max(lags)` and `max_lag = -min(lags)`. In other words, the latest feature we can create
            for `series` occurs at `series.end_time() + min_lag * series.freq`.
            2. We don't need to use the values of `series` to construct labels when we're creating prediction data,
            so we're able to create a feature for time `t` without having to worry about whether we can construct
            a corresponding label for this time.
        """
        times = series.time_index
        min_lag = -max(lags)
        times = times.union([
            times[-1] + i * series.freq for i in range(1, min_lag + 1)
        ])
        max_lag = -min(lags)
        times = times[max_lag:]
        return times

    @staticmethod
    def get_feature_times_future(
        future_covariates: TimeSeries,
        future_covariates_lags: Sequence[int],
    ) -> pd.Index:
        """
        Helper function called by `get_feature_times` that extracts all of the times within
        `future_covariates` that can be used to create features.

        Unlike the lag values for `target_series` and `past_covariates`, the values in
        `future_covariates_lags` can be negative, zero, or positive. This means that
        `min_lag = -max(future_covariates_lags)` and `max_lag = -min(future_covariates_lags)`
        are *not* guaranteed to be positive here: they could be negative (corresponding to
        a positive value in `future_covariates_lags`), zero, or positive (corresponding to
        a negative value in `future_covariates_lags`). With that being said, the relationship
        `min_lag <= max_lag` always holds.

        Consequently, we need to consider three scenarios when finding feature times
        for `future_covariates`:
            1. Both `min_lag` and `max_lag` are positive, which indicates that all of
            the lag values in `future_covariates_lags` are negative (i.e. only values before
            time `t` are used to create a feature from time `t`). In this case, `min_lag`
            and `max_lag` correspond to the smallest magnitude and largest magnitude *negative*
            lags in `future_covariates_lags` respectively. This means we *can* create features for
            times that extend beyond the end of `future_covariates`; additionally, we're unable
            to create features for the first `min_lag` times (see docstring for `get_feature_times_past`).
            2. Both `min_lag` and `max_lag` are non-positive. In this case, `abs(min_lag)` and `abs(max_lag)`
            correspond to the largest and smallest magnitude lags in `future_covariates_lags` respectively;
            note that, somewhat confusingly, `abs(max_lag) <= abs(min_lag)` here. This means that we *can* create f
            features for times that occur before the start of `future_covariates`; the reasoning for this is
            basically the inverse of Case 1 (i.e. we only need to know the values from times `t + abs(max_lag)`
            to `t + abs(min_lag)` to create a feature for time `t`). Additionally, we're unable to create features
            for the last `abs(min_lag)` times in the series, since these times do not have `abs(min_lag)` values
            after them.
            3. `min_lag` is non-positive (i.e. zero or negative), but `max_lag` is positive. In this case,
            `abs(min_lag)` is the magnitude of the largest *non-negative* lag value in `future_covariates_lags`
            and `max_lag` is the largest *negative* lag value in `future_covariates_lags`. This means that we
            *cannot* create features for times that occur before the start of `future_covariates`, nor for
            times that occur after the end of `future_covariates`; this is because we must have access to
            both times before *and* after time `t` to create a feature for this time, which clearly can't
            be acieved for times extending before the start or after the end of the series. Moreover,
            we must exclude the first `max_lag` times and the last `abs(min_lag)` times, since these
            times do not have enough values before or after them respectively.
        """
        times = future_covariates.time_index
        min_lag = -max(future_covariates_lags)
        max_lag = -min(future_covariates_lags)
        # Case 1:
        if (min_lag > 0) and (max_lag > 0):
            # Can create features for times extending after the end of `future_covariates`:
            times = times.union([
                times[-1] + i * future_covariates.freq for i in range(1, min_lag + 1)
            ])
            # Can't create features for first `max_lag` times in series:
            times = times[max_lag:]
        # Case 2:
        elif (min_lag <= 0) and (max_lag <= 0):
            # Can create features for times before the start of `future_covariates`:
            times = times.union([
                times[0] - i * future_covariates.freq
                for i in range(1, abs(max_lag) + 1)
            ])
            # Can't create features for last `abs(min_lag)` times in series:
            times = times[:min_lag] if min_lag != 0 else times
        # Case 3:
        elif (min_lag <= 0) and (max_lag > 0):
            # Can't create features for last `abs(min_lag)` times in series:
            times = times[:min_lag] if min_lag != 0 else times
            # Can't create features for first `max_lag` times in series:
            times = times[max_lag:]
        # Unexpected case:
        else:
            error_msg = (
                "Unexpected `future_covariates_lags` case encountered: "
                "`min_lag` is positive, but `max_lag` is negative. "
                f"Caused by `future_covariates_lags = {future_covariates_lags}`."
            )
            error = ValueError(error_msg)
            raise_log(error, get_logger(__name__))
        return times

    @staticmethod
    def construct_X_block(
        series: TimeSeries, feature_times: pd.Index, lags: Optional[Sequence[int]]
    ) -> np.array:
        """
        Helper function that creates the lagged features 'block' of a specific
        `series` (i.e. either `target_series`, `past_covariates`, or `future_covariates`);
        the feature matrix `X` is formed by concatenating the blocks of all of the specified
        series along the components axis. If `lags` is `None`, then `None` will be returned in
        lieu of an array. Please refer to the `create_lagged_features` docstring for further
        details about the structure of the `X` feature matrix.

        The returned `X_block` is constructed by looping over each time in `feature_times`,
        finding the index position of that time in the series, and then for each lag value in
        `lags`, offset this index position by a particular lag value; this offset index is then
        used to extract all of the components at a single lagged time.

        Unlike the implementation found in `darts.utils.data.tabularization`, this function doesn't
        use any 'vectorisation' tricks, which makes it slower to run, but more easily interpretable.

        Some of the times in `feature_times` may occur before the start *or* after the end of `series`;
        see the docstrings of `get_feature_times_past` and `get_feature_times_future` for why this is the
        case. Because of this, we need to prepend or append these 'extended times' to `series.time_index`
        before searching for the index of each time in the series. Even though the integer indices of the
        'extended times' won't be contained within the original `series`, offsetting these found indices
        by the requested lag value should 'bring us back' to a time within the original, unextended `series`.
        However, if we've prepended times to `series.time_index`, we have to note that all of the indices will
        be 'bumped up' by the number of values we've prepended, even after offsetting by a lag value. For example,
        if we extended `series.time_index` by prepending two values to the start, the integer index of the first
        actual value in `series` will occur at an index of `2` instead of `0`. To 'undo' this, we must subtract off
        the number of prepended value from the lag-offseted indices before retrieving values from `series`.
        """
        if lags is None:
            X_block = None
        else:
            series_times = series.time_index
            is_range_idx = isinstance(series_times[0], int)
            # If series ends before last time in `feature_times`:
            add_to_end = series_times[-1] < feature_times[-1]
            # If series starts after first time in `feature_times`:
            add_to_start = series_times[0] > feature_times[0]
            # Either add to start *OR* add to end, we can't have both (see `get_feature_times_future`):
            if add_to_end:
                num_prepended = 0
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
            elif add_to_start:
                num_prepended = (series_times[0] - feature_times[0]) // series.freq
                if is_range_idx:
                    # `+ 1` since `stop` index is exclusive:
                    series_times = pd.RangeIndex(
                        start=feature_times[0],
                        stop=series_times[-1] + 1,
                        step=series.freq,
                    )
                else:
                    series_times = pd.date_range(
                        start=feature_times[0], end=series_times[-1], freq=series.freq
                    )
            else:
                num_prepended = 0
            array_vals = series.all_values(copy=False)[:, :, 0]
            X_block = []
            for time in feature_times:
                # Find position of time within series:
                time_idx = np.searchsorted(series_times, time)
                X_row = []
                for lag in lags:
                    # Offset by particular lag value:
                    idx_to_get = time_idx + lag
                    # Account for prepended values:
                    idx_to_get -= num_prepended
                    raise_if_not(
                        idx_to_get >= 0,
                        f"Unexpected case encountered: `time_idx + lag - num_prepended = {idx_to_get} < 0`.",
                    )
                    # Extract all components at this lagged time:
                    X_row.append(array_vals[idx_to_get, :].reshape(-1))
                # Concatenate together all lagged values into a single row:
                X_row = np.concatenate(X_row, axis=0)
                X_block.append(X_row)
            # Concatenate all rows (i.e. observations) together into a block:
            X_block = np.stack(X_block, axis=0)
        return X_block

    #
    #   Generated Test Cases
    #

    # Input parameter combinations used to generate test cases:
    max_samples_per_ts_combos = (1, 2, None)
    target_lag_combos = past_lag_combos = (None, [-1, -3], [-3, -1])
    future_lag_combos = (*target_lag_combos, [0], [2, 1], [-1, 1], [0, 2])

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_prediction_data_equal_freq(self, series_type):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with time indices of equal
        frequencies. Since all of the timeseries are of the same frequency,
        the implementation of the 'moving window' method is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, and different values, but
        # they're all of the same frequency:
        if series_type == "integer":
            target = self.create_multivariate_linear_timeseries(
                n_components=2, start_value=0, end_value=10, start=2, end=20, freq=2
            )
            past = self.create_multivariate_linear_timeseries(
                n_components=3, start_value=10, end_value=20, start=4, end=23, freq=2
            )
            future = self.create_multivariate_linear_timeseries(
                n_components=4, start_value=20, end_value=30, start=6, end=26, freq=2
            )
        else:
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
        for lags, lags_past, lags_future, max_samples_per_ts in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.max_samples_per_ts_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags else None,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
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
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            assert X.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            # Check that outputs match:
            assert np.allclose(expected_X, X[:, :, 0])
            assert feats_times.equals(times[0])

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_prediction_data_unequal_freq(self, series_type):
        """
        Tests that `create_lagged_prediction_data` produces `X` and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times` and `construct_X_block`. Consistency is
        checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.covariates_lag_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with time indices of unequal
        frequencies. Since all of the timeseries are *not* of the same frequency,
        the implementation of the 'time intersection' method is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, different values, and of
        # different frequencies:
        if series_type == "integer":
            target = self.create_multivariate_linear_timeseries(
                n_components=2, start_value=0, end_value=10, start=2, end=20, freq=1
            )
            past = self.create_multivariate_linear_timeseries(
                n_components=3, start_value=10, end_value=20, start=4, end=23, freq=2
            )
            future = self.create_multivariate_linear_timeseries(
                n_components=4, start_value=20, end_value=30, start=6, end=26, freq=3
            )
        else:
            target = self.create_multivariate_linear_timeseries(
                n_components=2,
                start_value=0,
                end_value=10,
                start=pd.Timestamp("1/2/2000"),
                end=pd.Timestamp("1/20/2000"),
                freq="1d",
            )
            past = self.create_multivariate_linear_timeseries(
                n_components=3,
                start_value=10,
                end_value=20,
                start=pd.Timestamp("1/4/2000"),
                end=pd.Timestamp("1/23/2000"),
                freq="2d",
            )
            future = self.create_multivariate_linear_timeseries(
                n_components=4,
                start_value=20,
                end_value=30,
                start=pd.Timestamp("1/6/2000"),
                end=pd.Timestamp("1/26/2000"),
                freq="3d",
            )
        # Conduct test for each input parameter combo:
        for lags, lags_past, lags_future, max_samples_per_ts in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.max_samples_per_ts_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            X, times = create_lagged_prediction_data(
                target_series=target if lags else None,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
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
            X_target = self.construct_X_block(target, feats_times, lags)
            X_past = self.construct_X_block(past, feats_times, lags_past)
            X_future = self.construct_X_block(future, feats_times, lags_future)
            all_X = (X_target, X_past, X_future)
            to_concat = [X for X in all_X if X is not None]
            expected_X = np.concatenate(to_concat, axis=1)
            # Number of observations should match number of feature times:
            assert X.shape[0] == len(feats_times)
            assert X.shape[0] == len(times[0])
            # Check that outputs match:
            assert np.allclose(expected_X, X[:, :, 0])
            assert feats_times.equals(times[0])

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_prediction_data_method_consistency_range_index(self, series_type):
        """
        Tests that `create_lagged_prediction_data` produces the same result
        when `use_moving_windows = False` and when `use_moving_windows = True`
        for all of the parameter combinations used in the 'generated' test cases.

        Obviously, if both the 'Moving Window Method' and the 'Time Intersection'
        are both wrong in the same way, this test won't reveal any bugs. With this
        being said, if this test fails, something is definitely wrong in either
        one or both of the implemented methods.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, different values, and of
        # different frequencies:
        if series_type == "integer":
            target = self.create_multivariate_linear_timeseries(
                n_components=2, start_value=0, end_value=10, start=2, end=16, freq=2
            )
            past = self.create_multivariate_linear_timeseries(
                n_components=3, start_value=10, end_value=20, start=4, end=18, freq=2
            )
            future = self.create_multivariate_linear_timeseries(
                n_components=4, start_value=20, end_value=30, start=6, end=20, freq=2
            )
        else:
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
        for lags, lags_past, lags_future, max_samples_per_ts in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.max_samples_per_ts_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            # Using moving window method:
            X_mw, times_mw = create_lagged_prediction_data(
                target_series=target if lags else None,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=True,
            )
            # Using time intersection method:
            X_ti, times_ti = create_lagged_prediction_data(
                target_series=target if lags else None,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
                max_samples_per_ts=max_samples_per_ts,
                use_moving_windows=False,
            )
            assert np.allclose(X_mw, X_ti)
            assert times_mw[0].equals(times_ti[0])

    #
    #   Specified Cases Tests
    #

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [False, True]),
    )
    def test_lagged_prediction_data_single_lag_single_component_same_series(
        self, config
    ):
        """
        Tests that `create_lagged_prediction_data` correctly produces `X` and `times`
        when all the `series` inputs are identical, and all the `lags` inputs consist
        of a single value. In this situation, the expected `X` value can be found by
        concatenating three different slices of the same time series.
        """
        series_type, use_moving_windows = config
        if series_type == "integer":
            series = linear_timeseries(start=0, length=15)
        else:
            series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=15)
        lags = [-1]
        past_lags = [-3]
        future_lags = [2]
        # Can't create features for first 3 times (because `past_lags`) and last
        # two times (because `future_lags`):
        expected_times = series.time_index[3:-2]
        # Offset `3:-2` by `-1` lag:
        expected_X_target = series.all_values(copy=False)[2:-3, :, 0]
        # Offset `3:-2` by `-3` lag -> gives `0:-5`:
        expected_X_past = series.all_values(copy=False)[:-5, :, 0]
        # Offset `3:-2` by `+2` lag -> gives `5:None`:
        expected_X_future = series.all_values(copy=False)[5:, :, 0]
        expected_X = np.concatenate(
            [expected_X_target, expected_X_past, expected_X_future], axis=1
        )
        X, times = create_lagged_prediction_data(
            target_series=series,
            past_covariates=series,
            future_covariates=series,
            lags=lags,
            lags_past_covariates=past_lags,
            lags_future_covariates=future_lags,
            uses_static_covariates=False,
            use_moving_windows=use_moving_windows,
        )
        # Number of observations should match number of feature times:
        assert X.shape[0] == len(expected_times)
        assert X.shape[0] == len(times[0])
        # Check that outputs match:
        assert np.allclose(expected_X, X[:, :, 0])
        assert expected_times.equals(times[0])

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [False, True]),
    )
    def test_lagged_prediction_data_extend_past_and_future_covariates(self, config):
        """
        Tests that `create_lagged_prediction_data` correctly handles case where features
        can be created for a time that is *not* contained in `target_series`, `past_covariates`
        and/or `future_covariates`.

        More specifically, we define the series and lags such that a prediction feature
        can be generated for time `target.end_time() + target.freq`, even though this time
        isn't contained in any of the define series.
        """
        series_type, use_moving_windows = config
        if series_type == "integer":
            # Can create feature for time `t = 9`, but this time isn't in any of the three series:
            target = linear_timeseries(start=0, end=9, start_value=1, end_value=2)
            past = linear_timeseries(start=0, end=8, start_value=2, end_value=3)
            future = linear_timeseries(start=0, end=6, start_value=3, end_value=4)
        else:
            # Can create feature for time `t = '1/10/2000'`, but this time isn't in any of the three series:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                end=pd.Timestamp("1/10/2000"),
                start_value=1,
                end_value=2,
            )
            past = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                end=pd.Timestamp("1/9/2000"),
                start_value=2,
                end_value=3,
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                end=pd.Timestamp("1/7/2000"),
                start_value=3,
                end_value=4,
            )

        lags = [-1]
        lags_past = [-2]
        lags_future = [-4]
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from the very last values of each series:
        expected_X = np.concatenate([
            target.all_values(copy=False)[-1, :, 0],
            past.all_values(copy=False)[-1, :, 0],
            future.all_values(copy=False)[-1, :, 0],
        ]).reshape(1, -1)
        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        X, times = create_lagged_prediction_data(
            target,
            past_covariates=past,
            future_covariates=future,
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            uses_static_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            use_moving_windows=use_moving_windows,
        )
        assert times[0][0] == target.end_time() + target.freq
        assert np.allclose(expected_X, X[:, :, 0])

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [False, True]),
    )
    def test_lagged_prediction_data_single_point(self, config):
        """
        Tests that `create_lagged_prediction_data` correctly handles case
        where only one possible training point can be generated.
        """
        # Can only create feature using first value of target (i.e. `0`):
        series_type, use_moving_windows = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=1, start_value=0, end_value=1)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=1, start_value=0, end_value=1
            )

        expected_X = np.zeros((1, 1, 1))
        # Prediction time extend beyond end of series:
        lag = 5
        X, times = create_lagged_prediction_data(
            target,
            lags=[-lag],
            use_moving_windows=use_moving_windows,
            uses_static_covariates=False,
        )
        assert np.allclose(expected_X, X)
        # Should only have one sample, generated for
        # `t = target.end_time() + lag * target.freq`:
        assert len(times) == 1
        assert times[0] == target.end_time() + lag * target.freq

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [False, True]),
    )
    def test_lagged_prediction_data_zero_lags(self, config):
        """
        Tests that `create_lagged_prediction_data` correctly handles case when
        `0` is included in `lags_future_covariates` (i.e. when we're using the values
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point).
        """
        # Define `future` so that only value occurs at the same time as
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the value of `future` at the same time as the label (i.e. a lag
        # of `0` away from the only feature time):
        series_type, use_moving_windows = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=1, start_value=0, end_value=1)
            future = linear_timeseries(start=1, length=1, start_value=1, end_value=2)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=1, start_value=0, end_value=1
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=1, start_value=1, end_value=2
            )
        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        X, times = create_lagged_prediction_data(
            target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates=[0],
            uses_static_covariates=False,
            use_moving_windows=use_moving_windows,
        )
        assert np.allclose(expected_X, X)
        assert len(times[0]) == 1
        assert times[0][0] == future.start_time()

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [False, True]),
    )
    def test_lagged_prediction_data_positive_lags(self, config):
        """
        Tests that `create_lagged_prediction_data` correctly handles case when
        `0` is included in `lags_future_covariates` (i.e. when we're using the values
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point). This particular test checks this behaviour by using
        datetime index timeseries.
        """
        # Define `future` so that only value occurs one timestep *after*
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the value of `future` one timestep after the time of the label (i.e. a lag
        # of `1` away from the only feature time):
        series_type, use_moving_windows = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=1, start_value=0, end_value=1)
            future = linear_timeseries(start=2, length=1, start_value=1, end_value=2)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=1, start_value=0, end_value=1
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), length=1, start_value=1, end_value=2
            )
        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = np.array([0.0, 1.0]).reshape(1, 2, 1)
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        X, times = create_lagged_prediction_data(
            target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates=[1],
            uses_static_covariates=False,
            use_moving_windows=use_moving_windows,
        )
        assert np.allclose(expected_X, X)
        assert len(times[0]) == 1
        assert times[0][0] == target.end_time() + target.freq

    def test_lagged_prediction_data_sequence_inputs(self):
        """
        Tests that `create_lagged_prediction_data` correctly handles being
        passed a sequence of `TimeSeries` inputs, as opposed to individual
        `TimeSeries`.
        """
        # Define two simple tabularization problems:
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = linear_timeseries(start=6, end=11)
        lags = lags_past = lags_future = [-1]
        # Expected solution:
        expected_X_1 = np.concatenate(3 * [target_1.all_values(copy=False)], axis=1)
        expected_X_2 = np.concatenate(3 * [target_2.all_values(copy=False)], axis=1)
        expected_X = np.concatenate([expected_X_1, expected_X_2], axis=0)
        # Need to account for fact that we can create feature for last time in series:
        expected_times_1 = target_1.append_values([0]).time_index[1:]
        expected_times_2 = target_2.append_values([0]).time_index[1:]
        # Check when `concatenate = True`:
        X, times = create_lagged_prediction_data(
            target_series=(target_1, target_2),
            past_covariates=(past_1, past_2),
            future_covariates=(future_1, future_2),
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            uses_static_covariates=False,
        )
        assert np.allclose(X, expected_X)
        assert len(times) == 2
        assert times[0].equals(expected_times_1)
        assert times[1].equals(expected_times_2)
        # Check when `concatenate = False`:
        X, times = create_lagged_prediction_data(
            target_series=(target_1, target_2),
            past_covariates=(past_1, past_2),
            future_covariates=(future_1, future_2),
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            uses_static_covariates=False,
            concatenate=False,
        )
        assert len(X) == 2
        assert np.allclose(X[0], expected_X_1)
        assert np.allclose(X[1], expected_X_2)
        assert len(times) == 2
        assert times[0].equals(expected_times_1)
        assert times[1].equals(expected_times_2)

    def test_lagged_prediction_data_stochastic_series(self):
        """
        Tests that `create_lagged_prediction_data` is correctly vectorised
        over the sample axes of the input `TimeSeries`.
        """
        # Define two simple tabularization problems:
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = 2 * target_1
        target = target_1.concatenate(target_2, axis=2)
        past = past_1.concatenate(past_2, axis=2)
        future = future_1.concatenate(future_2, axis=2)
        lags = lags_past = lags_future = [-1]
        # Expected solution:
        expected_X = np.concatenate(3 * [target.all_values(copy=False)], axis=1)
        # Need to account for fact that we can create feature for last time in series:
        expected_times = target_1.append_values([0]).time_index[1:]
        X, times = create_lagged_prediction_data(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            uses_static_covariates=False,
        )
        assert np.allclose(X, expected_X)
        assert times[0].equals(expected_times)

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
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    target_series=series_1,
                    lags=lags,
                    past_covariates=series_2,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
            assert (
                "Specified series do not share any common times for which features can be created."
                == str(err.value)
            )
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    target_series=series_1,
                    lags=lags,
                    past_covariates=series_2,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                )
            assert (
                "Specified series do not share any common times for which features can be created."
                == str(err.value)
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
                with pytest.raises(ValueError) as err:
                    create_lagged_prediction_data(
                        target_series=series,
                        lags_future_covariates=[-1],
                        past_covariates=series,
                        uses_static_covariates=False,
                        use_moving_windows=use_moving_windows,
                    )
            assert "Must specify at least one series-lags pair." == str(err.value)

    def test_lagged_prediction_data_no_lags_specified_error(self):
        """
        Tests that `create_lagged_prediction_data` throws correct error
        when no lags are specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    target_series=target,
                    use_moving_windows=use_moving_windows,
                    uses_static_covariates=False,
                )
            assert (
                "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`."
                == str(err.value)
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
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    target_series=series,
                    lags=[-20, -1],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
            assert (
                "`target_series` must have at least "
                "`-min(lags) + max(lags) + 1` = 20 "
                "time steps; instead, it only has 2."
            ) == str(err.value)
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    past_covariates=series,
                    lags_past_covariates=[-20, -1],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
            assert (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 20 "
                "time steps; instead, it only has 2."
            ) == str(err.value)

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
        series = linear_timeseries(start=1, length=3, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Test invalid `lags` values:
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    target_series=series,
                    lags=[0],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
            assert (
                "`lags` must be a `Sequence` or `Dict` containing only `int` values less than 0."
            ) == str(err.value)
            # Test invalid `lags_past_covariates` values:
            with pytest.raises(ValueError) as err:
                create_lagged_prediction_data(
                    past_covariates=series,
                    lags_past_covariates=[0],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
            assert (
                "`lags_past_covariates` must be a `Sequence` or `Dict` containing only `int` values less than 0."
            ) == str(err.value)
            # This should *not* throw an error:
            create_lagged_prediction_data(
                future_covariates=series,
                lags_future_covariates=[-1, 0, 1],
                uses_static_covariates=False,
                use_moving_windows=use_moving_windows,
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
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                )
            # Specify `lags_future_covariates` but not `future_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    lags=lags,
                    lags_future_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
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
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
                assert len(w) == 2
                assert issubclass(w[0].category, UserWarning)
                assert issubclass(w[1].category, UserWarning)
                assert str(w[0].message) == (
                    "`past_covariates` was specified without accompanying "
                    "`lags_past_covariates` and, thus, will be ignored."
                )
                assert str(w[1].message) == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
                )
            # Specify `target_series` but not `lags` - this *should* throw
            # a warning when creating prediction data:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    target_series=series,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`target_series` was specified without accompanying "
                    "`lags` and, thus, will be ignored."
                )
            # Specify `target_series` but not `lags` - this *should* throw
            # a warning when creating prediction data:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_prediction_data(
                    lags=lags,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`lags` was specified without accompanying "
                    "`target_series` and, thus, will be ignored."
                )
