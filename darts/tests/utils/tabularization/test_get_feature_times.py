import itertools
import warnings
from collections.abc import Sequence
from itertools import product

import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.tabularization import _get_feature_times
from darts.utils.timeseries_generation import linear_timeseries


class TestGetFeatureTimes:
    """
    Tests `_get_feature_times` function defined in `darts.utils.data.tabularization`. There
    are broadly two 'groups' of tests defined in this module:
        1. 'Generated Test Cases': these test that `_get_feature_times` produces the same outputs
        as a simplified implementation of this same function. For these tests, the 'correct answer' is not
        directly specified; instead, it is generated from a set of input parameters using the set of simplified
        functions. The rationale behind this approach is that it allows for many different combinations of input
        values to be effortlessly tested. The drawback of this, however, is that the correctness of these tests
        assumes that the simplified functions have been implemented correctly - if this isn't the case, then these
        tests are not to be trusted. In saying this, these simplified functions are significantly easier to
        understand and debug than the `create_lagged_prediction_data` function they're helping to test.
        2. 'Specified Test Cases': these test that `_get_feature_times` returns an exactly specified output; these
        specified outputs are *not* 'generated' by another function. Although these 'specified' test cases tend to
        be simpler and less extensive than the 'generated' test cases, their correctness does not assume the correct
        implementation of any other function.
    """

    #
    #   Helper Functions for Generated Test Cases
    #

    @staticmethod
    def get_feature_times_target_training(
        target_series: TimeSeries,
        lags: Sequence[int],
        output_chunk_length: int,
        output_chunk_shift: int,
    ):
        """
        Helper function that returns all the times within `target_series` that can be used to
        create features and labels for training.

        More specifically:
            - The first `max_lag = -min(lags)` times are excluded, since these times
            have fewer than `max_lag` values after them, which means that we can't create
            features for these times.
            - The last `output_chunk_length - 1` times are excluded, since these times don't
            have `(output_chunk_length - 1)` values ahead of them and, therefore, we can't
            create labels for these times.
        """
        times = target_series.time_index
        # Exclude first `max_lag` times:
        max_lag = -min(lags)
        times = times[max_lag:]
        # Exclude last `output_chunk_length - 1` times:
        if output_chunk_length > 1:
            times = times[: -output_chunk_length + 1]
        if output_chunk_shift:
            times = times[:-output_chunk_shift]
        return times

    @staticmethod
    def get_feature_times_past(
        past_covariates: TimeSeries,
        past_covariates_lags: Sequence[int],
    ) -> pd.Index:
        """
        Helper function that returns all the times within `past_covariates` that can be used to
        create features for training or prediction.

        Unlike the `target_series` during training, features can be constructed for times that
        occur after the end of `past_covariates`; this is because:
            1. We don't need to have all the `past_covariates` values up to time `t` to construct
            a feature for this time; instead, we only need to have the values from time `t - min_lag`
            to `t - max_lag`, where `min_lag = -max(past_covariates_lags)` and
            `max_lag = -min(past_covariates_lags)`. In other words, the latest feature we can create
            for `past_covariates` occurs at `past_covariates.end_time() + min_lag * past_covariates.freq`.
            2. We don't need to use the values of `past_covariates` to construct labels, so we're able
            to create a feature for time `t` without having to worry about whether we can construct
            a corresponding label for this time.
        """
        times = past_covariates.time_index
        min_lag = -max(past_covariates_lags)
        # Add times after end of series for which we can create features:
        times = times.union([
            times[-1] + i * past_covariates.freq for i in range(1, min_lag + 1)
        ])
        max_lag = -min(past_covariates_lags)
        times = times[max_lag:]
        return times

    @staticmethod
    def get_feature_times_target_prediction(
        target_series: TimeSeries, lags: Sequence[int]
    ):
        """
        Helper function that returns all the times within `target_series` that can be used to
        create features for prediction.

        Since we don't need to worry about creating labels for prediction data, the process
        of constructing prediction features using the `target_series` is identical to
        constructing features for the `past_covariates` series.
        """
        return TestGetFeatureTimes.get_feature_times_past(target_series, lags)

    @staticmethod
    def get_feature_times_future(
        future_covariates: TimeSeries,
        future_covariates_lags: Sequence[int],
    ) -> pd.Index:
        """
        Helper function called by `_get_feature_times` that extracts all of the times within
        `future_covariates` that can be used to create features for training or prediction.

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

    #
    #   Generated Test Cases
    #

    # Input parameter combinations used to generate test cases:
    target_lag_combos = lags_past_combos = (
        [-1],
        [-2, -1],
        [-6, -4, -3],
        [-4, -6, -3],
    )
    lags_future_combos = (*target_lag_combos, [0], [0, 1], [1, 3], [-2, 2])
    ocl_combos = (1, 2, 5, 10)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            ["datetime", "integer"],
            [0, 1, 3],
        ),
    )
    def test_feature_times_training(self, config):
        """
        Tests that `_get_feature_times` produces the same `times` output as
        that generated by using the various `get_feature_times_*` helper
        functions defined in this module when `is_training = True`. Consistency
        is checked over all of the combinations of parameter values specified by
        `self.target_lag_combos`, `self.lags_past_combos`, `self.lags_future_combos`
        and `self.max_samples_per_ts_combos`. This particular test uses timeseries
        with range time indices.
        """
        # Define timeseries with different starting points, lengths, and frequencies:
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=1, length=20, freq=1)
            past = linear_timeseries(start=2, length=25, freq=2)
            future = linear_timeseries(start=3, length=30, freq=3)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=20, freq="1d"
            )
            past = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=25, freq="2d"
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), length=30, freq="3d"
            )
        for lags, lags_past, lags_future, ocl in product(
            self.target_lag_combos,
            self.lags_past_combos,
            self.lags_future_combos,
            self.ocl_combos,
        ):
            feature_times = _get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=ocl,
                is_training=True,
                output_chunk_shift=output_chunk_shift,
            )
            target_expected = self.get_feature_times_target_training(
                target, lags, ocl, output_chunk_shift=output_chunk_shift
            )
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            ["datetime", "integer"],
            [0, 1, 3],
        ),
    )
    def test_feature_times_prediction(self, config):
        """
        Tests that `_get_feature_times` produces the same `times` output as
        that generated by using the various `get_feature_times_*` helper
        functions defined in this module when `is_training = False` (i.e. when creaiting
        prediction data). Consistency is checked over all of the combinations of parameter
        values specified by `self.target_lag_combos`, `self.lags_past_combos`,
        `self.lags_future_combos` and `self.max_samples_per_ts_combos`. This particular test
        uses timeseries with range time indices.
        """
        # Define timeseries with different starting points, lengths, and frequencies:
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=1, length=20, freq=1)
            past = linear_timeseries(start=2, length=25, freq=2)
            future = linear_timeseries(start=3, length=30, freq=3)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=20, freq="1d"
            )
            past = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=25, freq="2d"
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), length=30, freq="3d"
            )

        for lags, lags_past, lags_future in product(
            self.target_lag_combos, self.lags_past_combos, self.lags_future_combos
        ):
            feature_times = _get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
                output_chunk_shift=output_chunk_shift,
            )
            target_expected = self.get_feature_times_target_prediction(target, lags)
            past_expected = self.get_feature_times_past(past, lags_past)
            future_expected = self.get_feature_times_future(future, lags_future)
            assert target_expected.equals(feature_times[0])
            assert past_expected.equals(feature_times[1])
            assert future_expected.equals(feature_times[2])

    #
    #   Specified Test Cases
    #

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_output_chunk_length_output_chunk_shift(self, config):
        """
        Tests that the last feature time for the `target_series`
        returned by `_get_feature_times` corresponds to
        `output_chunk_length - output_chunk_shift - 1` timesteps *before* the end of
        the target series; this is the last time point in
        `target_series` which has enough values in front of it
        to create a label. This particular test uses range time
        index series to check this behaviour.
        """
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=20, freq=2)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=20, freq="2d"
            )
        # Test multiple `output_chunk_length` values:
        for ocl in (1, 2, 3, 4, 5):
            feature_times = _get_feature_times(
                target_series=target,
                lags=[-2, -3, -5],
                output_chunk_length=ocl,
                is_training=True,
                output_chunk_shift=output_chunk_shift,
            )
            assert feature_times[0][-1] == target.end_time() - target.freq * (
                ocl + output_chunk_shift - 1
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_lags(self, config):
        """
        Tests that the first feature time for the `target_series`
        returned by `_get_feature_times` corresponds to the time
        that is `max_lags` timesteps *after* the start of
        the target series; this is the first time point in
        `target_series` which has enough values in preceeding it
        to create a feature. This particular test uses range time
        index series to check this behaviour.
        """
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=20, freq=2)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=20, freq="2d"
            )
        # Expect same behaviour when training and predicting:
        for is_training in (False, True):
            for max_lags in (-1, -2, -3, -4, -5):
                feature_times = _get_feature_times(
                    target_series=target,
                    lags=[-1, max_lags],
                    is_training=is_training,
                    output_chunk_shift=output_chunk_shift,
                )
                assert feature_times[0][0] == target.start_time() + target.freq * abs(
                    max_lags
                )

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_training_single_time(self, config):
        """
        Tests that `_get_feature_times` correctly handles case where only
        a single time can be used to create training features and labels.
        This particular test uses range index timeseries.
        """
        # Can only create feature and label for time `1` (`-1` lag behind is time `0`):
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=0, length=2 + output_chunk_shift, freq=1)
            # Can only create feature for time `6` (`-2` lags behind is time `2`):
            future = linear_timeseries(start=2, length=1, freq=2)
            exp_start_target, exp_start_future = 1, 6
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=2 + output_chunk_shift, freq="d"
            )
            # Can only create feature for "1/6/2000" (`-2` lags behind is "1/2/2000"):
            future = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=1, freq="2d"
            )
            exp_start_target, exp_start_future = (
                pd.Timestamp("1/2/2000"),
                pd.Timestamp("1/6/2000"),
            )

        lags = [-1]
        feature_times = _get_feature_times(
            target_series=target,
            output_chunk_length=1,
            lags=lags,
            is_training=True,
            output_chunk_shift=output_chunk_shift,
        )
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == exp_start_target

        future_lags = [-2]
        feature_times = _get_feature_times(
            target_series=target,
            future_covariates=future,
            output_chunk_length=1,
            lags=lags,
            lags_future_covariates=future_lags,
            is_training=True,
            output_chunk_shift=output_chunk_shift,
        )
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == exp_start_target
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == exp_start_future

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_prediction_single_time(self, config):
        """
        Tests that `_get_feature_times` correctly handles case where only
        a single time can be used to create prediction features.
        This particular test uses range index timeseries.
        """
        series_type, output_chunk_shift = config
        if series_type == "integer":
            # Can only create feature for time `1` (`-1` lag behind is time `0`):
            target = linear_timeseries(start=0, length=1, freq=1)
            # Can only create feature for time `6` (`-2` lags behind is time `2`):
            future = linear_timeseries(start=2, length=1, freq=2)
            exp_start_target, exp_start_future = 1, 6

        else:
            # Can only create feature for "1/2/2000" (`-1` lag behind is time "1/1/2000"):
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=1, freq="d"
            )
            # Can only create feature for "1/6/2000" (`-2` lag behind is time "1/2/2000"):
            future = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=1, freq="2d"
            )
            exp_start_target, exp_start_future = (
                pd.Timestamp("1/2/2000"),
                pd.Timestamp("1/6/2000"),
            )

        lags = [-1]
        feature_times = _get_feature_times(
            target_series=target,
            lags=lags,
            is_training=False,
        )
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == exp_start_target

        lags_future = [-2]
        feature_times = _get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=lags,
            lags_future_covariates=lags_future,
            is_training=False,
            output_chunk_shift=output_chunk_shift,
        )
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == exp_start_target
        assert len(feature_times[2]) == 1
        assert feature_times[2][0] == exp_start_future

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_extend_time_index_range_idx(self, config):
        """
        Tests that `_get_feature_times` is able to return feature
        times that occur after the end of a series or occur before
        the beginning of a series. This particular test uses range
        index time series.
        """
        # Feature times occur after end of series:
        series_type, output_chunk_shift = config
        if series_type == "integer":
            target = linear_timeseries(start=10, length=1, freq=3)
            past = linear_timeseries(start=2, length=1, freq=2)
            future = linear_timeseries(start=3, length=1, freq=1)
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/10/2000"), length=1, freq="3d"
            )
            past = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), length=1, freq="2d"
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), length=1, freq="1d"
            )
        lags = lags_past = lags_future_1 = [-4]
        feature_times = _get_feature_times(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=lags,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future_1,
            is_training=False,
            output_chunk_shift=output_chunk_shift,
        )
        assert len(feature_times[0]) == 1
        assert feature_times[0][0] == target.start_time() - lags[0] * target.freq
        assert len(feature_times[1]) == 1
        assert feature_times[1][0] == past.start_time() - lags_past[0] * past.freq
        assert len(feature_times[2]) == 1
        assert (
            feature_times[2][0] == future.start_time() - lags_future_1[0] * future.freq
        )
        # Feature time occurs before start of series:
        lags_future_2 = [4]
        feature_times = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future_2,
            is_training=False,
            output_chunk_shift=output_chunk_shift,
        )
        assert len(feature_times[2]) == 1
        assert (
            feature_times[2][0] == future.start_time() - lags_future_2[0] * future.freq
        )

    @pytest.mark.parametrize(
        "config",
        itertools.product(["datetime", "integer"], [0, 1, 3]),
    )
    def test_feature_times_future_lags(self, config):
        """
        Tests that `_get_feature_times` correctly handles the `lags_future_covariates`
        argument for the following three cases:
            1. `lags_future_covariates` contains only `0`
            2. `lags_future_covariates` contains only a positive lag
            3. `lags_future_covariates` contains a combination of positive,
            zero, and negative lags
        """
        series_type, output_chunk_shift = config
        if series_type == "integer":
            future = linear_timeseries(start=0, length=10, freq=2)
        else:
            future = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=10, freq="2d"
            )
        # Case 1 - Zero lag:
        lags_future = [0]
        feature_times = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future,
            is_training=False,
            output_chunk_shift=output_chunk_shift,
        )
        # All times will be feature times:
        assert len(feature_times[2]) == future.n_timesteps
        assert feature_times[2].equals(future.time_index)

        # Case 2 - Positive lag:
        lags_future = [1]
        feature_times = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future,
            is_training=False,
            output_chunk_shift=output_chunk_shift,
        )
        # Need to include new time at start of series; only last time will be excluded:
        extended_future = future.prepend_values([0])
        assert len(feature_times[2]) == extended_future.n_timesteps - 1
        assert feature_times[2].equals(extended_future.time_index[:-1])

        # Case 3 - Combo of negative, zero, and positive lags:
        lags_future = [-1, 0, 1]
        feature_times = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future,
            is_training=False,
        )
        # Only first and last times will be excluded:
        assert len(feature_times[2]) == future.n_timesteps - 2
        assert feature_times[2].equals(future.time_index[1:-1])

    def test_feature_times_unspecified_series(self):
        """
        Tests that `_get_feature_times` correctly returns
        `None` in place of a sequence of times if a particular
        series is not specified.
        """
        # Generate simple examples:
        target = linear_timeseries(start=1, length=20, freq=1)
        past = linear_timeseries(start=2, length=25, freq=2)
        future = linear_timeseries(start=3, length=30, freq=3)
        lags = [-1]
        lags_past = [-2]
        lags_future = [-3]
        # Need to extend series then exclude first few starting times:
        expected_target = target.append_values([0]).time_index[1:]
        expected_past = past.append_values(2 * [0]).time_index[2:]
        expected_future = future.append_values(3 * [0]).time_index[3:]

        # Specify only target, without past and future:
        feature_times = _get_feature_times(
            target_series=target, lags=lags, is_training=False
        )
        assert expected_target.equals(feature_times[0])
        assert feature_times[1] is None
        assert feature_times[2] is None

        # Specify only past, without target and future:
        feature_times = _get_feature_times(
            past_covariates=past,
            lags_past_covariates=lags_past,
            is_training=False,
        )
        assert feature_times[0] is None
        assert expected_past.equals(feature_times[1])
        assert feature_times[2] is None

        # Specify only future, without target and past:
        feature_times = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future,
            is_training=False,
        )
        assert feature_times[0] is None
        assert feature_times[1] is None
        assert expected_future.equals(feature_times[2])

        # Specify target and past, without future:
        feature_times = _get_feature_times(
            target_series=target,
            past_covariates=past,
            lags=lags,
            lags_past_covariates=lags_past,
            is_training=False,
        )
        assert expected_target.equals(feature_times[0])
        assert expected_past.equals(feature_times[1])
        assert feature_times[2] is None

        # Specify target and future, without past:
        feature_times = _get_feature_times(
            target_series=target,
            future_covariates=future,
            lags=lags,
            lags_future_covariates=lags_future,
            is_training=False,
        )
        assert expected_target.equals(feature_times[0])
        assert feature_times[1] is None
        assert expected_future.equals(feature_times[2])

        # Specify past and future, without target:
        feature_times = _get_feature_times(
            past_covariates=past,
            future_covariates=future,
            lags_past_covariates=lags_past,
            lags_future_covariates=lags_future,
            is_training=False,
        )
        assert feature_times[0] is None
        assert expected_past.equals(feature_times[1])
        assert expected_future.equals(feature_times[2])

    def test_feature_times_unspecified_lag_or_series_warning(self):
        """
        Tests that `_get_feature_times` throws correct warning when
        a series is specified by its corresponding lag is not, or
        vice versa. The only circumstance under which a warning
        should *not* be issued is when `target_series` is specified,
        but `lags` is not when `is_training = True`; this is because
        the user may not want to add autoregressive features to `X`,
        but they still need to specify `target_series` to create labels.
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
            _ = _get_feature_times(
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                is_training=False,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == (
                "`future_covariates` was specified without accompanying "
                "`lags_future_covariates` and, thus, will be ignored."
            )
        # Specify `lags_future_covariates` but not `future_covariates` when `is_training = False`
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                past_covariates=past,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == (
                "`lags_future_covariates` was specified without accompanying "
                "`future_covariates` and, thus, will be ignored."
            )
        # Specify `future_covariates` but not `lags_future_covariates` and
        # `target_series` but not lags, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                output_chunk_length=1,
                is_training=True,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == (
                "`future_covariates` was specified without accompanying "
                "`lags_future_covariates` and, thus, will be ignored."
            )
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `target_series` but not lags, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                is_training=True,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert str(w[0].message) == (
                "`lags_future_covariates` was specified without accompanying "
                "`future_covariates` and, thus, will be ignored."
            )
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `past_covariates` but not `lags_past_covariates`, when `is_training = True`
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                lags=lags,
                lags_future_covariates=lags_future,
                is_training=False,
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
        # Specify `lags_future_covariates` but not `future_covariates`, and
        # `past_covariates` but not `lags_past_covariates`, and `target_series`
        # but not `lags` when `is_training = False`:
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                lags_future_covariates=lags_future,
                output_chunk_length=1,
                is_training=True,
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
        # Specify `target_series` but not `lags` when `is_training = False`:
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=False,
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
        # Specify `target_series` but not `lags` when `is_training = True`;
        # this should *not* throw a warning:
        with warnings.catch_warnings(record=True) as w:
            _ = _get_feature_times(
                target_series=target,
                past_covariates=past,
                future_covariates=future,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                is_training=True,
            )
            assert len(w) == 0

    def test_feature_times_unspecified_training_inputs_error(self):
        """
        Tests that `_get_feature_times` throws correct error when
        `target_series` and/or `output_chunk_length` hasn't been
        specified when `is_training = True`.
        """
        output_chunk_length = 1
        # Don't specify `target_series`:
        with pytest.raises(ValueError) as err:
            _get_feature_times(
                output_chunk_length=output_chunk_length, is_training=True
            )
        assert ("Must specify `target_series` when `is_training = True`.") == str(
            err.value
        )
        # Don't specify neither `target_series` nor `output_chunk_length`
        with pytest.raises(ValueError) as err:
            _get_feature_times(is_training=True)
        assert ("Must specify `target_series` when `is_training = True`.") == str(
            err.value
        )

    def test_feature_times_no_lags_specified_error(self):
        """
        Tests that `_get_feature_times` throws correct error
        when no lags have been specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=target, is_training=False)
        assert (
            "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`."
            == str(err.value)
        )

    def test_feature_times_series_too_short_error(self):
        """
        Tests that `_get_feature_times` throws correct error
        when provided series are too short for specified
        lag and/or `output_chunk_length` values.
        """
        series = linear_timeseries(start=1, length=2, freq=1)
        # `target_series` too short when predicting:
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, lags=[-20, -1], is_training=False)
        assert (
            "`target_series` must have at least `-min(lags) + max(lags) + 1` = 20 "
            "time steps; instead, it only has 2."
        ) == str(err.value)
        # `target_series` too short when training:
        with pytest.raises(ValueError) as err:
            _get_feature_times(
                target_series=series,
                lags=[-20],
                output_chunk_length=5,
                is_training=True,
            )
        assert (
            "`target_series` must have at least `-min(lags) + output_chunk_length + output_chunk_shift` = 25 "
            "time steps; instead, it only has 2."
        ) == str(err.value)
        # `past_covariates` too short when training:
        with pytest.raises(ValueError) as err:
            _get_feature_times(
                target_series=series,
                past_covariates=series,
                lags_past_covariates=[-20, -1],
                output_chunk_length=1,
                is_training=True,
            )
        assert (
            "`past_covariates` must have at least "
            "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 20 time steps; "
            "instead, it only has 2."
        ) == str(err.value)

    def test_feature_times_invalid_lag_values_error(self):
        """
        Tests that `_get_feature_times` throws correct error
        when provided with invalid lag values (i.e. not less than
        0 if `lags`, or not less than 1 if `lags_past_covariates` or
        `lags_future_covariates`).
        """
        series = linear_timeseries(start=1, length=3, freq=1)
        # `lags` not <= -1:
        with pytest.raises(ValueError) as err:
            _get_feature_times(target_series=series, lags=[0], is_training=False)
        assert (
            "`lags` must be a `Sequence` or `Dict` containing only `int` values less than 0."
        ) == str(err.value)
        # `lags_past_covariates` not <= -1:
        with pytest.raises(ValueError) as err:
            _get_feature_times(
                past_covariates=series, lags_past_covariates=[0], is_training=False
            )
        assert (
            "`lags_past_covariates` must be a `Sequence` or `Dict` containing only `int` values less than 0."
        ) == str(err.value)
        # `lags_future_covariates` can be positive, negative, and/or zero - no error should be thrown:
        _get_feature_times(
            future_covariates=series,
            lags_future_covariates=[-1, 0, 1],
            is_training=False,
        )
