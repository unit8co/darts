import itertools
import warnings
from collections.abc import Sequence
from itertools import product
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts import concatenate as darts_concatenate
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.utils.data.tabularization import (
    create_lagged_component_names,
    create_lagged_training_data,
)
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.utils import freqs, generate_index


def helper_create_multivariate_linear_timeseries(
    n_components: int, components_names: Sequence[str] = None, **kwargs
) -> TimeSeries:
    """
    Helper function that creates a `linear_timeseries` with a specified number of
    components. To help distinguish each component from one another, `i` is added on
    to each value of the `i`th component. Any additional keyword arguments are passed
    to `linear_timeseries` (`start_value`, `end_value`, `start`, `end`, `length`, etc).
    """
    if components_names is None or len(components_names) < n_components:
        components_names = [f"lin_ts_{i}" for i in range(n_components)]
    timeseries = []
    for i in range(n_components):
        # Values of each component is 1 larger than the last:
        timeseries_i = linear_timeseries(column_name=components_names[i], **kwargs) + i
        timeseries.append(timeseries_i)
    return darts_concatenate(timeseries, axis=1)


class TestCreateLaggedTrainingData:
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
    #   Helper Functions for Generated Test Cases
    #

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
        output_chunk_shift: int,
    ):
        """
        Helper function that returns the times shared by all specified series that can be used
        to create features and labels. This is performed by using the helper functions
        `get_feature_times_target`, `get_feature_times_past`, and `get_feature_times_future` (all
        defined below) to extract the feature times from the target series, past covariates, and future
        covariates respectively, and then intersecting these features times with one another. A series is
        considered to be 'specified' if its corresponding lag (e.g. `lags` for `target`, or `lags_future`
        for `future`) is not `None`. If requested, the last `max_samples_per_ts` times are taken.

        This function is basically a simplified implementation of `get_feature_times` in `tabularization.py`
        that only works for `is_training = True`.
        """
        # Get feature times for `target_series`:
        times = TestCreateLaggedTrainingData.get_feature_times_target(
            target, lags, output_chunk_length, output_chunk_shift
        )
        # Intersect `times` with `past_covariates` feature times if past covariates to be added to `X`:
        if lags_past is not None:
            past_times = TestCreateLaggedTrainingData.get_feature_times_past(
                past, lags_past
            )
            times = times.intersection(past_times)
        # Intersect `times` with `future_covariates` feature times if future covariates to be added to `X`:
        if lags_future is not None:
            future_times = TestCreateLaggedTrainingData.get_feature_times_future(
                future, lags_future
            )
            times = times.intersection(future_times)
        # Take most recent `max_samples_per_ts` samples if requested:
        if (max_samples_per_ts is not None) and (len(times) > max_samples_per_ts):
            times = times[-max_samples_per_ts:]
        return times

    @staticmethod
    def get_feature_times_target(
        target_series: TimeSeries,
        lags: Optional[Sequence[int]],
        output_chunk_length: int,
        output_chunk_shift: int,
    ) -> pd.Index:
        """
        Helper function called by `get_feature_times` that extracts all times within a
        `target_series` that can be used to create a feature and label. More specifically,
        we can create features and labels for times within `target_series` that have *both*:
            1. At least `max_lag = -min(lags)` values preceding them, since these preceding
            values are required to construct a feature vector for that time. Since the first `max_lag`
            times do not fulfill this condition, they are excluded *if* values from `target_series` are
            to be added to `X`.
            2. At least `(output_chunk_length - 1)` values after them, because the all times from
            time `t` to time `t + output_chunk_length - 1` will be used as labels. Since the last
            `(output_chunk_length - 1)` times do not fulfil this condition, they are excluded.
        """
        times = target_series.time_index
        if lags is not None:
            max_lag = -min(lags)
            times = times[max_lag:]
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
        Helper function called by `get_feature_times` that extracts all times within
        `past_covariates` that can be used to create features. More specifically, we can create
        features for times within `past_covariates` that have at least `max_lag = -min(past_covariates_lags)`
        values preceeding them, since these preceeding values are required to construct a feature vector for
        that time. Since the first `max_lag` times do not fulfill this condition, they are exluded.

        Unlike the `target_series`, features can be constructed for times that occur after the end of
        `past_covariates`; this is because:
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
    def get_feature_times_future(
        future_covariates: TimeSeries,
        future_covariates_lags: Sequence[int],
    ) -> pd.Index:
        """
        Helper function called by `get_feature_times` that extracts all times within
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
        the feature matrix `X` is formed by concatenating the blocks of all specified
        series along the components axis. If `lags` is `None`, then `None` will be returned in
        lieu of an array. Please refer to the `create_lagged_features` docstring for further
        details about the structure of the `X` feature matrix.

        The returned `X_block` is constructed by looping over each time in `feature_times`,
        finding the index position of that time in the series, and then for each lag value in
        `lags`, offset this index position by a particular lag value; this offset index is then
        used to extract all components at a single lagged time.

        Unlike the implementation found in `darts.utils.data.tabularization`, this function doesn't
        use any 'vectorisation' tricks, which makes it slower to run, but more easily interpretable.

        Some of the times in `feature_times` may occur before the start *or* after the end of `series`;
        see the docstrings of `get_feature_times_past` and `get_feature_times_future` for why this is the
        case. Because of this, we need to prepend or append these 'extended times' to `series.time_index`
        before searching for the index of each time in the series. Even though the integer indices of the
        'extended times' won't be contained within the original `series`, offsetting these found indices
        by the requested lag value should 'bring us back' to a time within the original, unextended `series`.
        However, if we've prepended times to `series.time_index`, we have to note that all indices will
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

    @staticmethod
    def create_y(
        target: TimeSeries,
        feature_times: pd.Index,
        output_chunk_length: int,
        multi_models: bool,
        output_chunk_shift: int,
    ) -> np.ndarray:
        """
        Helper function that constructs the labels array `y` from the target series.
        This is done by by looping over each time in `feature_times`, finding the index
        position of that time in the target series, and then for each timestep ahead of
        this time we wish to predict, offset this time index by this timestep. This offset
        index is then used to extract all components of the target series at this time which
        is to be predicted. Please refer to the `create_lagged_features` docstring for further
        details about the structure of the `y` labels matrix.

        Unlike `construct_X_block`, we don't need to worry about times in `feature_times` lying
        outside of `target.time_index` here: each label *must* be contained in the `target`
        series already.
        """
        array_vals = target.all_values(copy=False)
        y = []
        for time in feature_times:
            raise_if(
                time < target.start_time(),
                f"Unexpected label time at {time}, but `series` starts at {target.start_time()}.",
            )
            raise_if(
                time > target.end_time(),
                f"Unexpected label time at {time}, but `series` ends at {target.end_time()}.",
            )
            time_idx = np.searchsorted(target.time_index, time)
            # If `multi_models = True`, want to predict all values from time `t` to
            # time `t + output_chunk_lenth - 1`; if `multi_models = False`, only want to
            # predict time `t + output_chunk_length - 1`:
            timesteps_ahead = (
                range(output_chunk_shift, output_chunk_length + output_chunk_shift)
                if multi_models
                else [output_chunk_length + output_chunk_shift - 1]
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

    @staticmethod
    def convert_lags_to_dict(ts_tg, ts_pc, ts_fc, lags_tg, lags_pc, lags_fc):
        """Convert lags to the dictionary format, assuming the lags are shared across the components"""
        lags_as_dict = dict()
        for ts_, lags_, name_ in zip(
            [ts_tg, ts_pc, ts_fc],
            [lags_tg, lags_pc, lags_fc],
            ["target", "past", "future"],
        ):
            single_ts = ts_[0] if isinstance(ts_, Sequence) else ts_
            if single_ts is None or lags_ is None:
                lags_as_dict[name_] = None
            # already in dict format
            elif isinstance(lags_, dict):
                lags_as_dict[name_] = lags_
            # from list
            elif isinstance(lags_, list):
                lags_as_dict[name_] = {c_name: lags_ for c_name in single_ts.components}
            else:
                raise ValueError(
                    f"Lags should be `None`, a list or a dictionary. Received {type(lags_)}."
                )
        return lags_as_dict

    def helper_create_expected_lagged_data(
        self,
        target: Optional[Union[TimeSeries, list[TimeSeries]]],
        past: Optional[Union[TimeSeries, list[TimeSeries]]],
        future: Optional[Union[TimeSeries, list[TimeSeries]]],
        lags: Optional[Union[list[int], dict[str, list[int]]]],
        lags_past: Optional[Union[list[int], dict[str, list[int]]]],
        lags_future: Optional[Union[list[int], dict[str, list[int]]]],
        output_chunk_length: int,
        output_chunk_shift: int,
        multi_models: bool,
        max_samples_per_ts: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Helper function to create the X and y arrays by building them block by block (one per covariates)."""
        feats_times = self.get_feature_times(
            target,
            past,
            future,
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            max_samples_per_ts,
            output_chunk_shift,
        )
        # Construct `X` by constructing each block, then concatenate these
        # blocks together along component axis:
        X_target = self.construct_X_block(target, feats_times, lags)
        X_past = self.construct_X_block(past, feats_times, lags_past)
        X_future = self.construct_X_block(future, feats_times, lags_future)
        all_X = (X_target, X_past, X_future)
        to_concat = [X for X in all_X if X is not None]
        expected_X = np.concatenate(to_concat, axis=1)
        expected_y = self.create_y(
            target,
            feats_times,
            output_chunk_length,
            multi_models,
            output_chunk_shift,
        )
        if len(expected_X.shape) == 2:
            expected_X = expected_X[:, :, np.newaxis]
        if len(expected_y.shape) == 2:
            expected_y = expected_y[:, :, np.newaxis]
        return expected_X, expected_y, feats_times

    def helper_check_lagged_data(
        self,
        convert_lags_to_dict: bool,
        expected_X: np.ndarray,
        expected_y: np.ndarray,
        expected_times_x,
        expected_times_y,
        target: Optional[Union[TimeSeries, list[TimeSeries]]],
        past_cov: Optional[Union[TimeSeries, list[TimeSeries]]],
        future_cov: Optional[Union[TimeSeries, list[TimeSeries]]],
        lags: Optional[Union[list[int], dict[str, list[int]]]],
        lags_past: Optional[Union[list[int], dict[str, list[int]]]],
        lags_future: Optional[Union[list[int], dict[str, list[int]]]],
        output_chunk_length: int,
        output_chunk_shift: int,
        use_static_covariates: bool,
        multi_models: bool,
        max_samples_per_ts: Optional[int],
        use_moving_windows: bool,
        concatenate: bool,
        **kwargs,
    ):
        """Helper function to call the `create_lagged_training_data()` method with lags argument either in the list
        format or the dictionary format (automatically convert them when they are identical across components).

        Assertions are different depending on the value of `concatenate` to account for the output shape.
        """
        if convert_lags_to_dict:
            lags_as_dict = self.convert_lags_to_dict(
                target,
                past_cov if lags_past else None,
                future_cov if lags_future else None,
                lags,
                lags_past,
                lags_future,
            )
            lags_ = lags_as_dict["target"]
            lags_past_ = lags_as_dict["past"]
            lags_future_ = lags_as_dict["future"]
        else:
            lags_ = lags
            lags_past_ = lags_past
            lags_future_ = lags_future

        # convert indexes to list of tuples to simplify processing
        expected_times_x = (
            expected_times_x
            if isinstance(expected_times_x, Sequence)
            else [expected_times_x]
        )
        expected_times_y = (
            expected_times_y
            if isinstance(expected_times_y, Sequence)
            else [expected_times_y]
        )

        X, y, times, _, _ = create_lagged_training_data(
            target_series=target,
            output_chunk_length=output_chunk_length,
            past_covariates=past_cov if lags_past_ else None,
            future_covariates=future_cov if lags_future_ else None,
            lags=lags_,
            lags_past_covariates=lags_past_,
            lags_future_covariates=lags_future_,
            uses_static_covariates=use_static_covariates,
            multi_models=multi_models,
            max_samples_per_ts=max_samples_per_ts,
            use_moving_windows=use_moving_windows,
            output_chunk_shift=output_chunk_shift,
            concatenate=concatenate,
        )
        # should have the exact same number of indexes
        assert len(times) == len(expected_times_x) == len(expected_times_y)

        # Check that time index(es) match:
        for time, exp_time in zip(times, expected_times_x):
            assert exp_time.equals(time)

        if concatenate:
            # Number of observations should match number of feature times:
            data_length = sum(len(time) for time in times)
            exp_length_x = sum(len(exp_time) for exp_time in expected_times_x)
            exp_length_y = sum(len(exp_time) for exp_time in expected_times_y)
            assert exp_length_x == exp_length_y
            assert X.shape[0] == exp_length_x == data_length
            assert y.shape[0] == exp_length_y == data_length

            # Check that outputs match:
            assert X.shape == expected_X.shape
            assert np.allclose(expected_X, X)
            assert y.shape == expected_y.shape
            assert np.allclose(expected_y, y)
        else:
            # Check the number of observation for each series
            for x_, exp_time_x, y_, exp_time_y, time in zip(
                X, expected_times_x, y, expected_times_y, times
            ):
                assert x_.shape[0] == len(time) == len(exp_time_x)
                assert y_.shape[0] == len(time) == len(exp_time_y)

            # Check that outputs match:
            for x_, y_ in zip(X, y):
                assert np.allclose(X, x_)
                assert np.allclose(y, y_)

    #
    #   Generated Test Cases
    #

    target_with_no_cov = helper_create_multivariate_linear_timeseries(
        n_components=1,
        components_names=["no_static"],
        start_value=0,
        end_value=10,
        start=2,
        length=10,
        freq=2,
    )
    n_comp = 2
    target_with_static_cov = helper_create_multivariate_linear_timeseries(
        n_components=n_comp,
        components_names=["static_0", "static_1"],
        start_value=0,
        end_value=10,
        start=2,
        length=10,
        freq=2,
    )
    target_with_static_cov = target_with_static_cov.with_static_covariates(
        pd.DataFrame({"dummy": [1]})  # leads to "global" static cov component name
    )
    target_with_static_cov2 = target_with_static_cov.with_static_covariates(
        pd.DataFrame({
            "dummy": [i for i in range(n_comp)]
        })  # leads to sharing target component names
    )
    target_with_static_cov3 = target_with_static_cov.with_static_covariates(
        pd.DataFrame({
            "dummy": [i for i in range(n_comp)],
            "dummy1": [i for i in range(n_comp)],
        })  # leads to sharing target component names
    )

    past = helper_create_multivariate_linear_timeseries(
        n_components=3,
        components_names=["past_0", "past_1", "past_2"],
        start_value=10,
        end_value=20,
        start=2,
        length=10,
        freq=2,
    )
    future = helper_create_multivariate_linear_timeseries(
        n_components=4,
        components_names=["future_0", "future_1", "future_2", "future_3"],
        start_value=20,
        end_value=30,
        start=2,
        length=10,
        freq=2,
    )

    # Input parameter combinations used to generate test cases:
    output_chunk_length_combos = (1, 3)
    output_chunk_shift_combos = (0, 1)
    multi_models_combos = (False, True)
    max_samples_per_ts_combos = (1, 2, None)
    # lags are sorted ascending as done by the models internally
    target_lag_combos = past_lag_combos = (None, [-3, -1], [-2, -1])
    future_lag_combos = (*target_lag_combos, [0], [1, 2], [-1, 1], [0, 2])

    # minimum series length
    min_n_ts = 8 + max(output_chunk_shift_combos)

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_training_data_equal_freq(self, series_type: str):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries with equal frequencies. Since all timeseries
        are of the same frequency, the implementation of the 'moving window' method is
        being tested here.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, and different values, but
        # they're all of the same frequency:
        if series_type == "integer":
            target = helper_create_multivariate_linear_timeseries(
                n_components=2,
                start_value=0,
                end_value=10,
                start=2,
                length=self.min_n_ts,
                freq=2,
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3,
                start_value=10,
                end_value=20,
                start=4,
                length=self.min_n_ts + 1,
                freq=2,
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4,
                start_value=20,
                end_value=30,
                start=6,
                length=self.min_n_ts + 2,
                freq=2,
            )
        else:
            target = helper_create_multivariate_linear_timeseries(
                n_components=2,
                start_value=0,
                end_value=10,
                start=pd.Timestamp("1/2/2000"),
                length=self.min_n_ts,
                freq="2D",
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3,
                start_value=10,
                end_value=20,
                start=pd.Timestamp("1/4/2000"),
                length=self.min_n_ts + 1,
                freq="2D",
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4,
                start_value=20,
                end_value=30,
                start=pd.Timestamp("1/6/2000"),
                length=self.min_n_ts + 1,
                freq="2D",
            )
        # Conduct test for each input parameter combo:
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
            output_chunk_shift,
        ) in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
            self.output_chunk_shift_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue

            expected_X, expected_y, expected_times = (
                self.helper_create_expected_lagged_data(
                    target,
                    past,
                    future,
                    lags,
                    lags_past,
                    lags_future,
                    output_chunk_length,
                    output_chunk_shift,
                    multi_models,
                    max_samples_per_ts,
                )
            )

            kwargs = {
                "expected_X": expected_X,
                "expected_y": expected_y,
                "expected_times_x": expected_times,
                "expected_times_y": expected_times,
                "target": target,
                "past_cov": past,
                "future_cov": future,
                "lags": lags,
                "lags_past": lags_past,
                "lags_future": lags_future,
                "output_chunk_length": output_chunk_length,
                "output_chunk_shift": output_chunk_shift,
                "use_static_covariates": False,
                "multi_models": multi_models,
                "max_samples_per_ts": max_samples_per_ts,
                "use_moving_windows": True,
                "concatenate": True,
            }

            self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_training_data_unequal_freq(self, series_type):
        """
        Tests that `create_lagged_training_data` produces `X`, `y`, and `times`
        outputs that are consistent with those generated by using the helper
        functions `get_feature_times`, `construct_X_block`, and `construct_labels`.
        Consistency is checked over all combinations of parameter values
        specified by `self.target_lag_combos`, `self.covariates_lag_combos`,
        `self.output_chunk_length_combos`, `self.multi_models_combos`, and
        `self.max_samples_per_ts_combos`.

        This particular test uses timeseries of unequal frequencies. Since all timeseries
        are *not* of the same frequency, the implementation of the 'time intersection' method
        is being tested here.
        """
        # Define range index timeseries - each has different number of components,
        # different start times, different lengths, different values, and different
        # frequencies:
        if series_type == "integer":
            target = helper_create_multivariate_linear_timeseries(
                n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3
            )
        else:
            target = helper_create_multivariate_linear_timeseries(
                n_components=2,
                start_value=0,
                end_value=10,
                start=pd.Timestamp("1/1/2000"),
                length=20,
                freq="D",
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3,
                start_value=10,
                end_value=20,
                start=pd.Timestamp("1/2/2000"),
                length=10,
                freq="2D",
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4,
                start_value=20,
                end_value=30,
                start=pd.Timestamp("1/3/2000"),
                length=7,
                freq="3D",
            )
        # Conduct test for each input parameter combo:
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
            output_chunk_shift,
        ) in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
            self.output_chunk_shift_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features and
            # labels for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue

            expected_X, expected_y, expected_times = (
                self.helper_create_expected_lagged_data(
                    target,
                    past,
                    future,
                    lags,
                    lags_past,
                    lags_future,
                    output_chunk_length,
                    output_chunk_shift,
                    multi_models,
                    max_samples_per_ts,
                )
            )

            kwargs = {
                "expected_X": expected_X,
                "expected_y": expected_y,
                "expected_times_x": expected_times,
                "expected_times_y": expected_times,
                "target": target,
                "past_cov": past,
                "future_cov": future,
                "lags": lags,
                "lags_past": lags_past,
                "lags_future": lags_future,
                "output_chunk_length": output_chunk_length,
                "output_chunk_shift": output_chunk_shift,
                "use_static_covariates": False,
                "multi_models": multi_models,
                "max_samples_per_ts": max_samples_per_ts,
                "use_moving_windows": False,
                "concatenate": True,
            }

            self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_lagged_training_data_method_consistency(self, series_type):
        """
        Tests that `create_lagged_training_data` produces the same result
        when `use_moving_windows = False` and when `use_moving_windows = True`
        for all parameter combinations used in the 'generated' test cases.

        Obviously, if both the 'Moving Window Method' and the 'Time Intersection'
        are both wrong in the same way, this test won't reveal any bugs. With this
        being said, if this test fails, something is definitely wrong in either
        one or both of the implemented methods.
        """
        # Define datetime index timeseries - each has different number of components,
        # different start times, different lengths, different values, and of
        # different frequencies:
        if series_type == "integer":
            target = helper_create_multivariate_linear_timeseries(
                n_components=2, start_value=0, end_value=10, start=2, length=20, freq=1
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3, start_value=10, end_value=20, start=4, length=10, freq=2
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4, start_value=20, end_value=30, start=6, length=7, freq=3
            )
        else:
            target = helper_create_multivariate_linear_timeseries(
                n_components=2,
                start_value=0,
                end_value=10,
                start=pd.Timestamp("1/2/2000"),
                end=pd.Timestamp("1/18/2000"),
                freq="2D",
            )
            past = helper_create_multivariate_linear_timeseries(
                n_components=3,
                start_value=10,
                end_value=20,
                start=pd.Timestamp("1/4/2000"),
                end=pd.Timestamp("1/20/2000"),
                freq="2D",
            )
            future = helper_create_multivariate_linear_timeseries(
                n_components=4,
                start_value=20,
                end_value=30,
                start=pd.Timestamp("1/6/2000"),
                end=pd.Timestamp("1/22/2000"),
                freq="2D",
            )
        # Conduct test for each input parameter combo:
        for (
            lags,
            lags_past,
            lags_future,
            output_chunk_length,
            multi_models,
            max_samples_per_ts,
            output_chunk_shift,
        ) in product(
            self.target_lag_combos,
            self.past_lag_combos,
            self.future_lag_combos,
            self.output_chunk_length_combos,
            self.multi_models_combos,
            self.max_samples_per_ts_combos,
            self.output_chunk_shift_combos,
        ):
            all_lags = (lags, lags_past, lags_future)
            # Skip test where all lags are `None` - can't assemble features
            # for this single combo of input params:
            lags_is_none = [x is None for x in all_lags]
            if all(lags_is_none):
                continue
            # Using moving window method:
            X_mw, y_mw, times_mw, _, _ = create_lagged_training_data(
                target_series=target,
                output_chunk_length=output_chunk_length,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
                max_samples_per_ts=max_samples_per_ts,
                multi_models=multi_models,
                use_moving_windows=True,
                output_chunk_shift=output_chunk_shift,
            )
            # Using time intersection method:
            X_ti, y_ti, times_ti, _, _ = create_lagged_training_data(
                target_series=target,
                output_chunk_length=output_chunk_length,
                past_covariates=past if lags_past else None,
                future_covariates=future if lags_future else None,
                lags=lags,
                lags_past_covariates=lags_past,
                lags_future_covariates=lags_future,
                uses_static_covariates=False,
                max_samples_per_ts=max_samples_per_ts,
                multi_models=multi_models,
                use_moving_windows=False,
                output_chunk_shift=output_chunk_shift,
            )
            assert np.allclose(X_mw, X_ti)
            assert np.allclose(y_mw, y_ti)
            assert times_mw[0].equals(times_ti[0])

    #
    #   Specified Cases Tests
    #

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3],
            [False, True],
            ["datetime", "integer"],
        ),
    )
    def test_lagged_training_data_single_lag_single_component_same_series(self, config):
        """
        Tests that `create_lagged_training_data` correctly produces `X`, `y` and `times`
        when all the `series` inputs are identical, all the `lags` inputs consist
        of a single value, and `output_chunk_length` is `1`. In this situation, the
        expected `X` values can be found by concatenating three different slices of the
        same time series, and the expected  `y` can be formed by taking a single slice
        from the `target`.
        """
        output_chunk_shift, use_moving_windows, series_type = config
        if series_type == "integer":
            series = linear_timeseries(start=0, length=15)
        else:
            series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=15)

        lags = [-1]
        output_chunk_length = 1
        past_lags = [-3]
        future_lags = [2]
        # Can't create features for first 3 times (because `past_lags`) and last
        # two times (because `future_lags`):
        # also up until output_chunk_shift>=2, the future_lags are the reason for pushing back the end time
        # of expected X; after that the output shift pushes back additionally.
        step_back = max(0, output_chunk_shift - 2)
        expected_times_x = series.time_index[3 : -2 - step_back]
        expected_times_y = expected_times_x + output_chunk_shift * series.freq
        expected_y = series.all_values(copy=False)[
            3 + output_chunk_shift : 3 + output_chunk_shift + len(expected_times_y),
            :,
            :,
        ]
        # Offset `3:-2` by `-1` lag:
        expected_X_target = series.all_values(copy=False)[
            2 : 2 + len(expected_times_x), :, 0
        ]
        # Offset `3:-2` by `-3` lag -> gives `0:-5`:
        expected_X_past = series.all_values(copy=False)[: len(expected_times_x), :, 0]
        # Offset `3:-2` by `+2` lag -> gives `5:None`:
        expected_X_future = series.all_values(copy=False)[
            5 : 5 + len(expected_times_x), :, 0
        ]
        expected_X = np.concatenate(
            [expected_X_target, expected_X_past, expected_X_future], axis=1
        )
        expected_X = np.expand_dims(expected_X, axis=-1)

        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times_x,
            "expected_times_y": expected_times_y,
            "target": series,
            "past_cov": series,
            "future_cov": series,
            "lags": lags,
            "lags_past": past_lags,
            "lags_future": future_lags,
            "output_chunk_length": output_chunk_length,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": True,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3],
            [False, True],
            list(itertools.product(["datetime"], ["D", "2D", freqs["ms"], freqs["YE"]]))
            + list(itertools.product(["integer"], [1, 2])),
        ),
    )
    def test_lagged_training_data_extend_past_and_future_covariates(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case where features
        and labels can be created for a time that is *not* contained in `past_covariates`
        and/or `future_covariates`.

        More specifically, we define the series and lags such that a training example can
        be generated for time `target.end_time()`, even though this time isn't contained in
        neither `past` nor `future`.
        """
        output_chunk_shift, use_moving_windows, (series_type, freq) = config
        if series_type == "integer":
            target = linear_timeseries(
                start=0, length=10, start_value=1, end_value=2, freq=freq
            )
            past = linear_timeseries(
                start=0, length=8, start_value=2, end_value=3, freq=freq
            )
            future = linear_timeseries(
                start=0, length=6, start_value=3, end_value=4, freq=freq
            )
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                start_value=1,
                end_value=2,
                length=11,
                freq=freq,
            )
            past = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                start_value=2,
                end_value=3,
                length=9,
                freq=freq,
            )
            future = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                start_value=3,
                end_value=4,
                length=7,
                freq=freq,
            )

        # Can create feature for time `t = 10`, but this time isn't in `past` or `future`:
        lags = [-1]
        lags_past = [-2]
        lags_future = [-4]
        # Only want to check very last generated observation:
        max_samples_per_ts = 1
        # Expect `X` to be constructed from second-to-last value of `target` (i.e.
        # the value immediately prior to the label), and the very last values of
        # `past` and `future`:
        expected_X = np.concatenate([
            target.all_values(copy=False)[-2 - output_chunk_shift, :, 0],
            past.all_values(copy=False)[-1 - output_chunk_shift, :, 0],
            future.all_values(copy=False)[-1 - output_chunk_shift, :, 0],
        ]).reshape(1, -1, 1)
        # Label is very last value of `target`:
        expected_y = target.all_values(copy=False)[-1:, :, :]

        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )

        # Check correctness for both 'moving window' method
        # and 'time intersection' method:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": past,
            "future_cov": future,
            "lags": lags,
            "lags_past": lags_past,
            "lags_future": lags_future,
            "output_chunk_length": 1,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": True,
            "max_samples_per_ts": max_samples_per_ts,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3], [False, True], ["datetime", "integer"], [False, True]
        ),
    )
    def test_lagged_training_data_single_point(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case
        where only one possible training point can be generated.
        """
        output_chunk_shift, use_moving_windows, series_type, multi_models = config
        # Can only create feature using first value of series (i.e. `0`)
        # and can only create label using last value of series (i.e. `1`)
        if series_type == "integer":
            target = linear_timeseries(
                start=0, length=2 + output_chunk_shift, start_value=0, end_value=1
            )
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                length=2 + output_chunk_shift,
                start_value=0,
                end_value=1,
            )

        output_chunk_length = 1
        lags = [-1]
        expected_X = np.zeros((1, 1, 1))
        expected_y = np.ones((1, 1, 1))
        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )
        # Test correctness for 'moving window' and for 'time intersection' methods, as well
        # as for different `multi_models` values:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": None,
            "future_cov": None,
            "lags": lags,
            "lags_past": None,
            "lags_future": None,
            "output_chunk_length": output_chunk_length,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": multi_models,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3], [False, True], ["datetime", "integer"], [False, True]
        ),
    )
    def test_lagged_training_data_zero_lags(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case when
        `0` is included in `lags_future_covariates` (i.e. when we're using the values
        `future_covariates` at time `t` to predict the value of `target_series` at
        that same time point).
        """
        # Define `future` so that only value occurs at the same time as
        # the only possible label that can be extracted from `target_series`; the
        # only possible feature that can be created using these series utilises
        # the value of `future` at the same time as the label (i.e. a lag
        # of `0` away from the only feature time):
        output_chunk_shift, use_moving_windows, series_type, multi_models = config

        if series_type == "integer":
            target = linear_timeseries(
                start=0, length=2 + output_chunk_shift, start_value=0, end_value=1
            )
            future = linear_timeseries(
                start=target.end_time() - output_chunk_shift * target.freq,
                length=1,
                start_value=1,
                end_value=2,
            )
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                length=2 + output_chunk_shift,
                start_value=0,
                end_value=1,
            )
            future = linear_timeseries(
                start=target.end_time() - output_chunk_shift * target.freq,
                length=1,
                start_value=1,
                end_value=2,
            )

        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = np.array([[[0.0], [1.0]]])
        expected_y = np.ones((1, 1, 1))
        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": None,
            "future_cov": future,
            "lags": [-1],
            "lags_past": None,
            "lags_future": [0],
            "output_chunk_length": 1,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": multi_models,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3],
            [False, True],
            ["datetime", "integer"],
            [False, True],
            [-1, 0, 1],
            [-2, 0, 2],
        ),
    )
    def test_lagged_training_data_no_target_lags_future_covariates(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case without target lags and different
        future covariates lags.
        This test should always result in one training sample.
        Additionally, we test that:
        - future starts before the target but extends far enough to create one training sample
        - future shares same time as target
        - future starts after target but target extends far enough to create one training sample.
        """
        (
            output_chunk_shift,
            use_moving_windows,
            series_type,
            multi_models,
            cov_start_shift,
            cov_lag,
        ) = config

        # adapt covariate start, length, and target length so that only 1 sample can be extracted
        target_length = 1 + output_chunk_shift + max(cov_start_shift, 0)
        cov_length = 1 - min(cov_start_shift, 0)
        if series_type == "integer":
            cov_start = 0 + cov_start_shift + cov_lag
            target = linear_timeseries(
                start=0, length=target_length, start_value=0, end_value=1
            )
            future = linear_timeseries(
                start=cov_start, length=cov_length, start_value=2, end_value=3
            )
        else:
            freq = pd.tseries.frequencies.to_offset("D")
            cov_start = pd.Timestamp("1/1/2000") + (cov_start_shift + cov_lag) * freq
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                length=target_length,
                start_value=0,
                end_value=1,
                freq=freq,
            )
            future = linear_timeseries(
                start=cov_start,
                length=cov_length,
                start_value=2,
                end_value=3,
                freq=freq,
            )

        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = future[-1].all_values(copy=False)
        expected_y = target[-1].all_values(copy=False)
        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": None,
            "future_cov": future,
            "lags": None,
            "lags_past": None,
            "lags_future": [cov_lag],
            "output_chunk_length": 1,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": multi_models,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3],
            [False, True],
            ["datetime", "integer"],
            [False, True],
            [-1, 0],
            [-2, -1],
        ),
    )
    def test_lagged_training_data_no_target_lags_past_covariates(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case without target lags and different
        past covariates lags.
        This test should always result in one training sample.
        Additionally, we test that:
        - past starts before the target but extends far enough to create one training sample
        - past shares same time as target
        """
        (
            output_chunk_shift,
            use_moving_windows,
            series_type,
            multi_models,
            cov_start_shift,
            cov_lag,
        ) = config

        # adapt covariate start, length, and target length so that only 1 sample can be extracted
        target_length = 1 + output_chunk_shift + max(cov_start_shift, 0)
        cov_length = 1 - min(cov_start_shift, 0)
        if series_type == "integer":
            cov_start = 0 + cov_start_shift + cov_lag
            target = linear_timeseries(
                start=0, length=target_length, start_value=0, end_value=1
            )
            past = linear_timeseries(
                start=cov_start, length=cov_length, start_value=2, end_value=3
            )
        else:
            freq = pd.tseries.frequencies.to_offset("D")
            cov_start = pd.Timestamp("1/1/2000") + (cov_start_shift + cov_lag) * freq
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                length=target_length,
                start_value=0,
                end_value=1,
                freq=freq,
            )
            past = linear_timeseries(
                start=cov_start,
                length=cov_length,
                start_value=2,
                end_value=3,
                freq=freq,
            )

        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = past[-1].all_values(copy=False)
        expected_y = target[-1].all_values(copy=False)
        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": past,
            "future_cov": None,
            "lags": None,
            "lags_past": [cov_lag],
            "lags_future": None,
            "output_chunk_length": 1,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": multi_models,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3], [False, True], ["datetime", "integer"], [False, True]
        ),
    )
    def test_lagged_training_data_positive_lags(self, config):
        """
        Tests that `create_lagged_training_data` correctly handles case when
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
        output_chunk_shift, use_moving_windows, series_type, multi_models = config

        if series_type == "integer":
            target = linear_timeseries(
                start=0, length=2 + output_chunk_shift, start_value=0, end_value=1
            )
            future = linear_timeseries(
                start=target.end_time() - (output_chunk_shift - 1) * target.freq,
                length=1,
                start_value=1,
                end_value=2,
            )
        else:
            target = linear_timeseries(
                start=pd.Timestamp("1/1/2000"),
                length=2 + output_chunk_shift,
                start_value=0,
                end_value=1,
            )
            future = linear_timeseries(
                start=target.end_time() - (output_chunk_shift - 1) * target.freq,
                length=1,
                start_value=1,
                end_value=2,
            )
        # X comprises of first value of `target` (i.e. 0) and only value in `future`:
        expected_X = np.array([[[0.0], [1.0]]])
        expected_y = np.ones((1, 1, 1))
        expected_times = generate_index(
            start=target.end_time() - output_chunk_shift * target.freq,
            length=1,
            freq=target.freq,
        )
        # Check correctness for 'moving windows' and 'time intersection' methods, as
        # well as for different `multi_models` values:
        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": None,
            "future_cov": future,
            "lags": [-1],
            "lags_past": None,
            "lags_future": [1],
            "output_chunk_length": 1,
            "output_chunk_shift": output_chunk_shift,
            "use_static_covariates": False,
            "multi_models": multi_models,
            "max_samples_per_ts": None,
            "use_moving_windows": use_moving_windows,
            "concatenate": True,
        }

        self.helper_check_lagged_data(convert_lags_to_dict=False, **kwargs)

        if use_moving_windows:
            self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
        else:
            with pytest.raises(ValueError) as err:
                self.helper_check_lagged_data(convert_lags_to_dict=True, **kwargs)
            assert str(err.value).startswith(
                "`use_moving_windows=False` is not supported when any of the lags"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 3],
            [1, 2],
            [True, False],
            ["datetime", "integer"],
        ),
    )
    def test_lagged_training_data_comp_wise_lags(self, config):
        """
        Tests that `create_lagged_training_data` generate the expected values when the
        lags are component-specific over multivariate series.

        Note that this is supported only when use_moving_window=True.
        """
        output_chunk_shift, output_chunk_length, multi_models, series_type = config

        lags_tg = {"target_0": [-4, -1], "target_1": [-4, -1]}
        lags_pc = [-3]
        lags_fc = {"future_0": [-1, 0], "future_1": [-2, 1]}

        if series_type == "integer":
            start_tg = 0
            start_pc = start_tg + 1
            start_fc = start_tg + 2
        else:
            start_tg = pd.Timestamp("2000-01-15")
            start_pc = pd.Timestamp("2000-01-16")
            start_fc = pd.Timestamp("2000-01-17")

        # length = max lag - min lag + 1 = -1 + 4 + 1 = 4
        target = helper_create_multivariate_linear_timeseries(
            n_components=2,
            components_names=["target_0", "target_1"],
            length=4 + output_chunk_shift + output_chunk_length,
            start=start_tg,
        )
        # length = max lag - min lag + 1 = -3 + 3 + 1 = 1
        past = (
            helper_create_multivariate_linear_timeseries(
                n_components=2,
                components_names=["past_0", "past_1"],
                length=1,
                start=start_pc,
            )
            + 100
        )
        # length = max lag - min lag + 1 = 1 + 2 + 1 = 4
        future = (
            helper_create_multivariate_linear_timeseries(
                n_components=2,
                components_names=["future_0", "future_1"],
                length=4 + output_chunk_shift + output_chunk_length,
                start=start_fc,
            )
            + 200
        )

        # extremes lags are manually computed, similarly to the model.lags attribute
        feats_times = self.get_feature_times(
            target,
            past,
            future,
            [-4, -1],  # min, max target lag
            [-3],  # unique past lag
            [-2, 1],  # min, max future lag
            output_chunk_length,
            None,
            output_chunk_shift,
        )

        # reorder the features to obtain target_0_lag-4, target_1_lag-4, target_0_lag-1, target_1_lag-1
        X_target = [
            self.construct_X_block(
                target["target_0"], feats_times, lags_tg["target_0"][0:1]
            ),
            self.construct_X_block(
                target["target_1"], feats_times, lags_tg["target_1"][0:1]
            ),
            self.construct_X_block(
                target["target_0"], feats_times, lags_tg["target_0"][1:2]
            ),
            self.construct_X_block(
                target["target_1"], feats_times, lags_tg["target_1"][1:2]
            ),
        ]
        # single lag for all the components, can be kept as is
        X_past = [
            self.construct_X_block(past[name], feats_times, lags_pc)
            for name in ["past_0", "past_1"]
        ]
        # reorder the features to obtain future_1_lag-2, future_0_lag-1, future_0_lag0, future_1_lag1
        X_future = [
            self.construct_X_block(
                future["future_1"], feats_times, lags_fc["future_1"][0:1]
            ),
            self.construct_X_block(
                future["future_0"], feats_times, lags_fc["future_0"][0:1]
            ),
            self.construct_X_block(
                future["future_0"], feats_times, lags_fc["future_0"][1:2]
            ),
            self.construct_X_block(
                future["future_1"], feats_times, lags_fc["future_1"][1:2]
            ),
        ]
        all_X = X_target + X_past + X_future
        expected_X = np.concatenate(all_X, axis=1)[:, :, np.newaxis]
        expected_y = self.create_y(
            target,
            feats_times,
            output_chunk_length,
            multi_models,
            output_chunk_shift,
        )[:, :, np.newaxis]

        # lags are already in dict format
        self.helper_check_lagged_data(
            convert_lags_to_dict=True,
            expected_X=expected_X,
            expected_y=expected_y,
            expected_times_x=feats_times,
            expected_times_y=feats_times,
            target=target,
            past_cov=past,
            future_cov=future,
            lags=lags_tg,
            lags_past=lags_pc,
            lags_future=lags_fc,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            use_static_covariates=False,
            multi_models=multi_models,
            max_samples_per_ts=None,
            use_moving_windows=True,
            concatenate=True,
        )

    def test_lagged_training_data_sequence_inputs(self):
        """
        Tests that `create_lagged_training_data` correctly handles being
        passed a sequence of `TimeSeries` inputs, as opposed to individual
        `TimeSeries`.
        """
        # Define two simple tabularization problems:
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = linear_timeseries(start=6, end=11)
        ts_tg = (target_1, target_2)
        ts_pc = (past_1, past_2)
        ts_fc = (future_1, future_2)
        lags = lags_past = lags_future = [-1]
        output_chunk_length = 1
        # Expected solution:
        expected_X_1 = np.concatenate(
            3 * [target_1.all_values(copy=False)[:-1, :, :]], axis=1
        )
        expected_X_2 = np.concatenate(
            3 * [target_2.all_values(copy=False)[:-1, :, :]], axis=1
        )
        expected_X = np.concatenate([expected_X_1, expected_X_2], axis=0)
        expected_y_1 = target_1.all_values(copy=False)[1:, :, :]
        expected_y_2 = target_2.all_values(copy=False)[1:, :, :]
        expected_y = np.concatenate([expected_y_1, expected_y_2], axis=0)
        expected_times_1 = target_1.time_index[1:]
        expected_times_2 = target_2.time_index[1:]

        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": [expected_times_1, expected_times_2],
            "expected_times_y": [expected_times_1, expected_times_2],
            "target": ts_tg,
            "past_cov": ts_pc,
            "future_cov": ts_fc,
            "lags": lags,
            "lags_past": lags_past,
            "lags_future": lags_future,
            "output_chunk_length": output_chunk_length,
            "output_chunk_shift": 0,
            "use_static_covariates": False,
            "multi_models": True,
            "max_samples_per_ts": None,
            "use_moving_windows": True,
        }

        # concatenate=True
        self.helper_check_lagged_data(
            convert_lags_to_dict=False, concatenate=True, **kwargs
        )
        self.helper_check_lagged_data(
            convert_lags_to_dict=True, concatenate=True, **kwargs
        )

        # concatenate=False
        self.helper_check_lagged_data(
            convert_lags_to_dict=False, concatenate=False, **kwargs
        )
        self.helper_check_lagged_data(
            convert_lags_to_dict=True, concatenate=False, **kwargs
        )

    def test_lagged_training_data_stochastic_series(self):
        """
        Tests that `create_lagged_training_data` is correctly vectorised
        over the sample axes of the input `TimeSeries`.
        """
        # Define two simple tabularization problems:
        target_1 = past_1 = future_1 = linear_timeseries(start=0, end=5)
        target_2 = past_2 = future_2 = 2 * target_1
        target = target_1.concatenate(target_2, axis=2)
        past = past_1.concatenate(past_2, axis=2)
        future = future_1.concatenate(future_2, axis=2)
        lags = lags_past = lags_future = [-1]
        output_chunk_length = 1
        # Expected solution:
        expected_X = np.concatenate(
            3 * [target.all_values(copy=False)[:-1, :, :]], axis=1
        )
        expected_y = target.all_values(copy=False)[1:, :, :]
        expected_times = target.time_index[1:]

        kwargs = {
            "expected_X": expected_X,
            "expected_y": expected_y,
            "expected_times_x": expected_times,
            "expected_times_y": expected_times,
            "target": target,
            "past_cov": past,
            "future_cov": future,
            "lags": lags,
            "lags_past": lags_past,
            "lags_future": lags_future,
            "output_chunk_length": output_chunk_length,
            "output_chunk_shift": 0,
            "use_static_covariates": False,
            "multi_models": True,
            "max_samples_per_ts": None,
            "use_moving_windows": True,
        }

        self.helper_check_lagged_data(
            convert_lags_to_dict=False, concatenate=True, **kwargs
        )
        self.helper_check_lagged_data(
            convert_lags_to_dict=True, concatenate=True, **kwargs
        )

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
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=series_1,
                    output_chunk_length=1,
                    lags=lags,
                    past_covariates=series_2,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "Specified series do not share any common times for which features can be created."
                == str(err.value)
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
                with pytest.raises(ValueError) as err:
                    create_lagged_training_data(
                        target_series=series_1,
                        output_chunk_length=1,
                        lags_past_covariates=lags,
                        uses_static_covariates=False,
                        use_moving_windows=use_moving_windows,
                        output_chunk_shift=0,
                    )
            assert "Must specify at least one series-lags pair." == str(err.value)
            # Warnings will be thrown indicating that `past_covariates`
            # is specified without `lags_past_covariates`, and that
            # `lags_future_covariates` specified without
            # `future_covariates` - ignore both warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError) as err:
                    create_lagged_training_data(
                        target_series=series_1,
                        output_chunk_length=1,
                        lags_future_covariates=lags,
                        past_covariates=series_2,
                        uses_static_covariates=False,
                        use_moving_windows=use_moving_windows,
                        output_chunk_shift=0,
                    )
            assert "Must specify at least one series-lags pair." == str(err.value)

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
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=0,
                    lags=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert "`output_chunk_length` must be a positive `int`." == str(err.value)
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=1.1,
                    lags=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert "`output_chunk_length` must be a positive `int`." == str(err.value)

    def test_lagged_training_data_no_lags_specified_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when no lags are specified.
        """
        target = linear_timeseries(start=1, length=20, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=target,
                    output_chunk_length=1,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`."
                == str(err.value)
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
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=5,
                    lags=[-20, -10],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "`target_series` must have at least "
                "`-min(lags) + output_chunk_length + output_chunk_shift` = 25 "
                "time steps; instead, it only has 2."
            ) == str(err.value)
            # `lags_past_covariates` too large test:
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    past_covariates=series,
                    lags_past_covariates=[-5, -3],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "`past_covariates` must have at least "
                "`-min(lags_past_covariates) + max(lags_past_covariates) + 1` = 3 "
                "time steps; instead, it only has 2."
            ) == str(err.value)

    def test_lagged_training_data_invalid_lag_values_error(self):
        """
        Tests that `create_lagged_training_data` throws correct
        error when invalid lag values are specified. More specifically:
            1. If `lags` contains any value greater than `-1`, an error
            should be thrown (since times > `-1` are used for labels).
            2. If `lags_past_covariates` contains any value greater than
            `-1` (since, by definition, past covariates are only 'know' )
            3. `lags_future_covariates` should be able to contain positive
            values, negative values, and/or zero without throwing any errors.
        """
        series = linear_timeseries(start=1, length=3, freq=1)
        # Check error thrown by 'moving windows' method and by 'time intersection' method:
        for use_moving_windows in (False, True):
            # Test invalid `lags` values:
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    lags=[0],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "`lags` must be a `Sequence` or `Dict` containing only `int` values less than 0."
            ) == str(err.value)
            # Test invalid `lags_past_covariates` values:
            with pytest.raises(ValueError) as err:
                create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    past_covariates=series,
                    lags_past_covariates=[0],
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
            assert (
                "`lags_past_covariates` must be a `Sequence` or `Dict` containing only `int` values less than 0."
            ) == str(err.value)
            # Test invalid `lags_future_covariates` values:
            create_lagged_training_data(
                target_series=series,
                output_chunk_length=1,
                future_covariates=series,
                lags_future_covariates=[-1, 0, 1],
                uses_static_covariates=False,
                use_moving_windows=use_moving_windows,
                output_chunk_shift=0,
            )

    def test_lagged_training_data_dict_lags_no_moving_window_error(self):
        """
        Tests that `create_lagged_training_data` throws correct error
        when `use_moving_window` is set to `False` and lags are provided
        as a dict for a multivariate series.
        """
        ts = linear_timeseries(start=1, length=20, freq=1, column_name="lin1")
        lags = [-1]
        lags_dict = {"lin1": [-1]}
        # one series, one set of lags are dict
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                target_series=ts,
                output_chunk_length=1,
                lags=lags_dict,
                uses_static_covariates=False,
                use_moving_windows=False,
                output_chunk_shift=0,
            )
        assert str(err.value).startswith(
            "`use_moving_windows=False` is not supported when any of the lags is provided as a dictionary."
        )
        # all the series are provided, only one passed as dict
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                target_series=ts,
                past_covariates=ts,
                future_covariates=ts,
                output_chunk_length=1,
                lags=lags,
                lags_past_covariates=lags_dict,
                lags_future_covariates=lags,
                uses_static_covariates=False,
                use_moving_windows=False,
                output_chunk_shift=0,
            )
        assert str(err.value).startswith(
            "`use_moving_windows=False` is not supported when any of the lags is provided as a dictionary."
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
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`future_covariates` was specified without accompanying "
                    "`lags_future_covariates` and, thus, will be ignored."
                )
            # Specify `lags_future_covariates`, but not `future_covariates`:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    lags=lags,
                    lags_future_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert str(w[0].message) == (
                    "`lags_future_covariates` was specified without accompanying "
                    "`future_covariates` and, thus, will be ignored."
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
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
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
            # Specify `target_series`, but not `lags` - unlike previous tests,
            # this should *not* throw a warning:
            with warnings.catch_warnings(record=True) as w:
                _ = create_lagged_training_data(
                    target_series=series,
                    output_chunk_length=1,
                    past_covariates=series,
                    lags_past_covariates=lags,
                    uses_static_covariates=False,
                    use_moving_windows=use_moving_windows,
                    output_chunk_shift=0,
                )
                assert len(w) == 0

    @pytest.mark.parametrize(
        "config",
        [
            # target no static covariate
            (
                target_with_no_cov,
                None,
                None,
                [-2, -1],
                None,
                None,
                False,
                1,
                ["no_static_target_lag-2", "no_static_target_lag-1"],
                ["no_static_target_hrz0"],
            ),
            # target with static covariate (but don't use them in feature names)
            (
                target_with_static_cov,
                None,
                None,
                [-4, -1],
                None,
                None,
                False,
                2,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-4",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                    "static_0_target_hrz1",
                    "static_1_target_hrz1",
                ],
            ),
            # target with static covariate (acting on global target components)
            (
                target_with_static_cov,
                None,
                None,
                [-4, -1],
                None,
                None,
                True,
                1,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-4",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                    "dummy_statcov_target_global_components",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
            # target with static covariate (component specific)
            (
                target_with_static_cov2,
                None,
                None,
                [-4, -1],
                None,
                None,
                True,
                1,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-4",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                    "dummy_statcov_target_static_0",
                    "dummy_statcov_target_static_1",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
            # target with static covariate (component specific & multivariate)
            (
                target_with_static_cov3,
                None,
                None,
                [-4, -1],
                None,
                None,
                True,
                1,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-4",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                    "dummy_statcov_target_static_0",
                    "dummy_statcov_target_static_1",
                    "dummy1_statcov_target_static_0",
                    "dummy1_statcov_target_static_1",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
            # target + past
            (
                target_with_no_cov,
                past,
                None,
                [-4, -3],
                [-1],
                None,
                False,
                1,
                [
                    "no_static_target_lag-4",
                    "no_static_target_lag-3",
                    "past_0_pastcov_lag-1",
                    "past_1_pastcov_lag-1",
                    "past_2_pastcov_lag-1",
                ],
                ["no_static_target_hrz0"],
            ),
            # target + future
            (
                target_with_no_cov,
                None,
                future,
                [-2, -1],
                None,
                [3],
                False,
                1,
                [
                    "no_static_target_lag-2",
                    "no_static_target_lag-1",
                    "future_0_futcov_lag3",
                    "future_1_futcov_lag3",
                    "future_2_futcov_lag3",
                    "future_3_futcov_lag3",
                ],
                ["no_static_target_hrz0"],
            ),
            # past + future
            (
                target_with_no_cov,
                past,
                future,
                None,
                [-1],
                [2],
                False,
                1,
                [
                    "past_0_pastcov_lag-1",
                    "past_1_pastcov_lag-1",
                    "past_2_pastcov_lag-1",
                    "future_0_futcov_lag2",
                    "future_1_futcov_lag2",
                    "future_2_futcov_lag2",
                    "future_3_futcov_lag2",
                ],
                ["no_static_target_hrz0"],
            ),
            # target with static (not used) + past + future
            (
                target_with_static_cov,
                past,
                future,
                [-2, -1],
                [-1],
                [2],
                False,
                1,
                [
                    "static_0_target_lag-2",
                    "static_1_target_lag-2",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                    "past_0_pastcov_lag-1",
                    "past_1_pastcov_lag-1",
                    "past_2_pastcov_lag-1",
                    "future_0_futcov_lag2",
                    "future_1_futcov_lag2",
                    "future_2_futcov_lag2",
                    "future_3_futcov_lag2",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
            # multiple series with same components names, including past/future covariates
            (
                [target_with_static_cov, target_with_static_cov],
                [past, past],
                [future, future],
                [-3],
                [-1],
                [2],
                False,
                1,
                [
                    "static_0_target_lag-3",
                    "static_1_target_lag-3",
                    "past_0_pastcov_lag-1",
                    "past_1_pastcov_lag-1",
                    "past_2_pastcov_lag-1",
                    "future_0_futcov_lag2",
                    "future_1_futcov_lag2",
                    "future_2_futcov_lag2",
                    "future_3_futcov_lag2",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
            # multiple series with different components will use the first series as reference
            (
                [
                    target_with_static_cov,
                    target_with_no_cov.stack(target_with_no_cov),
                ],
                [past, past],
                [future, past.stack(target_with_no_cov)],
                [-2, -1],
                [-1],
                [2],
                False,
                1,
                [
                    "static_0_target_lag-2",
                    "static_1_target_lag-2",
                    "static_0_target_lag-1",
                    "static_1_target_lag-1",
                    "past_0_pastcov_lag-1",
                    "past_1_pastcov_lag-1",
                    "past_2_pastcov_lag-1",
                    "future_0_futcov_lag2",
                    "future_1_futcov_lag2",
                    "future_2_futcov_lag2",
                    "future_3_futcov_lag2",
                ],
                [
                    "static_0_target_hrz0",
                    "static_1_target_hrz0",
                ],
            ),
        ],
    )
    def test_create_lagged_component_names(self, config):
        """
        Tests that `create_lagged_component_names` produces the expected features name depending
        on the lags, output_chunk_length and covariates.

        When lags are component-specific, they are identical across all the components.
        """
        (
            ts_tg,
            ts_pc,
            ts_fc,
            lags_tg,
            lags_pc,
            lags_fc,
            use_static_cov,
            ocl,
            expected_lagged_features,
            expected_lagged_labels,
        ) = config
        # lags as list
        created_lagged_features, created_lagged_labels = create_lagged_component_names(
            target_series=ts_tg,
            past_covariates=ts_pc,
            future_covariates=ts_fc,
            lags=lags_tg,
            lags_past_covariates=lags_pc,
            lags_future_covariates=lags_fc,
            concatenate=False,
            use_static_covariates=use_static_cov,
            output_chunk_length=ocl,
        )

        # converts lags to dictionary format
        lags_as_dict = self.convert_lags_to_dict(
            ts_tg,
            ts_pc,
            ts_fc,
            lags_tg,
            lags_pc,
            lags_fc,
        )

        created_lagged_features_dict_lags, created_lagged_labels_dict_lags = (
            create_lagged_component_names(
                target_series=ts_tg,
                past_covariates=ts_pc,
                future_covariates=ts_fc,
                lags=lags_as_dict["target"],
                lags_past_covariates=lags_as_dict["past"],
                lags_future_covariates=lags_as_dict["future"],
                concatenate=False,
                use_static_covariates=use_static_cov,
                output_chunk_length=ocl,
            )
        )
        assert expected_lagged_features == created_lagged_features
        assert expected_lagged_features == created_lagged_features_dict_lags
        assert expected_lagged_labels == created_lagged_labels
        assert expected_lagged_labels == created_lagged_labels_dict_lags

    @pytest.mark.parametrize(
        "config",
        [
            # lags have the same minimum
            (
                target_with_static_cov,
                None,
                None,
                {"static_0": [-4, -2], "static_1": [-4, -3]},
                None,
                None,
                False,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-4",
                    "static_1_target_lag-3",
                    "static_0_target_lag-2",
                ],
            ),
            # lags are not overlapping
            (
                target_with_static_cov,
                None,
                None,
                {"static_0": [-4, -1], "static_1": [-3, -2]},
                None,
                None,
                False,
                [
                    "static_0_target_lag-4",
                    "static_1_target_lag-3",
                    "static_1_target_lag-2",
                    "static_0_target_lag-1",
                ],
            ),
            # default lags for target, overlapping lags for past covariates
            (
                target_with_static_cov,
                past,
                None,
                {"static_0": [-3], "static_1": [-3]},
                {"past_0": [-4, -3], "past_1": [-3, -2], "past_2": [-2]},
                None,
                False,
                [
                    "static_0_target_lag-3",
                    "static_1_target_lag-3",
                    "past_0_pastcov_lag-4",
                    "past_0_pastcov_lag-3",
                    "past_1_pastcov_lag-3",
                    "past_1_pastcov_lag-2",
                    "past_2_pastcov_lag-2",
                ],
            ),
            # no lags for target, future covariates lags are not in the components order
            (
                target_with_static_cov,
                None,
                future,
                None,
                None,
                {
                    "future_3": [-2, 0, 2],
                    "future_0": [-4, 1],
                    "future_2": [1],
                    "future_1": [-2, 2],
                },
                False,
                [
                    "future_0_futcov_lag-4",
                    "future_1_futcov_lag-2",
                    "future_3_futcov_lag-2",
                    "future_3_futcov_lag0",
                    "future_0_futcov_lag1",
                    "future_2_futcov_lag1",
                    "future_1_futcov_lag2",
                    "future_3_futcov_lag2",
                ],
            ),
        ],
    )
    def test_create_lagged_component_names_different_lags(self, config):
        """
        Tests that `create_lagged_component_names` when lags are different across components.

        The lagged features should be sorted by lags, then by components.
        """
        (
            ts_tg,
            ts_pc,
            ts_fc,
            lags_tg,
            lags_pc,
            lags_fc,
            use_static_cov,
            expected_lagged_features,
        ) = config

        created_lagged_features, _ = create_lagged_component_names(
            target_series=ts_tg,
            past_covariates=ts_pc,
            future_covariates=ts_fc,
            lags=lags_tg,
            lags_past_covariates=lags_pc,
            lags_future_covariates=lags_fc,
            concatenate=False,
            use_static_covariates=use_static_cov,
        )
        assert expected_lagged_features == created_lagged_features

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [10, 50],
            [True, False],
            ["linear", "exponential"],
            ["D", "2D", 2],
            [True, False],
        ),
    )
    def test_correct_generated_weights_exponential(self, config):
        """Tests built in weights generation for:
        - varying target series sizes
        - with and without moving window tabularization
        - different weight functions
        - datetime and integer index
        - single and multiple series
        """
        training_size, use_moving_windows, sample_weight, freq, single_series = config

        if not isinstance(freq, int):
            freq = pd.tseries.frequencies.to_offset(freq)
            start = pd.Timestamp("2000-01-01")
        else:
            start = 1

        train_y = linear_timeseries(start=start, length=training_size, freq=freq)

        _, y, _, _, weights = create_lagged_training_data(
            lags=[-4, -1],
            target_series=train_y if single_series else [train_y] * 2,
            output_chunk_length=1,
            uses_static_covariates=False,
            sample_weight=sample_weight,
            output_chunk_shift=0,
            use_moving_windows=use_moving_windows,
        )

        len_y = len(y) if single_series else int(len(y) / 2)
        if sample_weight == "linear":
            expected_weights = np.linspace(0, 1, len(train_y))[-len_y:, None, None]
        else:  # exponential decay
            time_steps = np.linspace(0, 1, len(train_y))
            expected_weights = np.exp(-10 * (1 - time_steps))[-len_y:, None, None]

        if not single_series:
            expected_weights = np.concatenate([expected_weights] * 2, axis=0)

        assert weights.shape == y.shape
        np.testing.assert_array_almost_equal(weights, expected_weights)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [10, 20],
            [True, False],
            [True, False],
            [1, 2],
            [0, 1],
            ["D", "2D", 2],
            [True, False],
            [True, False],
        ),
    )
    def test_correct_user_weights(self, config):
        """Checks correct weights extraction for:
        - varying target series sizes
        - with and without moving window tabularization
        - weights with exact matching index and longer weights
        - single and multi horizon
        - with and without output chunk shift
        - datetime and integer index
        - single and multiple series
        - uni- and multivariate series
        """
        (
            training_size,
            use_moving_windows,
            weights_longer,
            ocl,
            ocs,
            freq,
            single_series,
            univar_series,
        ) = config
        if not isinstance(freq, int):
            freq = pd.tseries.frequencies.to_offset(freq)
            start = pd.Timestamp("2000-01-01")
        else:
            start = 1

        train_y = linear_timeseries(start=start, length=training_size, freq=freq)
        if not univar_series:
            train_y.stack(train_y)

        # weights are either longer or have the exact time index as the target series
        n_weights = len(train_y) + 2 * int(weights_longer)
        ts_weights = TimeSeries.from_times_and_values(
            times=generate_index(
                start=train_y.start_time() - int(weights_longer) * freq,
                length=n_weights,
                freq=freq,
            ),
            values=np.linspace(0, 1, n_weights),
        )
        if not univar_series:
            ts_weights.stack(ts_weights + 1.0)

        _, y, _, _, weights = create_lagged_training_data(
            lags=[-4, -1],
            target_series=train_y if single_series else [train_y] * 2,
            output_chunk_length=ocl,
            uses_static_covariates=False,
            sample_weight=ts_weights if single_series else [ts_weights] * 2,
            output_chunk_shift=ocs,
            use_moving_windows=use_moving_windows,
        )

        # weights shape must match label shape, since we have one
        # weight per sample and predict step
        assert weights.shape == y.shape

        # get the weights matching the index of the target series
        weights_exact = ts_weights.values()
        if weights_longer:
            weights_exact = weights_exact[1:-1]

        # the weights correspond to the same sample and time index as the `y` labels
        expected_weights = []
        len_y_single = len(y) if single_series else int(len(y) / 2)
        for i in range(ocl):
            mask = slice(-(i + len_y_single), -i if i else None)
            expected_weights.append(weights_exact[mask])
        expected_weights = np.concatenate(expected_weights, axis=1)[:, ::-1]
        if not single_series:
            expected_weights = np.concatenate([expected_weights] * 2, axis=0)
        np.testing.assert_array_almost_equal(weights[:, :, 0], expected_weights)

    @pytest.mark.parametrize(
        "use_moving_windows",
        [True, False],
    )
    def test_invalid_sample_weights(self, use_moving_windows):
        """Checks invalid weights raise error with and without moving window tabularization
        - too short series
        - not enough series
        - invalid string
        - weights shape does not match number of `series` components
        """
        training_size = 10

        train_y = linear_timeseries(length=training_size)
        weights_too_short = train_y[:-2]
        with pytest.raises(ValueError) as err:
            _ = create_lagged_training_data(
                lags=[-4, -1],
                target_series=train_y,
                output_chunk_length=1,
                uses_static_covariates=False,
                sample_weight=weights_too_short,
                output_chunk_shift=0,
                use_moving_windows=use_moving_windows,
            )
        assert (
            str(err.value)
            == "The `sample_weight` series must have at least the same times as the target `series`."
        )

        with pytest.raises(ValueError) as err:
            _ = create_lagged_training_data(
                lags=[-4, -1],
                target_series=[train_y] * 2,
                output_chunk_length=1,
                uses_static_covariates=False,
                sample_weight=[train_y],
                output_chunk_shift=0,
                use_moving_windows=use_moving_windows,
            )
        assert (
            str(err.value)
            == "The provided sequence of target `series` must have the same length as the provided sequence "
            "of `sample_weight`."
        )

        with pytest.raises(ValueError) as err:
            _ = create_lagged_training_data(
                lags=[-4, -1],
                target_series=[train_y] * 2,
                output_chunk_length=1,
                uses_static_covariates=False,
                sample_weight="invalid",
                output_chunk_shift=0,
                use_moving_windows=use_moving_windows,
            )
        assert str(err.value).startswith("Invalid `sample_weight` value: `'invalid'`. ")

        with pytest.raises(ValueError) as err:
            _ = create_lagged_training_data(
                lags=[-4, -1],
                target_series=train_y,
                output_chunk_length=1,
                uses_static_covariates=False,
                sample_weight=train_y.stack(train_y),
                output_chunk_shift=0,
                use_moving_windows=use_moving_windows,
            )
        assert str(err.value) == (
            "The number of components in `sample_weight` must either be `1` or "
            "match the number of target series components `1`."
        )
