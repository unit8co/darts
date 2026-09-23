from itertools import product

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.utils.data.tabularization import (
    StepwiseLaggedFeatures,
    _get_feature_times,
    create_lagged_component_names,
    create_lagged_prediction_data,
    create_lagged_training_data,
)


class TestStepwiseLaggedFeatures:
    """Tests the `StepwiseLaggedFeatures` container."""

    @staticmethod
    def helper_container(
        n_obs=5, n_feat=4, n_horizons=3, n_samples=None, step_cols=(1, 3)
    ):
        rng = np.random.default_rng(0)
        shape = (n_obs, n_feat) + ((n_samples,) if n_samples else ())
        base = rng.random(shape)
        step_values = rng.random((n_horizons, n_obs, len(step_cols)) + shape[2:])
        # invariant: horizon 0 of the step-wise columns is stored in `base`
        step_values[0] = base[:, list(step_cols)]
        return StepwiseLaggedFeatures(base, step_values, step_cols)

    @pytest.mark.parametrize("n_samples", [None, 2])
    def test_horizon(self, n_samples):
        X = self.helper_container(n_samples=n_samples)
        assert X.n_horizons == 3
        assert X.shape == X.base.shape
        assert len(X) == 5
        assert X.ndim == X.base.ndim
        # horizon 0 is `base` itself (no copy)
        assert X.horizon(0) is X.base
        other_cols = [0, 2]
        for h in range(1, X.n_horizons):
            X_h = X.horizon(h)
            assert X_h is not X.base
            np.testing.assert_array_equal(X_h[:, X.step_cols], X.step_values[h])
            np.testing.assert_array_equal(X_h[:, other_cols], X.base[:, other_cols])
        # `horizons()` yields every horizon in order
        for h, X_h in enumerate(X.horizons()):
            np.testing.assert_array_equal(X_h, X.horizon(h))
        with pytest.raises(IndexError):
            X.horizon(3)

    def test_format_fn(self):
        X = self.helper_container()
        X.format_fn = lambda arr: pd.DataFrame(arr)
        for h in range(X.n_horizons):
            X_h = X.horizon(h)
            assert isinstance(X_h, pd.DataFrame)
            X.format_fn = None
            np.testing.assert_array_equal(X_h.values, X.horizon(h))
            X.format_fn = lambda arr: pd.DataFrame(arr)

    def test_base_updates_are_shared(self):
        """Non step-wise columns are only stored in `base`: in-place updates reach every horizon."""
        X = self.helper_container()
        X.base[:, 0] = -1.0
        for h in range(X.n_horizons):
            assert (X.horizon(h)[:, 0] == -1.0).all()

    @pytest.mark.parametrize("n_samples", [None, 2])
    def test_getitem(self, n_samples):
        X = self.helper_container(n_samples=n_samples)
        sub = X[1:4]
        assert isinstance(sub, StepwiseLaggedFeatures)
        for h in range(X.n_horizons):
            np.testing.assert_array_equal(sub.horizon(h), X.horizon(h)[1:4])
        sub = X[::2]
        for h in range(X.n_horizons):
            np.testing.assert_array_equal(sub.horizon(h), X.horizon(h)[::2])
        if n_samples:
            sub = X[:, :, 0]
            assert sub.ndim == 2
            for h in range(X.n_horizons):
                np.testing.assert_array_equal(sub.horizon(h), X.horizon(h)[:, :, 0])
        # the features axis cannot be indexed
        with pytest.raises(IndexError):
            X[:, 0]
        with pytest.raises(IndexError):
            X[:, 1:3]
        # integer indexing of the observations axis would drop the axis
        with pytest.raises(IndexError):
            X[0]
        with pytest.raises(IndexError):
            X[(slice(None),) * (X.ndim + 1)]

    def test_repeat(self):
        X = self.helper_container()
        rep = X.repeat(3)
        assert rep.shape == (15, 4)
        for h in range(X.n_horizons):
            np.testing.assert_array_equal(
                rep.horizon(h), np.repeat(X.horizon(h), 3, axis=0)
            )
        with pytest.raises(ValueError):
            X.repeat(3, axis=1)

    def test_concatenate(self):
        X1, X2 = self.helper_container(n_obs=5), self.helper_container(n_obs=2)
        X = StepwiseLaggedFeatures.concatenate([X1, X2])
        assert X.shape == (7, 4)
        for h in range(X.n_horizons):
            np.testing.assert_array_equal(
                X.horizon(h), np.concatenate([X1.horizon(h), X2.horizon(h)])
            )
        with pytest.raises(ValueError):
            StepwiseLaggedFeatures.concatenate([
                X1,
                self.helper_container(step_cols=(0,)),
            ])
        with pytest.raises(ValueError):
            StepwiseLaggedFeatures.concatenate([
                X1,
                self.helper_container(n_horizons=2),
            ])
        with pytest.raises(ValueError):
            StepwiseLaggedFeatures.concatenate([])

    def test_with_base(self):
        X = self.helper_container()
        new_base = np.hstack([X.base, np.ones((5, 2))])
        X2 = X.with_base(new_base)
        assert X2.shape == (5, 6)
        np.testing.assert_array_equal(X2.step_cols, X.step_cols)
        for h in range(X.n_horizons):
            np.testing.assert_array_equal(X2.horizon(h)[:, :4], X.horizon(h))
            assert (X2.horizon(h)[:, 4:] == 1.0).all()

    def test_invalid_construction(self):
        base = np.zeros((5, 4))
        with pytest.raises(ValueError):
            # wrong number of dimensions
            StepwiseLaggedFeatures(base, np.zeros((5, 2)), [1, 3])
        with pytest.raises(ValueError):
            # wrong number of observations
            StepwiseLaggedFeatures(base, np.zeros((3, 4, 2)), [1, 3])
        with pytest.raises(ValueError):
            # wrong number of step-wise columns
            StepwiseLaggedFeatures(base, np.zeros((3, 5, 1)), [1, 3])
        with pytest.raises(ValueError):
            # wrong samples dimension
            StepwiseLaggedFeatures(np.zeros((5, 4, 2)), np.zeros((3, 5, 2, 1)), [1, 3])


class TestStepwiseTabularization:
    """
    Tests the tabularization of step-wise future covariates lags (`lags_future_covariates_stepwise`): for a
    step-wise component with lag `k`, the feature of horizon `h` must be the value of the covariate at time
    `t + h + k` (the tabularization functions expect `output_chunk_shift` to already be included in the lags).
    """

    lags_target = [-2, -1]
    lags_past = {"p0": [-1], "p1": [-3, -1]}
    lags_future = {"f0": [-1, 0, 2], "f1": [0, 1], "f2": [-2]}

    @staticmethod
    def helper_create_series(
        n_target: int = 30,
        n_covs: int = 30,
        series_type: str = "integer",
        n_samples: int = 1,
    ):
        """Series whose values encode the time step, so that lagged values can be checked exactly."""

        def values(offset, n_comps, n):
            vals = offset + np.arange(n, dtype=float)[:, None, None]
            vals = vals + 100 * np.arange(n_comps)[None, :, None]
            vals = vals + 0.001 * np.arange(n_samples)[None, None, :]
            return vals

        if series_type == "integer":
            start = 0
            kwargs = {}
        else:
            start = pd.Timestamp("2000-01-01")
            kwargs = {"freq": "D"}
        target = TimeSeries.from_times_and_values(
            pd.RangeIndex(start, start + n_target)
            if series_type == "integer"
            else pd.date_range(start, periods=n_target, **kwargs),
            values(0, 1, n_target),
            columns=["y"],
        )
        past = TimeSeries.from_times_and_values(
            pd.RangeIndex(start, start + n_covs)
            if series_type == "integer"
            else pd.date_range(start, periods=n_covs, **kwargs),
            values(1000, 2, n_covs),
            columns=["p0", "p1"],
        )
        future = TimeSeries.from_times_and_values(
            pd.RangeIndex(start, start + n_covs)
            if series_type == "integer"
            else pd.date_range(start, periods=n_covs, **kwargs),
            values(10000, 3, n_covs),
            columns=["f0", "f1", "f2"],
        )
        return target, past, future

    @staticmethod
    def helper_check_stepwise_values(
        X: StepwiseLaggedFeatures,
        times: pd.Index,
        future: TimeSeries,
        feature_names: list[str],
        stepwise: dict[str, bool],
        output_chunk_shift: int = 0,
    ):
        """Checks that `X.horizon(h)` contains the expected values for the step-wise and non step-wise columns."""
        step_cols = set(X.step_cols.tolist())
        # the step-wise columns are exactly the future covariates columns of the flagged components
        expected_step_cols = {
            i
            for i, name in enumerate(feature_names)
            if "_futcov_lag" in name
            and stepwise.get(
                name.split("_futcov_lag")[0], stepwise.get("default_lags", False)
            )
        }
        assert step_cols == expected_step_cols
        # invariant: horizon 0 of the step-wise columns is stored in `base`
        np.testing.assert_array_equal(X.step_values[0], X.base[:, X.step_cols])
        time_pos = future.time_index.get_indexer(times)
        assert (time_pos >= 0).all()
        fut_vals = future.all_values(copy=False)
        other_cols = [i for i in range(X.shape[1]) if i not in step_cols]
        for h in range(X.n_horizons):
            X_h = X.horizon(h)
            np.testing.assert_array_equal(X_h[:, other_cols], X.base[:, other_cols])
            for col in X.step_cols:
                comp, lag = feature_names[col].split("_futcov_lag")
                lag = int(lag) - output_chunk_shift
                comp_idx = list(future.components).index(comp)
                expected = fut_vals[
                    time_pos + h + lag + output_chunk_shift, comp_idx, :
                ]
                np.testing.assert_array_equal(X_h[:, col], expected)

    @pytest.mark.parametrize(
        "config",
        list(
            product(
                [1, 3],  # stride
                [0, 2],  # output_chunk_shift
                [None, 7],  # max_samples_per_ts
                ["integer", "datetime"],  # series_type
                [1, 2],  # n_samples
                [
                    {"f0": True},
                    {"f1": True, "f2": True},
                    {"default_lags": True},
                    {"default_lags": True, "f1": False},
                ],  # stepwise
            )
        ),
    )
    def test_training_data_stepwise_values(self, config):
        """
        Tests that the step-wise training features are the expected values, that the non step-wise columns and
        the labels are identical to the legacy tabularization (on the shared feature times), and that the last
        `output_chunk_length - 1` feature times are excluded.
        """
        (
            stride,
            output_chunk_shift,
            max_samples_per_ts,
            series_type,
            n_samples,
            stepwise,
        ) = config
        output_chunk_length = 4
        target, past, future = self.helper_create_series(
            series_type=series_type, n_samples=n_samples
        )
        # the tabularization expects the shift to already be included in the future lags
        lags_future = {
            comp: [lag + output_chunk_shift for lag in lags]
            for comp, lags in self.lags_future.items()
        }
        kwargs = dict(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            uses_static_covariates=False,
            stride=stride,
            max_samples_per_ts=max_samples_per_ts,
        )
        X, y, times, _, _ = create_lagged_training_data(
            lags_future_covariates_stepwise=stepwise, **kwargs
        )
        X_legacy, y_legacy, times_legacy, _, _ = create_lagged_training_data(**kwargs)
        assert isinstance(X, StepwiseLaggedFeatures)
        assert isinstance(X_legacy, np.ndarray)
        assert X.n_horizons == output_chunk_length
        assert X.shape[1:] == X_legacy.shape[1:]
        times, times_legacy = times[0], times_legacy[0]
        # the last anchor `t` requires `future[t + shift + k + output_chunk_length - 1]` for a step-wise lag
        # `k` (versus `future[t + shift + k]` for an absolute one), and the labels up to
        # `target[t + shift + output_chunk_length - 1]`; a component with a large absolute lag can be the
        # constraint on its own, in which case the step-wise ones cost no anchor at all
        max_lag = max(max(lags) for lags in self.lags_future.values())
        max_stepwise_lag = max(
            max(lags)
            for comp, lags in self.lags_future.items()
            if stepwise.get(comp, stepwise.get("default_lags", False))
        )
        last_target_anchor = len(target) - output_chunk_length - output_chunk_shift
        if max_samples_per_ts is None:
            last_anchor = min(
                last_target_anchor,
                len(future)
                - 1
                - output_chunk_shift
                - max(max_lag, max_stepwise_lag + output_chunk_length - 1),
            )
            last_legacy_anchor = min(
                last_target_anchor,
                len(future) - 1 - output_chunk_shift - max_lag,
            )
            assert times[-1] == future.time_index[last_anchor]
            assert times_legacy[-1] == future.time_index[last_legacy_anchor]
        else:
            assert len(times) == max_samples_per_ts
        # base features and labels are identical to the legacy ones for the shared anchors (with a
        # stride or `max_samples_per_ts`, the anchor sets only partially overlap)
        shared = times_legacy.get_indexer(times)
        overlap = shared >= 0
        if stride == 1:
            assert overlap.any()
        np.testing.assert_array_equal(X.base[overlap], X_legacy[shared[overlap]])
        np.testing.assert_array_equal(y[overlap], y_legacy[shared[overlap]])
        feature_names, _ = create_lagged_component_names(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=lags_future,
            output_chunk_length=output_chunk_length,
        )
        self.helper_check_stepwise_values(
            X, times, future, feature_names, stepwise, output_chunk_shift
        )
        # oracle: with consecutive anchors, `X_h[i] == X_0[i + h]` for the step-wise columns
        if stride == 1:
            for h in range(1, output_chunk_length):
                np.testing.assert_array_equal(
                    X.horizon(h)[:-h][:, X.step_cols],
                    X.horizon(0)[h:][:, X.step_cols],
                )

    @pytest.mark.parametrize(
        "config",
        list(
            product(
                [1, 3],
                ["integer", "datetime"],
                # `f1` is the case where the absolute `f0` already reaches further than the step-wise lags
                [{"f0": True}, {"f1": True}, {"default_lags": True}],
            )
        ),
    )
    def test_prediction_data_stepwise_values(self, config):
        stride, series_type, stepwise = config
        output_chunk_length = 3
        target, past, future = self.helper_create_series(series_type=series_type)
        kwargs = dict(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=self.lags_future,
            uses_static_covariates=False,
            stride=stride,
        )
        X, times = create_lagged_prediction_data(
            lags_future_covariates_stepwise=stepwise,
            output_chunk_length=output_chunk_length,
            **kwargs,
        )
        X_legacy, times_legacy = create_lagged_prediction_data(**kwargs)
        assert isinstance(X, StepwiseLaggedFeatures)
        assert X.n_horizons == output_chunk_length
        times, times_legacy = times[0], times_legacy[0]
        # the future covariates are the constraint at the end: the anchors are limited by the largest of the
        # absolute lags and of the step-wise lags, the latter read `output_chunk_length - 1` steps further
        max_lag = max(max(lags) for lags in self.lags_future.values())
        max_stepwise_lag = max(
            max(lags)
            for comp, lags in self.lags_future.items()
            if stepwise.get(comp, stepwise.get("default_lags", False))
        )
        last_anchor = (
            len(future) - 1 - max(max_lag, max_stepwise_lag + output_chunk_length - 1)
        )
        assert times[-1] == future.time_index[last_anchor]
        assert times_legacy[-1] == future.time_index[len(future) - 1 - max_lag]
        shared = times_legacy.get_indexer(times)
        overlap = shared >= 0
        if stride == 1:
            assert overlap.all()
        np.testing.assert_array_equal(X.base[overlap], X_legacy[shared[overlap]])
        feature_names, _ = create_lagged_component_names(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=self.lags_future,
            output_chunk_length=1,
        )
        self.helper_check_stepwise_values(X, times, future, feature_names, stepwise)

    def test_prediction_data_future_covariates_constraint(self):
        """When the future covariates are the constraint, `output_chunk_length - 1` anchors are lost."""
        output_chunk_length = 3
        target, past, future = self.helper_create_series(n_target=30, n_covs=30)
        kwargs = dict(
            target_series=target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates={"f0": [0, 2], "f1": [0], "f2": [0]},
            uses_static_covariates=False,
        )
        X, times = create_lagged_prediction_data(
            lags_future_covariates_stepwise={"f0": True},
            output_chunk_length=output_chunk_length,
            **kwargs,
        )
        _, times_legacy = create_lagged_prediction_data(**kwargs)
        assert times[0][-1] == times_legacy[0][-1] - (output_chunk_length - 1)
        # the last anchor uses the very last future covariates value at horizon `output_chunk_length - 1`
        last_val = X.horizon(output_chunk_length - 1)[-1, X.step_cols[-1], 0]
        assert last_val == future["f0"].all_values()[-1, 0, 0]

    @pytest.mark.parametrize("stepwise", [{"f0": True}, {"default_lags": True}])
    def test_only_stepwise_lag_zero(self, stepwise):
        """The issue #2968 use case: a single lag `0` per component, step-wise."""
        target, _, future = self.helper_create_series()
        output_chunk_length = 5
        X, y, times, _, _ = create_lagged_training_data(
            target_series=target,
            future_covariates=future,
            lags=[-1],
            lags_future_covariates={"f0": [0], "f1": [0], "f2": [0]},
            output_chunk_length=output_chunk_length,
            output_chunk_shift=0,
            uses_static_covariates=False,
            lags_future_covariates_stepwise=stepwise,
        )
        time_pos = future.time_index.get_indexer(times[0])
        fut_vals = future.all_values()
        for h in range(output_chunk_length):
            X_h = X.horizon(h)
            for col, comp in zip([1, 2, 3], ["f0", "f1", "f2"]):
                shift_h = h if stepwise.get(comp, stepwise.get("default_lags")) else 0
                np.testing.assert_array_equal(
                    X_h[:, col, 0], fut_vals[time_pos + shift_h, col - 1, 0]
                )
            # the label of horizon `h` and the step-wise feature of horizon `h` are aligned in time
            np.testing.assert_array_equal(
                y[:, h, 0], target.all_values()[time_pos + h, 0, 0]
            )

    @pytest.mark.parametrize("concatenate", [True, False])
    def test_multiple_series_and_static_covariates(self, concatenate):
        target, past, future = self.helper_create_series()
        target = target.with_static_covariates(pd.DataFrame({"s0": [7.0], "s1": [8.0]}))
        kwargs = dict(
            target_series=[target, target[:20]],
            past_covariates=[past, past],
            future_covariates=[future, future],
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=self.lags_future,
            output_chunk_length=3,
            output_chunk_shift=0,
            uses_static_covariates=True,
            concatenate=concatenate,
            lags_future_covariates_stepwise={"f0": True, "f2": True},
        )
        X, y, times, static_shape, _ = create_lagged_training_data(**kwargs)
        X_single = [
            create_lagged_training_data(**{
                **kwargs,
                "target_series": kwargs["target_series"][i],
                "past_covariates": past,
                "future_covariates": future,
                "concatenate": True,
            })[0]
            for i in range(2)
        ]
        assert static_shape == (1, 2)
        if concatenate:
            assert isinstance(X, StepwiseLaggedFeatures)
            assert len(X) == len(times[0]) + len(times[1])
            for h in range(3):
                np.testing.assert_array_equal(
                    X.horizon(h),
                    np.concatenate([X_single[0].horizon(h), X_single[1].horizon(h)]),
                )
                # static covariates are appended to the right, on every horizon
                assert (X.horizon(h)[:, -2:, 0] == [7.0, 8.0]).all()
        else:
            assert isinstance(X, list) and len(X) == 2
            for X_i, X_s in zip(X, X_single):
                assert isinstance(X_i, StepwiseLaggedFeatures)
                for h in range(3):
                    np.testing.assert_array_equal(X_i.horizon(h), X_s.horizon(h))
        # static covariates columns are never step-wise
        n_feats = X.shape[1] if concatenate else X[0].shape[1]
        step_cols = X.step_cols if concatenate else X[0].step_cols
        assert step_cols.max() < n_feats - 2

    @pytest.mark.parametrize(
        "config",
        [
            # `output_chunk_length == 1`: every horizon is the first one
            (1, {"f0": True}),
            # no component flagged
            (4, {"f0": False, "f1": False}),
            (4, {"default_lags": False}),
            # `None`
            (4, None),
        ],
    )
    def test_inactive_returns_ndarray(self, config):
        output_chunk_length, stepwise = config
        target, past, future = self.helper_create_series()
        kwargs = dict(
            target_series=target,
            past_covariates=past,
            future_covariates=future,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            lags_future_covariates=self.lags_future,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=0,
            uses_static_covariates=False,
        )
        X, y, times, _, _ = create_lagged_training_data(
            lags_future_covariates_stepwise=stepwise, **kwargs
        )
        X_legacy, y_legacy, times_legacy, _, _ = create_lagged_training_data(**kwargs)
        assert type(X) is np.ndarray
        np.testing.assert_array_equal(X, X_legacy)
        np.testing.assert_array_equal(y, y_legacy)
        assert times[0].equals(times_legacy[0])
        X, _ = create_lagged_prediction_data(
            target_series=target,
            future_covariates=future,
            lags=self.lags_target,
            lags_future_covariates=self.lags_future,
            uses_static_covariates=False,
            lags_future_covariates_stepwise=stepwise,
            output_chunk_length=output_chunk_length,
        )
        assert type(X) is np.ndarray

    def test_no_future_covariates_lags_is_noop(self):
        target, past, _ = self.helper_create_series()
        X, _, _, _, _ = create_lagged_training_data(
            target_series=target,
            past_covariates=past,
            lags=self.lags_target,
            lags_past_covariates=self.lags_past,
            output_chunk_length=4,
            output_chunk_shift=0,
            uses_static_covariates=False,
            lags_future_covariates_stepwise={"f0": True},
        )
        assert type(X) is np.ndarray

    def test_errors(self):
        target, past, future = self.helper_create_series()
        kwargs = dict(
            target_series=target,
            future_covariates=future,
            lags=self.lags_target,
            lags_future_covariates=self.lags_future,
            output_chunk_length=4,
            output_chunk_shift=0,
            uses_static_covariates=False,
        )
        # `multi_models=False` is not supported (the shift must be applied to the lags instead)
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True},
                multi_models=False,
                **kwargs,
            )
        assert "multi_models=False" in str(err.value)
        # must be a dictionary
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(lags_future_covariates_stepwise=True, **kwargs)
        assert "must be a dictionary" in str(err.value)
        # unknown components
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True, "unknown": True}, **kwargs
            )
        assert "unknown" in str(err.value)
        # `lags_future_covariates` must be component-specific
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True},
                **{**kwargs, "lags_future_covariates": [0, 1]},
            )
        assert "dictionary" in str(err.value)
        # `use_moving_windows=False` is not supported (already rejected by the dict lags)
        with pytest.raises(ValueError):
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True},
                use_moving_windows=False,
                **kwargs,
            )
        # too short future covariates: `output_chunk_length - 1` more values are required
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True},
                **{**kwargs, "future_covariates": future[:7]},
            )
        assert "output_chunk_length - 1" in str(err.value)
        # ... whereas the legacy tabularization accepts it (`-min(lags) + max(lags) + 1 = 5` values)
        create_lagged_training_data(**{**kwargs, "future_covariates": future[:7]})
        # different frequencies are not supported
        future_hourly = TimeSeries.from_times_and_values(
            pd.date_range("2000-01-01", periods=200, freq="h"),
            future.all_values()[:1].repeat(200, axis=0),
            columns=future.components,
        )
        target_daily = TimeSeries.from_times_and_values(
            pd.date_range("2000-01-01", periods=30, freq="D"),
            target.all_values(),
            columns=target.components,
        )
        with pytest.raises(ValueError) as err:
            create_lagged_training_data(
                lags_future_covariates_stepwise={"f0": True},
                **{
                    **kwargs,
                    "target_series": target_daily,
                    "future_covariates": future_hourly,
                },
            )
        assert "same frequency" in str(err.value)

    @pytest.mark.parametrize("stepwise_extension", [0, 3])
    def test_get_feature_times_stepwise_extension(self, stepwise_extension):
        """The `stepwise_extension` excludes the last feature times of the future covariates."""
        _, _, future = self.helper_create_series(n_covs=20)
        lags_future = [-1, 0, 2]
        times, min_lags, max_lags = _get_feature_times(
            future_covariates=future,
            lags_future_covariates=lags_future,
            is_training=False,
            return_min_and_max_lags=True,
            stepwise_extension=stepwise_extension,
        )
        times_future = times[2]
        # the first feature time requires `future[t - 1]`, the last one `future[t + 2 + extension]`
        assert times_future[0] == 1
        assert times_future[-1] == 19 - 2 - stepwise_extension
        assert max_lags[2] == 1
        assert min_lags[2] == -(2 + stepwise_extension)
