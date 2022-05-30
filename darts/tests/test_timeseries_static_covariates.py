import copy

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.tests.base_test_class import DartsBaseTestClass
from darts.timeseries import DEFAULT_GLOBAL_STATIC_COV_NAME
from darts.utils.timeseries_generation import _generate_index, linear_timeseries


class TimeSeriesStaticCovariateTestCase(DartsBaseTestClass):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        n_groups = 5
        len_ts = 10
        times = (
            pd.concat(
                [
                    pd.DataFrame(
                        _generate_index(start=pd.Timestamp(2010, 1, 1), length=len_ts)
                    )
                ]
                * n_groups,
                axis=0,
            )
            .reset_index(drop=True)
            .rename(columns={0: "times"})
        )

        x = pd.DataFrame(np.random.randn(n_groups * len_ts, 3), columns=["a", "b", "c"])
        static_multivar = pd.DataFrame(
            [
                [i, 0 if j < (len_ts // 2) else 1]
                for i in range(n_groups)
                for j in range(len_ts)
            ],
            columns=["st1", "st2"],
        )

        df_long_multi = pd.DataFrame(
            pd.concat([times, x, static_multivar], axis=1),
        )
        df_long_multi.loc[:, "constant"] = 1
        df_long_uni = df_long_multi.drop(columns=["st2"])

        cls.n_groups = n_groups
        cls.len_ts = len_ts
        cls.df_long_multi = df_long_multi
        cls.df_long_uni = df_long_uni

    def test_timeseries_from_longitudinal_df(self):
        # univariate static covs: only group by "st1", keep static covs "st1"
        value_cols = ["a", "b", "c"]
        ts_groups1 = TimeSeries.from_group_dataframe(
            df=self.df_long_uni,
            group_cols="st1",
            static_cols=None,
            time_col="times",
            value_cols=value_cols,
        )
        assert len(ts_groups1) == self.n_groups
        for i, ts in enumerate(ts_groups1):
            assert ts.static_covariates.index.equals(
                pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
            )
            assert ts.static_covariates.shape == (1, 1)
            assert ts.static_covariates.columns.equals(pd.Index(["st1"]))
            assert (ts.static_covariate_values(copy=False) == [[i]]).all()

        # multivariate static covs: only group by "st1", keep static covs "st1", "constant"
        ts_groups2 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1"],
            static_cols="constant",
            time_col="times",
            value_cols=value_cols,
        )
        assert len(ts_groups2) == self.n_groups
        for i, ts in enumerate(ts_groups2):
            assert ts.static_covariates.shape == (1, 2)
            assert ts.static_covariates.columns.equals(pd.Index(["st1", "constant"]))
            assert (ts.static_covariate_values(copy=False) == [[i, 1]]).all()

        # multivariate static covs: group by "st1" and "st2", keep static covs "st1", "st2", "constant"
        ts_groups3 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            static_cols=["constant"],
            time_col="times",
            value_cols=value_cols,
        )
        assert len(ts_groups3) == self.n_groups * 2
        for idx, ts in enumerate(ts_groups3):
            i = idx // 2
            j = idx % 2
            assert ts.static_covariates.shape == (1, 3)
            assert ts.static_covariates.columns.equals(
                pd.Index(["st1", "st2", "constant"])
            )
            assert (ts.static_covariate_values(copy=False) == [[i, j, 1]]).all()

        df = copy.deepcopy(self.df_long_multi)
        df.loc[:, "non_static"] = np.arange(len(df))
        # non static columns as static columns should raise an error
        with pytest.raises(ValueError):
            _ = TimeSeries.from_group_dataframe(
                df=df,
                group_cols=["st1"],
                static_cols=["non_static"],
                time_col="times",
                value_cols=value_cols,
            )

        # groups that are too short for TimeSeries requirements should raise an error
        with pytest.raises(ValueError):
            _ = TimeSeries.from_group_dataframe(
                df=df,
                group_cols=["st1", "non_static"],
                static_cols=None,
                time_col="times",
                value_cols=value_cols,
            )

    def test_with_static_covariates_univariate(self):
        ts = linear_timeseries(length=10)
        static_covs_series = pd.Series([0.0, 1.0], index=["st1", "st2"])
        static_covs_df = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])

        # check immutable
        ts.with_static_covariates(static_covs_series)
        assert not ts.has_static_covariates

        # from Series
        ts = ts.with_static_covariates(static_covs_series)
        assert ts.has_static_covariates
        np.testing.assert_almost_equal(
            ts.static_covariate_values(copy=False),
            np.expand_dims(static_covs_series.values, -1).T,
        )
        assert ts.static_covariates.index.equals(ts.components)

        # from DataFrame
        ts = ts.with_static_covariates(static_covs_df)
        assert ts.has_static_covariates
        np.testing.assert_almost_equal(
            ts.static_covariate_values(copy=False), static_covs_df.values
        )
        assert ts.static_covariates.index.equals(ts.components)

        # with None
        ts = ts.with_static_covariates(None)
        assert ts.static_covariates is None
        assert not ts.has_static_covariates

        # only pd.Series, pd.DataFrame or None
        with pytest.raises(ValueError):
            _ = ts.with_static_covariates([1, 2, 3])

        # multivariate does not work with univariate TimeSeries
        with pytest.raises(ValueError):
            static_covs_multi = pd.concat([static_covs_series] * 2, axis=1).T
            _ = ts.with_static_covariates(static_covs_multi)

    def test_static_covariate_values(self):
        ts = linear_timeseries(length=10)
        static_covs = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])
        ts = ts.with_static_covariates(static_covs)

        # changing values of copy should not change original DataFrame
        vals = ts.static_covariate_values(copy=True)
        vals[:] = -1.0
        assert (ts.static_covariate_values(copy=False) != -1.0).all()

        # changing values of view should change original DataFrame
        vals = ts.static_covariate_values(copy=False)
        vals[:] = -1.0
        assert (ts.static_covariate_values(copy=False) == -1.0).all()

        ts = ts.with_static_covariates(None)
        assert ts.static_covariate_values() is None

    def test_with_static_covariates_multivariate(self):
        ts = linear_timeseries(length=10)
        ts_multi = ts.stack(ts)
        static_covs = pd.DataFrame([[0.0, 1.0], [0.0, 1.0]], columns=["st1", "st2"])

        # from univariate static covariates
        ts_multi = ts_multi.with_static_covariates(static_covs.loc[0])
        assert ts_multi.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        assert ts_multi.static_covariates.columns.equals(static_covs.columns)
        np.testing.assert_almost_equal(
            ts_multi.static_covariate_values(copy=False), static_covs.loc[0:0].values
        )

        # from multivariate static covariates
        ts_multi = ts_multi.with_static_covariates(static_covs)
        assert ts_multi.static_covariates.index.equals(ts_multi.components)
        assert ts_multi.static_covariates.columns.equals(static_covs.columns)
        np.testing.assert_almost_equal(
            ts_multi.static_covariate_values(copy=False), static_covs.values
        )

        # raise an error if multivariate static covariates columns don't match the number of components in the series
        with pytest.raises(ValueError):
            _ = ts_multi.with_static_covariates(pd.concat([static_covs] * 2, axis=0))

    def test_stack(self):
        ts_uni = linear_timeseries(length=10)
        ts_multi = ts_uni.stack(ts_uni)

        static_covs_uni1 = pd.DataFrame([[0, 1]], columns=["st1", "st2"]).astype(int)
        static_covs_uni2 = pd.DataFrame([[3, 4]], columns=["st3", "st4"]).astype(int)
        static_covs_uni3 = pd.DataFrame(
            [[2, 3, 4]], columns=["st1", "st2", "st3"]
        ).astype(int)

        static_covs_multi = pd.DataFrame(
            [[0, 0], [1, 1]], columns=["st1", "st2"]
        ).astype(int)

        ts_uni = ts_uni.with_static_covariates(static_covs_uni1)
        ts_multi = ts_multi.with_static_covariates(static_covs_multi)

        # valid static covariates for concatenation/stack
        ts_stacked1 = ts_uni.stack(ts_uni)
        assert ts_stacked1.static_covariates.index.equals(ts_stacked1.components)
        np.testing.assert_almost_equal(
            ts_stacked1.static_covariate_values(copy=False),
            pd.concat([ts_uni.static_covariates] * 2, axis=0).values,
        )

        # valid static covariates for concatenation/stack: first only has static covs
        # -> this gives multivar ts with univar static covs
        ts_stacked2 = ts_uni.stack(ts_uni.with_static_covariates(None))
        np.testing.assert_almost_equal(
            ts_stacked2.static_covariate_values(copy=False),
            ts_uni.static_covariate_values(copy=False),
        )

        # mismatch between column names
        with pytest.raises(ValueError):
            _ = ts_uni.stack(ts_uni.with_static_covariates(static_covs_uni2))

        # mismatch between number of covariates
        with pytest.raises(ValueError):
            _ = ts_uni.stack(ts_uni.with_static_covariates(static_covs_uni3))

        # valid univar ts with univar static covariates + multivar ts with multivar static covariates
        ts_stacked3 = ts_uni.stack(ts_multi)
        np.testing.assert_almost_equal(
            ts_stacked3.static_covariate_values(copy=False),
            pd.concat(
                [ts_uni.static_covariates, ts_multi.static_covariates], axis=0
            ).values,
        )

        # invalid univar ts with univar static covariates + multivar ts with univar static covariates
        with pytest.raises(ValueError):
            _ = ts_uni.stack(ts_multi.with_static_covariates(static_covs_uni1))

    def test_concatenate_dim_component(self):
        """
        test concatenation with static covariates along component dimension (axis=1)
        Along component dimension, we concatenate/transfer the static covariates the series only if one of
        below cases applies:
        1)  concatenate when for each series the number of static cov components is equal to the number of
            components in the series. The static variable names (columns in series.static_covariates) must be
            identical across all series
        2)  if only the first series contains static covariates transfer only those
        3)  if `ignore_static_covarites=True`, case 1) is ignored and only the static covariates of the first
            series are transferred
        """
        ts_uni = linear_timeseries(length=10)
        ts_multi = ts_uni.stack(ts_uni)

        static_covs_uni1 = pd.DataFrame([[0, 1]], columns=["st1", "st2"]).astype(int)
        static_covs_uni2 = pd.DataFrame([[3, 4]], columns=["st3", "st4"]).astype(int)
        static_covs_uni3 = pd.DataFrame(
            [[2, 3, 4]], columns=["st1", "st2", "st3"]
        ).astype(int)

        static_covs_multi = pd.DataFrame(
            [[0, 0], [1, 1]], columns=["st1", "st2"]
        ).astype(int)

        ts_uni_static_uni1 = ts_uni.with_static_covariates(static_covs_uni1)
        ts_uni_static_uni2 = ts_uni.with_static_covariates(static_covs_uni2)
        ts_uni_static_uni3 = ts_uni.with_static_covariates(static_covs_uni3)

        ts_multi_static_uni1 = ts_multi.with_static_covariates(static_covs_uni1)
        ts_multi_static_multi = ts_multi.with_static_covariates(static_covs_multi)

        # concatenation without covariates
        ts_concat = concatenate([ts_uni, ts_uni], axis=1)
        assert ts_concat.static_covariates is None

        # concatenation along component dimension results in multi component static covariates
        ts_concat = concatenate([ts_uni_static_uni1, ts_uni_static_uni1], axis=1)
        assert ts_concat.static_covariates.shape == (2, 2)
        assert ts_concat.components.equals(ts_concat.static_covariates.index)
        np.testing.assert_almost_equal(
            ts_concat.static_covariate_values(copy=False),
            pd.concat([static_covs_uni1] * 2, axis=0).values,
        )

        # concatenation with inconsistent static variable names should fail ...
        with pytest.raises(ValueError):
            _ = concatenate([ts_uni_static_uni1, ts_uni_static_uni2], axis=1)

        # ... by ignoring the static covariates, it should work and take only the covariates of the first series
        ts_concat = concatenate(
            [ts_uni_static_uni1, ts_uni_static_uni2],
            axis=1,
            ignore_static_covariates=True,
        )
        assert ts_concat.static_covariates.shape == (1, 2)
        assert ts_concat.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        np.testing.assert_almost_equal(
            ts_concat.static_covariate_values(copy=False),
            ts_uni_static_uni1.static_covariate_values(copy=False),
        )

        # concatenation with inconsistent number of static covariates should fail ...
        with pytest.raises(ValueError):
            _ = concatenate([ts_uni_static_uni1, ts_uni_static_uni3], axis=1)

        # concatenation will only work if for each series the number of static cov components is equal to the
        # number of components in the series
        with pytest.raises(ValueError):
            _ = concatenate([ts_uni_static_uni1, ts_multi_static_uni1], axis=1)

        ts_concat = concatenate([ts_uni_static_uni1, ts_multi_static_multi], axis=1)
        assert ts_concat.static_covariates.shape == (ts_concat.n_components, 2)
        assert ts_concat.components.equals(ts_concat.static_covariates.index)
        np.testing.assert_almost_equal(
            ts_concat.static_covariate_values(copy=False),
            pd.concat([static_covs_uni1, static_covs_multi], axis=0),
        )

    def test_concatenate_dim_time(self):
        """
        Test concatenation with static covariates along time dimension (axis=0)
        Along time dimension, we only take the static covariates of the first series (as static covariates are
        time-independant).
        """
        static_covs_left = pd.DataFrame([[0, 1]], columns=["st1", "st2"]).astype(int)
        static_covs_right = pd.DataFrame([[3, 4]], columns=["st3", "st4"]).astype(int)

        ts_left = linear_timeseries(length=10).with_static_covariates(static_covs_left)
        ts_right = linear_timeseries(
            length=10, start=ts_left.end_time() + ts_left.freq
        ).with_static_covariates(static_covs_right)

        ts_concat = concatenate([ts_left, ts_right], axis=0)
        assert ts_concat.static_covariates.equals(ts_left.static_covariates)

    def test_concatenate_dim_samples(self):
        """
        Test concatenation with static covariates along sample dimension (axis=2)
        Along sample dimension, we only take the static covariates of the first series (as we components and
        time don't change).
        """
        static_covs_left = pd.DataFrame([[0, 1]], columns=["st1", "st2"]).astype(int)
        static_covs_right = pd.DataFrame([[3, 4]], columns=["st3", "st4"]).astype(int)

        ts_left = linear_timeseries(length=10).with_static_covariates(static_covs_left)
        ts_right = linear_timeseries(length=10).with_static_covariates(
            static_covs_right
        )

        ts_concat = concatenate([ts_left, ts_right], axis=2)
        assert ts_concat.static_covariates.equals(ts_left.static_covariates)

    def test_ts_methods_with_static_covariates(self):
        ts = linear_timeseries(length=10).astype("float64")
        static_covs = pd.Series([0, 1], index=["st1", "st2"]).astype(int)
        ts = ts.with_static_covariates(static_covs)

        assert ts.static_covariates.dtypes[0] == "float64"
        ts = ts.astype("float32")
        assert ts.static_covariates.dtypes[0] == "float32"

        ts_stochastic = ts.from_times_and_values(
            times=ts.time_index,
            values=np.random.randn(10, 1, 3),
            static_covariates=static_covs,
        )

        ts_check = ts.copy()
        assert ts_check.static_covariates.equals(ts.static_covariates)

        ts_check = ts.head()
        assert ts_check.static_covariates.equals(ts.static_covariates)

        ts_check = ts.tail()
        assert ts_check.static_covariates.equals(ts.static_covariates)

        # same values but different component names ("0" vs. "0_quantiles")
        ts_check = ts_stochastic.quantile_timeseries()
        assert not ts_check.components.equals(ts_stochastic.components)
        assert ts_stochastic.static_covariates.index.equals(ts_stochastic.components)
        np.testing.assert_almost_equal(
            ts_check.static_covariate_values(copy=False),
            ts_stochastic.static_covariate_values(copy=False),
        )

    def test_scalers_with_static_covariates(self):
        ts = linear_timeseries(start_value=1.0, end_value=2.0, length=10)
        static_covs = pd.Series([0.0, 2.0], index=["st1", "st2"])
        ts = ts.with_static_covariates(static_covs)

        for scaler_cls in [Scaler, BoxCox]:
            scaler = scaler_cls()
            ts_scaled = scaler.fit_transform(ts)
            assert ts_scaled.static_covariates.equals(ts.static_covariates)

            ts_inv = scaler.inverse_transform(ts_scaled)
            assert ts_inv.static_covariates.equals(ts.static_covariates)
