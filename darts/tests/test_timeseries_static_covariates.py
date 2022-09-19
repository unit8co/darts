import copy
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.tests.base_test_class import DartsBaseTestClass
from darts.timeseries import DEFAULT_GLOBAL_STATIC_COV_NAME, STATIC_COV_TAG
from darts.utils.timeseries_generation import generate_index, linear_timeseries


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
                        generate_index(start=pd.Timestamp(2010, 1, 1), length=len_ts)
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

    def setUp(self):
        self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.temp_work_dir)

    def test_ts_from_x(self):
        ts = linear_timeseries(length=10).with_static_covariates(
            pd.Series([0.0, 1.0], index=["st1", "st2"])
        )

        self.helper_test_cov_transfer(ts, TimeSeries.from_xarray(ts.data_array()))
        self.helper_test_cov_transfer(
            ts,
            TimeSeries.from_dataframe(
                ts.pd_dataframe(), static_covariates=ts.static_covariates
            ),
        )
        # ts.pd_series() loses component names -> static covariates have different components names
        self.helper_test_cov_transfer_values(
            ts,
            TimeSeries.from_series(
                ts.pd_series(), static_covariates=ts.static_covariates
            ),
        )
        self.helper_test_cov_transfer(
            ts,
            TimeSeries.from_times_and_values(
                times=ts.time_index,
                values=ts.all_values(),
                columns=ts.components,
                static_covariates=ts.static_covariates,
            ),
        )

        self.helper_test_cov_transfer(
            ts,
            TimeSeries.from_values(
                values=ts.all_values(),
                columns=ts.components,
                static_covariates=ts.static_covariates,
            ),
        )

        f_csv = os.path.join(self.temp_work_dir, "temp_ts.csv")
        f_pkl = os.path.join(self.temp_work_dir, "temp_ts.pkl")
        ts.to_csv(f_csv)
        ts.to_pickle(f_pkl)
        ts_json = ts.to_json()

        self.helper_test_cov_transfer(
            ts,
            TimeSeries.from_csv(
                f_csv, time_col="time", static_covariates=ts.static_covariates
            ),
        )
        self.helper_test_cov_transfer(ts, TimeSeries.from_pickle(f_pkl))
        self.helper_test_cov_transfer(
            ts, TimeSeries.from_json(ts_json, static_covariates=ts.static_covariates)
        )

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
            assert (ts.static_covariates_values(copy=False) == [[i]]).all()

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
            assert (ts.static_covariates_values(copy=False) == [[i, 1]]).all()

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
            assert (ts.static_covariates_values(copy=False) == [[i, j, 1]]).all()

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
            ts.static_covariates_values(copy=False),
            np.expand_dims(static_covs_series.values, -1).T,
        )
        assert ts.static_covariates.index.equals(ts.components)

        # from DataFrame
        ts = ts.with_static_covariates(static_covs_df)
        assert ts.has_static_covariates
        np.testing.assert_almost_equal(
            ts.static_covariates_values(copy=False), static_covs_df.values
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

    def test_static_covariates_values(self):
        ts = linear_timeseries(length=10)
        static_covs = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])
        ts = ts.with_static_covariates(static_covs)

        # changing values of copy should not change original DataFrame
        vals = ts.static_covariates_values(copy=True)
        vals[:] = -1.0
        assert (ts.static_covariates_values(copy=False) != -1.0).all()

        # changing values of view should change original DataFrame
        vals = ts.static_covariates_values(copy=False)
        vals[:] = -1.0
        assert (ts.static_covariates_values(copy=False) == -1.0).all()

        ts = ts.with_static_covariates(None)
        assert ts.static_covariates_values() is None

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
            ts_multi.static_covariates_values(copy=False), static_covs.loc[0:0].values
        )

        # from multivariate static covariates
        ts_multi = ts_multi.with_static_covariates(static_covs)
        assert ts_multi.static_covariates.index.equals(ts_multi.components)
        assert ts_multi.static_covariates.columns.equals(static_covs.columns)
        np.testing.assert_almost_equal(
            ts_multi.static_covariates_values(copy=False), static_covs.values
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
            ts_stacked1.static_covariates_values(copy=False),
            pd.concat([ts_uni.static_covariates] * 2, axis=0).values,
        )

        # valid static covariates for concatenation/stack: first only has static covs
        # -> this gives multivar ts with univar static covs
        ts_stacked2 = ts_uni.stack(ts_uni.with_static_covariates(None))
        np.testing.assert_almost_equal(
            ts_stacked2.static_covariates_values(copy=False),
            ts_uni.static_covariates_values(copy=False),
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
            ts_stacked3.static_covariates_values(copy=False),
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
        Along component dimension, we concatenate/transfer the static covariates of the series only if one of
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
            ts_concat.static_covariates_values(copy=False),
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
            ts_concat.static_covariates_values(copy=False),
            ts_uni_static_uni1.static_covariates_values(copy=False),
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
            ts_concat.static_covariates_values(copy=False),
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

    def test_non_numerical_static_covariates(self):
        static_covs = pd.DataFrame([["a", 0], ["b", 1]], columns=["cat", "num"])
        assert static_covs.dtypes["num"] == "int64"

        ts = TimeSeries.from_values(
            values=np.random.random((10, 2))
        ).with_static_covariates(static_covs)
        assert ts.static_covariates.dtypes["num"] == ts.dtype == "float64"
        assert ts.static_covariates.dtypes["cat"] == object

        ts = ts.astype(np.float32)
        assert ts.static_covariates.dtypes["num"] == ts.dtype == "float32"
        assert ts.static_covariates.dtypes["cat"] == object

    def test_get_item(self):
        # multi component static covariates
        static_covs = pd.DataFrame([["a", 0], ["b", 1]], columns=["cat", "num"])
        ts = TimeSeries.from_values(
            values=np.random.random((10, 2)), columns=["comp1", "comp2"]
        ).with_static_covariates(static_covs)

        assert ts.static_covariates.index.equals(ts.components)

        ts0 = ts[0]
        assert ts0.static_covariates.index.equals(ts.components)
        assert isinstance(ts0.static_covariates, pd.DataFrame)
        ts1 = ts["comp1"]
        assert ts1.static_covariates.index.equals(pd.Index(["comp1"]))
        assert isinstance(ts1.static_covariates, pd.DataFrame)
        ts2 = ts["comp2"]
        assert ts2.static_covariates.index.equals(pd.Index(["comp2"]))
        assert isinstance(ts2.static_covariates, pd.DataFrame)
        ts3 = ts["comp1":"comp2"]
        assert ts3.static_covariates.index.equals(pd.Index(["comp1", "comp2"]))
        assert isinstance(ts3.static_covariates, pd.DataFrame)
        ts4 = ts[["comp1", "comp2"]]
        assert ts4.static_covariates.index.equals(pd.Index(["comp1", "comp2"]))
        assert isinstance(ts4.static_covariates, pd.DataFrame)

        # uni/global component static covariates
        static_covs = pd.DataFrame([["a", 0]], columns=["cat", "num"])
        ts = TimeSeries.from_values(
            values=np.random.random((10, 3)), columns=["comp1", "comp2", "comp3"]
        ).with_static_covariates(static_covs)

        # 1) when static covs have 1 component but series is multivariate -> static covariate component name is set to
        # "global_components"
        assert ts.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        ts0 = ts[0]
        assert ts0.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        assert isinstance(ts0.static_covariates, pd.DataFrame)
        ts1 = ts["comp1":"comp3"]
        assert ts1.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        assert isinstance(ts1.static_covariates, pd.DataFrame)
        ts2 = ts[["comp1", "comp2", "comp3"]]
        assert ts2.static_covariates.index.equals(
            pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
        )
        assert isinstance(ts2.static_covariates, pd.DataFrame)

        # 2) if number of static cov components match the number of components in the series -> static covariate
        # component names are set to be equal to series component names
        ts3 = ts["comp1"]
        assert ts3.static_covariates.index.equals(pd.Index(["comp1"]))
        assert isinstance(ts3.static_covariates, pd.DataFrame)
        ts4 = ts["comp2"]
        assert ts4.static_covariates.index.equals(pd.Index(["comp2"]))
        assert isinstance(ts4.static_covariates, pd.DataFrame)

    def test_operations(self):
        static_covs = pd.DataFrame([[0, 1]], columns=["st1", "st2"])
        ts = TimeSeries.from_values(
            values=np.random.random((10, 2))
        ).with_static_covariates(static_covs)

        # arithmetics with series (left) and non-series (right)
        self.helper_test_cov_transfer(ts, ts / 3)
        self.helper_test_cov_transfer(ts, ts * 3)
        self.helper_test_cov_transfer(ts, ts**3)
        self.helper_test_cov_transfer(ts, ts + 3)
        self.helper_test_cov_transfer(ts, ts - 3)

        # conditions
        self.helper_test_cov_transfer_xa(ts, ts < 3)
        self.helper_test_cov_transfer_xa(ts, ts >= 3)
        self.helper_test_cov_transfer_xa(ts, ts > 3)
        self.helper_test_cov_transfer_xa(ts, ts >= 3)

        # arithmetics with non-series (left) and series (right)
        self.helper_test_cov_transfer(ts, 3 * ts)
        self.helper_test_cov_transfer(ts, 3 + ts)
        self.helper_test_cov_transfer(ts, 3 - ts)
        # conditions
        self.helper_test_cov_transfer_xa(ts, 3 > ts)
        self.helper_test_cov_transfer_xa(ts, 3 >= ts)
        self.helper_test_cov_transfer_xa(ts, 3 < ts)
        self.helper_test_cov_transfer_xa(ts, 3 <= ts)

        # arithmetics with two series
        self.helper_test_cov_transfer(ts, ts / ts)
        self.helper_test_cov_transfer(ts, ts * ts)
        self.helper_test_cov_transfer(ts, ts**ts)
        self.helper_test_cov_transfer(ts, ts + ts)
        self.helper_test_cov_transfer(ts, ts - ts)
        # conditions
        self.helper_test_cov_transfer_xa(ts, ts > ts)
        self.helper_test_cov_transfer_xa(ts, ts >= ts)
        self.helper_test_cov_transfer_xa(ts, ts < ts)
        self.helper_test_cov_transfer_xa(ts, ts <= ts)

        # other operations
        self.helper_test_cov_transfer(ts, abs(ts))
        self.helper_test_cov_transfer(ts, -ts)
        self.helper_test_cov_transfer(ts, round(ts, 2))

    def test_ts_methods_with_static_covariates(self):
        ts = linear_timeseries(length=10, start_value=1.0, end_value=2.0).astype(
            "float64"
        )
        static_covs = pd.Series([0, 1], index=["st1", "st2"]).astype(int)
        ts = ts.with_static_covariates(static_covs)

        assert ts.static_covariates.dtypes[0] == "float64"
        ts = ts.astype("float32")
        assert ts.static_covariates.dtypes[0] == "float32"

        ts_stoch = ts.from_times_and_values(
            times=ts.time_index,
            values=np.ones((10, 1, 3)),
            static_covariates=static_covs,
        )
        assert ts_stoch.static_covariates.index.equals(ts_stoch.components)

        self.helper_test_cov_transfer(ts, ts.with_values(ts.all_values()))
        self.helper_test_cov_transfer(
            ts, ts.with_columns_renamed(ts.components.tolist(), ts.components.tolist())
        )
        self.helper_test_cov_transfer(ts, ts.copy())
        self.helper_test_cov_transfer(ts, ts.mean())
        self.helper_test_cov_transfer(ts, ts.median())
        self.helper_test_cov_transfer(ts, ts.sum())
        self.helper_test_cov_transfer(ts, ts.min())
        self.helper_test_cov_transfer(ts, ts.max())
        self.helper_test_cov_transfer(ts, ts.head())
        self.helper_test_cov_transfer(ts, ts.tail())
        self.helper_test_cov_transfer(ts, ts.split_after(0.5)[0])
        self.helper_test_cov_transfer(ts, ts.split_after(0.5)[1])
        self.helper_test_cov_transfer(ts, ts.split_before(0.5)[0])
        self.helper_test_cov_transfer(ts, ts.split_before(0.5)[1])
        self.helper_test_cov_transfer(ts, ts.drop_before(0.5))
        self.helper_test_cov_transfer(ts, ts.drop_after(0.5))
        self.helper_test_cov_transfer(
            ts, ts.slice(ts.start_time() + ts.freq, ts.end_time() - ts.freq)
        )
        self.helper_test_cov_transfer(ts, ts.slice_n_points_after(ts.start_time(), 5))
        self.helper_test_cov_transfer(ts, ts.slice_n_points_before(ts.end_time(), 5))
        self.helper_test_cov_transfer(ts, ts.slice_intersect(ts[2:]))
        self.helper_test_cov_transfer(ts, ts.strip())
        self.helper_test_cov_transfer(ts, ts.longest_contiguous_slice())
        self.helper_test_cov_transfer(ts, ts.rescale_with_value(2.0))
        self.helper_test_cov_transfer(ts, ts.shift(2.0))
        self.helper_test_cov_transfer(ts, ts.diff())
        self.helper_test_cov_transfer(ts, ts.univariate_component(0))
        self.helper_test_cov_transfer(ts, ts.map(lambda x: x + 1))
        self.helper_test_cov_transfer(ts, ts.resample(ts.freq))
        self.helper_test_cov_transfer(ts, ts[:5].append(ts[5:]))
        self.helper_test_cov_transfer(ts, ts.append_values(ts.all_values()))

        self.helper_test_cov_transfer(ts_stoch, ts_stoch.var())
        self.helper_test_cov_transfer(ts_stoch, ts_stoch.std())
        self.helper_test_cov_transfer(ts_stoch, ts_stoch.skew())
        self.helper_test_cov_transfer(ts_stoch, ts_stoch.kurtosis())

        # will append "_quantile" to component names
        self.helper_test_cov_transfer_values(ts_stoch, ts_stoch.quantile_timeseries())
        self.helper_test_cov_transfer_values(ts_stoch, ts_stoch.quantile(0.5))
        # will change component names
        self.helper_test_cov_transfer_values(ts, ts.add_datetime_attribute("hour"))
        self.helper_test_cov_transfer_values(ts, ts.add_holidays("US"))

    def helper_test_cov_transfer(self, ts, ts_new):
        """static cov dataframes must be identical"""
        assert ts_new.static_covariates.equals(ts.static_covariates)

    def helper_test_cov_transfer_xa(self, ts, xa_new):
        """static cov dataframes must be identical between xarray and TimeSeries"""
        assert xa_new.attrs[STATIC_COV_TAG].equals(ts.static_covariates)

    def helper_test_cov_transfer_values(self, ts, ts_new):
        """values of static cov dataframes must match but not row index (component names).
        I.e. series.quantile_timeseries() adds "_quantiles" to component names
        """
        assert not ts_new.static_covariates.index.equals(ts.components)
        np.testing.assert_almost_equal(
            ts_new.static_covariates_values(copy=False),
            ts.static_covariates_values(copy=False),
        )
