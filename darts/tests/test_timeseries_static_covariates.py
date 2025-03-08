import copy
import os

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import BoxCox, Scaler
from darts.timeseries import (
    DEFAULT_GLOBAL_STATIC_COV_NAME,
    METADATA_TAG,
    STATIC_COV_TAG,
)
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.utils import generate_index


def setup_test_case():
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
    return n_groups, len_ts, df_long_uni, df_long_multi


def setup_tag(tag, ts):
    if tag == METADATA_TAG:
        x = {"st1": 0.0, "st2": 1.0}
        ts = ts.with_metadata(x)
    else:
        x = pd.Series([0.0, 1.0], index=["st1", "st2"])
        ts = ts.with_static_covariates(x)
    return ts, x


class TestTimeSeriesStaticCovariate:
    n_groups, len_ts, df_long_uni, df_long_multi = setup_test_case()

    @pytest.mark.parametrize("tag", [STATIC_COV_TAG, METADATA_TAG])
    def test_ts_from_x(self, tag, tmpdir_module):
        ts = linear_timeseries(length=10)
        ts, x = setup_tag(tag, ts)
        kwargs = {tag: x}

        self.helper_test_transfer(tag, ts, TimeSeries.from_xarray(ts.data_array()))
        self.helper_test_transfer(
            tag, ts, TimeSeries.from_dataframe(ts.to_dataframe(), **kwargs)
        )
        self.helper_test_transfer(
            tag, ts, TimeSeries.from_series(ts.to_series(), **kwargs)
        )
        self.helper_test_transfer(
            tag,
            ts,
            TimeSeries.from_times_and_values(
                times=ts.time_index,
                values=ts.all_values(),
                columns=ts.components,
                **kwargs,
            ),
        )

        self.helper_test_transfer(
            tag,
            ts,
            TimeSeries.from_values(
                values=ts.all_values(),
                columns=ts.components,
                **kwargs,
            ),
        )

        f_csv = os.path.join(tmpdir_module, "temp_ts.csv")
        f_pkl = os.path.join(tmpdir_module, "temp_ts.pkl")
        ts.to_csv(f_csv)
        ts.to_pickle(f_pkl)
        ts_json = ts.to_json()

        self.helper_test_transfer(
            tag, ts, TimeSeries.from_csv(f_csv, time_col="time", **kwargs)
        )
        self.helper_test_transfer(tag, ts, TimeSeries.from_pickle(f_pkl))
        self.helper_test_transfer(tag, ts, TimeSeries.from_json(ts_json, **kwargs))

    def test_invalid_metadata(self):
        ts = linear_timeseries(length=10)
        with pytest.raises(ValueError) as exc:
            _ = ts.with_metadata(0.0)
        assert (
            str(exc.value)
            == "`metadata` must be of type `dict` mapping metadata attributes to their values."
        )

    @pytest.mark.parametrize("index_type", ["int", "dt", "str"])
    def test_from_group_dataframe(self, index_type):
        """Tests correct extract of TimeSeries groups from a long DataFrame with unsorted (time/integer) index"""
        group = ["a", "a", "a", "b", "b", "b"]
        values = np.arange(len(group))

        if index_type == "int":
            index_expected = pd.RangeIndex(3)
            time = [2, 1, 0, 0, 1, 2]
        else:
            index_expected = pd.date_range("2024-01-01", periods=3)
            time = index_expected[::-1].append(index_expected)
            if index_type == "str":
                time = time.astype(str)

        # create a df with unsorted time
        df = pd.DataFrame({
            "group": group,
            "time": time,
            "x": values,
        })
        ts = TimeSeries.from_group_dataframe(df, group_cols="group", time_col="time")

        # check the time index
        assert ts[0].time_index.equals(index_expected)
        assert ts[1].time_index.equals(index_expected)

        # check the values
        assert (ts[0].values().flatten() == [values[2], values[1], values[0]]).all()
        assert (ts[1].values().flatten() == [values[3], values[4], values[5]]).all()

    def test_timeseries_from_longitudinal_df(self):
        # univariate static covs: only group by "st1", keep static covs "st1"
        value_cols = ["a", "b", "c"]
        ts_groups1 = TimeSeries.from_group_dataframe(
            df=self.df_long_uni,
            group_cols="st1",
            static_cols=None,
            time_col="times",
            value_cols=value_cols,
            metadata_cols=["st1", "constant"],
        )
        assert len(ts_groups1) == self.n_groups
        for i, ts in enumerate(ts_groups1):
            assert ts.static_covariates.index.equals(
                pd.Index([DEFAULT_GLOBAL_STATIC_COV_NAME])
            )
            assert ts.static_covariates.shape == (1, 1)
            assert ts.static_covariates.columns.equals(pd.Index(["st1"]))
            assert (ts.static_covariates_values(copy=False) == [[i]]).all()
            assert ts.metadata == {"st1": i, "constant": 1}

        # multivariate static covs: only group by "st1", keep static covs "st1", "constant"
        ts_groups2 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1"],
            static_cols="constant",
            time_col="times",
            value_cols=value_cols,
            metadata_cols=["st1", "constant"],
        )
        assert len(ts_groups2) == self.n_groups
        for i, ts in enumerate(ts_groups2):
            assert ts.static_covariates.shape == (1, 2)
            assert ts.static_covariates.columns.equals(pd.Index(["st1", "constant"]))
            assert (ts.static_covariates_values(copy=False) == [[i, 1]]).all()
            assert ts.metadata == {"st1": i, "constant": 1}

        # multivariate static covs: group by "st1" and "st2", keep static covs "st1", "st2", "constant"
        ts_groups3 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            static_cols=["constant"],
            time_col="times",
            value_cols=value_cols,
            metadata_cols=["st1", "st2", "constant"],
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
            assert ts.metadata == {"st1": i, "st2": j, "constant": 1}

        # drop group columns gives same time series with dropped static covariates
        # drop first column
        ts_groups4 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            static_cols=["constant"],
            time_col="times",
            value_cols=value_cols,
            drop_group_cols=["st1"],
            metadata_cols=["st1", "st2", "constant"],
        )
        assert len(ts_groups4) == self.n_groups * 2
        for idx, ts in enumerate(ts_groups4):
            i = idx // 2
            j = idx % 2
            assert ts.static_covariates.shape == (1, 2)
            assert ts.static_covariates.columns.equals(pd.Index(["st2", "constant"]))
            assert (ts.static_covariates_values(copy=False) == [[j, 1]]).all()
            assert ts.metadata == {"st1": i, "st2": j, "constant": 1}

        # drop last column
        ts_groups5 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            static_cols=["constant"],
            time_col="times",
            value_cols=value_cols,
            drop_group_cols=["st2"],
            metadata_cols=["st1", "st2", "constant"],
        )
        assert len(ts_groups5) == self.n_groups * 2
        for idx, ts in enumerate(ts_groups5):
            i = idx // 2
            j = idx % 2
            assert ts.static_covariates.shape == (1, 2)
            assert ts.static_covariates.columns.equals(pd.Index(["st1", "constant"]))
            assert (ts.static_covariates_values(copy=False) == [[i, 1]]).all()
            assert ts.metadata == {"st1": i, "st2": j, "constant": 1}

        # drop all columns
        ts_groups6 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            static_cols=["constant"],
            time_col="times",
            value_cols=value_cols,
            drop_group_cols=["st1", "st2"],
            metadata_cols="constant",
        )
        assert len(ts_groups6) == self.n_groups * 2
        for ts in ts_groups6:
            assert ts.static_covariates.shape == (1, 1)
            assert ts.static_covariates.columns.equals(pd.Index(["constant"]))
            assert (ts.static_covariates_values(copy=False) == [[1]]).all()
            assert ts.metadata == {"constant": 1}

        # drop all static covariates (no `static_cols`, all `group_cols` dropped) and no metadata cols
        ts_groups7 = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            time_col="times",
            value_cols=value_cols,
            drop_group_cols=["st1", "st2"],
        )
        assert len(ts_groups7) == self.n_groups * 2
        for ts in ts_groups7:
            assert ts.static_covariates is None
            assert ts.metadata is None

        ts_groups7_parallel = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1", "st2"],
            time_col="times",
            value_cols=value_cols,
            drop_group_cols=["st1", "st2"],
            n_jobs=-1,
        )
        assert ts_groups7_parallel == ts_groups7

    def test_from_group_dataframe_invalid_drop_cols(self):
        # drop col is not part of `group_cols`
        with pytest.raises(ValueError) as err:
            _ = TimeSeries.from_group_dataframe(
                df=self.df_long_multi,
                group_cols=["st1"],
                time_col="times",
                value_cols="a",
                drop_group_cols=["invalid"],
            )
        assert str(err.value).endswith("received: {'invalid'}.")

    def test_from_group_dataframe_groups_too_short(self):
        # groups that are too short for TimeSeries requirements should raise an error
        df = copy.deepcopy(self.df_long_multi)
        df.loc[:, "non_static"] = np.arange(len(df))
        with pytest.raises(ValueError) as err:
            _ = TimeSeries.from_group_dataframe(
                df=df,
                group_cols="non_static",
                time_col="times",
                value_cols="a",
            )
        assert str(err.value).startswith(
            "The time index of the provided DataArray is missing the freq attribute"
        )

    def test_from_group_dataframe_not_unique(self):
        # it is assumed that all static / metadata columns have only 1 unique value.
        # it will always pick the first encountered value
        series = TimeSeries.from_group_dataframe(
            df=self.df_long_multi,
            group_cols=["st1"],
            drop_group_cols="st1",
            static_cols="st2",
            time_col="times",
            value_cols="a",
            metadata_cols="st2",
        )
        first_values = [
            df["st2"].values[0] for idx, df in self.df_long_multi.groupby("st1")
        ]
        for s_, val in zip(series, first_values):
            assert s_.static_covariates_values()[0, 0] == val
            assert s_.metadata == {"st2": val}

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
        assert ts.static_covariates is None
        assert ts.static_covariates_values() is None

    def test_metadata_values(self):
        ts = linear_timeseries(length=10)
        assert not ts.has_metadata

        metadata = {"st1": 0, "st2": 1}
        ts = ts.with_metadata(metadata)

        assert ts.has_metadata
        assert ts.metadata == metadata
        # changing values of metadata is allowed
        ts.metadata["st1"] = 2
        assert ts.metadata == {"st1": 2, "st2": 1}

        # removing metadata
        ts = ts.with_metadata(None)
        assert ts.metadata is None

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

    def test_stack_metadata(self):
        metadata = {"a": 0, "b": 1}
        ts = linear_timeseries(length=10)
        ts_md = ts.with_metadata(metadata)

        # stacking will always use the metadata of `self`
        assert ts.stack(ts).metadata is None
        assert ts.stack(ts_md).metadata is None
        assert ts_md.stack(ts).metadata == metadata
        assert ts_md.stack(ts_md).metadata == metadata

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

    def test_concatenate_metadata(self):
        metadata = {"a": 0, "b": 1}
        ts = linear_timeseries(length=10)
        ts_md = ts.with_metadata(metadata)

        # concatenate will always use the metadata of `self`
        for axis in [1, 2]:
            assert ts.concatenate(ts, axis=axis).metadata is None
            assert ts.concatenate(ts_md, axis=axis).metadata is None
            assert ts_md.concatenate(ts, axis=axis).metadata == metadata
            assert ts_md.concatenate(ts_md, axis=axis).metadata == metadata

        # over axis=0 requires shifting
        assert ts.concatenate(ts_md.shift(10), axis=0).metadata is None
        assert ts_md.concatenate(ts.shift(10), axis=0).metadata == metadata

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
        assert isinstance(ts.static_covariates.dtypes["cat"], object)

        ts = ts.astype(np.float32)
        assert ts.static_covariates.dtypes["num"] == ts.dtype == "float32"
        assert isinstance(ts.static_covariates.dtypes["cat"], object)

    def test_non_numerical_metadata(self):
        metadata = {"a": 0, "b": "foo"}
        ts = linear_timeseries(length=10)
        assert ts.with_metadata(metadata).metadata == metadata

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

    @pytest.mark.parametrize(
        "get_item",
        [0, slice(0, 2), "comp1", slice("comp1", "comp2"), ["comp1", "comp2"]],
    )
    def test_get_item_metadata(self, get_item):
        metadata = {"a": 0, "b": 1}
        ts = TimeSeries.from_values(
            values=np.random.random((10, 2)), columns=["comp1", "comp2"]
        ).with_metadata(metadata)
        assert ts[get_item].metadata == metadata

    @pytest.mark.parametrize("tag", [STATIC_COV_TAG, METADATA_TAG])
    def test_operations(self, tag):
        ts = TimeSeries.from_values(values=np.random.random((10, 2)))
        ts, x = setup_tag(tag, ts)

        # arithmetics with series (left) and non-series (right)
        self.helper_test_transfer(tag, ts, ts / 3)
        self.helper_test_transfer(tag, ts, ts * 3)
        self.helper_test_transfer(tag, ts, ts**3)
        self.helper_test_transfer(tag, ts, ts + 3)
        self.helper_test_transfer(tag, ts, ts - 3)

        # conditions
        self.helper_test_transfer_xa(tag, ts, ts < 3)
        self.helper_test_transfer_xa(tag, ts, ts >= 3)
        self.helper_test_transfer_xa(tag, ts, ts > 3)
        self.helper_test_transfer_xa(tag, ts, ts >= 3)

        # arithmetics with non-series (left) and series (right)
        self.helper_test_transfer(tag, ts, 3 * ts)
        self.helper_test_transfer(tag, ts, 3 + ts)
        self.helper_test_transfer(tag, ts, 3 - ts)
        # conditions
        self.helper_test_transfer_xa(tag, ts, 3 > ts)
        self.helper_test_transfer_xa(tag, ts, 3 >= ts)
        self.helper_test_transfer_xa(tag, ts, 3 < ts)
        self.helper_test_transfer_xa(tag, ts, 3 <= ts)

        # arithmetics with two series
        self.helper_test_transfer(tag, ts, ts / ts)
        self.helper_test_transfer(tag, ts, ts * ts)
        self.helper_test_transfer(tag, ts, ts**ts)
        self.helper_test_transfer(tag, ts, ts + ts)
        self.helper_test_transfer(tag, ts, ts - ts)
        # conditions
        self.helper_test_transfer_xa(tag, ts, ts > ts)
        self.helper_test_transfer_xa(tag, ts, ts >= ts)
        self.helper_test_transfer_xa(tag, ts, ts < ts)
        self.helper_test_transfer_xa(tag, ts, ts <= ts)

        # other operations
        self.helper_test_transfer(tag, ts, abs(ts))
        self.helper_test_transfer(tag, ts, -ts)
        self.helper_test_transfer(tag, ts, round(ts, 2))

    @pytest.mark.parametrize("tag", [STATIC_COV_TAG, METADATA_TAG])
    def test_ts_methods(self, tag):
        ts = linear_timeseries(length=10, start_value=1.0, end_value=2.0).astype(
            "float64"
        )
        ts, x = setup_tag(tag, ts)
        kwargs = {tag: x}

        ts_stoch = ts.from_times_and_values(
            times=ts.time_index,
            values=np.random.randint(low=0, high=10, size=(10, 1, 3)),
            **kwargs,
        )

        if tag == STATIC_COV_TAG:
            assert ts.static_covariates.dtypes.iloc[0] == "float64"
            ts = ts.astype("float32")
            assert ts.static_covariates.dtypes.iloc[0] == "float32"
            assert ts_stoch.static_covariates.index.equals(ts_stoch.components)

        self.helper_test_transfer(tag, ts, ts.with_values(ts.all_values()))
        self.helper_test_transfer(
            tag,
            ts,
            ts.with_columns_renamed(ts.components.tolist(), ts.components.tolist()),
        )
        self.helper_test_transfer(tag, ts, ts.copy())
        self.helper_test_transfer(tag, ts, ts.mean())
        self.helper_test_transfer(tag, ts, ts.median())
        self.helper_test_transfer(tag, ts, ts.sum())
        self.helper_test_transfer(tag, ts, ts.min())
        self.helper_test_transfer(tag, ts, ts.max())
        self.helper_test_transfer(tag, ts, ts.head())
        self.helper_test_transfer(tag, ts, ts.tail())
        self.helper_test_transfer(tag, ts, ts.split_after(0.5)[0])
        self.helper_test_transfer(tag, ts, ts.split_after(0.5)[1])
        self.helper_test_transfer(tag, ts, ts.split_before(0.5)[0])
        self.helper_test_transfer(tag, ts, ts.split_before(0.5)[1])
        self.helper_test_transfer(tag, ts, ts.drop_before(0.5))
        self.helper_test_transfer(tag, ts, ts.drop_after(0.5))
        self.helper_test_transfer(
            tag, ts, ts.slice(ts.start_time() + ts.freq, ts.end_time() - ts.freq)
        )
        self.helper_test_transfer(tag, ts, ts.slice_n_points_after(ts.start_time(), 5))
        self.helper_test_transfer(tag, ts, ts.slice_n_points_before(ts.end_time(), 5))
        self.helper_test_transfer(tag, ts, ts.slice_intersect(ts[2:]))
        self.helper_test_transfer(tag, ts, ts.strip())
        self.helper_test_transfer(tag, ts, ts.longest_contiguous_slice())
        self.helper_test_transfer(tag, ts, ts.rescale_with_value(2.0))
        self.helper_test_transfer(tag, ts, ts.shift(2.0))
        self.helper_test_transfer(tag, ts, ts.diff())
        self.helper_test_transfer(tag, ts, ts.univariate_component(0))
        self.helper_test_transfer(tag, ts, ts.map(lambda x: x + 1))
        self.helper_test_transfer(tag, ts, ts.resample(ts.freq))
        self.helper_test_transfer(tag, ts, ts[:5].append(ts[5:]))
        self.helper_test_transfer(tag, ts, ts.append_values(ts.all_values()))

        self.helper_test_transfer(tag, ts_stoch, ts_stoch.var())
        self.helper_test_transfer(tag, ts_stoch, ts_stoch.std())
        self.helper_test_transfer(tag, ts_stoch, ts_stoch.skew())
        self.helper_test_transfer(tag, ts_stoch, ts_stoch.kurtosis())

        # will append "_quantile" to component names
        self.helper_test_transfer_values(tag, ts_stoch, ts_stoch.quantile_timeseries())
        self.helper_test_transfer_values(tag, ts_stoch, ts_stoch.quantile(0.5))
        # will change component names
        self.helper_test_transfer_values(tag, ts, ts.add_datetime_attribute("hour"))
        self.helper_test_transfer_values(tag, ts, ts.add_holidays("US"))

    @staticmethod
    def helper_test_transfer(tag, ts, ts_new):
        """static cov or metadata must be identical"""
        if tag == STATIC_COV_TAG:
            assert ts_new.static_covariates.equals(ts.static_covariates)
        else:  # metadata
            assert ts_new.metadata == ts.metadata

    @staticmethod
    def helper_test_transfer_xa(tag, ts, xa_new):
        """static cov or metadata must be identical between xarray and TimeSeries"""
        if tag == STATIC_COV_TAG:
            assert xa_new.attrs[STATIC_COV_TAG].equals(ts.static_covariates)
        else:  # metadata
            assert xa_new.attrs[METADATA_TAG] == ts.metadata

    @staticmethod
    def helper_test_transfer_values(tag, ts, ts_new):
        """values of static cov or metadata must match but not row index (component names).
        I.e. series.quantile_timeseries() adds "_quantiles" to component names
        """
        if tag == STATIC_COV_TAG:
            assert not ts_new.static_covariates.index.equals(ts.components)
            np.testing.assert_almost_equal(
                ts_new.static_covariates_values(copy=False),
                ts.static_covariates_values(copy=False),
            )
        else:  # metadata
            assert ts_new.metadata == ts.metadata
