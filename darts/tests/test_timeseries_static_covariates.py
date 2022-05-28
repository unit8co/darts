import copy

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import _generate_index, linear_timeseries


class TimeSeriesMultivariateTestCase(DartsBaseTestClass):
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
        ts_groups1 = TimeSeries.from_longitudinal_dataframe(
            df=self.df_long_uni,
            group_cols="st1",
            static_cols=None,
            time_col="times",
            value_cols=value_cols,
        )
        assert len(ts_groups1) == self.n_groups
        for i, ts in enumerate(ts_groups1):
            assert ts.static_covariates.shape == (1, 1)
            assert ts.static_covariates.index.equals(pd.Index(["st1"]))
            assert (ts.static_covariates.values == [[i]]).all()

        # multivariate static covs: only group by "st1", keep static covs "st1", "constant"
        ts_groups2 = TimeSeries.from_longitudinal_dataframe(
            df=self.df_long_multi,
            group_cols=["st1"],
            static_cols="constant",
            time_col="times",
            value_cols=value_cols,
        )
        assert len(ts_groups2) == self.n_groups
        for i, ts in enumerate(ts_groups2):
            assert ts.static_covariates.shape == (2, 1)
            assert ts.static_covariates.index.equals(pd.Index(["st1", "constant"]))
            assert (ts.static_covariates.values == [[i], [1]]).all()

        # multivariate static covs: group by "st1" and "st2", keep static covs "st1", "st2", "constant"
        ts_groups3 = TimeSeries.from_longitudinal_dataframe(
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
            assert ts.static_covariates.shape == (3, 1)
            assert ts.static_covariates.index.equals(
                pd.Index(["st1", "st2", "constant"])
            )
            assert (ts.static_covariates.values == [[i], [j], [1]]).all()

        df = copy.deepcopy(self.df_long_multi)
        df.loc[:, "non_static"] = np.arange(len(df))
        # non static columns as static columns should raise an error
        with pytest.raises(ValueError):
            _ = TimeSeries.from_longitudinal_dataframe(
                df=df,
                group_cols=["st1"],
                static_cols=["non_static"],
                time_col="times",
                value_cols=value_cols,
            )

        # groups that are too short for TimeSeries requirements should raise an error
        with pytest.raises(ValueError):
            _ = TimeSeries.from_longitudinal_dataframe(
                df=df,
                group_cols=["st1", "non_static"],
                static_cols=None,
                time_col="times",
                value_cols=value_cols,
            )

    def test_set_static_covariates_univariate(self):
        ts = linear_timeseries(length=10)
        static_covs = pd.Series([0.0, 1.0], index=["st1", "st2"])

        # inplace from Series for chained calls
        ts.set_static_covariates(static_covs)
        assert ts.static_covariates.equals(static_covs.to_frame())

        # from Series
        ts = ts.set_static_covariates(static_covs)
        assert ts.static_covariates.equals(static_covs.to_frame())

        # from DataFrame
        ts = ts.set_static_covariates(static_covs.to_frame())
        assert ts.static_covariates.equals(static_covs.to_frame())

        # with None
        ts = ts.set_static_covariates(None)
        assert ts.static_covariates is None

        # only pd.Series, pd.DataFrame or None
        with pytest.raises(ValueError):
            ts.set_static_covariates([1, 2, 3])

        # multivariate does not work with univariate TimeSeries
        with pytest.raises(ValueError):
            static_covs_multi = pd.concat([static_covs] * 2, axis=1)
            ts.set_static_covariates(static_covs_multi)

    def test_set_static_covariates_multivariate(self):
        ts = linear_timeseries(length=10)
        ts_multi = ts.stack(ts)
        static_covs = pd.DataFrame([[0.0, 1.0], [0.0, 1.0]], index=["st1", "st2"])

        # from univariate static covariates
        ts_multi = ts_multi.set_static_covariates(static_covs[static_covs.columns[0]])
        assert ts_multi.static_covariates.equals(
            static_covs[static_covs.columns[0]].to_frame()
        )

        # from multivariate static covariates
        ts_multi = ts_multi.set_static_covariates(static_covs)
        assert ts_multi.static_covariates.equals(static_covs)

        # raise an error if multivariate static covariates columns don't match the number of components in the series
        with pytest.raises(ValueError):
            _ = ts_multi.set_static_covariates(pd.concat([static_covs] * 2, axis=1))
