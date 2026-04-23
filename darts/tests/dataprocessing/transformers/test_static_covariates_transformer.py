import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils import timeseries_generation as tg


class TestStaticCovariatesTransformer:
    series = tg.linear_timeseries(length=10)
    static_covs1 = pd.DataFrame(
        data={
            "cont1": [0, 1, 2],
            "cat1": [1, 2, 3],
            "cont2": [0.1, 0.2, 0.3],
            "cat2": ["a", "b", "c"],
        }
    ).astype(dtype={"cat1": "O"})
    series1 = TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.concatenate([series.values()] * 3, axis=1),
        columns=["comp1", "comp2", "comp3"],
        static_covariates=static_covs1,
    )

    static_covs2 = pd.DataFrame(
        data={
            "cont1": [2, 3, 4],
            "cat1": [3, 4, 5],
            "cont2": [0.3, 0.4, 0.5],
            "cat2": ["c", "d", "e"],
        }
    ).astype(dtype={"cat1": "O"})
    series2 = TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.concatenate([series.values()] * 3, axis=1),
        columns=["comp1", "comp2", "comp3"],
        static_covariates=static_covs2,
    )

    @pytest.mark.parametrize("component_mask", [None, np.array([True, True, True])])
    def test_scaling_single_series(self, component_mask):
        # 3 categories for each categorical static covariate column (column idx 1 and 3)
        test_values = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.5, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ])
        for series in [self.series1, self.series2]:
            scaler = StaticCovariatesTransformer()
            self.helper_test_scaling(series, scaler, test_values)

        test_values = np.array([
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ])
        for series in [self.series1, self.series2]:
            scaler = StaticCovariatesTransformer(
                transformer_num=MinMaxScaler(feature_range=(-1, 1))
            )
            self.helper_test_scaling(series, scaler, test_values)

        test_values = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ])
        for series in [self.series1, self.series2]:
            scaler = StaticCovariatesTransformer(transformer_cat=OneHotEncoder())
            self.helper_test_scaling(series, scaler, test_values)

    def test_single_type_scaler(self):
        transformer_cont = StaticCovariatesTransformer()
        series_cont = self.series1.with_static_covariates(
            self.series1.static_covariates[["cont1", "cont2"]]
        )
        test_cont = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        self.helper_test_scaling(series_cont, transformer_cont, test_cont)

        transformer_cat = StaticCovariatesTransformer()
        series_cat = self.series1.with_static_covariates(
            self.series1.static_covariates[["cat1", "cat2"]]
        )
        test_cat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.helper_test_scaling(series_cat, transformer_cat, test_cat)

    def test_selected_columns(self):
        test_cont = (
            pd.DataFrame(
                [[0.0, 1, 0.0, "a"], [0.5, 2, 0.5, "b"], [1.0, 3, 1.0, "c"]],
            )
            .astype(dtype={1: "O", 3: "O"})
            .values
        )
        transformer_cont2 = StaticCovariatesTransformer(
            cols_num=["cont1", "cont2"], cols_cat=[]
        )
        self.helper_test_scaling(self.series1, transformer_cont2, test_cont)

        test_contcat = (
            pd.DataFrame(
                [[0.0, 1, 0.0, 0.0], [1.0, 2, 0.5, 1.0], [2.0, 3, 1.0, 2.0]],
            )
            .astype(dtype={1: "O"})
            .values
        )
        transformer_contcat = StaticCovariatesTransformer(
            cols_num=["cont2"], cols_cat=["cat2"]
        )
        self.helper_test_scaling(self.series1, transformer_contcat, test_contcat)

        test_cat = pd.DataFrame(
            [[0.0, 0.0, 0.1, 0.0], [1.0, 1.0, 0.2, 1], [2.0, 2.0, 0.3, 2.0]],
        ).values
        transformer_cat = StaticCovariatesTransformer(
            cols_num=[], cols_cat=["cat1", "cat2"]
        )
        self.helper_test_scaling(self.series1, transformer_cat, test_cat)

    def test_custom_scaler(self):
        # invalid scaler with missing inverse_transform
        class InvalidScaler:
            def fit(self):
                pass

            def transform(self):
                pass

        with pytest.raises(ValueError):
            _ = StaticCovariatesTransformer(transformer_num=InvalidScaler())

        with pytest.raises(ValueError):
            _ = StaticCovariatesTransformer(transformer_cat=InvalidScaler())

        class ValidScaler(InvalidScaler):
            def inverse_transform(self):
                pass

        _ = StaticCovariatesTransformer(transformer_num=ValidScaler())
        _ = StaticCovariatesTransformer(transformer_cat=ValidScaler())
        _ = StaticCovariatesTransformer(
            transformer_num=ValidScaler(), transformer_cat=ValidScaler()
        )

    def test_scaling_multi_series(self):
        # 5 categories in total for each categorical static covariate from multiple time series
        scaler = StaticCovariatesTransformer()
        series_tr2 = scaler.fit_transform([self.series1, self.series2])

        np.testing.assert_almost_equal(
            series_tr2[0].static_covariates_values(),
            np.array([
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 1.0, 0.25, 1.0],
                [0.5, 2.0, 0.5, 2.0],
            ]),
        )
        series_recovered2 = scaler.inverse_transform(series_tr2[0])
        assert self.series1.static_covariates.equals(
            series_recovered2.static_covariates
        )

        np.testing.assert_almost_equal(
            series_tr2[1].static_covariates_values(),
            np.array([
                [0.5, 2.0, 0.5, 2.0],
                [0.75, 3.0, 0.75, 3.0],
                [1.0, 4.0, 1.0, 4.0],
            ]),
        )
        series_recovered3 = scaler.inverse_transform(series_tr2[1])
        assert self.series2.static_covariates.equals(
            series_recovered3.static_covariates
        )

        series_recovered_multi = scaler.inverse_transform(series_tr2)
        assert self.series1.static_covariates.equals(
            series_recovered_multi[0].static_covariates
        )
        assert self.series2.static_covariates.equals(
            series_recovered_multi[1].static_covariates
        )

    def test_zero_cardinality_multi_series(self):
        """Check that inverse-transform works as expected when OneHotEncoder is used on several series with
        identical static covariates categories and values.
        """
        ts1 = self.series.with_static_covariates(
            pd.Series({
                "cov_a": "foo",
                "cov_b": "foo",
                "cov_c": "foo",
            })
        )
        ts2 = self.series.with_static_covariates(
            pd.Series({
                "cov_a": "foo",
                "cov_b": "foo",
                "cov_c": "bar",
            })
        )

        transformer = StaticCovariatesTransformer(transformer_cat=OneHotEncoder())
        transformer.fit([ts1, ts2])
        ts1_enc, ts2_enc = transformer.transform([ts1, ts2])
        ts1_inv, ts2_inv = transformer.inverse_transform([ts1_enc, ts2_enc])
        pd.testing.assert_frame_equal(ts1_inv.static_covariates, ts1.static_covariates)
        pd.testing.assert_frame_equal(ts2_inv.static_covariates, ts2.static_covariates)

    def test_cols_cat_order_different_from_data(self):
        series = [
            self.series.with_static_covariates(
                pd.DataFrame({"Country": ["US"], "City": ["New York"]})
            ),
            self.series.with_static_covariates(
                pd.DataFrame({"Country": ["China"], "City": ["Beijing"]})
            ),
        ]

        transformer = StaticCovariatesTransformer(
            transformer_cat=OneHotEncoder(sparse_output=False),
            cols_cat=["City", "Country"],
        )

        transformed = transformer.fit_transform(series)

        expected_columns = [
            "Country_China",
            "Country_US",
            "City_Beijing",
            "City_New York",
        ]
        assert transformed[0].static_covariates.columns.tolist() == expected_columns
        assert transformed[1].static_covariates.columns.tolist() == expected_columns

        # Series 0: Country="US", City="New York"
        # Expected: [Country_China=0.0, Country_US=1.0, City_Beijing=0.0, City_New York=1.0]
        first_static_covs = transformed[0].static_covariates
        assert first_static_covs.iloc[0].tolist() == [0.0, 1.0, 0.0, 1.0]

        # Series 1: Country="China", City="Beijing"
        # Expected: [Country_China=1.0, Country_US=0.0, City_Beijing=1.0, City_New York=0.0]
        second_static_covs = transformed[1].static_covariates
        assert second_static_covs.iloc[0].tolist() == [1.0, 0.0, 1.0, 0.0]

        recovered = transformer.inverse_transform(transformed)
        for i in range(2):
            pd.testing.assert_frame_equal(
                recovered[i].static_covariates,
                series[i].static_covariates,
            )

    @pytest.mark.parametrize("reverse_cols_order", [False, True])
    @pytest.mark.parametrize("drop", ["first", "if_binary"])
    def test_one_hot_encoder_with_drop_single_series(self, reverse_cols_order, drop):
        transformer = StaticCovariatesTransformer(
            transformer_cat=OneHotEncoder(sparse_output=False, drop=drop)
        )

        # pick edge-case cat column names ("cat1" is also prefix of "cat1_")
        data = {"cat1": [0, 1, 2], "col_num": [3.0, 4.0, 5.0], "cat1_": [10, 11, 11]}
        if reverse_cols_order:
            data = {k: v for k, v in list(data.items())[::-1]}
        sc_in1 = pd.DataFrame(
            data=data,
            index=self.series1.static_covariates.index,
        ).astype({"cat1": "O", "cat1_": "O"})
        series1 = self.series1.with_static_covariates(sc_in1)
        sc_in1 = series1.static_covariates

        data = {"cat1_0": [1, 0, 0]} if drop == "if_binary" else {}
        data = {
            **data,
            "cat1_1": [0, 1, 0],
            "cat1_2": [0, 0, 1],
            "col_num": [0.0, 0.5, 1.0],
            "cat1__11": [0, 1, 1],
        }
        if reverse_cols_order:
            data = {
                k: data[k]
                for k in ["cat1__11", "col_num"]
                + (["cat1_0"] if drop == "if_binary" else [])
                + ["cat1_1", "cat1_2"]
            }
        sc_tr_expected = pd.DataFrame(data=data, index=series1.static_covariates.index)
        sc_tr_expected = series1.with_static_covariates(
            sc_tr_expected
        ).static_covariates

        series_tr = transformer.fit_transform(series1)
        assert series_tr.static_covariates.equals(sc_tr_expected)

        series_recovered = transformer.inverse_transform(series_tr)
        assert series_recovered.static_covariates.equals(sc_in1)

    @pytest.mark.parametrize("drop", ["first", "if_binary"])
    def test_one_hot_encoder_with_drop_multi_series(self, drop):
        transformer = StaticCovariatesTransformer(
            transformer_cat=OneHotEncoder(sparse_output=False, drop=drop)
        )

        # pick edge-case cat column names ("cat1" is also prefix of "cat1_")
        sc_in1 = pd.DataFrame(
            data={"cat1": [0, 1, 2], "col_num": [3.0, 4.0, 5.0], "cat1_": [10, 11, 11]},
            index=self.series1.static_covariates.index,
        ).astype({"cat1": "O", "cat1_": "O"})
        series1 = self.series1.with_static_covariates(sc_in1)
        sc_in1 = series1.static_covariates

        sc_in2 = pd.DataFrame(
            data={"cat1": [3, 0, 3], "col_num": [5.0, 6.0, 7.0], "cat1_": [11, 10, 10]},
            index=self.series2.static_covariates.index,
        ).astype({"cat1": "O", "cat1_": "O"})
        series2 = self.series2.with_static_covariates(sc_in2)
        sc_in2 = series2.static_covariates

        data = {"cat1_0": [1, 0, 0]} if drop == "if_binary" else {}
        data = {
            **data,
            "cat1_1": [0, 1, 0],
            "cat1_2": [0, 0, 1],
            "cat1_3": [0, 0, 0],
            "col_num": [0.0, 0.25, 0.5],
            "cat1__11": [0, 1, 1],
        }
        sc_tr1_expected = pd.DataFrame(data=data, index=series1.static_covariates.index)
        sc_tr1_expected = series1.with_static_covariates(
            sc_tr1_expected
        ).static_covariates

        data = {"cat1_0": [0, 1, 0]} if drop == "if_binary" else {}
        data = {
            **data,
            "cat1_1": [0, 0, 0],
            "cat1_2": [0, 0, 0],
            "cat1_3": [1, 0, 1],
            "col_num": [0.5, 0.75, 1.0],
            "cat1__11": [1, 0, 0],
        }
        sc_tr2_expected = pd.DataFrame(data=data, index=series2.static_covariates.index)
        sc_tr2_expected = series2.with_static_covariates(
            sc_tr2_expected
        ).static_covariates

        series_list = [series1, series2]
        series_tr_list = transformer.fit_transform(series_list)
        for series_tr, sc_tr_expected in zip(
            series_tr_list, [sc_tr1_expected, sc_tr2_expected]
        ):
            assert series_tr.static_covariates.equals(sc_tr_expected)

        series_recovered_list = transformer.inverse_transform(series_tr_list)
        for recov, sc_in in zip(series_recovered_list, [sc_in1, sc_in2]):
            assert recov.static_covariates.equals(sc_in)

    @pytest.mark.parametrize(
        "enc_kwargs",
        [
            {"sparse_output": False, "min_frequency": 2},
            {"sparse_output": False, "max_categories": 3},
        ],
    )
    def test_one_hot_encoder_with_infrequent_sklearn(self, enc_kwargs):
        """Test for ``OneHotEncoder`` with ``min_frequency`` and ``max_categories``.

        Sklearn ≥ 1.1 groups low-frequency categories into a synthetic feature named  ``{col}_infrequent_sklearn``
        which does not appear in ``transformer_cat.categories_``. The column-name mapping must include this feature,
        otherwise ``transform`` silently drops the infrequent column and ``inverse_transform`` fails with a shape
        mismatch.
        """

        # Build a small sequence of series where one category ("JP") is
        # infrequent so that sklearn groups it into ``infrequent_sklearn``.
        def make_series(idx, country):
            return TimeSeries.from_times_and_values(
                times=pd.date_range("2024-01-01", periods=5),
                values=np.ones((5, 1)),
                columns=[f"comp_{idx}"],
                static_covariates=pd.DataFrame({"country": [country]}),
            )

        categories = (["US"] * 5) + (["KR"] * 2) + ["JP"]
        series_list = [make_series(i, country) for i, country in enumerate(categories)]

        vals_tr_expected = np.array(
            [[0, 1, 0]] * 5 + [[1, 0, 0]] * 2 + [[0, 0, 1]]
        ).astype("float64")
        # Every transformed series must carry the full one-hot output from the
        # underlying encoder, including the synthetic ``_infrequent_sklearn``
        # bucket (previously this column was silently dropped from the mapping).
        expected_cols = ["country_KR", "country_US", "country_infrequent_sklearn"]

        transformer = StaticCovariatesTransformer(
            transformer_cat=OneHotEncoder(**enc_kwargs)
        )

        transformed = transformer.fit_transform(series_list)
        vals_tr = []
        for ts in transformed:
            sc = ts.static_covariates
            vals_tr.append(sc.values)
            assert list(sc.columns) == expected_cols

        np.testing.assert_array_equal(np.concatenate(vals_tr), vals_tr_expected)

        # inverse transform must restore all frequent countries
        recovered = transformer.inverse_transform(transformed)
        for recov, series in zip(recovered[:-1], series_list[:-1]):
            assert recov.static_covariates.equals(series.static_covariates)

        # the infrequent inputs are mapped to sklearn's sentinel string
        assert recovered[-1].static_covariates.to_dict() == {
            "country": {"comp_7": "infrequent_sklearn"}
        }

    def helper_test_scaling(self, series, scaler, test_values):
        series_copy = series.copy()
        series_tr = scaler.fit_transform(series)
        assert all([
            a == b
            for a, b in zip(
                series_tr.static_covariates_values().flatten(),
                test_values.flatten(),
            )
        ])
        assert series == series_copy

        series_tr_copy = series_tr.copy()
        series_recovered = scaler.inverse_transform(series_tr)
        assert series.static_covariates.equals(series_recovered.static_covariates)
        assert series_tr == series_tr_copy
