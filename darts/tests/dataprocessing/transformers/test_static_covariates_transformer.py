import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg


class StaticCovariatesTransformerTestCase(DartsBaseTestClass):
    series = tg.linear_timeseries(length=10)
    static_covs1 = pd.DataFrame(
        data={
            "cont1": [0, 1, 2],
            "cat1": [1, 2, 3],
            "cont2": [0.1, 0.2, 0.3],
            "cat2": ["a", "b", "c"],
        }
    ).astype(dtype={"cat1": "O", "cat2": "O"})
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
    )
    static_covs2["cat1"] = static_covs2["cat1"].astype("O")
    series2 = TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.concatenate([series.values()] * 3, axis=1),
        columns=["comp1", "comp2", "comp3"],
        static_covariates=static_covs2,
    )

    def test_scaling_single_series(self):
        # 3 categories for each categorical static covariate column (column idx 1 and 3)
        test_values = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.5, 1.0], [1.0, 2.0, 1.0, 2.0]]
        )
        for series in [self.series1, self.series2]:
            scaler = StaticCovariatesTransformer()
            self.helper_test_scaling(series, scaler, test_values)

        test_values = np.array(
            [[-1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 2.0, 1.0, 2.0]]
        )
        for series in [self.series1, self.series2]:
            scaler = StaticCovariatesTransformer(
                transformer_num=MinMaxScaler(feature_range=(-1, 1))
            )
            self.helper_test_scaling(series, scaler, test_values)

        test_values = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )
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
            np.array(
                [[0.0, 0.0, 0.0, 0.0], [0.25, 1.0, 0.25, 1.0], [0.5, 2.0, 0.5, 2.0]]
            ),
        )
        series_recovered2 = scaler.inverse_transform(series_tr2[0])
        self.assertTrue(
            self.series1.static_covariates.equals(series_recovered2.static_covariates)
        )

        np.testing.assert_almost_equal(
            series_tr2[1].static_covariates_values(),
            np.array(
                [[0.5, 2.0, 0.5, 2.0], [0.75, 3.0, 0.75, 3.0], [1.0, 4.0, 1.0, 4.0]]
            ),
        )
        series_recovered3 = scaler.inverse_transform(series_tr2[1])
        self.assertTrue(
            self.series2.static_covariates.equals(series_recovered3.static_covariates)
        )

        series_recovered_multi = scaler.inverse_transform(series_tr2)
        self.assertTrue(
            self.series1.static_covariates.equals(
                series_recovered_multi[0].static_covariates
            )
        )
        self.assertTrue(
            self.series2.static_covariates.equals(
                series_recovered_multi[1].static_covariates
            )
        )

    def helper_test_scaling(self, series, scaler, test_values):
        series_tr = scaler.fit_transform(series)
        assert all(
            [
                a == b
                for a, b in zip(
                    series_tr.static_covariates_values().flatten(),
                    test_values.flatten(),
                )
            ]
        )

        series_recovered = scaler.inverse_transform(series_tr)
        self.assertTrue(
            series.static_covariates.equals(series_recovered.static_covariates)
        )
