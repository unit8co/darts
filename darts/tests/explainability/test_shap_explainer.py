import copy
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap
import sklearn
from dateutil.relativedelta import relativedelta
from numpy.testing import assert_array_equal
from sklearn.preprocessing import MinMaxScaler

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.explainability.explainability_result import ShapExplainabilityResult
from darts.explainability.shap_explainer import MIN_BACKGROUND_SAMPLE, ShapExplainer
from darts.models import (
    CatBoostModel,
    ExponentialSmoothing,
    LightGBMModel,
    LinearRegressionModel,
    NotImportedModule,
    RegressionModel,
    XGBModel,
)
from darts.utils.timeseries_generation import linear_timeseries

lgbm_available = not isinstance(LightGBMModel, NotImportedModule)
cb_available = not isinstance(CatBoostModel, NotImportedModule)


def extract_year(index):
    """Return year of time index entry, normalized"""
    return (index.year - 1950) / 50


class TestShapExplainer:
    np.random.seed(42)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    add_encoders = {
        "cyclic": {"past": ["month", "day"]},
        "datetime_attribute": {"future": ["hour", "dayofweek"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "custom": {"past": [extract_year]},
        "transformer": Scaler(scaler),
    }

    date_start = date(2012, 12, 12)
    date_end = date(2014, 6, 5)
    days = pd.date_range(date_start, date_end, freq="d")
    N = len(days)
    eps_1 = np.random.normal(0, 1, N).astype("float32")
    eps_2 = np.random.normal(0, 1, N).astype("float32")

    x_1 = np.zeros(N).astype("float32")
    x_2 = np.zeros(N).astype("float32")
    x_3 = np.zeros(N).astype("float32")

    days_past_cov = pd.date_range(
        date_start, date_start + timedelta(days=N - 2), freq="d"
    )

    past_cov_1 = np.random.normal(0, 1, N - 1).astype("float32")
    past_cov_2 = np.random.normal(0, 1, N - 1).astype("float32")
    past_cov_3 = np.random.normal(0, 1, N - 1).astype("float32")

    fut_cov_1 = np.random.normal(0, 1, N).astype("float32")
    fut_cov_2 = np.random.normal(0, 1, N).astype("float32")

    x_1[0] = eps_1[0]
    x_1[1] = eps_1[1]
    x_2[2] = eps_1[2]

    x_2[0] = eps_2[0]
    x_2[1] = eps_2[1]

    K_1 = 0.5
    K_2 = -0.25
    K_3 = 0.5
    K_4 = 0.9
    K_5 = -0.75
    K_6 = 0.75
    K_7 = 0.9

    # Multivariates Ex.2 independants
    for i in range(2, len(x_1)):
        x_1[i] = (
            K_1 * x_1[i - 1] + K_2 * past_cov_1[i - 2] + K_3 * fut_cov_1[i] + eps_1[i]
        )

    for i in range(1, len(x_2)):
        x_2[i] = (
            +K_4 * x_2[i - 1] + K_5 * past_cov_2[i - 1] + K_6 * fut_cov_2[i] + eps_2[i]
        )

    for i in range(2, len(x_3)):
        x_3[i] = K_7 * x_1[i - 1] + x_2[i - 2]

    target_ts = TimeSeries.from_times_and_values(
        days, np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1)
    ).with_columns_renamed(["0", "1"], ["price", "power"])

    target_ts_with_static_covs = TimeSeries.from_times_and_values(
        days,
        x_1.reshape(-1, 1),
        static_covariates=pd.DataFrame({"type": [0], "state": [1]}),
    ).with_columns_renamed(["0"], ["price"])
    target_ts_with_multi_component_static_covs = TimeSeries.from_times_and_values(
        days,
        np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1),
        static_covariates=pd.DataFrame({"type": [0, 1], "state": [2, 3]}),
    ).with_columns_renamed(["0", "1"], ["price", "power"])
    target_ts_multiple_series_with_different_static_covs = [
        TimeSeries.from_times_and_values(
            days, x_1.reshape(-1, 1), static_covariates=pd.DataFrame({"type": [0]})
        ).with_columns_renamed(["0"], ["price"]),
        TimeSeries.from_times_and_values(
            days, x_2.reshape(-1, 1), static_covariates=pd.DataFrame({"state": [1]})
        ).with_columns_renamed(["0"], ["price"]),
    ]

    past_cov_ts = TimeSeries.from_times_and_values(
        days_past_cov,
        np.concatenate(
            [
                past_cov_1.reshape(-1, 1),
                past_cov_2.reshape(-1, 1),
                past_cov_3.reshape(-1, 1),
            ],
            axis=1,
        ),
    )

    fut_cov_ts = TimeSeries.from_times_and_values(
        days,
        np.concatenate([fut_cov_1.reshape(-1, 1), fut_cov_2.reshape(-1, 1)], axis=1),
    )

    def test_creation(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        # Model should be fitted first
        m = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=4,
            add_encoders=self.add_encoders,
        )
        with pytest.raises(ValueError):
            ShapExplainer(m, self.target_ts, self.past_cov_ts, self.fut_cov_ts)

        # Model should be a RegressionModel
        m = ExponentialSmoothing()
        m.fit(self.target_ts["price"])
        with pytest.raises(ValueError):
            ShapExplainer(m)

        # For now, multi_models=False not allowed
        m = LinearRegressionModel(lags=1, output_chunk_length=2, multi_models=False)
        m.fit(
            series=self.target_ts,
        )
        with pytest.raises(ValueError):
            ShapExplainer(
                m,
                self.target_ts,
            )

        m = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=4,
            add_encoders=self.add_encoders,
        )

        m.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        # Should have the same number of target, past and futures in the respective lists
        with pytest.raises(ValueError):
            ShapExplainer(
                m,
                [self.target_ts, self.target_ts],
                self.past_cov_ts,
                self.fut_cov_ts,
            )

        # Missing a future covariate if you choose to use a new background
        with pytest.raises(ValueError):
            ShapExplainer(
                m, self.target_ts, background_past_covariates=self.past_cov_ts
            )

        # Missing a past covariate if you choose to use a new background
        with pytest.raises(ValueError):
            ShapExplainer(
                m, self.target_ts, background_future_covariates=self.fut_cov_ts
            )

        # Good type of explainers
        shap_explain = ShapExplainer(m)
        if m._supports_native_multioutput:
            # since xgboost > 2.1.0, model supports native multi-output regression
            # CatBoostModel supports multi-output for certain loss functions
            assert isinstance(shap_explain.explainers.explainers, shap.explainers.Tree)
        else:
            assert isinstance(
                shap_explain.explainers.explainers[0][0], shap.explainers.Tree
            )

        # Linear model - also not a MultiOutputRegressor
        m = LinearRegressionModel(
            lags=1,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=2,
        )
        m.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(m)
        assert isinstance(shap_explain.explainers.explainers, shap.explainers.Linear)

        # ExtraTreesRegressor - also not a MultiOutputRegressor
        m = RegressionModel(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=2,
            model=sklearn.tree.ExtraTreeRegressor(),
        )
        m.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(m)
        assert isinstance(shap_explain.explainers.explainers, shap.explainers.Tree)

        # No past or future covariates
        m = LinearRegressionModel(
            lags=1,
            output_chunk_length=2,
        )
        m.fit(
            series=self.target_ts,
        )

        shap_explain = ShapExplainer(m)
        assert isinstance(shap_explain.explainers.explainers, shap.explainers.Linear)

        # CatBoost
        model_cls = CatBoostModel if cb_available else XGBModel
        m = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -6],
            lags_future_covariates=[0],
            output_chunk_length=4,
        )
        m.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(m)
        if m._supports_native_multioutput:
            assert isinstance(shap_explain.explainers.explainers, shap.explainers.Tree)
        else:
            assert isinstance(
                shap_explain.explainers.explainers[0][0], shap.explainers.Tree
            )

        # Bad choice of shap explainer
        with pytest.raises(ValueError):
            ShapExplainer(m, shap_method="bad_choice")

    def test_explain(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        m = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=4,
            add_encoders=self.add_encoders,
        )
        m.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(m)
        with pytest.raises(ValueError):
            _ = shap_explain.explain(horizons=[1, 5])  # horizon > output_chunk_length
        with pytest.raises(ValueError):
            _ = shap_explain.explain(
                horizons=[1, 2], target_components=["test"]
            )  # wrong name

        results = shap_explain.explain()
        with pytest.raises(ValueError):
            results.get_explanation(horizon=5, component="price")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=5, component="price")
        with pytest.raises(ValueError):
            results.get_shap_explanation_object(horizon=5, component="price")
        with pytest.raises(ValueError):
            results.get_explanation(horizon=1, component="test")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=1, component="test")
        with pytest.raises(ValueError):
            results.get_shap_explanation_object(horizon=1, component="test")

        results = shap_explain.explain(horizons=[1, 3], target_components=["power"])
        with pytest.raises(ValueError):
            results.get_explanation(horizon=2, component="power")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=2, component="power")
        with pytest.raises(ValueError):
            results.get_shap_explanation_object(horizon=2, component="power")
        with pytest.raises(ValueError):
            results.get_explanation(horizon=1, component="test")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=1, component="test")
        with pytest.raises(ValueError):
            results.get_shap_explanation_object(horizon=1, component="test")

        explanation = results.get_explanation(horizon=1, component="power")
        assert len(explanation) == 537
        feature_vals = results.get_feature_values(horizon=1, component="power")
        assert len(feature_vals) == 537

        # list of foregrounds: encoders have to be corrected first.
        results = shap_explain.explain(
            foreground_series=[self.target_ts, self.target_ts[:100]],
            foreground_past_covariates=[self.past_cov_ts, self.past_cov_ts[:40]],
            foreground_future_covariates=[self.fut_cov_ts, self.fut_cov_ts[:40]],
        )
        ts_res = results.get_explanation(horizon=2, component="power")
        assert len(ts_res) == 2
        feature_vals = results.get_feature_values(horizon=2, component="power")
        assert len(feature_vals) == 2

        # explain with a new foreground, minimum required. We should obtain one
        # timeseries with only one time element
        results = shap_explain.explain(
            foreground_series=self.target_ts[-5:],
            foreground_past_covariates=self.past_cov_ts[-4:],
            foreground_future_covariates=self.fut_cov_ts[-1],
        )

        ts_res = results.get_explanation(horizon=2, component="power")
        assert len(ts_res) == 1
        assert ts_res.time_index[-1] == pd.Timestamp(2014, 6, 5)
        feature_vals = results.get_feature_values(horizon=2, component="power")
        assert len(feature_vals) == 1
        assert feature_vals.time_index[-1] == pd.Timestamp(2014, 6, 5)

        with pytest.raises(ValueError):
            results.get_explanation(horizon=5, component="price")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=5, component="price")
        with pytest.raises(ValueError):
            results.get_explanation(horizon=1, component="test")
        with pytest.raises(ValueError):
            results.get_feature_values(horizon=1, component="test")

        # right instance
        assert isinstance(results, ShapExplainabilityResult)

        components_list = [
            "price_target_lag-4",
            "power_target_lag-4",
            "price_target_lag-3",
            "power_target_lag-3",
            "price_target_lag-2",
            "power_target_lag-2",
            "price_target_lag-1",
            "power_target_lag-1",
            "0_past_cov_lag-3",
            "1_past_cov_lag-3",
            "2_past_cov_lag-3",
            "darts_enc_pc_cyc_month_sin_past_cov_lag-3",
            "darts_enc_pc_cyc_month_cos_past_cov_lag-3",
            "darts_enc_pc_cyc_day_sin_past_cov_lag-3",
            "darts_enc_pc_cyc_day_cos_past_cov_lag-3",
            "darts_enc_pc_pos_relative_past_cov_lag-3",
            "darts_enc_pc_cus_custom_past_cov_lag-3",
            "0_past_cov_lag-2",
            "1_past_cov_lag-2",
            "2_past_cov_lag-2",
            "darts_enc_pc_cyc_month_sin_past_cov_lag-2",
            "darts_enc_pc_cyc_month_cos_past_cov_lag-2",
            "darts_enc_pc_cyc_day_sin_past_cov_lag-2",
            "darts_enc_pc_cyc_day_cos_past_cov_lag-2",
            "darts_enc_pc_pos_relative_past_cov_lag-2",
            "darts_enc_pc_cus_custom_past_cov_lag-2",
            "0_past_cov_lag-1",
            "1_past_cov_lag-1",
            "2_past_cov_lag-1",
            "darts_enc_pc_cyc_month_sin_past_cov_lag-1",
            "darts_enc_pc_cyc_month_cos_past_cov_lag-1",
            "darts_enc_pc_cyc_day_sin_past_cov_lag-1",
            "darts_enc_pc_cyc_day_cos_past_cov_lag-1",
            "darts_enc_pc_pos_relative_past_cov_lag-1",
            "darts_enc_pc_cus_custom_past_cov_lag-1",
            "0_fut_cov_lag0",
            "1_fut_cov_lag0",
            "hour_fut_cov_lag0",
            "dayofweek_fut_cov_lag0",
            "relative_idx_fut_cov_lag0",
        ]

        results = shap_explain.explain()

        # all the features explained are here, in the right order
        assert [
            results.get_explanation(i, "price").components.to_list() == components_list
            for i in range(1, 5)
        ]

        # No past or future covariates
        m = LinearRegressionModel(
            lags=1,
            output_chunk_length=2,
        )
        m.fit(
            series=self.target_ts,
        )
        shap_explain = ShapExplainer(m)

        assert isinstance(shap_explain.explain(), ShapExplainabilityResult)

    def test_explain_with_lags_future_covariates_series_of_same_length_as_target(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[2],
            output_chunk_length=1,
        )

        model.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        for component in ["power", "price"]:
            explanation = explanation_results.get_explanation(
                horizon=1, component=component
            )

            # The fut_cov_ts have the same length as the target_ts. Hence, if we pass lags_future_covariates this means
            # that the last prediction can be made max(lags_future_covariates) time periods before the end of the
            # series (in this case 2 time periods).
            assert explanation.end_time() == self.target_ts.end_time() - relativedelta(
                days=2
            )

    def test_explain_with_lags_future_covariates_series_extending_into_future(self):
        # Constructing future covariates TimeSeries that extends further into the future than the target series
        date_start = date(2012, 12, 12)
        date_end = date(2014, 6, 7)
        days = pd.date_range(date_start, date_end, freq="d")
        fut_cov = np.random.normal(0, 1, len(days)).astype("float32")
        fut_cov_ts = TimeSeries.from_times_and_values(days, fut_cov.reshape(-1, 1))

        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[2],
            output_chunk_length=1,
        )

        model.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=fut_cov_ts,
        )

        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        for component in ["power", "price"]:
            explanation = explanation_results.get_explanation(
                horizon=1, component=component
            )

            # The fut_cov_ts extends further into the future than the target_ts. Hence, at prediction time we know the
            # values of lagged future covariates and we thus no longer expect the end_time() of the explanation
            # TimeSeries to differ from the end_time() of the target TimeSeries
            assert explanation.end_time() == self.target_ts.end_time()

    def test_explain_with_lags_covariates_series_older_timestamps_than_target(self):
        # Constructing covariates TimeSeries with older timestamps than target
        date_start = date(2012, 12, 10)
        date_end = date(2014, 6, 5)
        days = pd.date_range(date_start, date_end, freq="d")
        fut_cov = np.random.normal(0, 1, len(days)).astype("float32")
        fut_cov_ts = TimeSeries.from_times_and_values(days, fut_cov.reshape(-1, 1))
        past_cov = np.random.normal(0, 1, len(days)).astype("float32")
        past_cov_ts = TimeSeries.from_times_and_values(days, past_cov.reshape(-1, 1))

        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=None,
            lags_past_covariates=[-1, -2],
            lags_future_covariates=[-1, -2],
            output_chunk_length=1,
        )

        model.fit(
            series=self.target_ts,
            past_covariates=past_cov_ts,
            future_covariates=fut_cov_ts,
        )

        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        for component in ["power", "price"]:
            explanation = explanation_results.get_explanation(
                horizon=1, component=component
            )

            # The covariates series (past and future) start two time periods earlier than the target series. This in
            # combination with the LightGBM configuration (lags=None and 'largest' covariates lags equal to -2) means
            # that at the start of the target series we have sufficient information to explain the prediction.
            assert explanation.start_time() == self.target_ts.start_time()

    def test_plot(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        m_0 = model_cls(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=4,
            add_encoders=self.add_encoders,
        )
        m_0.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(m_0)

        # We need at least 5 points for force_plot
        with pytest.raises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:104],
                self.past_cov_ts[100:104],
                self.fut_cov_ts[100:104],
                2,
                "power",
            )

        fplot = shap_explain.force_plot_from_ts(
            self.target_ts[100:105],
            self.past_cov_ts[100:105],
            self.fut_cov_ts[100:105],
            2,
            "power",
        )
        assert isinstance(fplot, shap.plots._force.BaseVisualizer)
        plt.close()

        # no component name -> multivariate error
        with pytest.raises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                1,
            )

        # fake component
        with pytest.raises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                2,
                "fake",
            )

        # horizon 0
        with pytest.raises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                0,
                "power",
            )

        # Check the dimensions of returned values
        dict_shap_values = shap_explain.summary_plot(show=False)
        # One nested dict per horizon
        assert len(dict_shap_values) == m_0.output_chunk_length
        # Size of nested dict match number of component
        for i in range(1, m_0.output_chunk_length + 1):
            assert len(dict_shap_values[i]) == self.target_ts.width

        # Wrong component name
        with pytest.raises(ValueError):
            shap_explain.summary_plot(horizons=[1], target_components=["test"])

        # Wrong horizon
        with pytest.raises(ValueError):
            shap_explain.summary_plot(horizons=[0], target_components=["test"])
        with pytest.raises(ValueError):
            shap_explain.summary_plot(horizons=[10], target_components=["test"])

        # No past or future covariates
        m = LinearRegressionModel(
            lags=1,
            output_chunk_length=2,
        )
        m.fit(
            series=self.target_ts,
        )

        shap_explain = ShapExplainer(m)
        fplot = shap_explain.force_plot_from_ts(
            foreground_series=self.target_ts[100:105],
            horizon=1,
            target_component="power",
        )
        assert isinstance(fplot, shap.plots._force.BaseVisualizer)
        plt.close()

    def test_feature_values_align_with_input(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            output_chunk_length=1,
        )
        model.fit(
            series=self.target_ts,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        df = pd.merge(
            self.target_ts.to_dataframe(),
            explanation_results.get_feature_values(
                horizon=1, component="price"
            ).to_dataframe(),
            how="left",
            left_index=True,
            right_index=True,
        )
        df[["price_shift_4", "power_shift_4"]] = df[["price", "power"]].shift(4)

        assert_array_equal(
            df[["price_shift_4", "power_shift_4"]].values,
            df[["price_target_lag-4", "power_target_lag-4"]].values,
        )

    def test_feature_values_align_with_raw_output_shap(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            output_chunk_length=1,
        )
        model.fit(
            series=self.target_ts,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()

        feature_values = explanation_results.get_feature_values(
            horizon=1, component="price"
        )
        comparison = explanation_results.get_shap_explanation_object(
            horizon=1, component="price"
        ).data

        assert_array_equal(feature_values.values(), comparison)
        assert (
            feature_values.values().shape
            == explanation_results.get_explanation(horizon=1, component="price")
            .values()
            .shape
        ), "The shape of the feature values should be the same as the shap values"

    def test_shap_explanation_object_validity(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            lags_past_covariates=2,
            lags_future_covariates=[1],
            output_chunk_length=1,
        )
        model.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()

        assert isinstance(
            explanation_results.get_shap_explanation_object(
                horizon=1, component="power"
            ),
            shap.Explanation,
        )

    @pytest.mark.parametrize(
        "config",
        [
            (XGBModel, {}),
            (
                LightGBMModel if lgbm_available else XGBModel,
                {"likelihood": "quantile", "quantiles": [0.5]},
            ),
        ],
    )
    def test_shap_selected_components(self, config):
        """Test selected components with and without Darts' MultiOutputRegressor"""
        model_cls, model_kwargs = config
        # model_cls = XGBModel
        model = model_cls(
            lags=4,
            lags_past_covariates=2,
            lags_future_covariates=[1],
            output_chunk_length=1,
            **model_kwargs,
        )
        model.fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        # check that explain() with selected components gives identical results
        for comp in self.target_ts.components:
            explanation_comp = shap_explain.explain(target_components=[comp])
            assert explanation_comp.available_components == [comp]
            assert explanation_comp.available_horizons == [1]
            # explained forecasts
            fc_res_tmp = copy.deepcopy(explanation_results.explained_forecasts)
            fc_res_tmp[1] = {str(comp): fc_res_tmp[1][comp]}
            assert explanation_comp.explained_forecasts == fc_res_tmp

            # feature values
            fv_res_tmp = copy.deepcopy(explanation_results.feature_values)
            fv_res_tmp[1] = {str(comp): fv_res_tmp[1][comp]}
            assert explanation_comp.explained_forecasts == fc_res_tmp

            # shap objects
            assert (
                len(explanation_comp.shap_explanation_object[1]) == 1
                and comp in explanation_comp.shap_explanation_object[1]
            )

    def test_shapley_with_static_cov(self):
        ts = self.target_ts_with_static_covs
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            output_chunk_length=1,
        )
        model.fit(
            series=ts,
        )
        shap_explain = ShapExplainer(model)

        # different static covariates dimensions should raise an error
        with pytest.raises(ValueError):
            shap_explain.explain(
                ts.with_static_covariates(ts.static_covariates["state"])
            )

        # without static covariates should raise an error
        with pytest.raises(ValueError):
            shap_explain.explain(ts.with_static_covariates(None))

        explanation_results = shap_explain.explain(ts)
        assert len(explanation_results.explained_forecasts[1]["price"].columns) == (
            -(min(model.lags["target"])) + model.static_covariates.shape[1]
        )

        model.fit(
            series=self.target_ts_with_multi_component_static_covs,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        assert len(explanation_results.feature_values[1]) == 2
        for comp in self.target_ts_with_multi_component_static_covs.components:
            comps_out = explanation_results.explained_forecasts[1][comp].columns
            assert len(comps_out) == (
                -(min(model.lags["target"])) * model.input_dim["target"]
                + model.input_dim["target"] * model.static_covariates.shape[1]
            )
            assert comps_out[-4:].tolist() == [
                "type_statcov_target_price",
                "type_statcov_target_power",
                "state_statcov_target_price",
                "state_statcov_target_power",
            ]

    def test_shapley_multiple_series_with_different_static_covs(self):
        model_cls = LightGBMModel if lgbm_available else XGBModel
        model = model_cls(
            lags=4,
            output_chunk_length=1,
        )
        model.fit(
            series=self.target_ts_multiple_series_with_different_static_covs,
        )
        shap_explain = ShapExplainer(
            model,
            background_series=self.target_ts_multiple_series_with_different_static_covs,
        )
        explanation_results = shap_explain.explain()

        assert len(explanation_results.feature_values) == 2

        # model trained on multiple series will take column names of first series -> even though
        # static covs have different names, the output will show the same names
        for explained_forecast in explanation_results.explained_forecasts:
            comps_out = explained_forecast[1]["price"].columns.tolist()
            assert comps_out[-1] == "type_statcov_target_price"

    def test_shap_regressor_component_specific_lags(self):
        model = LinearRegressionModel(
            lags={"price": [-3, -2], "power": [-1]},
            output_chunk_length=1,
        )
        # multivariate ts as short as possible
        min_ts_length = MIN_BACKGROUND_SAMPLE * np.abs(min(model.lags["target"]))
        ts = linear_timeseries(
            start_value=1,
            end_value=min_ts_length,
            length=min_ts_length,
            column_name="price",
        ).stack(
            linear_timeseries(
                start_value=102,
                end_value=100 + 2 * min_ts_length,
                length=min_ts_length,
                column_name="power",
            )
        )
        model.fit(ts)
        shap_explain = ShapExplainer(model)

        # one column per lag, grouped by components
        expected_columns = [
            "price_target_lag-3",
            "price_target_lag-2",
            "power_target_lag-1",
        ]
        expected_df = pd.DataFrame(
            data=np.stack(
                [np.arange(1, 29), np.arange(3, 31), np.arange(106, 161, 2)], axis=1
            ),
            columns=expected_columns,
        )

        # check that the appropriate lags are extracted
        assert all(shap_explain.explainers.background_X == expected_df)
        assert model.lagged_feature_names == list(expected_df.columns)

        # check that explain() can be called
        explanation_results = shap_explain.explain()
        plt.close()
        for comp in ts.components:
            comps_out = explanation_results.explained_forecasts[1][comp].columns
            assert all(comps_out == expected_columns)
