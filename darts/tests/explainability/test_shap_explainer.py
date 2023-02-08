from datetime import date, timedelta

import numpy as np
import pandas as pd
import shap
import sklearn
from dateutil.relativedelta import relativedelta
from numpy.testing import assert_array_equal
from sklearn.preprocessing import MinMaxScaler

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.explainability.explainability import ExplainabilityResult
from darts.explainability.shap_explainer import ShapExplainer
from darts.models import (
    CatBoostModel,
    ExponentialSmoothing,
    LightGBMModel,
    LinearRegressionModel,
    RegressionModel,
)
from darts.tests.base_test_class import DartsBaseTestClass


class ShapExplainerTestCase(DartsBaseTestClass):

    np.random.seed(42)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    add_encoders = {
        "cyclic": {"past": ["month", "day"]},
        "datetime_attribute": {"future": ["hour", "dayofweek"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "custom": {"past": [lambda idx: (idx.year - 1950) / 50]},
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

    models = []
    models.append(
        LightGBMModel(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=4,
            add_encoders=add_encoders,
        )
    )

    models.append(
        CatBoostModel(
            lags=4,
            lags_past_covariates=[-1, -2, -6],
            lags_future_covariates=[0],
            output_chunk_length=4,
        )
    )
    models.append(
        LinearRegressionModel(
            lags=1,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=2,
        )
    )

    models.append(
        RegressionModel(
            lags=4,
            lags_past_covariates=[-1, -2, -3],
            lags_future_covariates=[0],
            output_chunk_length=2,
            model=sklearn.tree.ExtraTreeRegressor(),
        )
    )

    models.append(
        LinearRegressionModel(
            lags=1,
            output_chunk_length=2,
        )
    )

    def test_creation(self):

        # Model should be fitted first
        with self.assertRaises(ValueError):
            ShapExplainer(
                self.models[0], self.target_ts, self.past_cov_ts, self.fut_cov_ts
            )

        # Model should be a RegressionModel
        m = ExponentialSmoothing()
        m.fit(self.target_ts["price"])
        with self.assertRaises(ValueError):
            ShapExplainer(m)

        m = self.models[0].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        # SHould have the same number of target, past and futures in the respective lists
        with self.assertRaises(ValueError):
            ShapExplainer(
                self.models[0],
                [self.target_ts, self.target_ts],
                self.past_cov_ts,
                self.fut_cov_ts,
            )

        # Missing a future covariate if you choose to use a new background
        with self.assertRaises(ValueError):
            ShapExplainer(
                m, self.target_ts, background_past_covariates=self.past_cov_ts
            )

        # Missing a past covariate if you choose to use a new background
        with self.assertRaises(ValueError):
            ShapExplainer(
                m, self.target_ts, background_future_covariates=self.fut_cov_ts
            )

        # good type of explainers
        shap_explain = ShapExplainer(m)
        self.assertTrue(
            isinstance(shap_explain.explainers.explainers[0][0], shap.explainers.Tree)
        )

        # Linear model - also not a MultiOuputRegressor
        m = self.models[2].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(m)
        self.assertTrue(
            isinstance(shap_explain.explainers.explainers, shap.explainers.Linear)
        )

        # ExtraTreesRegressor - also not a MultiOuputRegressor
        m = self.models[3].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(m)
        self.assertTrue(
            isinstance(shap_explain.explainers.explainers, shap.explainers.Tree)
        )

        # No past or future covariates
        m = self.models[4].fit(
            series=self.target_ts,
        )

        shap_explain = ShapExplainer(m)
        self.assertTrue(
            isinstance(shap_explain.explainers.explainers, shap.explainers.Linear)
        )

        # CatBoost
        m = self.models[1].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )
        shap_explain = ShapExplainer(m)
        self.assertTrue(
            isinstance(shap_explain.explainers.explainers[0][0], shap.explainers.Tree)
        )

        # Bad choice of shap explainer
        with self.assertRaises(ValueError):
            shap_explain = ShapExplainer(m, shap_method="bad_choice")

    def test_explain(self):

        m_0 = self.models[0].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(m_0)

        with self.assertRaises(ValueError):
            # horizon > output_chunk_length
            results = shap_explain.explain(horizons=[1, 5])
            # wrong name
            results = shap_explain.explain(horizons=[1, 2], target_components=["test"])

        results = shap_explain.explain()

        with self.assertRaises(ValueError):
            # wrong horizon
            results.get_explanation(horizon=5, component="price")
            results.get_feature_values(horizon=5, component="price")
            results.get_shap_explanation_object(horizon=5, component="price")
            # wrong component name
            results.get_explanation(horizon=1, component="test")
            results.get_feature_values(horizon=1, component="test")
            results.get_shap_explanation_object(horizon=1, component="test")

        results = shap_explain.explain(horizons=[1, 3], target_components=["power"])
        with self.assertRaises(ValueError):
            # wrong horizon
            results.get_explanation(horizon=2, component="power")
            results.get_feature_values(horizon=2, component="power")
            results.get_shap_explanation_object(horizon=2, component="power")
            # wrong component name
            results.get_explanation(horizon=1, component="test")
            results.get_feature_values(horizon=1, component="test")
            results.get_shap_explanation_object(horizon=1, component="test")

        explanation = results.get_explanation(horizon=1, component="power")
        self.assertEqual(len(explanation), 537)
        feature_vals = results.get_feature_values(horizon=1, component="power")
        self.assertEqual(len(feature_vals), 537)

        # list of foregrounds: encoders have to be corrected first.
        results = shap_explain.explain(
            foreground_series=[self.target_ts, self.target_ts[:100]],
            foreground_past_covariates=[self.past_cov_ts, self.past_cov_ts[:40]],
            foreground_future_covariates=[self.fut_cov_ts, self.fut_cov_ts[:40]],
        )
        ts_res = results.get_explanation(horizon=2, component="power")
        self.assertEqual(len(ts_res), 2)
        feature_vals = results.get_feature_values(horizon=2, component="power")
        self.assertEqual(len(feature_vals), 2)

        # explain with a new foreground, minimum required. We should obtain one
        # timeseries with only one time element
        results = shap_explain.explain(
            foreground_series=self.target_ts[-5:],
            foreground_past_covariates=self.past_cov_ts[-4:],
            foreground_future_covariates=self.fut_cov_ts[-1],
        )

        ts_res = results.get_explanation(horizon=2, component="power")
        self.assertTrue(len(ts_res) == 1)
        self.assertTrue(ts_res.time_index[-1] == pd.Timestamp(2014, 6, 5))
        feature_vals = results.get_feature_values(horizon=2, component="power")
        self.assertTrue(len(feature_vals) == 1)
        self.assertTrue(feature_vals.time_index[-1] == pd.Timestamp(2014, 6, 5))

        with self.assertRaises(ValueError):
            # wrong horizon
            results.get_explanation(horizon=5, component="price")
            results.get_feature_values(horizon=5, component="price")
            # wrong component name
            results.get_explanation(horizon=1, component="test")
            results.get_feature_values(horizon=1, component="test")

        # right instance
        self.assertTrue(isinstance(results, ExplainabilityResult))

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
        self.assertTrue(
            [
                results.get_explanation(i, "price").components.to_list()
                == components_list
                for i in range(1, 5)
            ]
        )

        # No past or future covariates
        m = self.models[4].fit(
            series=self.target_ts,
        )
        shap_explain = ShapExplainer(m)

        self.assertTrue(isinstance(shap_explain.explain(), ExplainabilityResult))

    def test_explain_with_lags_future_covariates_series_of_same_length_as_target(self):
        model = LightGBMModel(
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
            self.assertEqual(
                explanation.end_time(),
                self.target_ts.end_time() - relativedelta(days=2),
            )

    def test_explain_with_lags_future_covariates_series_extending_into_future(self):

        # Constructing future covariates TimeSeries that extends further into the future than the target series
        date_start = date(2012, 12, 12)
        date_end = date(2014, 6, 7)
        days = pd.date_range(date_start, date_end, freq="d")
        fut_cov = np.random.normal(0, 1, len(days)).astype("float32")
        fut_cov_ts = TimeSeries.from_times_and_values(days, fut_cov.reshape(-1, 1))

        model = LightGBMModel(
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
            self.assertEqual(explanation.end_time(), self.target_ts.end_time())

    def test_explain_with_lags_covariates_series_older_timestamps_than_target(self):

        # Constructing covariates TimeSeries with older timestamps than target
        date_start = date(2012, 12, 10)
        date_end = date(2014, 6, 5)
        days = pd.date_range(date_start, date_end, freq="d")
        fut_cov = np.random.normal(0, 1, len(days)).astype("float32")
        fut_cov_ts = TimeSeries.from_times_and_values(days, fut_cov.reshape(-1, 1))
        past_cov = np.random.normal(0, 1, len(days)).astype("float32")
        past_cov_ts = TimeSeries.from_times_and_values(days, past_cov.reshape(-1, 1))

        model = LightGBMModel(
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
            self.assertEqual(explanation.start_time(), self.target_ts.start_time())

    def test_plot(self):

        m_0 = self.models[0].fit(
            series=self.target_ts,
            past_covariates=self.past_cov_ts,
            future_covariates=self.fut_cov_ts,
        )

        shap_explain = ShapExplainer(m_0)

        # We need at least 5 points for force_plot
        with self.assertRaises(ValueError):
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
        self.assertTrue(isinstance(fplot, shap.plots._force.BaseVisualizer))

        # no component name -> multivariate error
        with self.assertRaises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                1,
            )

        # fake component
        with self.assertRaises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                2,
                "fake",
            )

        # horizon 0
        with self.assertRaises(ValueError):
            shap_explain.force_plot_from_ts(
                self.target_ts[100:108],
                self.past_cov_ts[100:108],
                self.fut_cov_ts[100:108],
                0,
                "power",
            )

        # Wrong component name
        with self.assertRaises(ValueError):
            shap_explain.summary_plot(horizons=[1], target_components=["test"])

        # Wrong horizon
        with self.assertRaises(ValueError):
            shap_explain.summary_plot(horizons=[0], target_components=["test"])
        with self.assertRaises(ValueError):
            shap_explain.summary_plot(horizons=[10], target_components=["test"])

        # No past or future covariates
        m = self.models[4].fit(
            series=self.target_ts,
        )

        shap_explain = ShapExplainer(m)
        fplot = shap_explain.force_plot_from_ts(
            foreground_series=self.target_ts[100:105],
            horizon=1,
            target_component="power",
        )
        self.assertTrue(isinstance(fplot, shap.plots._force.BaseVisualizer))

    def test_feature_values_validity(self):
        model = LightGBMModel(
            lags=4,
            output_chunk_length=1,
        )
        model.fit(
            series=self.target_ts,
        )
        shap_explain = ShapExplainer(model)
        explanation_results = shap_explain.explain()
        df = pd.merge(
            self.target_ts.pd_dataframe(),
            explanation_results.get_feature_values(
                horizon=1, component="price"
            ).pd_dataframe(),
            how="left",
            left_index=True,
            right_index=True,
        )
        df[["price_shift_4", "power_shift_4"]] = df[["price", "power"]].shift(4)

        assert_array_equal(
            df[["price_shift_4", "power_shift_4"]].values,
            df[["price_target_lag-4", "power_target_lag-4"]].values,
        )

    def test_shap_explanation_object_validity(self):
        model = LightGBMModel(
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

        self.assertIsInstance(
            explanation_results.get_shap_explanation_object(
                horizon=1, component="power"
            ),
            shap.Explanation,
        )
