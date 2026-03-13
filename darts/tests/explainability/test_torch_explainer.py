import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.datasets import make_regression

from darts.tests.conftest import NF_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.explainability import TorchExplainer
from darts.explainability.torch_explainer import (
    MAX_BACKGROUND_SAMPLE,
    MIN_BACKGROUND_SAMPLE,
    _available_shap_methods,
)
from darts.models import (
    BlockRNNModel,
    DLinearModel,
    NBEATSModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
    TSMixerModel,
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.likelihood_models.torch import (
    CauchyLikelihood,
    GaussianLikelihood,
    QuantileRegression,
    TorchLikelihood,
)

N_PAST_COVARIATES = 3
N_FUTURE_COVARIATES = 2
N_TARGETS = 3

chronos2_local_dir = (
    Path(__file__).parent.parent
    / "models"
    / "forecasting"
    / "artefacts"
    / "chronos2"
    / "tiny_chronos2"
).absolute()


def encode_year(idx):
    return (idx.year - 1950) / 50


ADD_ENCODERS = {
    "cyclic": {"future": ["month"]},
    "custom": {"past": [encode_year]},
    "transformer": Scaler(),
    "tz": "CET",
}

ALL_MODELS = [
    (RNNModel, {"model": "LSTM"}),
    (BlockRNNModel, {"add_encoders": ADD_ENCODERS}),
    (NBEATSModel, {"num_stacks": 2, "num_layers": 2, "layer_widths": 16}),
    (NHiTSModel, {"layer_widths": 16}),
    (TiDEModel, {"hidden_size": 32, "add_encoders": ADD_ENCODERS}),
    (DLinearModel, {"add_encoders": ADD_ENCODERS}),
    (
        TSMixerModel,
        {"hidden_size": 8, "ff_size": 8, "num_blocks": 1, "add_encoders": ADD_ENCODERS},
    ),
    # (Chronos2Model, {"local_dir": chronos2_local_dir, "batch_size": 4}),
]
SHAP_METHODS = [
    "kernel",
    "sampling",
    "partition",
    "permutation",
]
LIKELIHOODS = [
    (GaussianLikelihood, None),
    (CauchyLikelihood, None),
    (QuantileRegression, {"quantiles": [0.1, 0.5, 0.9]}),
]


if NF_AVAILABLE:
    from darts.models import NeuralForecastModel

    ALL_MODELS += [
        (
            NeuralForecastModel,
            {"model": "MLPMultivariate", "model_kwargs": {"hidden_size": 16}},
        ),
        (
            NeuralForecastModel,
            {
                "model": "PatchTST",
                "model_kwargs": {
                    "encoder_layers": 1,
                    "patch_len": 3,
                    "stride": 3,
                    "n_heads": 4,
                    "hidden_size": 16,
                    "linear_hidden_size": 32,
                },
            },
        ),
    ]


kwargs = {
    "n_epochs": 1,
    **tfm_kwargs,
}


class TestSKLearnExplainer:
    # set random seed
    np.random.seed(42)

    X, Y = make_regression(
        n_samples=100,
        n_features=N_PAST_COVARIATES + N_FUTURE_COVARIATES,
        n_informative=3,
        n_targets=N_TARGETS,
        noise=1,
        random_state=42,
    )
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    multivariate_series = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=80, freq="D"),
        values=Y[:80],
        columns=[f"T_{i}" for i in range(N_TARGETS)],
    ).with_static_covariates(pd.DataFrame({"S_0": [1] * N_TARGETS}))
    univariate_series = multivariate_series.univariate_component(0)
    past_covariates = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=80, freq="D"),
        values=X[:80, :N_PAST_COVARIATES],
        columns=[f"P_{i}" for i in range(N_PAST_COVARIATES)],
    )
    future_covariates = TimeSeries.from_times_and_values(
        times=pd.date_range("20200101", periods=97, freq="D"),
        # shift forward by 3 so that future covariate may influence target at time t
        values=X[3:, N_PAST_COVARIATES:],
        columns=[f"F_{i}" for i in range(N_FUTURE_COVARIATES)],
    )
    multiple_multivariate_series = [
        multivariate_series,
        multivariate_series + 1,
    ]

    @pytest.mark.parametrize("model_cls, model_kwargs", ALL_MODELS)
    def test_creation(
        self,
        model_cls: type[TorchForecastingModel],
        model_kwargs: dict | None,
        tmpdir_fn,
    ):
        model = model_cls(
            input_chunk_length=10,
            output_chunk_length=5,
            **(model_kwargs or {}),
            **kwargs,
        )

        # cannot create explainer with unfitted model
        with pytest.raises(ValueError, match="must be fitted before instantiating."):
            explainer = TorchExplainer(model)

        # fit the model
        model.fit(series=self.multivariate_series)

        # create explainer with fitted model
        explainer = TorchExplainer(model)

        # check explainer attributes
        assert explainer.model == model
        assert explainer.n == model.output_chunk_length

        # save and load the model to check explainer works with loaded models
        save_path = os.path.join(tmpdir_fn, "model.pt")
        model.save(save_path)
        loaded_model = model_cls.load(save_path)
        loaded_explainer = TorchExplainer(loaded_model)

        assert loaded_explainer.model == loaded_model
        assert loaded_explainer.n == loaded_model.output_chunk_length

    @pytest.mark.parametrize("model_cls, model_kwargs", ALL_MODELS)
    def test_creation_multiple_series(
        self,
        model_cls: type[TorchForecastingModel],
        model_kwargs: dict | None,
        tmpdir_fn,
    ):
        model = model_cls(
            input_chunk_length=10,
            output_chunk_length=5,
            **(model_kwargs or {}),
            **kwargs,
        )

        # cannot create explainer with unfitted model
        with pytest.raises(ValueError, match="must be fitted before instantiating."):
            explainer = TorchExplainer(model)

        # fit the model
        model.fit(series=self.multiple_multivariate_series)

        # create explainer with multiple series but no background raises error
        with pytest.raises(ValueError, match="`background_series` must be provided"):
            explainer = TorchExplainer(model)

        # create explainer with multiple series and background
        explainer = TorchExplainer(
            model, background_series=self.multiple_multivariate_series
        )

        # check explainer attributes
        assert explainer.model == model
        assert explainer.n == model.output_chunk_length

        # save and load the model to check explainer works with loaded models
        save_path = os.path.join(tmpdir_fn, "model.pt")
        model.save(save_path)
        loaded_model = model_cls.load(save_path)
        loaded_explainer = TorchExplainer(
            model, background_series=self.multiple_multivariate_series
        )

        assert loaded_explainer.n == loaded_model.output_chunk_length

    @pytest.mark.parametrize("model_cls, model_kwargs", ALL_MODELS)
    def test_explain(
        self,
        model_cls: type[TorchForecastingModel],
        model_kwargs: dict | None,
        tmpdir_fn,
    ):
        model = model_cls(
            input_chunk_length=7,
            output_chunk_length=3,  # RNN model ignores output_chunk_length which is set to 1 internally
            **(model_kwargs or {}),
            **kwargs,
        )

        # prepare training data
        series = self.multivariate_series
        past_covariates = (
            self.past_covariates if model.supports_past_covariates else None
        )
        future_covariates = (
            self.future_covariates if model.supports_future_covariates else None
        )

        # prepare background data
        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = (
            future_covariates.split_before(background_series.start_time())
            if future_covariates is not None
            else (None, None)
        )

        # prepare foreground data (past/future covariates can be reused)
        foreground_series = series[-10:]

        # fit the model
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # create explainer with fitted model
        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )

        # explain the foreground
        results = explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        valid_horizon = 1 if isinstance(model, RNNModel) else 2
        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        if past_covariates is not None:
            components.update({
                f"{name}_pastcov_lag-{lag + 1}"
                for name in past_covariates.columns
                for lag in range(model.input_chunk_length)
            })
        if future_covariates is not None:
            components.update({
                f"{name}_futcov_lag{lag}"
                for name in future_covariates.columns
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
            })
        if (
            model.supports_static_covariates
            and foreground_series.static_covariates is not None
        ):
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        if model_kwargs is not None and "add_encoders" in model_kwargs:
            components.update({
                f"{prefix}_lag{lag}"
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
                for prefix in [
                    "darts_enc_fc_cyc_month_cos_futcov",
                    "darts_enc_fc_cyc_month_sin_futcov",
                ]
            })
            components.update({
                f"{prefix}_lag-{lag + 1}"
                for lag in range(model.input_chunk_length)
                for prefix in [
                    "darts_enc_pc_cus_custom_pastcov",
                ]
            })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_explanation(horizon=4, component="T_0")

        # check explanation is returned for valid horizon, component, and input features
        explanation = results.get_explanation(horizon=valid_horizon, component="T_0")
        assert isinstance(explanation, TimeSeries)
        assert (
            explanation.n_timesteps
            == foreground_series.n_timesteps - model.input_chunk_length + 1
        )
        assert set(explanation.components) == components

        # check explanation values are finite
        assert np.isfinite(explanation.values()).all()
        # check explanation values are additive, i.e., sum of SHAP values across all features equals the difference
        # between the prediction and the base value
        # base values should be approximately equal across all time steps since the same background is used for all
        # predictions
        pred = model.historical_forecasts(
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=valid_horizon,
            last_points_only=True,
            overlap_end=True,
            retrain=False,
        )
        assert isinstance(pred, TimeSeries)
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_0"].values().ravel() - shap_values_sum
        # assert all base values are approximately equal
        assert np.allclose(base_values, base_values[0], rtol=1e-5, atol=1e-8)

        # invalid component or horizon raises error for feature values as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_feature_values(horizon=4, component="T_0")

        # check feature values are returned for valid horizon and component
        feature_values = results.get_feature_values(
            horizon=valid_horizon,
            component="T_1",
        )
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == explanation.n_timesteps
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        # invalid component or horizon raises error for shap explanation object as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_shap_explanation_object(horizon=4, component="T_0")

        # check shap explanation object is returned for valid horizon and component
        shap_explanation_object = results.get_shap_explanation_object(
            horizon=valid_horizon,
            component="T_1",
        )
        explanation = results.get_explanation(horizon=valid_horizon, component="T_1")
        assert isinstance(explanation, TimeSeries)
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data,
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

        # save and load the model to check explainer works with loaded models
        save_path = os.path.join(tmpdir_fn, "model.pt")
        model.save(save_path)
        loaded_model = model_cls.load(save_path)
        loaded_explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )
        assert loaded_explainer.n == loaded_model.output_chunk_length

        loaded_results = loaded_explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )
        loaded_explanation = loaded_results.get_explanation(
            horizon=valid_horizon, component="T_1"
        )
        loaded_feature_values = loaded_results.get_feature_values(
            horizon=valid_horizon,
            component="T_1",
        )
        loaded_shap_explanation_object = loaded_results.get_shap_explanation_object(
            horizon=valid_horizon,
            component="T_1",
        )
        assert isinstance(loaded_explanation, TimeSeries)
        assert loaded_explanation.n_timesteps == explanation.n_timesteps
        assert set(loaded_explanation.components) == components
        assert np.isfinite(loaded_explanation.values()).all()
        assert isinstance(loaded_feature_values, TimeSeries)
        assert loaded_feature_values.n_timesteps == feature_values.n_timesteps
        assert set(loaded_feature_values.components) == components
        assert np.isfinite(loaded_feature_values.values()).all()
        assert isinstance(loaded_shap_explanation_object, shap.Explanation)

        np.testing.assert_array_equal(
            loaded_shap_explanation_object.values,
            loaded_explanation.values(),
        )
        np.testing.assert_array_equal(
            loaded_shap_explanation_object.data,
            loaded_feature_values.values(),
        )

        # unfortunately, shap and base values are not exactly the same across
        # runs with the same background data, even with fixed random seed,
        # due to some non-determinism in the SHAP implementation.

    @pytest.mark.parametrize("uses_past_covariates", [True, False])
    @pytest.mark.parametrize("uses_future_covariates", [True, False])
    @pytest.mark.parametrize("uses_static_covariates", [True, False])
    def test_explain_without_foreground(
        self,
        uses_past_covariates: bool,
        uses_future_covariates: bool,
        uses_static_covariates: bool,
    ):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=4,
            use_static_covariates=uses_static_covariates,
            **(model_kwargs or {}),
            **kwargs,
        )

        # prepare training data
        series = self.multivariate_series
        past_covariates = self.past_covariates if uses_past_covariates else None
        future_covariates = self.future_covariates if uses_future_covariates else None

        # prepare background data
        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = (
            future_covariates.split_before(background_series.start_time())
            if future_covariates is not None
            else (None, None)
        )

        # fit the model
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # create explainer with fitted model
        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )

        # explain the foreground
        results = explainer.explain()

        valid_horizon = 1 if isinstance(model, RNNModel) else 2
        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in background_series.columns
            for lag in range(model.input_chunk_length)
        }
        if past_covariates is not None:
            components.update({
                f"{name}_pastcov_lag-{lag + 1}"
                for name in past_covariates.columns
                for lag in range(model.input_chunk_length)
            })
        if future_covariates is not None:
            components.update({
                f"{name}_futcov_lag{lag}"
                for name in future_covariates.columns
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
            })
        if uses_static_covariates and background_series.static_covariates is not None:
            components.update({
                f"{name}_statcov_target_{target}"
                for name in background_series.static_covariates.columns
                for target in background_series.columns
            })
        if model_kwargs is not None and "add_encoders" in model_kwargs:
            components.update({
                f"{prefix}_lag{lag}"
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
                for prefix in [
                    "darts_enc_fc_cyc_month_cos_futcov",
                    "darts_enc_fc_cyc_month_sin_futcov",
                ]
            })
            components.update({
                f"{prefix}_lag-{lag + 1}"
                for lag in range(model.input_chunk_length)
                for prefix in [
                    "darts_enc_pc_cus_custom_pastcov",
                ]
            })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 5 is not available."):
            results.get_explanation(horizon=5, component="T_0")

        # check explanation is returned for valid horizon, component, and input features
        explanation = results.get_explanation(horizon=valid_horizon, component="T_0")
        assert isinstance(explanation, TimeSeries)
        # background series would use `generate_fit_encodings` rather than `generate_fit_predict_encodings`
        # thus, the length is shorter by `model.output_chunk_length` compared to using the foreground
        assert (
            explanation.n_timesteps
            == background_series.n_timesteps
            - model.input_chunk_length
            - model.output_chunk_length
            + 1
        )
        assert set(explanation.components) == components

        # check explanation values are finite
        assert np.isfinite(explanation.values()).all()
        # check explanation values are additive, i.e., sum of SHAP values across all features equals the difference
        # between the prediction and the base value
        # base values should be approximately equal across all time steps since the same background is used for all
        # predictions
        pred = model.historical_forecasts(
            series=background_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=valid_horizon,
            last_points_only=True,
            overlap_end=True,
            retrain=False,
        )
        assert isinstance(pred, TimeSeries)
        pred = pred[: len(explanation)]
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_0"].values().ravel() - shap_values_sum
        # assert all base values are approximately equal
        np.testing.assert_allclose(base_values, base_values[0], rtol=1e-5, atol=1e-8)

        # invalid component or horizon raises error for feature values as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 5 is not available."):
            results.get_feature_values(horizon=5, component="T_0")

        # check feature values are returned for valid horizon and component
        feature_values = results.get_feature_values(
            horizon=valid_horizon,
            component="T_1",
        )
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == explanation.n_timesteps
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        # invalid component or horizon raises error for shap explanation object as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 5 is not available."):
            results.get_shap_explanation_object(horizon=5, component="T_0")

        # check shap explanation object is returned for valid horizon and component
        shap_explanation_object = results.get_shap_explanation_object(
            horizon=valid_horizon,
            component="T_1",
        )
        explanation = results.get_explanation(horizon=valid_horizon, component="T_1")
        assert isinstance(explanation, TimeSeries)
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data,
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.parametrize("shap_method", SHAP_METHODS)
    def test_explain_shap_methods(
        self,
        shap_method: str,
    ):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=8,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        # prepare training data
        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        # prepare background data
        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = (
            future_covariates.split_before(background_series.start_time())
            if future_covariates is not None
            else (None, None)
        )

        # prepare foreground data (past/future covariates can be reused)
        foreground_series = series[-10:]

        # fit the model
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # create explainer with fitted model
        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            shap_method=shap_method,
        )

        # explain the foreground
        results = explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        valid_horizon = 1 if isinstance(model, RNNModel) else 2
        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        if past_covariates is not None:
            components.update({
                f"{name}_pastcov_lag-{lag + 1}"
                for name in past_covariates.columns
                for lag in range(model.input_chunk_length)
            })
        if future_covariates is not None:
            components.update({
                f"{name}_futcov_lag{lag}"
                for name in future_covariates.columns
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
            })
        if (
            model.supports_static_covariates
            and foreground_series.static_covariates is not None
        ):
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        if model_kwargs is not None and "add_encoders" in model_kwargs:
            components.update({
                f"{prefix}_lag{lag}"
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
                for prefix in [
                    "darts_enc_fc_cyc_month_cos_futcov",
                    "darts_enc_fc_cyc_month_sin_futcov",
                ]
            })
            components.update({
                f"{prefix}_lag-{lag + 1}"
                for lag in range(model.input_chunk_length)
                for prefix in [
                    "darts_enc_pc_cus_custom_pastcov",
                ]
            })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_explanation(horizon=4, component="T_0")

        # check explanation is returned for valid horizon, component, and input features
        explanation = results.get_explanation(horizon=valid_horizon, component="T_0")
        assert isinstance(explanation, TimeSeries)
        assert (
            explanation.n_timesteps
            == foreground_series.n_timesteps - model.input_chunk_length + 1
        )
        assert set(explanation.components) == components

        # check explanation values are finite
        assert np.isfinite(explanation.values()).all()
        # check explanation values are additive, i.e., sum of SHAP values across all features equals the difference
        # between the prediction and the base value
        # base values should be approximately equal across all time steps since the same background is used for all
        # predictions
        pred = model.historical_forecasts(
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=valid_horizon,
            last_points_only=True,
            overlap_end=True,
            retrain=False,
        )
        assert isinstance(pred, TimeSeries)
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_0"].values().ravel() - shap_values_sum
        # assert all base values are approximately equal
        assert np.allclose(base_values, base_values[0], rtol=1e-5, atol=1e-8)

        # invalid component or horizon raises error for feature values as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_feature_values(horizon=4, component="T_0")

        # check feature values are returned for valid horizon and component
        feature_values = results.get_feature_values(
            horizon=valid_horizon,
            component="T_1",
        )
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == explanation.n_timesteps
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        # invalid component or horizon raises error for shap explanation object as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_shap_explanation_object(horizon=4, component="T_0")

        # check shap explanation object is returned for valid horizon and component
        shap_explanation_object = results.get_shap_explanation_object(
            horizon=valid_horizon,
            component="T_1",
        )
        explanation = results.get_explanation(horizon=valid_horizon, component="T_1")
        assert isinstance(explanation, TimeSeries)
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data,
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.parametrize("likelihood_cls, likelihood_kwargs", LIKELIHOODS)
    def test_explain_probabilistic_model(
        self,
        likelihood_cls: type[TorchLikelihood],
        likelihood_kwargs: dict | None,
    ):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=5,
            output_chunk_length=2,
            likelihood=likelihood_cls(**(likelihood_kwargs or {})),
            **(model_kwargs or {}),
            **kwargs,
        )

        # prepare training data
        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        # prepare background data
        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = (
            future_covariates.split_before(background_series.start_time())
            if future_covariates is not None
            else (None, None)
        )

        # prepare foreground data (past/future covariates can be reused)
        foreground_series = series[-10:]

        # fit the model
        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # create explainer with fitted model
        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )

        # explain the foreground
        results = explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        assert model.likelihood is not None
        likelihood_components = model.likelihood.component_names(series)
        assert set(explainer.explainer.target_components_likelihood) == set(
            likelihood_components
        )

        # probabilistic models should have explanations for all components of the likelihood,
        # but not for pre-likelihood components
        with pytest.raises(ValueError, match='Component "T_0" is not available'):
            results.get_explanation(horizon=1, component="T_0")

        valid_horizon = 1 if isinstance(model, RNNModel) else 2
        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        if past_covariates is not None:
            components.update({
                f"{name}_pastcov_lag-{lag + 1}"
                for name in past_covariates.columns
                for lag in range(model.input_chunk_length)
            })
        if future_covariates is not None:
            components.update({
                f"{name}_futcov_lag{lag}"
                for name in future_covariates.columns
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
            })
        if (
            model.supports_static_covariates
            and foreground_series.static_covariates is not None
        ):
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        if model_kwargs is not None and "add_encoders" in model_kwargs:
            components.update({
                f"{prefix}_lag{lag}"
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
                for prefix in [
                    "darts_enc_fc_cyc_month_cos_futcov",
                    "darts_enc_fc_cyc_month_sin_futcov",
                ]
            })
            components.update({
                f"{prefix}_lag-{lag + 1}"
                for lag in range(model.input_chunk_length)
                for prefix in [
                    "darts_enc_pc_cus_custom_pastcov",
                ]
            })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_explanation(horizon=4, component=likelihood_components[0])

        # check explanation is returned for valid horizon, component, and input features
        explanation = results.get_explanation(
            horizon=valid_horizon, component=likelihood_components[0]
        )
        assert isinstance(explanation, TimeSeries)
        assert (
            explanation.n_timesteps
            == foreground_series.n_timesteps - model.input_chunk_length + 1
        )
        assert set(explanation.components) == components

        # check explanation values are finite
        assert np.isfinite(explanation.values()).all()
        # check explanation values are additive, i.e., sum of SHAP values across all features equals the difference
        # between the prediction and the base value
        # base values should be approximately equal across all time steps since the same background is used for all
        # predictions
        pred = model.historical_forecasts(
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=valid_horizon,
            last_points_only=True,
            overlap_end=True,
            retrain=False,
            predict_likelihood_parameters=True,  # output likelihood parameters directly
        )
        assert isinstance(pred, TimeSeries)
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred[likelihood_components[0]].values().ravel() - shap_values_sum
        # assert all base values are approximately equal
        assert np.allclose(base_values, base_values[0], rtol=1e-5, atol=1e-8)

        # invalid component or horizon raises error for feature values as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_feature_values(horizon=4, component=likelihood_components[0])

        # check feature values are returned for valid horizon and component
        feature_values = results.get_feature_values(
            horizon=valid_horizon,
            component=likelihood_components[0],
        )
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == explanation.n_timesteps
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        # invalid component or horizon raises error for shap explanation object as well
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=4, component=None)
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(horizon=2, component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(horizon=valid_horizon, component="T_11")
        with pytest.raises(ValueError, match="Horizon 4 is not available."):
            results.get_shap_explanation_object(
                horizon=4, component=likelihood_components[-1]
            )

        # check shap explanation object is returned for valid horizon and component
        shap_explanation_object = results.get_shap_explanation_object(
            horizon=valid_horizon,
            component=likelihood_components[-1],
        )
        explanation = results.get_explanation(
            horizon=valid_horizon, component=likelihood_components[-1]
        )
        assert isinstance(explanation, TimeSeries)
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data,
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = pred[likelihood_components[-1]].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-4,
            atol=1e-8,
        )

    def test_explain_multiple_series(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multiple_multivariate_series
        past_covariates = [self.past_covariates, self.past_covariates + 1]
        future_covariates = [self.future_covariates, self.future_covariates + 1]

        background_series = [ts[-20:] for ts in series]
        background_past_covariates = [ts[-20:] for ts in past_covariates]
        background_future_covariates = [
            future_cov.split_before(background.start_time())[1]
            for future_cov, background in zip(future_covariates, background_series)
        ]

        foreground_series = [ts[-10:] for ts in series]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )
        results = explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        explanation = results.get_explanation(horizon=2, component="T_0")
        feature_values = results.get_feature_values(horizon=2, component="T_0")
        shap_explanation_object = results.get_shap_explanation_object(
            horizon=2, component="T_0"
        )
        assert isinstance(explanation, list)
        assert isinstance(feature_values, list)
        assert isinstance(shap_explanation_object, list)
        assert len(explanation) == len(series)
        assert len(feature_values) == len(series)
        assert len(shap_explanation_object) == len(series)

        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series[0].columns
            for lag in range(model.input_chunk_length)
        }
        if past_covariates is not None:
            components.update({
                f"{name}_pastcov_lag-{lag + 1}"
                for name in past_covariates[0].columns
                for lag in range(model.input_chunk_length)
            })
        if future_covariates is not None:
            components.update({
                f"{name}_futcov_lag{lag}"
                for name in future_covariates[0].columns
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
            })
        if (
            model.supports_static_covariates
            and foreground_series[0].static_covariates is not None
        ):
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series[0].static_covariates.columns
                for target in foreground_series[0].columns
            })
        if model_kwargs is not None and "add_encoders" in model_kwargs:
            components.update({
                f"{prefix}_lag{lag}"
                for lag in range(-model.input_chunk_length, model.output_chunk_length)
                for prefix in [
                    "darts_enc_fc_cyc_month_cos_futcov",
                    "darts_enc_fc_cyc_month_sin_futcov",
                ]
            })
            components.update({
                f"{prefix}_lag-{lag + 1}"
                for lag in range(model.input_chunk_length)
                for prefix in [
                    "darts_enc_pc_cus_custom_pastcov",
                ]
            })

        for i in range(len(series)):
            assert isinstance(explanation[i], TimeSeries)
            assert (
                explanation[i].n_timesteps
                == foreground_series[i].n_timesteps - model.input_chunk_length + 1
            )
            assert np.isfinite(explanation[i].values()).all()
            assert set(explanation[i].components) == components

            assert isinstance(feature_values[i], TimeSeries)
            assert feature_values[i].n_timesteps == explanation[i].n_timesteps
            assert np.isfinite(feature_values[i].values()).all()
            assert set(feature_values[i].components) == components

            assert isinstance(shap_explanation_object[i], shap.Explanation)
            np.testing.assert_array_equal(
                shap_explanation_object[i].values,
                explanation[i].values(),
            )
            np.testing.assert_array_equal(
                shap_explanation_object[i].data,
                feature_values[i].values(),
            )

    def test_explain_single(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )
        results = explainer.explain_single(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        components.update({
            f"{name}_pastcov_lag-{lag + 1}"
            for name in past_covariates.columns
            for lag in range(model.input_chunk_length)
        })
        components.update({
            f"{name}_futcov_lag{lag}"
            for name in future_covariates.columns
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
        })
        if foreground_series.static_covariates is not None:
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        components.update({
            f"{prefix}_lag{lag}"
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
            for prefix in [
                "darts_enc_fc_cyc_month_cos_futcov",
                "darts_enc_fc_cyc_month_sin_futcov",
            ]
        })
        components.update({
            f"{prefix}_lag-{lag + 1}"
            for lag in range(model.input_chunk_length)
            for prefix in ["darts_enc_pc_cus_custom_pastcov"]
        })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(component="T_11")

        explanation = results.get_explanation(component="T_0")
        assert isinstance(explanation, TimeSeries)
        assert explanation.n_timesteps == model.output_chunk_length
        assert set(explanation.components) == components
        assert np.isfinite(explanation.values()).all()

        prediction = model.predict(
            n=model.output_chunk_length,
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        assert isinstance(prediction, TimeSeries)
        assert prediction.n_timesteps == explanation.n_timesteps

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(component="T_11")

        feature_values = results.get_feature_values(component="T_1")
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == 1
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(component="T_11")

        shap_explanation_object = results.get_shap_explanation_object(component="T_1")
        explanation = results.get_explanation(component="T_1")
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data[:1],
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = prediction["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_explain_single_without_foreground(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=4,
            output_chunk_length=5,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )
        results = explainer.explain_single()

        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in background_series.columns
            for lag in range(model.input_chunk_length)
        }
        components.update({
            f"{name}_pastcov_lag-{lag + 1}"
            for name in background_past_covariates.columns
            for lag in range(model.input_chunk_length)
        })
        components.update({
            f"{name}_futcov_lag{lag}"
            for name in background_future_covariates.columns
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
        })
        if background_series.static_covariates is not None:
            components.update({
                f"{name}_statcov_target_{target}"
                for name in background_series.static_covariates.columns
                for target in background_series.columns
            })
        components.update({
            f"{prefix}_lag{lag}"
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
            for prefix in [
                "darts_enc_fc_cyc_month_cos_futcov",
                "darts_enc_fc_cyc_month_sin_futcov",
            ]
        })
        components.update({
            f"{prefix}_lag-{lag + 1}"
            for lag in range(model.input_chunk_length)
            for prefix in ["darts_enc_pc_cus_custom_pastcov"]
        })

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(component="T_11")

        explanation = results.get_explanation(component="T_0")
        assert isinstance(explanation, TimeSeries)
        assert explanation.n_timesteps == model.output_chunk_length
        assert set(explanation.components) == components
        assert np.isfinite(explanation.values()).all()

        # for single explanation without foreground, the equivalent prediction
        # is backshifted by `output_chunk_length` since the explanation is based
        # on the background data where future covariates are only available up
        # to the end of the background series.
        prediction = model.predict(
            n=model.output_chunk_length,
            series=background_series[: -model.output_chunk_length],
            past_covariates=background_past_covariates[: -model.output_chunk_length],
            future_covariates=background_future_covariates,
        )
        assert isinstance(prediction, TimeSeries)
        assert prediction.n_timesteps == explanation.n_timesteps

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(component="T_11")

        feature_values = results.get_feature_values(component="T_1")
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == 1
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(component="T_11")

        shap_explanation_object = results.get_shap_explanation_object(component="T_1")
        explanation = results.get_explanation(component="T_1")
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data[:1],
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = prediction["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.parametrize("shap_method", SHAP_METHODS)
    def test_explain_single_shap_methods(
        self,
        shap_method: str,
    ):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            shap_method=shap_method,
        )
        results = explainer.explain_single(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
            target_components=["T_0", "T_1"],
        )

        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        components.update({
            f"{name}_pastcov_lag-{lag + 1}"
            for name in past_covariates.columns
            for lag in range(model.input_chunk_length)
        })
        components.update({
            f"{name}_futcov_lag{lag}"
            for name in future_covariates.columns
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
        })
        if foreground_series.static_covariates is not None:
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        components.update({
            f"{prefix}_lag{lag}"
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
            for prefix in [
                "darts_enc_fc_cyc_month_cos_futcov",
                "darts_enc_fc_cyc_month_sin_futcov",
            ]
        })
        components.update({
            f"{prefix}_lag-{lag + 1}"
            for lag in range(model.input_chunk_length)
            for prefix in ["darts_enc_pc_cus_custom_pastcov"]
        })

        explanation = results.get_explanation(component="T_0")
        assert isinstance(explanation, TimeSeries)
        assert explanation.n_timesteps == model.output_chunk_length
        assert set(explanation.components) == components
        assert np.isfinite(explanation.values()).all()

        prediction = model.predict(
            n=model.output_chunk_length,
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        assert isinstance(prediction, TimeSeries)
        assert prediction.n_timesteps == explanation.n_timesteps

        feature_values = results.get_feature_values(component="T_1")
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == 1
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        shap_explanation_object = results.get_shap_explanation_object(component="T_1")
        explanation = results.get_explanation(component="T_1")
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data[:1],
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = prediction["T_1"].values().ravel() - shap_values_sum
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-5,
            atol=1e-8,
        )

    @pytest.mark.parametrize("likelihood_cls, likelihood_kwargs", LIKELIHOODS)
    def test_explain_single_probabilistic_model(
        self,
        likelihood_cls: type[TorchLikelihood],
        likelihood_kwargs: dict | None,
    ):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=5,
            output_chunk_length=2,
            likelihood=likelihood_cls(**(likelihood_kwargs or {})),
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = (
            past_covariates[-20:] if past_covariates is not None else None
        )
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
        )
        results = explainer.explain_single(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        assert model.likelihood is not None
        likelihood_components = model.likelihood.component_names(series)
        assert set(explainer.explainer.target_components_likelihood) == set(
            likelihood_components
        )

        with pytest.raises(ValueError, match='Component "T_0" is not available'):
            results.get_explanation(component="T_0")
        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_explanation(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_explanation(component="T_11")

        components = {
            f"{name}_target_lag-{lag + 1}"
            for name in foreground_series.columns
            for lag in range(model.input_chunk_length)
        }
        components.update({
            f"{name}_pastcov_lag-{lag + 1}"
            for name in past_covariates.columns
            for lag in range(model.input_chunk_length)
        })
        components.update({
            f"{name}_futcov_lag{lag}"
            for name in future_covariates.columns
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
        })
        if foreground_series.static_covariates is not None:
            components.update({
                f"{name}_statcov_target_{target}"
                for name in foreground_series.static_covariates.columns
                for target in foreground_series.columns
            })
        components.update({
            f"{prefix}_lag{lag}"
            for lag in range(-model.input_chunk_length, model.output_chunk_length)
            for prefix in [
                "darts_enc_fc_cyc_month_cos_futcov",
                "darts_enc_fc_cyc_month_sin_futcov",
            ]
        })
        components.update({
            f"{prefix}_lag-{lag + 1}"
            for lag in range(model.input_chunk_length)
            for prefix in ["darts_enc_pc_cus_custom_pastcov"]
        })

        explanation = results.get_explanation(component=likelihood_components[0])
        assert isinstance(explanation, TimeSeries)
        assert explanation.n_timesteps == model.output_chunk_length
        assert set(explanation.components) == components
        assert np.isfinite(explanation.values()).all()

        prediction = model.predict(
            n=model.output_chunk_length,
            series=foreground_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            predict_likelihood_parameters=True,
        )
        assert isinstance(prediction, TimeSeries)
        assert prediction.n_timesteps == explanation.n_timesteps

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_feature_values(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_feature_values(component="T_11")

        feature_values = results.get_feature_values(component=likelihood_components[0])
        assert isinstance(feature_values, TimeSeries)
        assert feature_values.n_timesteps == 1
        assert set(feature_values.components) == components
        assert np.isfinite(feature_values.values()).all()

        with pytest.raises(ValueError, match="component parameter is required"):
            results.get_shap_explanation_object(component=None)
        with pytest.raises(ValueError, match='Component "T_11" is not available'):
            results.get_shap_explanation_object(component="T_11")

        shap_explanation_object = results.get_shap_explanation_object(
            component=likelihood_components[-1]
        )
        explanation = results.get_explanation(component=likelihood_components[-1])
        assert isinstance(explanation, TimeSeries)
        assert isinstance(shap_explanation_object, shap.Explanation)
        np.testing.assert_array_equal(
            shap_explanation_object.values,
            explanation.values(),
        )
        np.testing.assert_array_equal(
            shap_explanation_object.data[:1],
            feature_values.values(),
        )
        shap_values_sum = explanation.values().sum(axis=1)
        base_values = (
            prediction[likelihood_components[-1]].values().ravel() - shap_values_sum
        )
        # sometimes probabilistic models can have more variability in the base values due to the nature of the
        # likelihood outputs, so we use a slightly looser tolerance here
        np.testing.assert_allclose(
            shap_explanation_object.base_values,
            base_values,
            rtol=1e-3,
            atol=1e-8,
        )

    def test_summary_plot(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = past_covariates[-20:]
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            background_num_samples=10,
        )

        dict_shap_values = explainer.summary_plot(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
            num_samples=2,
            show=False,
        )
        assert len(dict_shap_values) == model.output_chunk_length
        for horizon in range(1, model.output_chunk_length + 1):
            assert len(dict_shap_values[horizon]) == series.width
            for component in series.components:
                assert isinstance(
                    dict_shap_values[horizon][component], shap.Explanation
                )

        with pytest.raises(ValueError, match="Invalid `target_components`"):
            explainer.summary_plot(horizons=[1], target_components=["test"])
        with pytest.raises(ValueError, match=r"All `horizons` must be `>=1`\."):
            explainer.summary_plot(horizons=[0], target_components=["T_0"])
        with pytest.raises(
            ValueError,
            match=r"At least one of the `horizons` is larger than `output_chunk_length`\.",
        ):
            explainer.summary_plot(
                horizons=[model.output_chunk_length + 1],
                target_components=["T_0"],
            )

    def test_force_plot(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = past_covariates[-20:]
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            background_num_samples=10,
        )

        force_plot = explainer.force_plot(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
            horizon=2,
            target_component="T_0",
        )
        assert isinstance(force_plot, shap.plots._force.BaseVisualizer)

        with pytest.raises(ValueError, match=r"`target_component` is required"):
            explainer.force_plot(
                foreground_series=foreground_series,
                foreground_past_covariates=past_covariates,
                foreground_future_covariates=future_covariates,
                horizon=1,
            )
        with pytest.raises(ValueError, match="Invalid `target_components`"):
            explainer.force_plot(
                foreground_series=foreground_series,
                foreground_past_covariates=past_covariates,
                foreground_future_covariates=future_covariates,
                horizon=1,
                target_component="test",
            )
        with pytest.raises(ValueError, match=r"All `horizons` must be `>=1`\."):
            explainer.force_plot(
                foreground_series=foreground_series,
                foreground_past_covariates=past_covariates,
                foreground_future_covariates=future_covariates,
                horizon=0,
                target_component="T_0",
            )
        with pytest.raises(
            ValueError,
            match=r"At least one of the `horizons` is larger than `output_chunk_length`\.",
        ):
            explainer.force_plot(
                foreground_series=foreground_series,
                foreground_past_covariates=past_covariates,
                foreground_future_covariates=future_covariates,
                horizon=model.output_chunk_length + 1,
                target_component="T_0",
            )

    def test_waterfall_plot(self):
        model_kwargs = {"add_encoders": ADD_ENCODERS}
        model = DLinearModel(
            input_chunk_length=7,
            output_chunk_length=4,
            **(model_kwargs or {}),
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        background_series = series[-20:]
        background_past_covariates = past_covariates[-20:]
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        foreground_series = series[-10:]

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            background_num_samples=10,
        )

        results = explainer.explain(
            foreground_series=foreground_series,
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
        )

        for horizon in range(1, model.output_chunk_length + 1):
            for component in series.components:
                shap_explanation_object = results.get_shap_explanation_object(
                    horizon=horizon, component=component
                )
                assert isinstance(shap_explanation_object, shap.Explanation)

                waterfall_plot = shap.plots.waterfall(
                    shap_explanation_object[0], show=False
                )
                assert waterfall_plot is not None

    def test_validation_and_helper_branches(self):
        assert _available_shap_methods() == SHAP_METHODS

        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            add_encoders=ADD_ENCODERS,
            **kwargs,
        )

        series = self.univariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        background_series = series[-20:]
        background_past_covariates = past_covariates[-20:]
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        with pytest.raises(ValueError, match="Invalid `shap_method`=invalid"):
            TorchExplainer(
                model,
                background_series=background_series,
                background_past_covariates=background_past_covariates,
                background_future_covariates=background_future_covariates,
                shap_method="invalid",
            )

        with pytest.raises(
            ValueError,
            match=(
                "`background_num_samples` must be less than or equal to "
                f"MAX_BACKGROUND_SAMPLE={MAX_BACKGROUND_SAMPLE}"
            ),
        ):
            TorchExplainer(
                model,
                background_series=background_series,
                background_past_covariates=background_past_covariates,
                background_future_covariates=background_future_covariates,
                background_num_samples=MAX_BACKGROUND_SAMPLE + 1,
            )

        short_background_series = series[
            -(
                model.input_chunk_length
                + model.output_chunk_length
                + MIN_BACKGROUND_SAMPLE
                - 4
            ) :
        ]
        short_background_past_covariates = past_covariates[
            -len(short_background_series) :
        ]
        _, short_background_future_covariates = future_covariates.split_before(
            short_background_series.start_time()
        )
        with pytest.raises(
            ValueError,
            match=(
                "Background series must contain at least "
                f"{MIN_BACKGROUND_SAMPLE} samples"
            ),
        ):
            TorchExplainer(
                model,
                background_series=short_background_series,
                background_past_covariates=short_background_past_covariates,
                background_future_covariates=short_background_future_covariates,
            )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            background_num_samples=10,
        )

        force_plot = explainer.force_plot(
            foreground_series=series[-10:],
            foreground_past_covariates=past_covariates,
            foreground_future_covariates=future_covariates,
            horizon=1,
        )
        assert isinstance(force_plot, shap.plots._force.BaseVisualizer)

        with pytest.raises(ValueError, match="`n_samples` must be less than or equal"):
            explainer.explainer.create_shap_array(
                series[-10:],
                past_covariates,
                future_covariates,
                n_samples=1000,
            )

        assert explainer.explainer._batch_collate_np([(None,)], [0]) is None

    def test_helper_sampling_and_single_target_filtering(self):
        model = DLinearModel(
            input_chunk_length=6,
            output_chunk_length=3,
            add_encoders=ADD_ENCODERS,
            **kwargs,
        )

        series = self.multivariate_series
        past_covariates = self.past_covariates
        future_covariates = self.future_covariates

        model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        background_series = series[-20:]
        background_past_covariates = past_covariates[-20:]
        _, background_future_covariates = future_covariates.split_before(
            background_series.start_time()
        )

        explainer = TorchExplainer(
            model,
            background_series=background_series,
            background_past_covariates=background_past_covariates,
            background_future_covariates=background_future_covariates,
            background_num_samples=10,
        )

        long_length = (
            model.input_chunk_length
            + model.output_chunk_length
            + MAX_BACKGROUND_SAMPLE
            + 25
        )
        times = pd.date_range("20210101", periods=long_length, freq="D")
        long_background_series = TimeSeries.from_times_and_values(
            times=times,
            values=np.tile(series.values(copy=False), (16, 1))[:long_length],
            columns=series.components,
        ).with_static_covariates(series.static_covariates)
        long_background_past_covariates = TimeSeries.from_times_and_values(
            times=times,
            values=np.tile(past_covariates.values(copy=False), (16, 1))[:long_length],
            columns=past_covariates.components,
        )
        long_background_future_covariates = TimeSeries.from_times_and_values(
            times=pd.date_range("20210101", periods=long_length + 20, freq="D"),
            values=np.tile(future_covariates.values(copy=False), (16, 1))[
                : long_length + 20
            ],
            columns=future_covariates.components,
        )

        with pytest.raises(
            ValueError,
            match="`background_num_samples` must be less than or equal to the number of samples in the dataset",
        ):
            explainer.explainer.create_shap_array(
                long_background_series,
                long_background_past_covariates,
                long_background_future_covariates,
                n_samples=MAX_BACKGROUND_SAMPLE + 100,
                train=True,
            )

        sampled_background, _, _ = explainer.explainer.create_shap_array(
            long_background_series,
            long_background_past_covariates,
            long_background_future_covariates,
            train=True,
        )
        assert sampled_background.shape[0] == MAX_BACKGROUND_SAMPLE

    def test_invalid_model_type_check(self):
        with pytest.raises(
            ValueError,
            match="Only models of type `TorchForecastingModel` are supported",
        ):
            TorchExplainer(object())
