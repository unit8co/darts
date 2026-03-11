import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from darts.tests.conftest import NF_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries
from darts.explainability import TorchExplainer
from darts.models import (
    BlockRNNModel,
    Chronos2Model,
    DLinearModel,
    NBEATSModel,
    NeuralForecastModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
    TSMixerModel,
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

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

ALL_MODELS = [
    (RNNModel, {"model": "LSTM"}),
    (BlockRNNModel, None),
    (NBEATSModel, {"num_stacks": 2, "num_layers": 2, "layer_widths": 16}),
    (NHiTSModel, {"layer_widths": 16}),
    (TiDEModel, {"hidden_size": 32}),
    (DLinearModel, None),
    (TSMixerModel, {"hidden_size": 16, "ff_size": 16}),
    (Chronos2Model, {"local_dir": chronos2_local_dir}),
]

if NF_AVAILABLE:
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
                    "patch_len": 3,
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

        assert loaded_explainer.model.model_params == loaded_model.model_params
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

        # save and load the model to check explainer works with loaded models
