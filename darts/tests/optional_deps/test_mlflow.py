import os

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts.tests.conftest import MLFLOW_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev

if not MLFLOW_AVAILABLE:
    pytest.skip(
        f"MLflow not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import mlflow

from darts.models import ExponentialSmoothing, LinearRegressionModel
from darts.utils.mlflow import (
    autolog,
    load_model,
    log_model,
    save_model,
)

if TORCH_AVAILABLE:
    from darts.models import NBEATSModel


class TestMLflow:
    ts_univariate = tg.linear_timeseries(
        start_value=10, end_value=50, length=50
    ).astype("float32")
    ts_multivariate = ts_univariate.stack(ts_univariate * 1.5)
    ts_with_static = ts_univariate.with_static_covariates(
        pd.DataFrame({"static_feat": [1.0]})
    )
    ts_past_cov = tg.sine_timeseries(length=62).astype("float32")
    ts_future_cov = tg.constant_timeseries(value=1.0, length=62).astype("float32")

    def test_save_load_statistical_model(self, tmpdir_fn):
        """Test save/load round-trip for statistical model"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)
        assert os.path.exists(os.path.join(model_path, "MLmodel"))

        loaded_model = load_model(f"file://{model_path}")
        pred_original = model.predict(n=5)
        pred_loaded = loaded_model.predict(n=5)

        np.testing.assert_array_almost_equal(
            pred_original.values(), pred_loaded.values(), decimal=4
        )

    def test_save_load_regression_model(self, tmpdir_fn):
        """Test save/load round-trip for regression model"""
        model = LinearRegressionModel(lags=5)
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        loaded_model = load_model(f"file://{model_path}")
        pred_original = model.predict(n=3)
        pred_loaded = loaded_model.predict(n=3)

        np.testing.assert_array_almost_equal(
            pred_original.values(), pred_loaded.values(), decimal=4
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_save_load_torch_model(self, tmpdir_fn):
        """Test save/load round-trip for torch model"""
        model = NBEATSModel(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        loaded_model = load_model(f"file://{model_path}")
        pred_original = model.predict(n=2)
        pred_loaded = loaded_model.predict(n=2)

        np.testing.assert_array_almost_equal(
            pred_original.values(), pred_loaded.values(), decimal=4
        )

    def test_log_model_basic(self, tmpdir_fn):
        """Test basic log_model functionality"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        loaded_model = load_model(log_info.model_uri)
        pred_loaded = loaded_model.predict(n=5)
        pred_original = model.predict(n=5)

        assert len(pred_loaded) == 5
        np.testing.assert_array_almost_equal(
            pred_original.values(), pred_loaded.values(), decimal=4
        )

    def test_log_model_with_params(self, tmpdir_fn):
        """Test that log_params=True logs model parameters"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)
            run_id = mlflow.active_run().info.run_id

        run = mlflow.get_run(run_id)
        assert run.data.params["lags"] == "5"
        assert run.data.params["lags_past_covariates"] == "3"
        assert run.data.params["n_past_covariates"] == "1"
        assert run.data.params["n_future_covariates"] == "0"
        assert run.data.params["n_static_covariates"] == "0"

    def test_log_model_with_covariates(self, tmpdir_fn):
        """Test that covariate info is logged with correct values"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)
            run_id = mlflow.active_run().info.run_id

        run = mlflow.get_run(run_id)
        # Check covariate usage tags have the correct boolean values
        assert run.data.tags["uses_past_covariates"] == "true"
        assert run.data.tags["uses_future_covariates"] == "false"
        assert run.data.tags["uses_static_covariates"] == "false"
        # Check covariate count params
        assert run.data.params["n_past_covariates"] == "1"
        assert run.data.params["n_future_covariates"] == "0"
        assert run.data.params["n_static_covariates"] == "0"

    def test_autolog_enable_disable(self, tmpdir_fn):
        """Test autolog can be enabled and disabled"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        autolog()

        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        runs = mlflow.search_runs()
        assert len(runs) == 1

        # Verify the run has the expected model class tag
        assert runs.iloc[0]["tags.darts.model_class"] == "ExponentialSmoothing"

        autolog(disable=True)

        model2 = ExponentialSmoothing()
        model2.fit(self.ts_univariate)

        runs_after_disable = mlflow.search_runs()
        assert len(runs_after_disable) == 1

    def test_autolog_parameters(self, tmpdir_fn):
        """Test that autolog logs model parameters"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        autolog()

        model = ExponentialSmoothing(seasonal_periods=12)
        model.fit(self.ts_univariate)

        runs = mlflow.search_runs()
        assert len(runs) == 1

        last_run = runs.iloc[0]
        assert last_run["params.seasonal_periods"] == "12"
        assert last_run["tags.darts.model_class"] == "ExponentialSmoothing"

        autolog(disable=True)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_torch_metrics(self, tmpdir_fn):
        """Test that autolog logs training metrics for torch models"""
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
        mlflow.set_experiment("test_experiment")

        autolog()

        model = NBEATSModel(
            input_chunk_length=4, output_chunk_length=2, n_epochs=2, **tfm_kwargs_dev
        )
        train, val = self.ts_univariate.split_before(0.7)
        model.fit(train, val_series=val)

        runs = mlflow.search_runs()
        assert len(runs) == 1
        last_run_id = runs.iloc[0]["run_id"]
        assert runs.iloc[0]["tags.darts.model_class"] == "NBEATSModel"

        client = mlflow.tracking.MlflowClient()
        metrics = client.get_metric_history(last_run_id, "train_loss")

        assert len(metrics) > 0
        # All logged loss values should be finite and non-negative
        for m in metrics:
            assert np.isfinite(m.value), f"train_loss is not finite: {m.value}"
            assert m.value >= 0, f"train_loss is negative: {m.value}"

        autolog(disable=True)

    def test_load_nonexistent_model(self):
        """Test that loading nonexistent model raises appropriate error"""
        with pytest.raises(Exception):
            load_model("runs:/fake_run_id/model")

    @pytest.mark.parametrize(
        "model_cls,fit_kwargs",
        [
            (ExponentialSmoothing, {}),
            (LinearRegressionModel, {"lags": 5}),
        ],
    )
    def test_save_load_multiple_models(self, tmpdir_fn, model_cls, fit_kwargs):
        """Test save/load for multiple model types"""
        if fit_kwargs:
            model = model_cls(**fit_kwargs)
        else:
            model = model_cls()

        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)
        loaded = load_model(f"file://{model_path}")

        pred1 = model.predict(n=5)
        pred2 = loaded.predict(n=5)
        np.testing.assert_array_almost_equal(pred1.values(), pred2.values(), decimal=4)
