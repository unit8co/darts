import json
import os

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts.tests.conftest import MLFLOW_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev
from darts.utils.utils import PL_AVAILABLE

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


@pytest.fixture
def mlflow_tracking(tmpdir_fn):
    """Set up MLflow tracking with a temporary database."""
    mlflow.set_tracking_uri(f"sqlite:///{tmpdir_fn}/mlflow.db")
    return mlflow.tracking.MlflowClient()


@pytest.fixture
def autolog_context():
    """Context manager to safely enable/disable autolog for a test.

    Usage:
        with autolog_context():            # default autolog
        with autolog_context(log_training_metrics=True):   # custom kwargs
    """
    from contextlib import contextmanager

    @contextmanager
    def _autolog_context(**kwargs):
        autolog(disable=True)  # clean state
        autolog(**kwargs)  # enable with custom kwargs
        try:
            yield
        finally:
            autolog(disable=True)  # clean up

    return _autolog_context


def assert_mlflow_artifacts_exist(path: str, is_torch: bool = False):
    """Assert that all required MLflow artifact files exist."""
    assert os.path.exists(os.path.join(path, "MLmodel"))
    assert os.path.exists(os.path.join(path, "conda.yaml"))
    assert os.path.exists(os.path.join(path, "requirements.txt"))
    assert os.path.exists(os.path.join(path, "python_env.yaml"))

    if is_torch:
        assert os.path.exists(os.path.join(path, "data", "model.pt"))
        assert os.path.exists(os.path.join(path, "data", "model.pt.ckpt"))
    else:
        assert os.path.exists(os.path.join(path, "data", "model.pkl"))


def assert_predictions_equal(model1, model2, n: int, decimal: int = 4, series=None):
    """Assert that two models produce equivalent predictions. If series is provided,
    it will be passed to the second model's predict method (for global models that require it)."""
    pred1 = model1.predict(n=n)
    if series is not None:
        pred2 = model2.predict(n=n, series=series)
    else:
        pred2 = model2.predict(n=n)
    np.testing.assert_array_almost_equal(
        pred1.values(), pred2.values(), decimal=decimal
    )


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

        assert_mlflow_artifacts_exist(model_path, is_torch=False)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=5)

    def test_save_load_regression_model(self, tmpdir_fn):
        """Test save/load round-trip for regression model"""
        model = LinearRegressionModel(lags=5)
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        assert_mlflow_artifacts_exist(model_path, is_torch=False)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=3, series=self.ts_univariate)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_save_load_torch_model(self, tmpdir_fn):
        """Test save/load round-trip for torch model"""
        model = NBEATSModel(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        assert_mlflow_artifacts_exist(model_path, is_torch=True)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=2, series=self.ts_univariate)

    def test_log_model_basic(self, mlflow_tracking):
        """Test basic log_model functionality"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        loaded_model = load_model(log_info.model_uri)
        assert_predictions_equal(model, loaded_model, n=5)

    def test_log_model_with_params(self, mlflow_tracking):
        """Test that log_params=True logs model parameters"""
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

    def test_log_model_with_covariates(self, mlflow_tracking):
        """Test that covariate info is logged with correct values"""
        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)
            run_id = mlflow.active_run().info.run_id

            # get artifact while run is still active
            artifact_uri = mlflow.get_artifact_uri("covariates.json")
            artifact_path = artifact_uri.replace("file://", "")
            assert os.path.exists(artifact_path), (
                "covariates.json artifact should exist"
            )

            with open(artifact_path) as f:
                cov_data = json.load(f)

        run = mlflow.get_run(run_id)
        # check covariate usage tags have the correct boolean values
        assert run.data.tags["uses_past_covariates"] == "true"
        assert run.data.tags["uses_future_covariates"] == "false"
        assert run.data.tags["uses_static_covariates"] == "false"
        # check covariate count params
        assert run.data.params["n_past_covariates"] == "1"
        assert run.data.params["n_future_covariates"] == "0"
        assert run.data.params["n_static_covariates"] == "0"

        # verify structure and content
        assert "past_covariates" in cov_data
        assert "future_covariates" in cov_data
        assert "static_covariates" in cov_data

        # check past covariates data
        assert cov_data["past_covariates"]["used"] is True
        assert cov_data["past_covariates"]["count"] == 1
        assert len(cov_data["past_covariates"]["names"]) == 1

        # check future and static covariates not used
        assert cov_data["future_covariates"]["used"] is False
        assert cov_data["future_covariates"]["count"] == 0
        assert cov_data["static_covariates"]["used"] is False
        assert cov_data["static_covariates"]["count"] == 0

    def test_log_model_with_all_covariate_types(self, mlflow_tracking):
        """Test logging model with past, future, and static covariates"""
        # use a model that supports all covariate types
        model = LinearRegressionModel(
            lags=5, lags_past_covariates=3, lags_future_covariates=[0, 1]
        )
        model.fit(
            self.ts_with_static[:40],
            past_covariates=self.ts_past_cov[:40],
            future_covariates=self.ts_future_cov[:50],
        )

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)
            run_id = mlflow.active_run().info.run_id

            # get artifact while run is still active
            artifact_uri = mlflow.get_artifact_uri("covariates.json")
            artifact_path = artifact_uri.replace("file://", "")

            with open(artifact_path) as f:
                cov_data = json.load(f)

        run = mlflow.get_run(run_id)

        # verify all covariate types are tracked
        assert run.data.tags["uses_past_covariates"] == "true"
        assert run.data.tags["uses_future_covariates"] == "true"
        assert run.data.tags["uses_static_covariates"] == "true"

        # verify covariate counts
        assert run.data.params["n_past_covariates"] == "1"
        assert run.data.params["n_future_covariates"] == "1"
        assert run.data.params["n_static_covariates"] == "1"

        # all covariate types should be used
        assert cov_data["past_covariates"]["used"] is True
        assert cov_data["past_covariates"]["count"] == 1
        assert cov_data["future_covariates"]["used"] is True
        assert cov_data["future_covariates"]["count"] == 1
        assert cov_data["static_covariates"]["used"] is True
        assert cov_data["static_covariates"]["count"] == 1

    def test_autolog_enable_disable(self, mlflow_tracking, autolog_context):
        """Test autolog can be enabled and disabled"""
        with autolog_context():
            model = ExponentialSmoothing()
            model.fit(self.ts_univariate)

            runs = mlflow.search_runs()
            assert len(runs) == 1, "Expected exactly one run after autolog fit"

            # verify the run has expected content
            last_run = runs.iloc[0]
            assert last_run["tags.darts.model_class"] == "ExponentialSmoothing"
            assert last_run["tags.mlflow.runName"] is not None

        # after context exits, autolog should be disabled
        model2 = ExponentialSmoothing()
        model2.fit(self.ts_univariate)

        runs_after_disable = mlflow.search_runs()
        assert len(runs_after_disable) == 1, (
            "No new run should be created after disable"
        )

    def test_autolog_parameters(self, mlflow_tracking, autolog_context):
        """Test that autolog logs model parameters"""
        with autolog_context():
            model = ExponentialSmoothing(seasonal_periods=12)
            model.fit(self.ts_univariate)

            runs = mlflow.search_runs()
            assert len(runs) == 1

            last_run = runs.iloc[0]
            assert last_run["params.seasonal_periods"] == "12"
            assert last_run["tags.darts.model_class"] == "ExponentialSmoothing"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_torch_metrics(self, mlflow_tracking, autolog_context):
        """Test that autolog logs training metrics for torch models"""
        with autolog_context():
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=2,
                n_epochs=2,
                **tfm_kwargs_dev,
            )
            train, val = self.ts_univariate.split_before(0.7)
            model.fit(train, val_series=val)

            runs = mlflow.search_runs()
            assert len(runs) == 1, "Expected exactly one run"
            last_run = runs.iloc[0]
            last_run_id = last_run["run_id"]
            assert last_run["tags.darts.model_class"] == "NBEATSModel"

            client = mlflow.tracking.MlflowClient()

            # check train_loss metrics
            train_metrics = client.get_metric_history(last_run_id, "train_loss")
            assert len(train_metrics) > 0, "Expected train_loss metrics to be logged"
            assert len(train_metrics) <= 2, "Expected at most 2 epochs of train_loss"

            for m in train_metrics:
                assert np.isfinite(m.value), f"train_loss is not finite: {m.value}"
                assert m.value >= 0, f"train_loss is negative: {m.value}"
                assert m.step >= 0, "Metric step should be non-negative"

            val_metrics = client.get_metric_history(last_run_id, "val_loss")
            if val_metrics:
                for m in val_metrics:
                    assert np.isfinite(m.value), f"val_loss is not finite: {m.value}"
                    assert m.value >= 0, f"val_loss is negative: {m.value}"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_injects_callback(self, mlflow_tracking, autolog_context):
        """Test that autolog injects MLflow callback into torch models"""
        with autolog_context():
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=2,
                n_epochs=2,
                **tfm_kwargs_dev,
            )

            # record initial callbacks from trainer_params (before fit)
            initial_callbacks = model.trainer_params.get("callbacks", [])
            initial_callback_count = len(initial_callbacks) if initial_callbacks else 0

            train, val = self.ts_univariate.split_before(0.7)
            model.fit(train, val_series=val)

            # verify callback was injected after fit
            final_callbacks = model.trainer_params.get("callbacks", [])
            assert final_callbacks is not None, "Callbacks should not be None"
            assert len(final_callbacks) > initial_callback_count, (
                "MLflow callback should be added"
            )

            # check that the _DartsMlflowCallback is present
            from darts.utils.mlflow import _DartsMlflowCallback

            has_mlflow_callback = any(
                isinstance(cb, _DartsMlflowCallback) for cb in final_callbacks
            )
            assert has_mlflow_callback, (
                f"_DartsMlflowCallback not found in {[type(cb).__name__ for cb in final_callbacks]}"
            )

    def test_covariate_artifact_schema(self, mlflow_tracking):
        """Test that covariate artifact has correct JSON schema"""
        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)

            artifact_uri = mlflow.get_artifact_uri("covariates.json")
            artifact_path = artifact_uri.replace("file://", "")

            with open(artifact_path) as f:
                cov_data = json.load(f)

        # validate schema structure
        required_keys = ["past_covariates", "future_covariates", "static_covariates"]
        assert all(key in cov_data for key in required_keys), (
            "Missing required covariate keys"
        )

        for cov_type in required_keys:
            cov_info = cov_data[cov_type]
            assert "used" in cov_info and isinstance(cov_info["used"], bool)
            assert "count" in cov_info and isinstance(cov_info["count"], int)
            assert "names" in cov_info and isinstance(cov_info["names"], list)
            assert cov_info["count"] == len(cov_info["names"])

    def test_multivariate_with_all_covariate_types(self, mlflow_tracking):
        """Test saving/loading multivariate series with all covariate types"""
        # create multivariate target with static covariates
        target = self.ts_multivariate.with_static_covariates(
            pd.DataFrame({"static_feat_1": [1.0], "static_feat_2": [2.0]})
        )

        model = LinearRegressionModel(
            lags=5, lags_past_covariates=3, lags_future_covariates=[0, 1]
        )
        model.fit(
            target[:40],
            past_covariates=self.ts_past_cov[:40],
            future_covariates=self.ts_future_cov[:50],
        )

        with mlflow.start_run():
            log_model(model, name="model", log_params=True)
            run_id = mlflow.active_run().info.run_id

        run = mlflow.get_run(run_id)

        # verify all covariate types detected
        assert run.data.tags["uses_past_covariates"] == "true"
        assert run.data.tags["uses_future_covariates"] == "true"
        assert run.data.tags["uses_static_covariates"] == "true"

        # verify correct component counts
        assert run.data.params["n_past_covariates"] == "1"
        assert run.data.params["n_future_covariates"] == "1"
        assert run.data.params["n_static_covariates"] == "2"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_callback_injection_with_existing_callbacks(
        self, mlflow_tracking, autolog_context
    ):
        """Test callback injection when model already has callbacks"""
        # create model with existing callback
        if PL_AVAILABLE:
            import pytorch_lightning as pl

            existing_callback = pl.callbacks.EarlyStopping(monitor="train_loss")
        else:
            pytest.skip("PyTorch Lightning not available")

        with autolog_context():
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=2,
                n_epochs=2,
                pl_trainer_kwargs={"callbacks": [existing_callback]},
                **{k: v for k, v in tfm_kwargs_dev.items() if k != "pl_trainer_kwargs"},
            )

            train, val = self.ts_univariate.split_before(0.7)
            model.fit(train, val_series=val)

            # verify both callbacks present
            callbacks = model.trainer_params.get("callbacks", [])
            assert len(callbacks) == 2, "Should have both existing and MLflow callbacks"

            from darts.utils.mlflow import _DartsMlflowCallback

            has_mlflow = any(isinstance(cb, _DartsMlflowCallback) for cb in callbacks)
            has_existing = any(
                isinstance(cb, pl.callbacks.EarlyStopping) for cb in callbacks
            )
            assert has_mlflow and has_existing, "Both callbacks should be present"

    def test_autolog_multiple_fits(self, mlflow_tracking, autolog_context):
        """Test that multiple fits with autolog create separate runs"""
        with autolog_context():
            # since managed_run=True, subsequent fits will reuse the existing run,
            # so we explicitly start runs for each fit
            with mlflow.start_run():
                model2 = LinearRegressionModel(lags=5)
                model2.fit(self.ts_univariate)
            with mlflow.start_run():
                model1 = ExponentialSmoothing()
                model1.fit(self.ts_univariate)

        runs = mlflow.search_runs()
        assert len(runs) == 2, "Expected two separate runs for two fits"

        # verify different model classes logged
        model_classes = set(runs["tags.darts.model_class"])
        assert "ExponentialSmoothing" in model_classes
        assert "LinearRegressionModel" in model_classes

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_torch_model_multiple_fits(self, mlflow_tracking, autolog_context):
        """Test autolog with multiple fits of a torch model"""
        with autolog_context():
            with mlflow.start_run():
                model1 = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=1,
                    **tfm_kwargs_dev,
                )
                train, val = self.ts_univariate.split_before(0.7)
                model1.fit(train, val_series=val)

            with mlflow.start_run():
                model2 = LinearRegressionModel(lags=5)
                model2.fit(self.ts_univariate)

        runs = mlflow.search_runs()
        assert len(runs) == 2, "Expected two separate runs for two fits"

        for _, run in runs.iterrows():
            assert run["tags.darts.model_class"] in [
                "NBEATSModel",
                "LinearRegressionModel",
            ]
            assert run["tags.mlflow.runName"] is not None

    def test_save_load_preserves_series_metadata(self, tmpdir_fn):
        """Test that save/load preserves multivariate and static covariate structure"""
        target = self.ts_multivariate.with_static_covariates(
            pd.DataFrame({"stat1": [1.0], "stat2": [2.0]})
        )

        model = LinearRegressionModel(lags=5)
        model.fit(target)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        loaded_model = load_model(f"file://{model_path}")

        pred_original = model.predict(n=3)
        pred_loaded = loaded_model.predict(n=3, series=target)

        # verify multivariate structure preserved
        assert pred_original.width == pred_loaded.width == 2, (
            "Should maintain 2 components"
        )
        assert pred_original.n_components == pred_loaded.n_components == 2

        np.testing.assert_array_almost_equal(
            pred_original.values(), pred_loaded.values(), decimal=4
        )

    def test_load_nonexistent_model(self):
        """Test that loading nonexistent model raises appropriate error"""
        with pytest.raises(Exception):
            load_model("runs:/fake_run_id/model")

    def test_load_invalid_uri_fails(self):
        """Test that loading with invalid URI raises an error"""
        with pytest.raises(Exception):
            load_model("invalid://bad/uri")

        with pytest.raises(Exception):
            load_model("file:///nonexistent/path/to/model")

    def test_load_corrupted_mlmodel_fails(self, tmpdir_fn):
        """Test that loading with corrupted MLmodel file fails"""
        # save a valid model
        model = LinearRegressionModel(lags=5)
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        # corrupt the MLmodel file
        mlmodel_path = os.path.join(model_path, "MLmodel")
        with open(mlmodel_path, "w") as f:
            f.write("corrupted content that is not valid YAML {[[")

        # loading should fail
        with pytest.raises(Exception):
            load_model(f"file://{model_path}")

    def test_load_missing_model_file_fails(self, tmpdir_fn):
        """Test that loading with missing model data file fails"""
        # save a valid model
        model = LinearRegressionModel(lags=5)
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        # remove the model data file
        model_data_path = os.path.join(model_path, "data", "model.pkl")
        os.remove(model_data_path)

        # loading should fail
        with pytest.raises(Exception):
            load_model(f"file://{model_path}")

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

        # Only pass series for global models (LinearRegressionModel)
        # Local models (ExponentialSmoothing) don't need it
        series = self.ts_univariate if fit_kwargs else None
        assert_predictions_equal(model, loaded, n=5, series=series)

    @pytest.mark.parametrize(
        "series,series_name",
        [
            ("ts_multivariate", "multivariate"),
            ("ts_with_static", "static_covariates"),
        ],
    )
    def test_save_load_with_special_series(self, tmpdir_fn, series, series_name):
        """Test save/load with multivariate and static covariate series"""
        test_series = getattr(self, series)

        model = LinearRegressionModel(lags=5)
        model.fit(test_series)

        model_path = os.path.join(tmpdir_fn, f"test_model_{series_name}")
        save_model(model, model_path)

        loaded_model = load_model(f"file://{model_path}")

        assert_predictions_equal(model, loaded_model, n=3, series=test_series)

        # verify the series dimensions are preserved
        pred_original = model.predict(n=3)
        pred_loaded = loaded_model.predict(n=3, series=test_series)
        assert pred_original.width == pred_loaded.width

    def test_autolog_training_metrics_regression(
        self, mlflow_tracking, autolog_context
    ):
        """Test that autolog computes and logs in-sample training metrics for regression models."""
        with autolog_context(log_training_metrics=True):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, output_chunk_length=3)
                model.fit(self.ts_univariate)

        run_data = mlflow.get_run(run.info.run_id).data

        for metric_name in ["train_mae", "train_mse", "train_rmse", "train_mape"]:
            assert metric_name in run_data.metrics, f"{metric_name} should be logged"
            assert np.isfinite(run_data.metrics[metric_name])

    def test_autolog_extra_metrics(self, mlflow_tracking, autolog_context):
        """Test that extra_metrics are logged alongside defaults."""
        from darts.metrics import r2_score, smape

        with autolog_context(
            log_training_metrics=True, extra_metrics=[smape, r2_score]
        ):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, output_chunk_length=3)
                model.fit(self.ts_univariate)

        run_data = mlflow.get_run(run.info.run_id).data

        # defaults
        for metric_name in ["train_mae", "train_mse", "train_rmse", "train_mape"]:
            assert metric_name in run_data.metrics
        # extras
        assert "train_smape" in run_data.metrics
        assert "train_r2_score" in run_data.metrics

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_training_metrics_torch(self, mlflow_tracking, autolog_context):
        """Test that autolog logs training metrics for torch models including epochs_trained."""
        with autolog_context(
            log_training_metrics=True, inject_per_epoch_callbacks=True
        ):
            with mlflow.start_run() as run:
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=2,
                    **tfm_kwargs_dev,
                )
                train, val = self.ts_univariate.split_before(0.7)
                model.fit(train)

        run_data = mlflow.get_run(run.info.run_id).data

        assert "train_loss" in run_data.metrics
        for metric_name in ["train_mae", "train_mse", "train_rmse", "train_mape"]:
            assert metric_name in run_data.metrics, (
                f"{metric_name} should be logged for torch model"
            )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_validation_metrics_torch(self, mlflow_tracking, autolog_context):
        """Test that validation metrics are computed on val_series for torch models."""
        with autolog_context(log_training_metrics=True, log_validation_metrics=True):
            with mlflow.start_run() as run:
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=2,
                    **tfm_kwargs_dev,
                )
                train, val = self.ts_univariate.split_before(0.7)
                model.fit(train, val_series=val)

        run_data = mlflow.get_run(run.info.run_id).data

        # validation forecasting metrics
        for metric_name in ["val_mae", "val_mse", "val_rmse", "val_mape"]:
            assert metric_name in run_data.metrics, (
                f"{metric_name} should be logged for torch model with val_series"
            )
            assert np.isfinite(run_data.metrics[metric_name])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_validation_metrics_disabled(
        self, mlflow_tracking, autolog_context
    ):
        """Test that validation metrics are NOT logged when log_validation_metrics=False."""
        with autolog_context(log_training_metrics=True, log_validation_metrics=False):
            with mlflow.start_run() as run:
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=2,
                    **tfm_kwargs_dev,
                )
                train, val = self.ts_univariate.split_before(0.7)
                model.fit(train, val_series=val)

        run_data = mlflow.get_run(run.info.run_id).data

        # training metrics should still be present
        assert "train_mae" in run_data.metrics
        # validation metrics should NOT be present
        for metric_name in ["val_mae", "val_mse", "val_rmse", "val_mape"]:
            assert metric_name not in run_data.metrics, (
                f"{metric_name} should NOT be logged when validation disabled"
            )

    def test_autolog_training_metrics_multiple_series(
        self, mlflow_tracking, autolog_context
    ):
        """Test that backtest-based training metrics work with multiple series."""
        ts1 = tg.linear_timeseries(start_value=10, end_value=50, length=50).astype(
            "float32"
        )
        ts2 = tg.linear_timeseries(start_value=20, end_value=60, length=50).astype(
            "float32"
        )

        with autolog_context(log_training_metrics=True):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, output_chunk_length=3)
                model.fit([ts1, ts2])

        run_data = mlflow.get_run(run.info.run_id).data

        for metric_name in ["train_mae", "train_mse", "train_rmse", "train_mape"]:
            assert metric_name in run_data.metrics, (
                f"{metric_name} should be logged for multiple series"
            )
            assert np.isfinite(run_data.metrics[metric_name])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_training_metrics_multiple_series_torch(
        self, mlflow_tracking, autolog_context
    ):
        """Test that backtest-based training metrics work with multiple series on torch models."""
        ts1 = tg.linear_timeseries(start_value=10, end_value=50, length=50).astype(
            "float32"
        )
        ts2 = tg.linear_timeseries(start_value=20, end_value=60, length=50).astype(
            "float32"
        )

        with autolog_context(log_training_metrics=True):
            with mlflow.start_run() as run:
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=2,
                    pl_trainer_kwargs={
                        "accelerator": "cpu",
                        "enable_progress_bar": False,
                        "enable_model_summary": False,
                    },
                )
                model.fit([ts1, ts2])

        run_data = mlflow.get_run(run.info.run_id).data

        for metric_name in ["train_mae", "train_mse", "train_rmse", "train_mape"]:
            assert metric_name in run_data.metrics, (
                f"{metric_name} should be logged for torch multiple series"
            )
            assert np.isfinite(run_data.metrics[metric_name])
