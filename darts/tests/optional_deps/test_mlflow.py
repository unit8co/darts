import logging
import os

import numpy as np
import pandas as pd
import pytest

import darts.metrics as dm
import darts.metrics.metrics as dmm
import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.tests.conftest import MLFLOW_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev

if not MLFLOW_AVAILABLE:
    pytest.skip(
        f"MLflow not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import mlflow
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient

from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveSeasonal,
    RegressionEnsembleModel,
)
from darts.utils.mlflow import (
    _infer_metric_axes,
    _log_backtest_metrics,
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
        assert os.path.exists(os.path.join(path, "model.pt"))
        assert os.path.exists(os.path.join(path, "model.pt.ckpt"))
    else:
        assert os.path.exists(os.path.join(path, "model.pkl"))


def assert_predictions_equal(
    model1: ForecastingModel,
    model2: ForecastingModel,
    n: int,
    decimal: int = 4,
    is_global: bool = True,
    series: TimeSeries | None = None,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
):
    """Assert that two models produce equivalent predictions. If series is provided,
    it will be passed to the second model's predict method (for global models that require it)."""
    if is_global:
        assert isinstance(model1, GlobalForecastingModel)
        assert isinstance(model2, GlobalForecastingModel)
        pred1 = model1.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        pred2 = model2.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
    else:
        pred1 = model1.predict(n=n)
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
    # binary classification series with values {0.0, 1.0}
    ts_binary = tg.constant_timeseries(value=0.0, length=50).with_values(
        np.random.default_rng(42)
        .choice([0.0, 1.0], size=50)
        .astype(np.float32)
        .reshape(-1, 1)
    )

    def test_save_load_statistical_model(self, tmpdir_fn):
        """Test save/load round-trip for statistical model"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path)

        assert_mlflow_artifacts_exist(model_path, is_torch=False)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=5, is_global=False)

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

        # save(clean=True) strips pl_trainer_kwargs; explicitly restore accelerator
        # so Lightning doesn't default to MPS on Github macOS runner
        loaded_model = load_model(
            f"file://{model_path}",
            pl_trainer_kwargs=tfm_kwargs_dev.get("pl_trainer_kwargs", {}),
        )
        assert_predictions_equal(model, loaded_model, n=2, series=self.ts_univariate)

    def test_log_model_basic(self, mlflow_tracking):
        """Test basic log_model functionality"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        loaded_model = load_model(log_info.model_uri)
        assert_predictions_equal(model, loaded_model, n=5, is_global=False)

    def test_log_model_with_covariates(self, mlflow_tracking):
        """Test that covariate info is logged with correct values"""
        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        with mlflow.start_run():
            log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        loaded_model = load_model(f"runs:/{run_id}/model")
        assert_predictions_equal(
            model,
            loaded_model,
            n=5,
            series=self.ts_univariate[:40],
            past_covariates=self.ts_past_cov,
        )

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
            log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        loaded_model = load_model(f"runs:/{run_id}/model")
        assert_predictions_equal(
            model,
            loaded_model,
            n=5,
            series=self.ts_with_static[:40],
            past_covariates=self.ts_past_cov,
            future_covariates=self.ts_future_cov,
        )

    def test_autolog_enable_disable(self, mlflow_tracking, autolog_context):
        """Test autolog can be enabled and disabled"""
        with autolog_context():
            with mlflow.start_run():
                model = ExponentialSmoothing()
                model.fit(self.ts_univariate)

            runs = mlflow.search_runs()
            assert len(runs) == 1, "Expected exactly one run after autolog fit"

            # verify the run has expected content
            last_run = runs.iloc[0]
            assert last_run["tags.model_class"] == "ExponentialSmoothing"
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
            with mlflow.start_run():
                model = ExponentialSmoothing(seasonal_periods=12)
                model.fit(self.ts_univariate)

            runs = mlflow.search_runs()
            assert len(runs) == 1

            last_run = runs.iloc[0]
            assert last_run["params.seasonal_periods"] == "12"
            assert last_run["tags.model_class"] == "ExponentialSmoothing"

    def test_autolog_model_params_json_artifact(self, mlflow_tracking, autolog_context):
        """The model_params.json artifact mirrors model.model_params, preserving
        JSON-native types and falling back to str() for non-serializable values
        (e.g. enums) instead of the flat params store's blanket stringification."""
        with autolog_context():
            with mlflow.start_run() as run:
                model = ExponentialSmoothing(seasonal_periods=12)
                model.fit(self.ts_univariate)

        params = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/model_params.json"
        )
        assert params["seasonal_periods"] == 12
        assert params["trend"] == str(model.model_params["trend"])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_torch_metrics(self, mlflow_tracking, autolog_context):
        """Test that autolog logs training metrics for torch models"""
        with autolog_context():
            with mlflow.start_run():
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
            assert last_run["tags.model_class"] == "NBEATSModel"

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
    def test_autolog_pytorch_autolog_enabled(self, mlflow_tracking, autolog_context):
        """Test that autolog enables mlflow.pytorch.autolog and logs per-epoch
        train_loss, val_loss, and custom torch_metrics with finite non-negative values."""
        import torchmetrics
        from mlflow.utils.autologging_utils import autologging_is_disabled

        n_epochs = 2

        def assert_metric(history, key):
            assert len(history) > 0, f"{key} not logged"
            assert len(history) <= n_epochs, f"too many {key} entries"
            assert all(np.isfinite(m.value) and m.value >= 0 for m in history)

        with autolog_context():
            assert not autologging_is_disabled("pytorch")

            with mlflow.start_run():
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=n_epochs,
                    torch_metrics=torchmetrics.MeanAbsoluteError(),
                    **tfm_kwargs_dev,
                )
                train, val = self.ts_univariate.split_before(0.7)
                model.fit(train, val_series=val)

            runs = mlflow.search_runs()
            assert len(runs) == 1
            run_id = runs.iloc[0]["run_id"]
            assert runs.iloc[0]["tags.model_class"] == "NBEATSModel"

            client = mlflow.tracking.MlflowClient()
            assert_metric(client.get_metric_history(run_id, "train_loss"), "train_loss")
            assert_metric(client.get_metric_history(run_id, "val_loss"), "val_loss")
            # custom torch_metrics: in normal use both train_/val_ prefixes are logged, but
            # fast_dev_run suppresses the Lightning logger during traininge
            assert_metric(
                client.get_metric_history(run_id, "val_MeanAbsoluteError"),
                "val_MeanAbsoluteError",
            )

        assert autologging_is_disabled("pytorch")

    def test_autolog_covariate_info_single_series(
        self, mlflow_tracking, autolog_context
    ):
        """The covariates.json artifact reports usage, count, and names for a
        single-series fit."""
        with autolog_context():
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, lags_past_covariates=3)
                model.fit(
                    self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40]
                )

        covariates = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/covariates.json"
        )
        assert covariates["past_covariates"]["used"] is True
        assert covariates["past_covariates"]["count"] == 1
        assert (
            covariates["past_covariates"]["names"]
            == self.ts_past_cov.components.tolist()
        )
        assert covariates["future_covariates"]["used"] is False
        assert covariates["static_covariates"]["used"] is False

    def test_autolog_covariate_info_multi_series(
        self, mlflow_tracking, autolog_context
    ):
        """Fitting on a list of series doesn't leave past_covariates unreported:
        `model.past_covariate_series` stays None for a multi-series fit, so the
        info must come from the actual `fit()` call arguments instead."""
        series = [self.ts_univariate, self.ts_univariate * 1.2]
        past_covs = [self.ts_past_cov[:50], self.ts_past_cov[:50]]
        with autolog_context():
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, lags_past_covariates=3)
                model.fit(series, past_covariates=past_covs)

        covariates = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/covariates.json"
        )
        assert covariates["past_covariates"]["used"] is True
        assert covariates["past_covariates"]["count"] == 1
        assert (
            covariates["past_covariates"]["names"]
            == self.ts_past_cov.components.tolist()
        )

    def test_autolog_covariate_info_static_is_global(
        self, mlflow_tracking, autolog_context
    ):
        """`is_global` is based on row count vs. the number of series
        components, not the number of static-covariate columns."""
        # one shared row over 2 components -> global
        global_target = self.ts_multivariate.with_static_covariates(
            pd.DataFrame({"a": [1.0], "b": [2.0]})
        )
        with autolog_context():
            with mlflow.start_run() as run:
                LinearRegressionModel(lags=5).fit(global_target)
        covariates = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/covariates.json"
        )
        assert covariates["static_covariates"]["is_global"] is True

        # one row per component (2 components, 2 rows) -> component-specific
        per_component_target = self.ts_multivariate.with_static_covariates(
            pd.DataFrame({"a": [1.0, 2.0]})
        )
        with autolog_context():
            with mlflow.start_run() as run:
                LinearRegressionModel(lags=5).fit(per_component_target)
        covariates = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/covariates.json"
        )
        assert covariates["static_covariates"]["is_global"] is False

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
            log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        loaded_model = load_model(f"runs:/{run_id}/model")
        assert_predictions_equal(
            loaded_model,
            model,
            n=5,
            series=target[:40],
            past_covariates=self.ts_past_cov,
            future_covariates=self.ts_future_cov,
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_pytorch_autolog_with_existing_callbacks(
        self, mlflow_tracking, autolog_context
    ):
        """Test pytorch autolog works when model already has callbacks"""
        # create model with existing callback
        if TORCH_AVAILABLE:
            import pytorch_lightning as pl

            existing_callback = pl.callbacks.EarlyStopping(monitor="train_loss")
        else:
            pytest.skip("PyTorch Lightning not available")

        with autolog_context():
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=2,
                n_epochs=2,
                pl_trainer_kwargs={
                    **tfm_kwargs_dev.get("pl_trainer_kwargs", {}),
                    "callbacks": [existing_callback],
                },
                **{k: v for k, v in tfm_kwargs_dev.items() if k != "pl_trainer_kwargs"},
            )

            train, val = self.ts_univariate.split_before(0.7)
            model.fit(train, val_series=val)

            # verify existing callback is still present (not removed by autolog)
            callbacks = model.trainer_params.get("callbacks", [])
            has_existing = any(
                isinstance(cb, pl.callbacks.EarlyStopping) for cb in callbacks
            )
            assert has_existing, "Existing EarlyStopping callback should be preserved"

            # verify metrics were still logged via mlflow.pytorch.autolog
            runs = mlflow.search_runs()
            assert len(runs) >= 1, "Expected at least one run"
            last_run_id = runs.iloc[0]["run_id"]
            client = mlflow.tracking.MlflowClient()
            train_metrics = client.get_metric_history(last_run_id, "train_loss")
            assert len(train_metrics) > 0, (
                "Expected train_loss metrics to be logged via pytorch autolog"
            )

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
        model_classes = set(runs["tags.model_class"])
        assert "ExponentialSmoothing" in model_classes
        assert "LinearRegressionModel" in model_classes

    def test_autolog_ensemble_model_fit_logs_once(
        self, mlflow_tracking, autolog_context
    ):
        """Fitting a composite/ensemble model must log exactly one model artifact
        for the outer model, not one per sub-model too.

        RegressionEnsembleModel.fit() internally calls fit() on each of its
        forecasting_models. Since every ForecastingModel subclass is patched
        independently, an unguarded patch would re-trigger autologging for each
        inner fit() call as well as the outer one.
        """
        model = RegressionEnsembleModel(
            forecasting_models=[
                NaiveSeasonal(K=12),
                LinearRegressionModel(lags=12),
            ],
            regression_train_n_points=12,
        )

        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model.fit(self.ts_univariate)

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1, (
            f"Expected exactly one logged model, got {len(logged_models)}: "
            f"{[m.name for m in logged_models]}"
        )
        assert logged_models[0].name == "RegressionEnsembleModel"

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
            assert run["tags.model_class"] in [
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
        model_data_path = os.path.join(model_path, "model.pkl")
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
        if isinstance(model, GlobalForecastingModel):
            assert_predictions_equal(model, loaded, n=5, series=self.ts_univariate)
        else:
            assert_predictions_equal(model, loaded, n=5, is_global=False)

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

    def test_autolog_metric_logging_scalar(self, mlflow_tracking, autolog_context):
        """Calling a darts metric inside an active run logs a scalar to MLflow."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                result = dm.mae(self.ts_univariate, self.ts_univariate * 1.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mae" in run_data.metrics, "mae should be logged to MLflow"
        assert np.isfinite(run_data.metrics["mae"])
        assert np.isscalar(result)
        assert np.isfinite(float(result))

    def test_autolog_metric_repeated_call(self, mlflow_tracking, autolog_context):
        """Calling the same metric twice overwrites the value (last-value-wins)."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.rmse(self.ts_univariate, self.ts_univariate * 1.1)
                dm.rmse(self.ts_univariate, self.ts_univariate * 1.2)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "rmse" in run_data.metrics, "rmse should be logged to MLflow"

    def test_autolog_metric_per_component(self, mlflow_tracking, autolog_context):
        """Non-scalar metric results logged per-component as {name}_{component_name}.

        ts_multivariate = ts_univariate.stack(ts_univariate * 1.5), whose component
        names are ['linear', 'linear_1'].  With component_reduction=None the result
        is a 1-D array (one value per component), so the expected keys are
        'mae_linear' and 'mae_linear_1'.
        """
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.mae(
                    self.ts_multivariate,
                    self.ts_multivariate * 1.1,
                    component_reduction=None,
                )

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mae_linear" in run_data.metrics, (
            "Component 'linear' should be logged as mae_linear"
        )
        assert "mae_linear_1" in run_data.metrics, (
            "Component 'linear_1' should be logged as mae_linear_1"
        )
        assert np.isfinite(run_data.metrics["mae_linear"])
        assert np.isfinite(run_data.metrics["mae_linear_1"])

    def test_autolog_metric_no_active_run(self, mlflow_tracking, autolog_context):
        """Calling a metric without an active run does not raise and returns correctly."""
        with autolog_context(log_metrics=True):
            # called outside any start_run — must not raise
            result = dm.mse(self.ts_univariate, self.ts_univariate * 1.1)

        assert np.isscalar(result)
        assert np.isfinite(float(result))

    def test_autolog_metric_returns_correct_value(
        self, mlflow_tracking, autolog_context
    ):
        """The patched metric returns the same value whether inside or outside a run."""
        with autolog_context(log_metrics=True):
            pred = self.ts_univariate * 1.05

            with mlflow.start_run():
                result_inside = dm.mae(self.ts_univariate, pred)

            # call outside a run — no logging, same computation
            result_outside = dm.mae(self.ts_univariate, pred)

        np.testing.assert_almost_equal(result_inside, result_outside, decimal=6)
        assert np.isfinite(result_inside)

    def test_autolog_log_metrics_false(self, mlflow_tracking, autolog_context):
        """autolog(log_metrics=False) leaves metrics unpatched — nothing is logged."""
        with autolog_context(log_metrics=False):
            with mlflow.start_run() as run:
                dm.mape(self.ts_univariate, self.ts_univariate * 1.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mape" not in run_data.metrics, (
            "mape should NOT be logged when log_metrics=False"
        )

    def test_autolog_metric_any_import_path_logs(
        self, mlflow_tracking, autolog_context
    ):
        """Metrics log identically regardless of which module path was used to
        call them. The mlflow hook lives inside `multi_ts_support` (baked into
        the function itself) rather than patching a specific module
        attribute, and `darts.metrics.mae` and `darts.metrics.metrics.mae` are
        the same function object.
        """
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_public:
                dm.mae(self.ts_univariate, self.ts_univariate * 1.1)
            with mlflow.start_run() as run_impl:
                dmm.mae(self.ts_univariate, self.ts_univariate * 1.1)

        assert "mae" in mlflow.get_run(run_public.info.run_id).data.metrics
        assert "mae" in mlflow.get_run(run_impl.info.run_id).data.metrics

    def test_autolog_metric_import_order_independent(
        self, mlflow_tracking, autolog_context
    ):
        """A metric imported via `from darts.metrics import <name>` *before*
        autolog() is enabled still logs correctly, since the hook lives inside
        the metric's own `multi_ts_support` decorator rather than patching a
        module attribute after the fact."""
        from darts.metrics import mae as mae_imported_before_autolog

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                mae_imported_before_autolog(
                    self.ts_univariate, self.ts_univariate * 1.1
                )

        assert "mae" in mlflow.get_run(run.info.run_id).data.metrics

    def test_autolog_metric_internal_composite_call_not_double_logged(
        self, mlflow_tracking, autolog_context
    ):
        """rmse calls mse internally via `_get_wrapped_metric`, which bypasses
        `multi_ts_support` entirely, so autologging must fire once (for rmse),
        not twice (rmse and the internal mse call)."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.rmse(self.ts_univariate, self.ts_univariate * 1.1)

        metrics = mlflow.get_run(run.info.run_id).data.metrics
        assert "rmse" in metrics
        assert "mse" not in metrics

    def test_autolog_metric_per_timestep(self, mlflow_tracking, autolog_context):
        """A per-timestep metric (ae) logs one value per timestep across MLflow steps.

        time_reduction=None (ae's default) means the result keeps a per-timestep
        axis, which is mapped to the MLflow step (mirroring the backtest path)
        rather than being mislabeled as per-component.
        """
        train = self.ts_univariate[:40]
        model = LinearRegressionModel(lags=4)
        model.fit(train)
        pred = model.predict(n=10)
        actual = self.ts_univariate[40:]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.ae(actual, pred)

        ref = np.asarray(ref, dtype=float)  # shape (n_timesteps,)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "ae")
        assert len(history) == len(ref), "Expected one step per timestep"
        steps = sorted(m.step for m in history)
        assert steps == list(range(len(ref)))
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, ref, atol=1e-5)

    def test_autolog_metric_aligns_time_axis_by_end_date(
        self, mlflow_tracking, autolog_context
    ):
        """A shorter series' time axis aligns from the end, not the start, so
        its steps overlap the tail of a longer series rather than the head."""
        ts_long = self.ts_univariate  # length 50
        ts_short = ts_long[10:]  # length 40, same end, starts 10 steps later

        # distinct constant error per series, so a step's mean reveals exactly
        # which series contributed to it
        pred_long = ts_long + 1.0
        pred_short = ts_short + 2.0

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.ae([ts_long, ts_short], [pred_long, pred_short])

        history = mlflow_tracking.get_metric_history(run.info.run_id, "ae")
        by_step = {m.step: m.value for m in history}
        assert len(by_step) == 50
        for step in range(10):
            assert by_step[step] == pytest.approx(1.0, abs=1e-4), step
        for step in range(10, 50):
            assert by_step[step] == pytest.approx(1.5, abs=1e-4), step

        rows = self._read_per_series_table(run.info.run_id)
        steps_by_series = {0: set(), 1: set()}
        for r in rows:
            steps_by_series[r["series_index"]].add(r["step"])
        assert steps_by_series[0] == set(range(50))
        assert steps_by_series[1] == set(range(10, 50))

    def test_autolog_metric_quantile(self, mlflow_tracking, autolog_context):
        """A quantile metric (mql) logs one key per quantile with matching values."""
        train = self.ts_univariate[:40]
        model = LinearRegressionModel(
            lags=4, likelihood="quantile", quantiles=[0.1, 0.5, 0.9]
        )
        model.fit(train)
        pred = model.predict(n=10, num_samples=200)
        actual = self.ts_univariate[40:]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mql(actual, pred, q=[0.1, 0.5, 0.9])

        ref = np.asarray(ref, dtype=float)  # shape (n_quantiles,)
        m = mlflow.get_run(run.info.run_id).data.metrics
        for i, key in enumerate(("mql_q0_100", "mql_q0_500", "mql_q0_900")):
            assert key in m, f"Expected quantile key {key}"
            assert m[key] == pytest.approx(ref[i], abs=1e-5)

    def test_autolog_metric_quantile_interval(self, mlflow_tracking, autolog_context):
        """A quantile interval metric (miw) logs one key per interval."""
        train = self.ts_univariate[:40]
        model = LinearRegressionModel(
            lags=4, likelihood="quantile", quantiles=[0.1, 0.5, 0.9]
        )
        model.fit(train)
        pred = model.predict(n=10, num_samples=200)
        actual = self.ts_univariate[40:]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.miw(actual, pred, q_interval=(0.1, 0.9))

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert "miw_qi_80_000" in m
        assert m["miw_qi_80_000"] == pytest.approx(float(ref), abs=1e-5)

    def test_autolog_metric_multi_series(self, mlflow_tracking, autolog_context):
        """A list of series logs the mean over series; per-series values go to a table."""
        series = [self.ts_univariate, self.ts_univariate * 1.2]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred)

        ref = np.asarray(ref, dtype=float)  # shape (n_series,)
        m = mlflow.get_run(run.info.run_id).data.metrics
        # aggregate = mean over series, no per-series _s{i} keys
        assert m["mae"] == pytest.approx(float(np.mean(ref)), abs=1e-5)
        assert not any(k.startswith("mae_s") for k in m)
        # granular per-series breakdown written to a table artifact
        rows = self._read_per_series_table(run.info.run_id)
        by_series = {int(row["series_index"]): float(row["value"]) for row in rows}
        assert by_series == pytest.approx({0: ref[0], 1: ref[1]}, abs=1e-5)

    def test_autolog_metric_multi_series_custom_agg_func(
        self, mlflow_tracking, autolog_context
    ):
        """autolog()'s agg_func controls how per-series values are aggregated
        into the single logged metric (default np.mean)."""
        # 3 series with distinct, asymmetric per-series errors so median != mean
        series = [self.ts_univariate * f for f in (1.0, 1.2, 5.0)]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True, agg_func=np.median):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred)

        ref = np.asarray(ref, dtype=float)
        assert float(np.median(ref)) != pytest.approx(float(np.mean(ref)), abs=1e-3)
        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["mae"] == pytest.approx(float(np.median(ref)), abs=1e-5)

    def test_autolog_metric_multi_series_per_component(
        self, mlflow_tracking, autolog_context
    ):
        """A list of multivariate series with component_reduction=None logs the
        per-component mean over series; the CSV carries one row per (component, series)."""
        series = [self.ts_multivariate, self.ts_multivariate * 1.2]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred, component_reduction=None)

        ref = np.asarray(ref, dtype=float)  # shape (n_series, n_components)
        m = mlflow.get_run(run.info.run_id).data.metrics
        # aggregate per component = mean over series, no per-series _s{i} keys
        assert m["mae_linear"] == pytest.approx(float(ref[:, 0].mean()), abs=1e-5)
        assert m["mae_linear_1"] == pytest.approx(float(ref[:, 1].mean()), abs=1e-5)
        assert not any(k.endswith(("_s0", "_s1")) for k in m)
        # granular CSV: one row per (component, series)
        rows = self._read_per_series_table(run.info.run_id)
        got = {
            (row["key"], int(row["series_index"])): float(row["value"]) for row in rows
        }
        assert got[("mae_linear", 0)] == pytest.approx(ref[0, 0], abs=1e-5)
        assert got[("mae_linear", 1)] == pytest.approx(ref[1, 0], abs=1e-5)
        assert got[("mae_linear_1", 0)] == pytest.approx(ref[0, 1], abs=1e-5)
        assert got[("mae_linear_1", 1)] == pytest.approx(ref[1, 1], abs=1e-5)

    def test_autolog_metric_name_override(self, mlflow_tracking, autolog_context):
        """The metric `name` kwarg overrides only the metric-name token in the key,
        keeping the backtest prefix and the quantile/axis suffixes."""
        actual = self.ts_univariate
        train = self.ts_univariate[:40]
        qmodel = self._fit_qlr(train)
        pred = qmodel.predict(n=10, num_samples=200)
        target = self.ts_univariate[40:]

        with autolog_context(log_metrics=True):
            # direct call: name replaces the metric token; suffix (_q0_500) preserved
            with mlflow.start_run() as run_direct:
                dm.mae(actual, actual * 1.1, name="custom")
                dm.mql(target, pred, q=0.5, name="myq")
            # backtest: name replaces the metric token; backtest_ prefix preserved
            with mlflow.start_run() as run_bt:
                self._fit_lr().backtest(
                    self.ts_univariate,
                    metric=dm.mae,
                    metric_kwargs={"name": "custom"},
                    retrain=False,
                    stride=10,
                )

        direct = mlflow.get_run(run_direct.info.run_id).data.metrics
        assert "custom" in direct
        assert "mae" not in direct, "default metric name should be replaced"
        assert "myq_q0_500" in direct, "quantile suffix should be preserved"

        bt = mlflow.get_run(run_bt.info.run_id).data.metrics
        assert "backtest_custom" in bt
        assert "backtest_mae" not in bt, "default metric name should be replaced"

    def test_autolog_metric_multi_series_classification_labels_explicit(
        self, mlflow_tracking, autolog_context
    ):
        """f1 with label_reduction=None and explicit labels on a list of binary
        series logs the per-label mean over series and writes the per-series table.

        ``labels`` must be explicit (``label_reduction=None`` without it raises,
        since the number of output labels can't be determined ahead of time)."""
        # two independent binary series (same classes, deterministic)
        binary1 = tg.constant_timeseries(value=0.0, length=50).with_values(
            np.array([0.0, 1.0] * 25, dtype=np.float32).reshape(-1, 1)
        )
        binary2 = tg.constant_timeseries(value=0.0, length=50).with_values(
            np.array([1.0, 0.0] * 25, dtype=np.float32).reshape(-1, 1)
        )
        series = [binary1, binary2]
        pred = series  # perfect predictions → f1 == 1.0 per label per series

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.f1(series, pred, label_reduction=None, labels=[0, 1])

        ref = [np.asarray(r, dtype=float).flatten() for r in ref]
        m = mlflow.get_run(run.info.run_id).data.metrics
        # aggregate per label = mean over series, no per-series _s{i} keys
        assert m["f1_label0"] == pytest.approx(
            float(np.mean([ref[0][0], ref[1][0]])), abs=1e-5
        )
        assert m["f1_label1"] == pytest.approx(
            float(np.mean([ref[0][1], ref[1][1]])), abs=1e-5
        )
        assert not any(k.endswith(("_s0", "_s1")) for k in m)
        # granular CSV: one row per (label, series)
        rows = self._read_per_series_table(run.info.run_id)
        got = {
            (row["key"], int(row["series_index"])): float(row["value"]) for row in rows
        }
        for i in range(2):
            assert got[("f1_label0", i)] == pytest.approx(ref[i][0], abs=1e-5)
            assert got[("f1_label1", i)] == pytest.approx(ref[i][1], abs=1e-5)

    def test_autolog_metric_component_count_mismatch_raises(
        self, mlflow_tracking, autolog_context
    ):
        """A list of series with different numbers of components raises rather
        than taking component names from the first series only and silently
        mislabeling the rest."""
        series = [self.ts_univariate, self.ts_multivariate]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                with pytest.raises(ValueError, match="same number of components"):
                    dm.mae(series, pred)

        assert not mlflow.get_run(run.info.run_id).data.metrics

    def test_autolog_metric_size_mismatch_raises(
        self, mlflow_tracking, autolog_context, caplog
    ):
        """When the inferred C-axis size doesn't divide the result, logging raises
        (and logs an error), propagating out of the public metric call — the
        metric callback isn't invoked through MLflow's safe_patch, so nothing
        catches it. No metrics are written for the failed call."""
        actual = self.ts_univariate[40:]
        # mae with component_reduction=None on a univariate series produces shape (T,),
        # which is size T — divisible by c_size=1 (1 component × 1 quantile), so we
        # need to force a mismatch.  We do that by monkey-patching _infer_metric_axes
        # to report has_comp_axis=True with a fake 3-component count, making c_size=3
        # while the actual result is shape (T,).
        train = self.ts_univariate[:40]
        model = LinearRegressionModel(lags=4)
        model.fit(train)
        pred = model.predict(n=10)

        import unittest.mock as mock

        from darts.utils import mlflow as mlflow_utils

        fake_axes = (False, True, ["_c0", "_c1", "_c2"])
        with mock.patch.object(
            mlflow_utils, "_infer_metric_axes", return_value=fake_axes
        ):
            with autolog_context(log_metrics=True):
                with mlflow.start_run() as run:
                    with caplog.at_level(logging.ERROR, logger="darts"):
                        with pytest.raises(ValueError, match="not divisible"):
                            dm.mae(actual, pred)

        assert any("not divisible" in record.message for record in caplog.records), (
            "Expected a 'not divisible' error to be logged when axes don't match"
        )
        # no metrics should have been written for the (faked) mismatched call
        run_data = mlflow.get_run(run.info.run_id).data.metrics
        assert not any("mae" in k for k in run_data), (
            "No mae metrics should be logged when the size check fails"
        )

    def test_autolog_metric_per_series_table_schema_and_single_series_skip(
        self, mlflow_tracking, autolog_context
    ):
        """The per-series table has the expected schema for multi-series input, and no
        artifact is written for single-series input (mean == the value itself)."""
        # multi-series: artifact exists with the documented columns
        multi = [self.ts_univariate, self.ts_univariate * 1.2]
        pred_multi = [s * 1.1 for s in multi]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_multi:
                dm.mae(multi, pred_multi)

        rows = self._read_per_series_table(run_multi.info.run_id)
        assert list(rows[0].keys()) == ["key", "series_index", "step", "value"]
        assert {int(r["series_index"]) for r in rows} == {0, 1}

        # single-series: no per-series table artifact should be created
        single = self.ts_univariate
        pred_single = single * 1.1
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_single:
                dm.mae(single, pred_single)

        artifacts = mlflow_tracking.list_artifacts(run_single.info.run_id)
        assert not any(a.path == "metrics_per_series.json" for a in artifacts), (
            "Single-series input should not write a per-series table artifact"
        )

    def test_autolog_backtest_scalar(self, mlflow_tracking, autolog_context):
        """Default (reduced) backtest of a single univariate series logs one scalar."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr().backtest(
                    self.ts_univariate, metric=dm.mae, retrain=False, stride=10
                )

        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae" in run_data.metrics
        assert run_data.metrics["backtest_mae"] == pytest.approx(float(ref), abs=1e-5)

    def test_autolog_backtest_per_window_steps(self, mlflow_tracking, autolog_context):
        """reduction=None logs per-window values as consecutive steps of one key."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr().backtest(
                    self.ts_univariate,
                    metric=dm.mae,
                    retrain=False,
                    stride=10,
                    reduction=None,
                )

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        assert len(history) > 1, "Expected multiple per-window steps"
        steps = sorted(m.step for m in history)
        assert steps == list(range(len(history)))
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, np.asarray(ref, dtype=float), atol=1e-5)

    def test_autolog_backtest_per_component(self, mlflow_tracking, autolog_context):
        """component_reduction=None logs one key per component name."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr(self.ts_multivariate).backtest(
                    self.ts_multivariate,
                    metric=dm.mae,
                    retrain=False,
                    stride=10,
                    metric_kwargs={"component_reduction": None},
                )

        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae_linear" in run_data.metrics
        assert "backtest_mae_linear_1" in run_data.metrics
        ref = np.asarray(ref, dtype=float)
        assert run_data.metrics["backtest_mae_linear"] == pytest.approx(
            ref[0], abs=1e-5
        )
        assert run_data.metrics["backtest_mae_linear_1"] == pytest.approx(
            ref[1], abs=1e-5
        )

    def test_autolog_backtest_multi_metric(self, mlflow_tracking, autolog_context):
        """Multiple metrics are logged under one key each."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr().backtest(
                    self.ts_univariate,
                    metric=[dm.mae, dm.rmse],
                    retrain=False,
                    stride=10,
                )

        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae" in run_data.metrics
        assert "backtest_rmse" in run_data.metrics
        assert run_data.metrics["backtest_mae"] == pytest.approx(
            float(ref[0]), abs=1e-5
        )
        assert run_data.metrics["backtest_rmse"] == pytest.approx(
            float(ref[1]), abs=1e-5
        )

    def test_autolog_backtest_multi_series(self, mlflow_tracking, autolog_context):
        """A list of series logs the mean over series; per-series values go to a table."""
        series = [self.ts_univariate, self.ts_univariate * 1.2]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr(series).backtest(
                    series, metric=dm.mae, retrain=False, stride=10
                )

        run_data = mlflow.get_run(run.info.run_id).data
        # aggregate = mean over series, no per-series _s{i} keys
        assert run_data.metrics["backtest_mae"] == pytest.approx(
            float(np.mean(ref)), abs=1e-5
        )
        assert not any(k.startswith("backtest_mae_s") for k in run_data.metrics)
        # granular per-series breakdown written to a table artifact
        rows = self._read_per_series_table(run.info.run_id)
        by_series = {int(row["series_index"]): float(row["value"]) for row in rows}
        assert by_series == pytest.approx(
            {0: float(ref[0]), 1: float(ref[1])}, abs=1e-5
        )

    def test_autolog_backtest_multi_series_custom_agg_func(self, mlflow_tracking):
        """autolog()'s agg_func also controls the backtest() aggregation
        (default np.mean). Calls _log_backtest_metrics directly with a
        fabricated result so the per-series values are exact."""
        backtest_args = {
            "metric": dm.mae,
            "metric_kwargs": {},
            "series": [self.ts_univariate, self.ts_univariate, self.ts_univariate],
            "forecast_horizon": 1,
            "reduction": np.mean,
            "last_points_only": True,
        }
        result = [1.0, 2.0, 100.0]  # asymmetric -> median != mean

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _log_backtest_metrics(
                client, run.info.run_id, result, backtest_args, agg_func=np.median
            )
            client.flush(synchronous=True)

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["backtest_mae"] == pytest.approx(2.0)

    def test_autolog_backtest_per_timestep_scalar(
        self, mlflow_tracking, autolog_context
    ):
        """A per-timestep metric (ae) under default reduction collapses to one scalar."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr().backtest(
                    self.ts_univariate, metric=dm.ae, retrain=False, stride=10
                )

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        assert len(history) == 1, "Default reduction should yield a single value"
        assert history[0].value == pytest.approx(float(ref), abs=1e-5)

    def test_autolog_backtest_per_timestep_per_window(
        self, mlflow_tracking, autolog_context
    ):
        """ae + reduction=None + forecast_horizon>1 logs one key per window, with
        one step per forecast horizon timestep."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = self._fit_lr().backtest(
                    self.ts_univariate,
                    metric=dm.ae,
                    retrain=False,
                    stride=10,
                    forecast_horizon=4,
                    reduction=None,
                )

        ref = np.asarray(ref, dtype=float)  # shape (n_windows, forecast_horizon)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae_w0")
        assert len(history) == 4, "Expected one step per forecast horizon timestep"
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, ref[0], atol=1e-5)

    def test_autolog_backtest_historical_forecasts_horizon_inferred(
        self, mlflow_tracking, autolog_context
    ):
        """When `historical_forecasts` is user-supplied, `backtest()` ignores the
        `forecast_horizon` argument, autologging must infer the true window
        length from the historical forecasts themselves, not from the (unused,
        defaulted) `forecast_horizon` argument."""
        model = self._fit_lr()
        hf = model.historical_forecasts(
            self.ts_univariate,
            retrain=False,
            stride=10,
            forecast_horizon=4,
            last_points_only=False,
        )
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                # forecast_horizon is not passed (defaults to 1) and would be
                # wrong; the real horizon (4) must come from `hf`.
                ref = model.backtest(
                    self.ts_univariate,
                    historical_forecasts=hf,
                    metric=dm.ae,
                    reduction=None,
                )

        ref = np.asarray(ref, dtype=float)  # shape (n_windows, forecast_horizon)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae_w0")
        assert len(history) == 4, "Expected one step per forecast horizon timestep"
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, ref[0], atol=1e-5)

    def test_autolog_backtest_quantile(self, mlflow_tracking, autolog_context):
        """A quantile metric (mql) logs one key per quantile."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                self._fit_qlr().backtest(
                    self.ts_univariate,
                    metric=dm.mql,
                    metric_kwargs={"q": [0.1, 0.5, 0.9]},
                    retrain=False,
                    stride=10,
                    num_samples=200,
                )

        m = mlflow.get_run(run.info.run_id).data.metrics
        for key in (
            "backtest_mql_q0_100",
            "backtest_mql_q0_500",
            "backtest_mql_q0_900",
        ):
            assert key in m, f"Expected quantile key {key}"
            assert np.isfinite(m[key])

    def test_autolog_backtest_inconsistent_axes_flat_fallback(
        self, mlflow_tracking, autolog_context
    ):
        """Metrics with mismatched axes (mae has no time axis, ae does) cannot be
        merged into a structured layout, so values are logged flat by index."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                self._fit_lr().backtest(
                    self.ts_univariate,
                    metric=[dm.mae, dm.ae],
                    retrain=False,
                    stride=10,
                )

        m = mlflow.get_run(run.info.run_id).data.metrics
        flat_keys = [k for k in m if k.startswith("backtest_metrics_")]
        assert flat_keys, "Expected flat fallback keys for inconsistent axes"
        assert "backtest_mae" not in m, "No structured key on flat fallback"

    def test_autolog_backtest_classification_labels_in_data(
        self, mlflow_tracking, autolog_context
    ):
        """f1 with explicit labels present in the series logs finite per-label keys."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                self._fit_lr(self.ts_binary).backtest(
                    self.ts_binary,
                    metric=dm.f1,
                    metric_kwargs={"label_reduction": None, "labels": [0, 1]},
                    retrain=False,
                    stride=10,
                )

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert "backtest_f1_label0" in m
        assert "backtest_f1_label1" in m
        assert np.isfinite(m["backtest_f1_label0"])
        assert np.isfinite(m["backtest_f1_label1"])

    def test_autolog_backtest_classification_labels_not_in_data(
        self, mlflow_tracking, autolog_context
    ):
        """f1 with explicit labels absent from the series still creates the keys, but
        the scores are NaN (the labels never appear in any window)."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                self._fit_lr(self.ts_binary).backtest(
                    self.ts_binary,
                    metric=dm.f1,
                    metric_kwargs={"label_reduction": None, "labels": [5, 10]},
                    retrain=False,
                    stride=10,
                )

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert "backtest_f1_label5" in m
        assert "backtest_f1_label10" in m
        assert np.isnan(m["backtest_f1_label5"])
        assert np.isnan(m["backtest_f1_label10"])

    def test_log_backtest_metrics_component_count_mismatch_raises(
        self, mlflow_tracking
    ):
        """A list of series with different numbers of components raises rather
        than taking component names from the first series only.

        Calls _log_backtest_metrics directly so the raise is not swallowed by
        MLflow's safe_patch wrapper.
        """
        backtest_args = {
            "metric": dm.mae,
            "metric_kwargs": {},
            "series": [self.ts_univariate, self.ts_multivariate],
            "forecast_horizon": 1,
            "reduction": np.mean,
            "last_points_only": True,
        }
        result = [1.0, np.array([1.0, 2.0])]

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            with pytest.raises(ValueError, match="same number of components"):
                _log_backtest_metrics(client, run.info.run_id, result, backtest_args)

        assert not mlflow.get_run(run.info.run_id).data.metrics

    def test_log_backtest_metrics_unknown_labels_raises(self, mlflow_tracking):
        """label_reduction=None without explicit labels raises rather than
        inferring class names from the series at runtime.

        This is tested by calling _log_backtest_metrics directly so the raise
        is not swallowed by MLflow's safe_patch wrapper.
        """
        ts_bin = tg.constant_timeseries(value=0.0, length=10)
        backtest_args = {
            "metric": dm.f1,
            "metric_kwargs": {"label_reduction": None},
            "series": ts_bin,
            "forecast_horizon": 1,
            "reduction": np.mean,
            "last_points_only": True,
        }

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            with pytest.raises(ValueError, match="requires explicit `labels`"):
                _log_backtest_metrics(
                    client, run.info.run_id, np.array([0.5]), backtest_args
                )

    def test_log_backtest_metrics_label_count_mismatch(self, mlflow_tracking):
        """When the explicit label count does not divide the metric output size,
        logging raises rather than silently producing incomplete metrics.

        This is tested by calling _log_backtest_metrics directly so the raise is
        not swallowed by MLflow's safe_patch wrapper.
        """
        ts_3class = tg.constant_timeseries(value=0.0, length=50)

        # Simulate a backtest result with only 2 entries — as if the metric was
        # evaluated on windows that only contained 2 of the 3 explicit labels.
        # axis_size is 3 (len(labels)) but result has 2 → mismatch.
        fake_result = np.array([0.8, 0.6], dtype=float)

        backtest_args = {
            "metric": dm.f1,
            "metric_kwargs": {"label_reduction": None, "labels": [0, 1, 2]},
            "series": ts_3class,
            "forecast_horizon": 1,
            "reduction": np.mean,  # not None → has_windows=False → single window
            "last_points_only": True,
        }

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            with pytest.raises(ValueError, match="not divisible"):
                _log_backtest_metrics(
                    client, run.info.run_id, fake_result, backtest_args
                )

        assert not mlflow.get_run(run.info.run_id).data.metrics

    def test_log_backtest_metrics_aligns_windows_by_end_date(self, mlflow_tracking):
        """A shorter series' window axis aligns from the end, not the start,
        so its windows overlap the tail of a longer series rather than the
        head. Calls _log_backtest_metrics directly with a fabricated result
        so the window counts per series are exact."""
        backtest_args = {
            "metric": dm.mae,
            "metric_kwargs": {},
            "series": [self.ts_univariate, self.ts_univariate],
            "forecast_horizon": 1,
            "reduction": None,
            "last_points_only": False,
        }
        # series 0 has 5 windows; series 1 (shorter, later-starting) has 3
        result = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 20.0, 30.0])]

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _log_backtest_metrics(client, run.info.run_id, result, backtest_args)
            client.flush(synchronous=True)

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({0: 1.0, 1: 2.0, 2: 6.5, 3: 12.0, 4: 17.5})

        rows = self._read_per_series_table(run.info.run_id)
        by_step = {(r["series_index"], r["step"]): r["value"] for r in rows}
        assert by_step[(0, 0)] == pytest.approx(1.0)
        assert by_step[(0, 4)] == pytest.approx(5.0)
        assert by_step[(1, 2)] == pytest.approx(10.0)
        assert by_step[(1, 4)] == pytest.approx(30.0)
        assert (1, 0) not in by_step
        assert (1, 1) not in by_step

    def test_log_backtest_metrics_aligns_last_points_only_time_axis(
        self, mlflow_tracking
    ):
        """last_points_only stitches windows into one series scored per real
        timestep -- a separate code path from the window-axis case above that
        needs the same end-date alignment."""
        backtest_args = {
            "metric": dm.ae,
            "metric_kwargs": {},
            "series": [self.ts_univariate, self.ts_univariate],
            "forecast_horizon": 1,
            "reduction": None,
            "last_points_only": True,
        }
        # series 0 has 5 timesteps; series 1 (shorter, later-starting) has 3
        result = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([10.0, 20.0, 30.0])]

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _log_backtest_metrics(client, run.info.run_id, result, backtest_args)
            client.flush(synchronous=True)

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({0: 1.0, 1: 2.0, 2: 6.5, 3: 12.0, 4: 17.5})

    @staticmethod
    def _read_per_series_table(run_id):
        """Load the run's consolidated per-series metric table into row dicts."""
        df = mlflow.load_table(
            artifact_file="metrics_per_series.json", run_ids=[run_id]
        )
        return df.to_dict("records")

    def _fit_lr(self, series=None):
        """Fit and return a fresh LinearRegressionModel (no active run)."""
        model = LinearRegressionModel(lags=4)
        model.fit(series if series is not None else self.ts_univariate)
        return model

    def _fit_qlr(self, series=None):
        """Fit and return a fresh quantile LinearRegressionModel (no active run)."""
        model = LinearRegressionModel(
            lags=4, likelihood="quantile", quantiles=[0.1, 0.5, 0.9]
        )
        model.fit(series if series is not None else self.ts_univariate)
        return model


@pytest.mark.parametrize(
    "metric_name, metric_kwargs, expected",
    [
        ("mae", {}, dict(has_time_axis=False, has_comp_axis=False, axis_size=1)),
        ("ae", {}, dict(has_time_axis=True, has_comp_axis=False, axis_size=1)),
        ("mae", {"component_reduction": None}, dict(has_comp_axis=True)),
    ],
)
def test_infer_metric_axes_reductions(metric_name, metric_kwargs, expected):
    has_time_axis, has_comp_axis, axis_labels = _infer_metric_axes(
        getattr(dm, metric_name), metric_kwargs
    )
    actual = {
        "has_time_axis": has_time_axis,
        "has_comp_axis": has_comp_axis,
        "axis_size": len(axis_labels),
    }
    for attr, value in expected.items():
        assert actual[attr] == value


def test_infer_metric_axes_quantiles():
    _, _, axis_labels = _infer_metric_axes(dm.mql, {"q": [0.1, 0.5, 0.9]})
    assert axis_labels == ["_q0.100", "_q0.500", "_q0.900"]


def test_infer_metric_axes_quantile_interval():
    has_time, _, axis_labels = _infer_metric_axes(dm.iw, {"q_interval": (0.1, 0.9)})
    assert axis_labels == ["_qi_80.000"]
    assert has_time is True


def test_infer_metric_axes_unknown_labels_raises():
    """label_reduction=None with no explicit labels cannot determine the number
    of output labels ahead of time, so this raises rather than falling back."""
    with pytest.raises(ValueError, match="requires explicit `labels`"):
        _infer_metric_axes(dm.f1, {"label_reduction": None})
