import os

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
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
            assert runs.iloc[0]["tags.darts.model_class"] == "NBEATSModel"

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

    def test_covariate_artifact_schema(self, mlflow_tracking):
        """Test that covariate artifact has correct JSON schema"""
        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(self.ts_univariate[:40], past_covariates=self.ts_past_cov[:40])

        # TODO: use autolog

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
            from darts.metrics import mae

            with mlflow.start_run() as run:
                result = mae(self.ts_univariate, self.ts_univariate * 1.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mae" in run_data.metrics, "mae should be logged to MLflow"
        assert np.isfinite(run_data.metrics["mae"])
        assert np.isscalar(result)
        assert np.isfinite(float(result))

    def test_autolog_metric_repeated_call(self, mlflow_tracking, autolog_context):
        """Calling the same metric twice overwrites the value (last-value-wins)."""
        with autolog_context(log_metrics=True):
            from darts.metrics import rmse

            with mlflow.start_run() as run:
                rmse(self.ts_univariate, self.ts_univariate * 1.1)
                rmse(self.ts_univariate, self.ts_univariate * 1.2)

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
            from darts.metrics import mae

            with mlflow.start_run() as run:
                mae(
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
            from darts.metrics import mse

            # called outside any start_run — must not raise
            result = mse(self.ts_univariate, self.ts_univariate * 1.1)

        assert np.isscalar(result)
        assert np.isfinite(float(result))

    def test_autolog_metric_returns_correct_value(
        self, mlflow_tracking, autolog_context
    ):
        """The patched metric returns the same value whether inside or outside a run."""
        with autolog_context(log_metrics=True):
            from darts.metrics import mae

            pred = self.ts_univariate * 1.05

            with mlflow.start_run():
                result_inside = mae(self.ts_univariate, pred)

            # call outside a run — no logging, same computation
            result_outside = mae(self.ts_univariate, pred)

        np.testing.assert_almost_equal(result_inside, result_outside, decimal=6)
        assert np.isfinite(result_inside)

    def test_autolog_log_metrics_false(self, mlflow_tracking, autolog_context):
        """autolog(log_metrics=False) leaves metrics unpatched — nothing is logged."""
        with autolog_context(log_metrics=False):
            from darts.metrics import mape

            with mlflow.start_run() as run:
                mape(self.ts_univariate, self.ts_univariate * 1.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mape" not in run_data.metrics, (
            "mape should NOT be logged when log_metrics=False"
        )

    def test_autolog_public_namespace_patched(self, mlflow_tracking, autolog_context):
        """Only darts.metrics (public namespace) is patched; darts.metrics.metrics is not.

        Patching only the public namespace avoids breaking internal metric-to-metric
        calls within the implementation module (e.g. rmse calling mse internally).
        """
        with autolog_context(log_metrics=True):
            import darts.metrics as dm
            import darts.metrics.metrics as dmm

            # public namespace → patched: call inside a run should log
            with mlflow.start_run() as run_public:
                dm.mae(self.ts_univariate, self.ts_univariate * 1.1)

            # implementation module → NOT patched: call inside a run should not log
            with mlflow.start_run() as run_impl:
                dmm.mae(self.ts_univariate, self.ts_univariate * 1.1)

        run_data_public = mlflow.get_run(run_public.info.run_id).data
        run_data_impl = mlflow.get_run(run_impl.info.run_id).data
        assert "mae" in run_data_public.metrics, (
            "darts.metrics.mae should log to MLflow (public namespace is patched)"
        )
        assert "mae" not in run_data_impl.metrics, (
            "darts.metrics.metrics.mae should NOT log (implementation module is not patched)"
        )
