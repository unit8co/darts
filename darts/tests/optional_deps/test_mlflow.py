import copy
import itertools
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import patch

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
from darts.tests.conftest import (
    MLFLOW_AVAILABLE,
    TORCH_AVAILABLE,
    tfm_kwargs,
    tfm_kwargs_dev,
)

if not MLFLOW_AVAILABLE:
    pytest.skip(
        f"MLflow not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import mlflow
import yaml
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient

from darts.metrics.utils import unregister_metric_callback
from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveSeasonal,
    RegressionEnsembleModel,
)
from darts.utils.mlflow import (
    _build_metric_keys,
    _flush_logged_metrics,
    _infer_metric_axes,
    _log_backtest_metrics,
    _log_per_series_table,
    _reshape_metric_result,
    _resolve_backtest_layout,
    _run_has_logged_model,
    autolog,
    get_default_conda_env,
    load_model,
    log_model,
    save_model,
)

if TORCH_AVAILABLE:
    from darts.models import NBEATSModel


TS_UNIVARIATE = tg.sine_timeseries(length=50).astype("float32")
TS_MULTIVARIATE = TS_UNIVARIATE.stack(TS_UNIVARIATE * 1.5)
TS_WITH_STATIC = TS_UNIVARIATE.with_static_covariates(
    pd.DataFrame({"static_feat": [1.0]})
)
TS_PAST_COV = tg.sine_timeseries(length=62).astype("float32")
TS_FUTURE_COV = tg.constant_timeseries(value=1.0, length=62).astype("float32")
# binary classification series with values {0.0, 1.0}
TS_BINARY = tg.constant_timeseries(value=0.0, length=50).with_values(
    np.random.default_rng(42)
    .choice([0.0, 1.0], size=50)
    .astype(np.float32)
    .reshape(-1, 1)
)


@pytest.fixture(scope="module")
def _mlflow_store(tmp_path_factory):
    """Shared MLflow backend for the whole module (one SQLite DB)."""
    store_dir = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{store_dir / 'mlflow.db'}")
    return store_dir


@pytest.fixture
def mlflow_tracking(_mlflow_store, request):
    """Isolated MLflow experiment per test on the shared module store."""
    exp_name = f"darts_mlflow_{request.node.name}"
    artifact_root = _mlflow_store / f"artifacts_{uuid.uuid4().hex}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = mlflow.tracking.MlflowClient()
    exp_id = client.create_experiment(
        exp_name,
        artifact_location=str(artifact_root),
    )
    mlflow.set_experiment(experiment_id=exp_id)

    autolog(disable=True)
    yield client
    autolog(disable=True)


@contextmanager
def _autolog_context(**kwargs):
    autolog(disable=True)
    autolog(**kwargs)
    try:
        yield
    finally:
        autolog(disable=True)


@pytest.fixture
def autolog_context():
    """Context manager to safely enable/disable autolog for a test.

    Usage:
        with autolog_context():            # default autolog
        with autolog_context(log_models=True):   # custom kwargs
    """
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


BT_REQUIRED_DEFAUTLS = {
    "historical_forecasts": None,
    "forecast_horizon": 1,
    "last_points_only": False,
    "past_covariates": None,
    "future_covariates": None,
    "reduction": np.nanmean,
    "metric": dm.mape,
    "metric_kwargs": None,
}


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


def _read_per_series_table(run_id):
    """Load the run's consolidated per-series metric table into row dicts."""
    df = mlflow.load_table(artifact_file="metrics_per_series.json", run_ids=[run_id])
    return df.to_dict("records")


def _fit_lr(series=None):
    """Fit and return a LinearRegressionModel (no active run)."""
    series = TS_UNIVARIATE if series is None else series
    model = LinearRegressionModel(lags=1)
    model.fit(series)
    return model


def _fit_qlr(series=None):
    """Fit and return a quantile LinearRegressionModel (no active run)."""
    series = TS_UNIVARIATE if series is None else series
    model = LinearRegressionModel(
        lags=1, likelihood="quantile", quantiles=[0.1, 0.5, 0.9]
    )
    model.fit(series)
    return model


class TestMLflow:
    def test_save_load_statistical_model(self, tmp_path):
        """Test save/load round-trip for statistical model"""
        model = ExponentialSmoothing()
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        assert_mlflow_artifacts_exist(str(model_path), is_torch=False)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=5, is_global=False)

    def test_save_load_regression_model(self, tmp_path):
        """Test save/load round-trip for regression model"""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        assert_mlflow_artifacts_exist(str(model_path), is_torch=False)

        loaded_model = load_model(f"file://{model_path}")
        assert_predictions_equal(model, loaded_model, n=3, series=TS_UNIVARIATE)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_save_load_torch_model(self, tmp_path):
        """Test save/load round-trip for torch model"""
        model = NBEATSModel(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        assert_mlflow_artifacts_exist(str(model_path), is_torch=True)

        # save(clean=True) strips pl_trainer_kwargs; explicitly restore accelerator
        # so Lightning doesn't default to MPS on Github macOS runner
        loaded_model = load_model(
            f"file://{model_path}",
            pl_trainer_kwargs=tfm_kwargs_dev.get("pl_trainer_kwargs", {}),
        )
        assert_predictions_equal(model, loaded_model, n=2, series=TS_UNIVARIATE)

    def test_log_model_basic(self, mlflow_tracking):
        """Test basic log_model functionality"""
        model = ExponentialSmoothing()
        model.fit(TS_UNIVARIATE)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        loaded_model = load_model(log_info.model_uri)
        assert_predictions_equal(model, loaded_model, n=5, is_global=False)

    def test_log_model_with_covariates(self, mlflow_tracking):
        """Test that covariate info is logged with correct values"""
        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(TS_UNIVARIATE[:40], past_covariates=TS_PAST_COV[:40])

        with mlflow.start_run():
            log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        loaded_model = load_model(f"runs:/{run_id}/model")
        assert_predictions_equal(
            model,
            loaded_model,
            n=5,
            series=TS_UNIVARIATE[:40],
            past_covariates=TS_PAST_COV,
        )

    def test_log_model_with_all_covariate_types(self, mlflow_tracking):
        """Test logging model with past, future, and static covariates"""
        # use a model that supports all covariate types
        model = LinearRegressionModel(
            lags=5, lags_past_covariates=3, lags_future_covariates=[0, 1]
        )
        model.fit(
            TS_WITH_STATIC[:40],
            past_covariates=TS_PAST_COV[:40],
            future_covariates=TS_FUTURE_COV[:50],
        )

        with mlflow.start_run():
            log_model(model, name="model")
            run_id = mlflow.active_run().info.run_id

        loaded_model = load_model(f"runs:/{run_id}/model")
        assert_predictions_equal(
            model,
            loaded_model,
            n=5,
            series=TS_WITH_STATIC[:40],
            past_covariates=TS_PAST_COV,
            future_covariates=TS_FUTURE_COV,
        )

    def test_autolog_enable_disable(self, mlflow_tracking, autolog_context):
        """Test autolog can be enabled and disabled"""
        client = mlflow_tracking
        runs = mlflow.search_runs()
        assert len(runs) == 0

        autolog()
        with mlflow.start_run():
            runs = mlflow.search_runs()
            assert len(runs) == 1
            run_id = runs.iloc[0]["run_id"]

            # metric is auto-logged
            dm.mae(TS_UNIVARIATE, TS_UNIVARIATE, name="auto_logged")
            assert len(client.get_metric_history(run_id, "auto_logged")) == 1

            autolog(disable=True)
            dm.mae(TS_UNIVARIATE, TS_UNIVARIATE, name="not_logged")
            assert len(client.get_metric_history(run_id, "not_logged")) == 0

    def test_autolog_parameters(self, mlflow_tracking, autolog_context):
        """Test that autolog logs model parameters"""
        with autolog_context():
            with mlflow.start_run():
                model = ExponentialSmoothing(seasonal_periods=12)
                model.fit(TS_UNIVARIATE)

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
                model.fit(TS_UNIVARIATE)

        params = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/model_params.json"
        )
        assert params["seasonal_periods"] == 12
        assert params["trend"] == str(model.model_params["trend"])

    @pytest.mark.parametrize("single_series", [True, False])
    def test_autolog_series_info_single_series(
        self, mlflow_tracking, autolog_context, single_series
    ):
        """The series_info.json artifact reports the target series' component
        names/count, plus covariate usage, count, and names, for a
        single-series fit."""
        series = TS_UNIVARIATE[:40]
        past_covs = TS_PAST_COV[:40]
        if not single_series:
            series = [series] * 2
            past_covs = [past_covs] * 2

        with autolog_context():
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5, lags_past_covariates=3)
                model.fit(series, past_covariates=past_covs)

        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        assert series_info["series"]["count"] == TS_UNIVARIATE.n_components
        assert series_info["series"]["names"] == TS_UNIVARIATE.components.tolist()
        assert series_info["past_covariates"]["used"] is True
        assert series_info["past_covariates"]["count"] == 1
        assert (
            series_info["past_covariates"]["names"] == TS_PAST_COV.components.tolist()
        )
        assert series_info["future_covariates"]["used"] is False
        assert series_info["static_covariates"]["used"] is False

    def test_autolog_series_info_add_encoders(self, mlflow_tracking, autolog_context):
        """Covariates generated purely via `add_encoders` (no explicit
        covariate argument passed to `fit()`) are still reported in
        `series_info.json`."""
        with autolog_context():
            with mlflow.start_run() as run:
                model = LinearRegressionModel(
                    lags=5,
                    lags_future_covariates=[0],
                    add_encoders={"datetime_attribute": {"future": ["month"]}},
                )
                model.fit(TS_UNIVARIATE[:40])

        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        expected_names = model.encoders.future_components.tolist()
        assert expected_names  # sanity check: encoders actually generated something
        assert series_info["future_covariates"]["used"] is True
        assert series_info["future_covariates"]["count"] == 0
        assert series_info["future_covariates"]["encodings"] == expected_names
        assert series_info["past_covariates"]["used"] is False

    def test_autolog_series_info_static_covariates(
        self, mlflow_tracking, autolog_context
    ):
        # one shared row over 2 components -> global
        global_target = TS_MULTIVARIATE.with_static_covariates(
            pd.DataFrame({"a": [1.0], "b": [2.0]})
        )
        with autolog_context():
            with mlflow.start_run() as run:
                LinearRegressionModel(lags=5).fit(global_target)
        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        assert series_info["static_covariates"]["count"] == 2
        assert series_info["static_covariates"]["names"] == ["a", "b"]

        # one row per component (2 components, 2 rows) -> component-specific
        per_component_target = TS_MULTIVARIATE.with_static_covariates(
            pd.DataFrame({"a": [1.0, 2.0]})
        )
        with autolog_context():
            with mlflow.start_run() as run:
                LinearRegressionModel(lags=5).fit(per_component_target)
        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        assert series_info["static_covariates"]["count"] == 1
        assert series_info["static_covariates"]["names"] == ["a"]

    @pytest.mark.parametrize("method", ["historical_forecasts", "backtest"])
    def test_autolog_historical_forecasts_series_info_covariates(
        self, mlflow_tracking, autolog_context, method
    ):
        """HF and backtest without a prior fit() still reports model params and series_info,
        and logs an untrained model template when log_models=True."""
        target = TS_UNIVARIATE.with_static_covariates(
            pd.DataFrame({"static_feat": [1.0]})
        )
        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(
                    lags=5,
                    lags_past_covariates=3,
                    lags_future_covariates=[0],
                    add_encoders={"datetime_attribute": {"future": ["month"]}},
                )
                _ = getattr(model, method)(
                    series=target,
                    past_covariates=TS_PAST_COV,
                    forecast_horizon=1,
                    retrain=True,
                    start=-1,
                )

        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        assert series_info["past_covariates"]["used"] is True
        assert (
            series_info["past_covariates"]["names"] == TS_PAST_COV.components.tolist()
        )
        assert series_info["future_covariates"]["used"] is True
        assert series_info["static_covariates"]["used"] is True
        assert series_info["static_covariates"]["names"] == ["static_feat"]
        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert tags["model_uses_future_covariates"] == "True"
        assert tags["model_uses_past_covariates"] == "True"
        assert tags["model_uses_static_covariates"] == "True"
        assert tags["darts.model_is_pretrained"] == "False"

        params = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/model_params.json"
        )
        assert params["lags"] == 5

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1
        loaded_model = load_model(logged_models[0].model_uri)
        assert loaded_model._fit_called is False

    def test_autolog_historical_forecasts_retrain_false_skips_model_setup(
        self, mlflow_tracking, autolog_context
    ):
        """historical_forecasts(retrain=False) does not log model setup."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE[:40])
        with autolog_context():
            with mlflow.start_run() as run:
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=False,
                    start=0.5,
                )

        client = mlflow.tracking.MlflowClient()
        artifact_paths = [a.path for a in client.list_artifacts(run.info.run_id)]
        assert "model_params.json" not in artifact_paths
        assert "series_info.json" not in artifact_paths

    def test_autolog_historical_forecasts_overwrites_fit_setup(
        self, mlflow_tracking, autolog_context
    ):
        """fit() then historical_forecasts(retrain=True) in the same run
        overwrites model_params / series_info with the HF call's series."""
        with autolog_context():
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5)
                model.fit(TS_UNIVARIATE[:30])
                model.historical_forecasts(
                    series=TS_MULTIVARIATE,
                    forecast_horizon=1,
                    retrain=True,
                    start=0.5,
                )

        series_info = mlflow.artifacts.load_dict(
            f"runs:/{run.info.run_id}/series_info.json"
        )
        assert series_info["series"]["names"] == TS_MULTIVARIATE.components.tolist()

    def test_autolog_historical_forecasts_retrain_logs_untrained_model(
        self, mlflow_tracking, autolog_context
    ):
        """HF(retrain=True) without fit() logs an untrained model template."""
        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5)
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=True,
                    start=0.5,
                )

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1
        assert logged_models[0].name == "LinearRegressionModel"

        loaded_model = load_model(logged_models[0].model_uri)
        assert loaded_model._fit_called is False

        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert tags["darts.model_is_pretrained"] == "False"

    def test_autolog_fit_then_historical_forecasts_logs_one_pretrained_model(
        self, mlflow_tracking, autolog_context
    ):
        """fit() then HF(retrain=True) keeps the fit artifact and pretrained flag."""
        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model = LinearRegressionModel(lags=5)
                model.fit(TS_UNIVARIATE[:30])
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=True,
                    start=0.5,
                )

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1

        loaded_model = load_model(logged_models[0].model_uri)
        assert loaded_model._fit_called is True

        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert tags["darts.model_is_pretrained"] == "True"

    def test_autolog_manual_log_model_then_historical_forecasts(
        self, mlflow_tracking, autolog_context
    ):
        """Manual log_model() before HF keeps the user's artifact only."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE[:30])

        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                log_model(model, name="manual_model")
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=True,
                    start=0.5,
                )

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1
        assert logged_models[0].name == "manual_model"

        loaded_model = load_model(logged_models[0].model_uri)
        assert loaded_model._fit_called is True

        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert "darts.model_is_pretrained" not in tags

    def test_autolog_historical_forecasts_retrain_false_no_logged_model(
        self, mlflow_tracking, autolog_context
    ):
        """historical_forecasts(retrain=False) does not log a model artifact."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE[:40])
        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=False,
                    start=0.5,
                )

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 0

    def test_autolog_ensemble_historical_forecasts_retrain_logs_once(
        self, mlflow_tracking, autolog_context
    ):
        """Ensemble HF(retrain=True) logs exactly one untrained template."""
        model = RegressionEnsembleModel(
            forecasting_models=[
                NaiveSeasonal(K=12),
                LinearRegressionModel(lags=12),
            ],
            regression_train_n_points=12,
        )

        with autolog_context(log_models=True):
            with mlflow.start_run() as run:
                model.historical_forecasts(
                    series=TS_UNIVARIATE,
                    forecast_horizon=1,
                    retrain=True,
                    start=0.5,
                )

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1
        assert logged_models[0].name == "RegressionEnsembleModel"

        loaded_model = load_model(logged_models[0].model_uri)
        assert loaded_model._fit_called is False

        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert tags["darts.model_is_pretrained"] == "False"

    def test_multivariate_with_all_covariate_types(self, mlflow_tracking):
        """Test saving/loading multivariate series with all covariate types"""
        # create multivariate target with static covariates
        target = TS_MULTIVARIATE.with_static_covariates(
            pd.DataFrame({"static_feat_1": [1.0], "static_feat_2": [2.0]})
        )

        model = LinearRegressionModel(
            lags=5, lags_past_covariates=3, lags_future_covariates=[0, 1]
        )
        model.fit(
            target[:40],
            past_covariates=TS_PAST_COV[:40],
            future_covariates=TS_FUTURE_COV[:50],
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
            past_covariates=TS_PAST_COV,
            future_covariates=TS_FUTURE_COV,
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_autolog_pytorch(self, mlflow_tracking, autolog_context):
        """Test that autolog logs training metrics for torch models"""
        import pytorch_lightning as pl
        import torchmetrics
        from mlflow.utils.autologging_utils import autologging_is_disabled

        existing_callback = pl.callbacks.EarlyStopping(monitor="train_loss")
        tfm_kwargs_ = copy.deepcopy(tfm_kwargs)
        tfm_kwargs_["pl_trainer_kwargs"] = {
            **tfm_kwargs_["pl_trainer_kwargs"],
            "callbacks": [existing_callback],
        }

        with autolog_context():
            assert not autologging_is_disabled("pytorch")
            with mlflow.start_run():
                model = NBEATSModel(
                    input_chunk_length=4,
                    output_chunk_length=2,
                    n_epochs=2,
                    torch_metrics=torchmetrics.MeanAbsoluteError(),
                    **tfm_kwargs_,
                )
                train, val = TS_UNIVARIATE.split_before(0.7)
                model.fit(train, val_series=val)
        assert autologging_is_disabled("pytorch")

        # verify existing callback is still present (not removed by autolog)
        callbacks = model.trainer_params.get("callbacks", [])
        has_existing = any(
            isinstance(cb, pl.callbacks.EarlyStopping) for cb in callbacks
        )
        assert has_existing, "Existing EarlyStopping callback should be preserved"

        runs = mlflow.search_runs()
        assert len(runs) == 1, "Expected exactly one run"
        last_run = runs.iloc[0]
        last_run_id = last_run["run_id"]
        assert last_run["tags.model_class"] == "NBEATSModel"

        client = mlflow.tracking.MlflowClient()

        # check train_loss metrics
        train_metrics = client.get_metric_history(last_run_id, "train_loss")
        val_metrics = client.get_metric_history(last_run_id, "val_loss")
        torch_metrics = client.get_metric_history(last_run_id, "val_MeanAbsoluteError")

        for metrics in [train_metrics, val_metrics, torch_metrics]:
            assert len(metrics) == 2
            for idx, m in enumerate(metrics):
                assert np.isfinite(m.value)
                assert m.step == idx

    def test_autolog_multiple_fits(self, mlflow_tracking, autolog_context):
        """Test that multiple fits with autolog create separate runs"""
        with autolog_context():
            # since managed_run=True, subsequent fits will reuse the existing run,
            # so we explicitly start runs for each fit
            with mlflow.start_run():
                model2 = LinearRegressionModel(lags=5)
                model2.fit(TS_UNIVARIATE)
            with mlflow.start_run():
                model1 = ExponentialSmoothing()
                model1.fit(TS_UNIVARIATE)

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
                model.fit(TS_UNIVARIATE)

        logged_models = mlflow_tracking.search_logged_models(
            experiment_ids=[run.info.experiment_id]
        )
        assert len(logged_models) == 1, (
            f"Expected exactly one logged model, got {len(logged_models)}: "
            f"{[m.name for m in logged_models]}"
        )
        assert logged_models[0].name == "RegressionEnsembleModel"

        assert_predictions_equal(
            model,
            load_model(logged_models[0].model_uri),
            n=5,
            series=TS_UNIVARIATE,
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

    def test_load_corrupted_mlmodel_fails(self, tmp_path):
        """Test that loading with corrupted MLmodel file fails"""
        # save a valid model
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        # corrupt the MLmodel file
        mlmodel_path = model_path / "MLmodel"
        with open(mlmodel_path, "w") as f:
            f.write("corrupted content that is not valid YAML {[[")

        # loading should fail
        with pytest.raises(Exception):
            load_model(f"file://{model_path}")

    def test_load_missing_model_file_fails(self, tmp_path):
        """Test that loading with missing model data file fails"""
        # save a valid model
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        # remove the model data file
        model_data_path = model_path / "model.pkl"
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
    def test_save_load_multiple_models(self, tmp_path, model_cls, fit_kwargs):
        """Test save/load for multiple model types"""
        if fit_kwargs:
            model = model_cls(**fit_kwargs)
        else:
            model = model_cls()

        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))
        loaded = load_model(f"file://{model_path}")

        # Only pass series for global models (LinearRegressionModel)
        # Local models (ExponentialSmoothing) don't need it
        if isinstance(model, GlobalForecastingModel):
            assert_predictions_equal(model, loaded, n=5, series=TS_UNIVARIATE)
        else:
            assert_predictions_equal(model, loaded, n=5, is_global=False)

    @pytest.mark.parametrize(
        "series,series_name",
        [
            (TS_MULTIVARIATE, "multivariate"),
            (TS_WITH_STATIC, "static_covariates"),
        ],
    )
    def test_save_load_with_special_series(self, tmp_path, series, series_name):
        """Test save/load with multivariate and static covariate series"""
        model = LinearRegressionModel(lags=5)
        model.fit(series)

        model_path = tmp_path / f"test_model_{series_name}"
        save_model(model, str(model_path))

        loaded_model = load_model(f"file://{model_path}")

        assert_predictions_equal(model, loaded_model, n=3, series=series)

        # verify the series dimensions are preserved
        pred_original = model.predict(n=3)
        pred_loaded = loaded_model.predict(n=3, series=series)
        assert pred_original.width == pred_loaded.width

    def test_save_model_invalid_model_type(self, tmp_path):
        """save_model rejects non-ForecastingModel instances."""
        with pytest.raises(ValueError, match="ForecastingModel"):
            save_model(object(), str(tmp_path / "bad_model"))

    def test_save_model_with_signature_metadata_and_input_example(self, tmp_path):
        """Optional MLflow Model fields are persisted when provided."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        signature = ModelSignature(
            inputs=Schema([ColSpec("double", "input")]),
            outputs=Schema([ColSpec("double", "output")]),
        )
        metadata = {"author": "darts-test"}
        save_model(
            model,
            str(model_path),
            signature=signature,
            input_example={"n": 3},
            metadata=metadata,
        )

        with open(model_path / "MLmodel") as f:
            mlmodel_conf = yaml.safe_load(f)
        assert mlmodel_conf["signature"] is not None
        assert mlmodel_conf["metadata"] == metadata
        assert os.path.exists(model_path / "input_example.json")

    def test_save_model_custom_pip_requirements(self, tmp_path):
        """Explicit pip_requirements override the default dependency list."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        custom_reqs = ["numpy>=1.0"]
        save_model(model, str(model_path), pip_requirements=custom_reqs)

        with open(model_path / "requirements.txt") as f:
            requirements = f.read()
        assert "numpy>=1.0" in requirements

    def test_save_model_pip_constraints(self, tmp_path):
        """Constraint files referenced in pip_requirements are written out."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        constraints_file = tmp_path / "constraints.txt"
        constraints_file.write_text("numpy>=1.0")

        model_path = tmp_path / "test_model"
        save_model(
            model,
            str(model_path),
            pip_requirements=[f"-c {constraints_file}"],
        )
        assert (model_path / "constraints.txt").exists()

    def test_load_model_non_forecasting_model_class(self, tmp_path):
        """load_model rejects flavor configs whose model_class is not a ForecastingModel."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE)

        model_path = tmp_path / "test_model"
        save_model(model, str(model_path))

        mlmodel_path = model_path / "MLmodel"
        with open(mlmodel_path) as f:
            mlmodel_conf = yaml.safe_load(f)
        mlmodel_conf["flavors"]["darts"]["model_class"] = "builtins.str"
        with open(mlmodel_path, "w") as f:
            yaml.safe_dump(mlmodel_conf, f)

        with pytest.raises(ValueError, match="ForecastingModel"):
            load_model(f"file://{model_path}")

    def test_get_default_conda_env(self):
        """Default conda env includes darts as a pip dependency."""
        env = get_default_conda_env()
        assert env["name"] == "mlflow-env"
        pip_deps = env["dependencies"][-1]["pip"]
        assert any("darts" in dep for dep in pip_deps)

    def test_autolog_invalid_series_align(self):
        """autolog rejects unsupported series_align values."""
        with pytest.raises(ValueError, match="series_align"):
            autolog(series_align="middle")

    def test_autolog_pytorch_import_error_is_ignored(self, autolog_context):
        """Missing mlflow.pytorch does not break autolog when log_torch_metrics=True."""
        real_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow.pytorch":
                raise ImportError("no pytorch flavor")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with autolog_context(log_torch_metrics=True):
                with mlflow.start_run():
                    LinearRegressionModel(lags=5).fit(TS_UNIVARIATE[:40])

    def test_autolog_disable_pytorch_autolog_exception_is_ignored(self):
        """Errors while disabling mlflow.pytorch autolog are swallowed."""
        mock_pytorch = patch("mlflow.pytorch.autolog", side_effect=RuntimeError("boom"))
        with mock_pytorch:
            autolog(log_torch_metrics=True, disable=True)

    def test_autolog_fit_no_active_run(self, autolog_context):
        """fit() with autolog enabled but no active run completes without logging."""
        with autolog_context(log_models=True):
            model = LinearRegressionModel(lags=5)
            model.fit(TS_UNIVARIATE[:40])

        assert model._fit_called

    def test_autolog_historical_forecasts_no_active_run(self, autolog_context):
        """historical_forecasts(retrain=True) without a run does not raise."""
        model = LinearRegressionModel(lags=5)
        model.fit(TS_UNIVARIATE[:40])
        with autolog_context(log_models=True):
            model.historical_forecasts(
                series=TS_UNIVARIATE,
                forecast_horizon=1,
                retrain=True,
                start=0.5,
            )

    def test_autolog_backtest_no_active_run(self, autolog_context):
        """backtest() with autolog enabled but no active run returns normally."""
        with autolog_context(log_metrics=True):
            ref = _fit_lr().backtest(
                TS_UNIVARIATE, metric=dm.mae, retrain=False, start=-2
            )
        assert np.isfinite(float(np.asarray(ref).ravel()[0]))


class TestAutoLogStandaloneMetrics:
    def test_autolog_metric_no_active_run(self, mlflow_tracking, autolog_context):
        """Calling a metric without an active run does not raise and returns correctly."""
        with autolog_context(log_metrics=True):
            # called outside any start_run — must not raise
            result = dm.mse(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)

        assert np.isscalar(result)
        assert np.isfinite(float(result))

    def test_autolog_metric_returns_correct_value(
        self, mlflow_tracking, autolog_context
    ):
        """The patched metric returns the same value whether inside or outside a run."""
        with autolog_context(log_metrics=True):
            pred = TS_UNIVARIATE * 1.05

            with mlflow.start_run():
                result_inside = dm.mae(TS_UNIVARIATE, pred)

            # call outside a run — no logging, same computation
            result_outside = dm.mae(TS_UNIVARIATE, pred)

        np.testing.assert_almost_equal(result_inside, result_outside, decimal=6)
        assert np.isfinite(result_inside)

    def test_autolog_log_metrics_false(self, mlflow_tracking, autolog_context):
        """autolog(log_metrics=False) leaves metrics unpatched — nothing is logged."""
        with autolog_context(log_metrics=False):
            with mlflow.start_run() as run:
                dm.mape(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mape" not in run_data.metrics, (
            "mape should NOT be logged when log_metrics=False"
        )

    def test_autolog_metric_series_reduction(self, mlflow_tracking, autolog_context):
        """series_reduction collapses multi-series input to one logged scalar."""
        series = [TS_UNIVARIATE, TS_UNIVARIATE * 1.2]
        pred = [TS_UNIVARIATE * 1.1, TS_UNIVARIATE * 1.3]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred, series_reduction=np.mean)

        assert np.isscalar(ref)
        run_data = mlflow.get_run(run.info.run_id).data
        assert "mae" in run_data.metrics
        assert run_data.metrics["mae"] == pytest.approx(ref)

    def test_autolog_metric_any_import_path_logs(
        self, mlflow_tracking, autolog_context
    ):
        """Metrics log identically regardless of which module path was used to
        call them. The mlflow hook lives inside `multi_ts_support` (baked into
        the function itself) rather than patching a specific module
        attribute, and `darts.metrics.mae` and `darts.metrics.metrics.mae` are
        the same function object.
        """
        from darts.metrics import mae

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_public:
                dm.mae(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)
            with mlflow.start_run() as run_module:
                dmm.mae(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)
            with mlflow.start_run() as run_explicit:
                mae(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)

        assert "mae" in mlflow.get_run(run_public.info.run_id).data.metrics
        assert "mae" in mlflow.get_run(run_module.info.run_id).data.metrics
        assert "mae" in mlflow.get_run(run_explicit.info.run_id).data.metrics

    def test_autolog_metric_internal_composite_call_not_double_logged(
        self, mlflow_tracking, autolog_context
    ):
        """rmse calls mse internally via `_get_wrapped_metric`, which bypasses
        `multi_ts_support` entirely, so autologging must fire once (for rmse),
        not twice (rmse and the internal mse call)."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.rmse(TS_UNIVARIATE, TS_UNIVARIATE * 1.1)

        metrics = mlflow.get_run(run.info.run_id).data.metrics
        assert "rmse" in metrics
        assert "mse" not in metrics

    def test_autolog_metric_logging_scalar(self, mlflow_tracking, autolog_context):
        """Calling a darts metric inside an active run logs a scalar to MLflow."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                result = dm.mae(TS_UNIVARIATE, TS_UNIVARIATE + 0.1)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "mae" in run_data.metrics, "mae should be logged to MLflow"
        assert run_data.metrics["mae"] == result == pytest.approx(0.1, abs=1e-5)

    def test_autolog_metric_repeated_call(self, mlflow_tracking, autolog_context):
        """Calling the same metric twice overwrites the value (last-value-wins)."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                dm.rmse(TS_UNIVARIATE, TS_UNIVARIATE)
                result = dm.rmse(TS_UNIVARIATE, TS_UNIVARIATE + 2.0)

        run_data = mlflow.get_run(run.info.run_id).data
        assert "rmse" in run_data.metrics, "rmse should be logged to MLflow"
        assert run_data.metrics["rmse"] == result == pytest.approx(2.0, abs=1e-5)

    def test_autolog_metric_per_component(self, mlflow_tracking, autolog_context):
        """Non-scalar metric results logged per-component as {name}_{component_name}.

        ts_multivariate = ts_univariate.stack(ts_univariate * 1.5), whose component
        names are ['sine', 'sine_1'].  With component_reduction=None the result
        is a 1-D array (one value per component), so the expected keys are
        'mae_sine' and 'mae_sine_1'.
        """
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                result = dm.mae(
                    TS_MULTIVARIATE,
                    TS_MULTIVARIATE * 1.1,
                    component_reduction=None,
                )
        comps = TS_MULTIVARIATE.components
        assert result.shape == (len(comps),)
        run_data = mlflow.get_run(run.info.run_id).data
        for comp, value in zip(comps, result):
            assert f"mae_{comp}" in run_data.metrics
            assert run_data.metrics[f"mae_{comp}"] == value

    def test_autolog_metric_per_component_and_q_label(
        self, mlflow_tracking, autolog_context
    ):
        """Non-scalar metric results logged per-component and quantile / label as
        {name}_{component_name}_q{quantile / label}.

        E.g. ["mae_comp0_q0.100", ..., "mae_comp0_q0.900", "mae_comp1_q0.100", ..., "mae_comp1_q0.900"]
        """
        vals = TS_MULTIVARIATE.all_values()
        series_prob = TS_MULTIVARIATE.with_values(
            np.concatenate([vals * 0.9, vals, vals * 1.2], axis=2)
        )
        quantiles = [0.1, 0.5, 0.9]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                result = dm.mae(
                    TS_MULTIVARIATE,
                    series_prob,
                    component_reduction=None,
                    q=quantiles,
                )
        comps = TS_MULTIVARIATE.components
        assert result.shape == (len(comps) * len(quantiles),)
        run_data = mlflow.get_run(run.info.run_id).data
        for c_idx, comp in enumerate(comps):
            for q_idx, q in enumerate(quantiles):
                value = result[c_idx * len(quantiles) + q_idx]
                assert f"mae_{comp}_q{q:.3f}" in run_data.metrics
                assert run_data.metrics[f"mae_{comp}_q{q:.3f}"] == pytest.approx(
                    value, abs=1e-5
                )

    def test_autolog_metric_per_timestep(self, mlflow_tracking, autolog_context):
        """A per-timestep metric (ae) logs one value per timestep across MLflow steps.

        time_reduction=None (ae's default) means the result keeps a per-timestep
        axis, which is mapped to the MLflow step (mirroring the backtest path)
        rather than being mislabeled as per-component.
        """
        actual = TS_UNIVARIATE[40:]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.ae(actual, actual + 1.0)

        ref = np.asarray(ref, dtype=float)  # shape (n_timesteps,)
        assert ref.shape == (len(actual),)

        history = mlflow_tracking.get_metric_history(run.info.run_id, "ae")
        assert len(history) == len(ref), "Expected one step per timestep"
        steps = sorted(m.step for m in history)
        assert steps == list(range(len(ref)))
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, ref, atol=1e-5)

    def test_autolog_metric_per_timestep_component_and_q_label(
        self, mlflow_tracking, autolog_context
    ):
        """A per-timestep metric (ae) logs one value per timestep and component and quantile / label as
        {name}_{component_name}_q{quantile / label}.

        E.g. ["mae_comp0_q0.100", ..., "mae_comp0_q0.900", "mae_comp1_q0.100", ..., "mae_comp1_q0.900"]
        """
        actual = TS_MULTIVARIATE[40:]
        vals = actual.all_values()
        series_prob = actual.with_values(
            np.concatenate([vals * 0.9, vals, vals * 1.2], axis=2)
        )
        quantiles = [0.1, 0.5, 0.9]
        comps = actual.components

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.ae(actual, series_prob, component_reduction=None, q=quantiles)

        ref = np.asarray(ref, dtype=float)
        assert ref.shape == (len(actual), len(comps) * len(quantiles))

        for c_idx, comp in enumerate(comps):
            for q_idx, q in enumerate(quantiles):
                history = mlflow_tracking.get_metric_history(
                    run.info.run_id, f"ae_{comp}_q{q:.3f}"
                )
                assert len(history) == len(ref)
                steps = sorted(m.step for m in history)
                assert steps == list(range(len(actual)))
                logged = [m.value for m in sorted(history, key=lambda m: m.step)]
                np.testing.assert_allclose(
                    logged, ref[:, c_idx * len(quantiles) + q_idx], atol=1e-5
                )

    def test_autolog_metric_aligns_time_axis_by_forecast_position(
        self, mlflow_tracking, autolog_context
    ):
        """Each series starts at forecast position zero."""
        ts_long = TS_UNIVARIATE  # length 50
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
        for step in range(40):
            assert by_step[step] == pytest.approx(1.5, abs=1e-4), step
        for step in range(40, 50):
            assert by_step[step] == pytest.approx(1.0, abs=1e-4), step

        rows = _read_per_series_table(run.info.run_id)
        steps_by_series = {0: set(), 1: set()}
        for r in rows:
            steps_by_series[r["series_index"]].add(r["step"])
        assert steps_by_series[0] == set(range(50))
        assert steps_by_series[1] == set(range(40))

    def test_autolog_metric_quantile(self, mlflow_tracking, autolog_context):
        """A quantile metric (mql) logs one key per quantile with matching values."""
        actual = TS_UNIVARIATE[40:]
        vals = actual.all_values()
        pred = actual.with_values(
            np.concatenate([vals - 1.0, vals, vals + 1.0], axis=2)
        )

        quantiles = [0.1, 0.5, 0.9]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mql(actual, pred, q=quantiles)

        ref = np.asarray(ref, dtype=float)  # shape (n_quantiles,)
        assert ref.shape == (len(quantiles),)

        m = mlflow.get_run(run.info.run_id).data.metrics
        for i, key in enumerate(("mql_q0.100", "mql_q0.500", "mql_q0.900")):
            assert key in m, f"Expected quantile key {key}"
            assert m[key] == pytest.approx(ref[i], abs=1e-5)

    def test_autolog_metric_quantile_interval(self, mlflow_tracking, autolog_context):
        """A quantile interval metric (miw) logs one key per interval."""
        actual = TS_UNIVARIATE[40:]
        vals = actual.all_values()
        pred = actual.with_values(
            np.concatenate([vals - 1.0, vals, vals + 1.0], axis=2)
        )

        q_intervals = [(0.1, 0.9), (0.2, 0.8)]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.miw(
                    actual,
                    pred,
                    q_interval=q_intervals,
                )

        ref = np.atleast_1d(ref)
        assert ref.shape == (len(q_intervals),)
        m = mlflow.get_run(run.info.run_id).data.metrics
        for i, key in enumerate(("miw_qi0.800", "miw_qi0.600")):
            assert key in m, f"Expected quantile key {key}"
            assert m[key] == pytest.approx(ref[i], abs=1e-5)

    def test_autolog_metric_multi_series(self, mlflow_tracking, autolog_context):
        """A list of series logs the mean over series; per-series values go to a table."""
        series = [TS_UNIVARIATE, TS_UNIVARIATE * 1.2]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred)

        ref = np.atleast_1d(ref)
        assert ref.shape == (len(series),)
        m = mlflow.get_run(run.info.run_id).data.metrics
        # agg_func = np.nanmean over series, no per-series _s{i} keys
        assert m["mae"] == pytest.approx(float(np.nanmean(ref)), abs=1e-5)
        assert not any(k.startswith("mae_s") for k in m)
        # granular per-series breakdown written to a table artifact
        rows = _read_per_series_table(run.info.run_id)
        by_series = {int(row["series_index"]): float(row["value"]) for row in rows}
        assert by_series == pytest.approx({0: ref[0], 1: ref[1]}, abs=1e-5)

    def test_autolog_metric_multi_series_custom_agg_func(
        self, mlflow_tracking, autolog_context
    ):
        """autolog()'s agg_func controls how per-series values are aggregated
        into the single logged metric (default np.mean)."""
        # 3 series with distinct, asymmetric per-series errors so median != mean
        series = [TS_UNIVARIATE * f for f in (1.0, 1.2, 5.0)]
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
        series = [TS_MULTIVARIATE, TS_MULTIVARIATE * 1.2]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred, component_reduction=None)

        ref = np.asarray(ref, dtype=float)
        assert ref.shape == (len(series), len(series[0].components))

        m = mlflow.get_run(run.info.run_id).data.metrics
        # aggregate per component = mean over series, no per-series _s{i} keys
        assert m["mae_sine"] == pytest.approx(float(ref[:, 0].mean()), abs=1e-5)
        assert m["mae_sine_1"] == pytest.approx(float(ref[:, 1].mean()), abs=1e-5)
        assert not any(k.endswith(("_s0", "_s1")) for k in m)
        # granular CSV: one row per (component, series)
        rows = _read_per_series_table(run.info.run_id)
        got = {
            (row["key"], int(row["series_index"])): float(row["value"]) for row in rows
        }
        assert got[("mae_sine", 0)] == pytest.approx(ref[0, 0], abs=1e-5)
        assert got[("mae_sine", 1)] == pytest.approx(ref[1, 0], abs=1e-5)
        assert got[("mae_sine_1", 0)] == pytest.approx(ref[0, 1], abs=1e-5)
        assert got[("mae_sine_1", 1)] == pytest.approx(ref[1, 1], abs=1e-5)

    def test_autolog_metric_name_override(self, mlflow_tracking, autolog_context):
        """The metric `name` kwarg overrides only the metric-name token in the key,
        keeping the backtest prefix and the quantile/axis suffixes."""
        actual = TS_UNIVARIATE
        train = TS_UNIVARIATE[:40]
        qmodel = _fit_qlr(train)
        pred = qmodel.predict(n=10, num_samples=200)
        target = TS_UNIVARIATE[40:]

        with autolog_context(log_metrics=True):
            # direct call: name replaces the metric token; suffix (_q0_500) preserved
            with mlflow.start_run() as run_direct:
                dm.mae(actual, actual * 1.1, name="custom")
                dm.mql(target, pred, q=0.5, name="myq")
            # backtest: name replaces the metric token; backtest_ prefix preserved
            with mlflow.start_run() as run_bt:
                _fit_lr().backtest(
                    TS_UNIVARIATE,
                    forecast_horizon=1,
                    start=-1,
                    metric=dm.mae,
                    metric_kwargs={"name": "custom"},
                    retrain=False,
                )

        direct = mlflow.get_run(run_direct.info.run_id).data.metrics
        assert "custom" in direct
        assert "mae" not in direct, "default metric name should be replaced"
        assert "myq_q0.500" in direct, "quantile suffix should be preserved"

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

        labels = [0, 1]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.f1(series, pred, label_reduction=None, labels=labels)

        ref = np.array(ref)
        assert ref.shape == (len(series), len(labels))
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
        rows = _read_per_series_table(run.info.run_id)
        got = {
            (row["key"], int(row["series_index"])): float(row["value"]) for row in rows
        }
        for i in range(2):
            assert got[("f1_label0", i)] == pytest.approx(ref[i][0], abs=1e-5)
            assert got[("f1_label1", i)] == pytest.approx(ref[i][1], abs=1e-5)

    def test_autolog_metric_component_count_mismatch_allowed_when_reduced(
        self, mlflow_tracking, autolog_context
    ):
        """Default mae reduces components to scalars, so mixed component counts
        are valid and log under a single aggregated key."""
        series = [TS_UNIVARIATE, TS_MULTIVARIATE]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = dm.mae(series, pred)

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["mae"] == pytest.approx(float(np.mean(ref)), abs=1e-5)

    def test_autolog_metric_component_count_mismatch_raises(
        self, mlflow_tracking, autolog_context
    ):
        """When components are preserved, mixed component counts raise rather
        than taking names from the first series and mislabeling the rest."""
        series = [TS_UNIVARIATE, TS_MULTIVARIATE]
        pred = [s * 1.1 for s in series]

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                with pytest.raises(ValueError, match="same number of components"):
                    dm.mae(series, pred, component_reduction=None)

        assert not mlflow.get_run(run.info.run_id).data.metrics

    def test_autolog_metric_size_mismatch_raises(
        self, mlflow_tracking, autolog_context, caplog
    ):
        """When the inferred C-axis size doesn't divide the result, logging raises
        (and logs an error), propagating out of the public metric call — the
        metric callback isn't invoked through MLflow's safe_patch, so nothing
        catches it. No metrics are written for the failed call."""
        actual = TS_UNIVARIATE[40:]
        # mae with component_reduction=None on a univariate series produces shape (T,),
        # which is size T — divisible by c_size=1 (1 component × 1 quantile), so we
        # need to force a mismatch.  We do that by monkey-patching _infer_metric_axes
        # to report has_comp_axis=True with a fake 3-component count, making c_size=3
        # while the actual result is shape (T,).
        train = TS_UNIVARIATE[:40]
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
        multi = [TS_UNIVARIATE, TS_UNIVARIATE * 1.2]
        pred_multi = [s * 1.1 for s in multi]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_multi:
                ref = np.asarray(dm.mae(multi, pred_multi), dtype=float)

        assert ref.shape == (len(multi),)
        rows = _read_per_series_table(run_multi.info.run_id)
        assert list(rows[0].keys()) == [
            "key",
            "series_index",
            "step",
            "window_index",
            "value",
        ]
        for idx, (row, ref_i) in enumerate(zip(rows, ref)):
            assert row["key"] == "mae"
            assert int(row["series_index"]) == idx
            assert np.isnan(row["step"])
            assert np.isnan(row["window_index"])
            assert row["value"] == pytest.approx(ref_i, abs=1e-5)

        # single-series: no per-series table artifact should be created
        single = TS_UNIVARIATE
        pred_single = single * 1.1
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run_single:
                dm.mae(single, pred_single)

        artifacts = mlflow_tracking.list_artifacts(run_single.info.run_id)
        assert not any(a.path == "metrics_per_series.json" for a in artifacts), (
            "Single-series input should not write a per-series table artifact"
        )


class TestAutoLogBacktestMetrics:
    def test_autolog_backtest_scalar(self, mlflow_tracking, autolog_context):
        """Default (reduced) backtest of a single univariate series logs one scalar."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE, metric=dm.mae, retrain=False, start=-2
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (1,)
        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae" in run_data.metrics
        np.testing.assert_almost_equal(run_data.metrics["backtest_mae"], ref)

    def test_autolog_backtest_log_metrics_false(self, mlflow_tracking, autolog_context):
        """autolog(log_metrics=False) skips backtest metric logging."""
        with autolog_context(log_metrics=False):
            with mlflow.start_run() as run:
                _fit_lr().backtest(
                    TS_UNIVARIATE, metric=dm.mae, retrain=False, start=-2
                )

        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae" not in run_data.metrics

    def test_autolog_backtest_per_window_steps(self, mlflow_tracking, autolog_context):
        """reduction=None logs per-window values at steps."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.mae,
                    retrain=False,
                    start=-2,
                    reduction=None,
                )

        ref = np.atleast_1d(ref)
        assert ref.shape == (2,)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        assert len(history) == 2
        steps = sorted(m.step for m in history)
        assert steps == list(range(len(history)))
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_array_almost_equal(logged, ref)

    def test_autolog_backtest_per_component(self, mlflow_tracking, autolog_context):
        """component_reduction=None logs one key per component name."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr(TS_MULTIVARIATE).backtest(
                    TS_MULTIVARIATE,
                    metric=dm.mae,
                    retrain=False,
                    start=-2,
                    metric_kwargs={"component_reduction": None},
                )

        ref = np.atleast_1d(ref)
        assert ref.shape == (TS_MULTIVARIATE.n_components,)
        run_data = mlflow.get_run(run.info.run_id).data
        assert "backtest_mae_sine" in run_data.metrics
        assert "backtest_mae_sine_1" in run_data.metrics
        ref = np.asarray(ref, dtype=float)
        assert run_data.metrics["backtest_mae_sine"] == pytest.approx(ref[0], abs=1e-5)
        assert run_data.metrics["backtest_mae_sine_1"] == pytest.approx(
            ref[1], abs=1e-5
        )

    def test_autolog_backtest_per_component_and_q_label(
        self, mlflow_tracking, autolog_context
    ):
        """Aggregated windows; component_reduction=None logs one key per component name and
        quantile / label as {name}_{component_name}_q{quantile / label}.

        E.g. ["backtest_mae_comp0_q0.100", ..., "backtest_mae_comp0_q0.900", "backtest_mae_comp1_q0.100",
        ..., "backtest_mae_comp1_q0.900"]
        """
        quantiles = [0.1, 0.5, 0.9]
        comps = TS_MULTIVARIATE.components
        n_windows = 2
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_qlr(TS_MULTIVARIATE).backtest(
                    TS_MULTIVARIATE,
                    metric=dm.mae,
                    retrain=False,
                    start=-n_windows,
                    num_samples=3,
                    metric_kwargs={"component_reduction": None, "q": quantiles},
                    reduction=np.nanmean,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (len(comps) * len(quantiles),)

        run_data = mlflow.get_run(run.info.run_id).data
        for c_idx, comp in enumerate(comps):
            for q_idx, q in enumerate(quantiles):
                name = f"backtest_mae_{comp}_q{q:.3f}"
                assert name in run_data.metrics
                assert run_data.metrics[name] == pytest.approx(
                    ref[c_idx * len(quantiles) + q_idx], abs=1e-5
                )

    def test_autolog_backtest_per_window_component_and_q_label(
        self, mlflow_tracking, autolog_context
    ):
        """Stepped window metrics; component_reduction=None logs one key per window, component name and
        quantile / label as {name}_{component_name}_q{quantile / label}.

        E.g. ["backtest_mae_comp0_q0.100", ..., "backtest_mae_comp0_q0.900", "backtest_mae_comp1_q0.100",
        ..., "backtest_mae_comp1_q0.900"]
        """
        quantiles = [0.1, 0.5, 0.9]
        comps = TS_MULTIVARIATE.components
        n_windows = 2
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_qlr(TS_MULTIVARIATE).backtest(
                    TS_MULTIVARIATE,
                    metric=dm.mae,
                    retrain=False,
                    start=-n_windows,
                    num_samples=3,
                    metric_kwargs={"component_reduction": None, "q": quantiles},
                    reduction=None,
                )
        ref = np.atleast_2d(ref)
        assert ref.shape == (n_windows, len(comps) * len(quantiles))

        for c_idx, comp in enumerate(comps):
            for q_idx, q in enumerate(quantiles):
                history = mlflow_tracking.get_metric_history(
                    run.info.run_id, f"backtest_mae_{comp}_q{q:.3f}"
                )
                assert len(history) == len(ref)
                steps = sorted(m.step for m in history)
                assert steps == list(range(len(ref)))
                logged = [m.value for m in sorted(history, key=lambda m: m.step)]
                np.testing.assert_allclose(
                    logged, ref[:, c_idx * len(quantiles) + q_idx], atol=1e-5
                )

    @pytest.mark.parametrize("reduction", [None, np.nanmean])
    def test_autolog_backtest_per_timestep_component_and_q_label(
        self, mlflow_tracking, autolog_context, reduction
    ):
        """Stepped horizon metrics; component_reduction=None logs one key per window, component name and
        quantile / label as {name}_{component_name}_q{quantile / label}.

        E.g. ["backtest_mae_comp0_q0.100", ..., "backtest_mae_comp0_q0.900", "backtest_mae_comp1_q0.100",
        ..., "backtest_mae_comp1_q0.900"]
        """
        quantiles = [0.1, 0.5, 0.9]
        comps = TS_MULTIVARIATE.components
        horizon = 3
        n_windows = 2
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_qlr(TS_MULTIVARIATE).backtest(
                    TS_MULTIVARIATE,
                    metric=dm.ae,
                    retrain=False,
                    start=-(horizon + n_windows - 1),
                    forecast_horizon=horizon,
                    num_samples=3,
                    metric_kwargs={"component_reduction": None, "q": quantiles},
                    reduction=reduction,
                )
        if reduction is None:
            assert ref.shape == (n_windows, horizon, len(comps) * len(quantiles))
            ref = np.nanmean(ref, axis=0)

        ref = np.atleast_2d(ref)
        assert ref.shape == (horizon, len(comps) * len(quantiles))

        for c_idx, comp in enumerate(comps):
            for q_idx, q in enumerate(quantiles):
                history = mlflow_tracking.get_metric_history(
                    run.info.run_id, f"backtest_ae_{comp}_q{q:.3f}"
                )
                assert len(history) == len(ref)
                steps = sorted(m.step for m in history)
                assert steps == list(range(len(ref)))
                logged = [m.value for m in sorted(history, key=lambda m: m.step)]
                np.testing.assert_allclose(
                    logged, ref[:, c_idx * len(quantiles) + q_idx], atol=1e-5
                )

    def test_autolog_backtest_multi_metric(self, mlflow_tracking, autolog_context):
        """Multiple metrics are logged under one key each."""
        metrics = [dm.mae, dm.rmse]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=metrics,
                    retrain=False,
                    start=-2,
                )

        ref = np.array(ref)
        assert ref.shape == (len(metrics),)
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
        series = [TS_UNIVARIATE, TS_UNIVARIATE * 1.2]
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr(series).backtest(
                    series, metric=dm.mae, retrain=False, start=-2
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (len(series),)

        run_data = mlflow.get_run(run.info.run_id).data
        # aggregate = mean over series, no per-series _s{i} keys
        assert run_data.metrics["backtest_mae"] == pytest.approx(
            float(np.mean(ref)), abs=1e-5
        )
        assert not any(k.startswith("backtest_mae_s") for k in run_data.metrics)
        # granular per-series breakdown written to a table artifact
        rows = _read_per_series_table(run.info.run_id)
        by_series = {int(row["series_index"]): float(row["value"]) for row in rows}
        assert by_series == pytest.approx(
            {0: float(ref[0]), 1: float(ref[1])}, abs=1e-5
        )

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_multi_series_custom_agg_func(
        self, autolog_context, mlflow_tracking, horizon
    ):
        """autolog()'s agg_func also controls the backtest() aggregation
        (default np.mean). Calls _log_backtest_metrics directly with a
        fabricated result so the per-series values are exact."""
        series = TS_UNIVARIATE
        series = [series] * 3
        with autolog_context(log_metrics=True, agg_func=np.median):
            with mlflow.start_run() as run:
                ref = _fit_lr(series[0]).backtest(
                    metric=dm.mae,
                    series=series,
                    last_points_only=True,
                    start=-(horizon + 1),
                    forecast_horizon=horizon,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (len(series),)  # (n_series,)
        m = mlflow.get_run(run.info.run_id).data.metrics

        assert m["backtest_mae"] == pytest.approx(np.median(ref))

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_per_timestep_with_reduction(
        self, mlflow_tracking, autolog_context, horizon
    ):
        """A per-timestep metric (ae) under default reduction collapses to one
        scalar per forecast horizon."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.ae,
                    retrain=False,
                    start=-(horizon + 1),
                    last_points_only=False,
                    forecast_horizon=horizon,
                    reduction=np.nanmean,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (horizon,)  # (n_windows, horizon)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        assert len(history) == horizon, "Default reduction should yield a single value"
        values = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_array_almost_equal(values, ref)

        # no per-series table due to reduction
        with pytest.raises(mlflow.exceptions.MlflowException):
            _ = _read_per_series_table(run.info.run_id)

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_per_timestep_without_reduction(
        self, mlflow_tracking, autolog_context, horizon
    ):
        """A per-timestep metric (ae) under default reduction collapses to one
        scalar per forecast horizon."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.ae,
                    retrain=False,
                    start=-(horizon + 1),
                    last_points_only=False,
                    forecast_horizon=horizon,
                    reduction=None,
                )
        n_windows = 2
        ref = np.atleast_2d(ref)
        ref = ref.T if ref.shape[1] == n_windows else ref
        assert ref.shape == (n_windows, horizon)  # (n_windows, horizon)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        assert len(history) == horizon, "Default reduction should yield a single value"
        values = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_array_almost_equal(values, np.nanmean(ref, axis=0))

        rows = _read_per_series_table(run.info.run_id)
        assert len(rows) == ref.size
        for t_idx in range(horizon):
            for w_idx in range(n_windows):  # windows
                row = rows[t_idx * n_windows + w_idx]
                assert row["key"] == "backtest_ae"
                assert row["series_index"] == 0
                assert row["step"] == t_idx
                assert row["window_index"] == w_idx
                assert row["value"] == pytest.approx(ref[w_idx, t_idx])

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_historical_forecasts_horizon_inferred(
        self, mlflow_tracking, autolog_context, horizon
    ):
        """When `historical_forecasts` is user-supplied, `backtest()` ignores the
        `forecast_horizon` argument, autologging must infer the true window
        length from the historical forecasts themselves, not from the (unused,
        defaulted) `forecast_horizon` argument."""
        model = _fit_lr()
        hf = model.historical_forecasts(
            TS_UNIVARIATE,
            retrain=False,
            start=-(horizon + 1),
            forecast_horizon=horizon,
            last_points_only=False,
        )
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                # forecast_horizon is not passed (defaults to 1) and would be
                # wrong; the real horizon (4) must come from `hf`.
                ref = model.backtest(
                    TS_UNIVARIATE,
                    historical_forecasts=hf,
                    metric=dm.ae,
                    reduction=None,
                )

        n_windows = 2
        ref = np.atleast_2d(ref)
        ref = ref.T if ref.shape[1] == n_windows else ref
        assert ref.shape == (n_windows, horizon)
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        assert len(history) == horizon, (
            "Expected one step per forecast horizon timestep"
        )
        logged = [m.value for m in sorted(history, key=lambda m: m.step)]
        np.testing.assert_allclose(logged, np.nanmean(ref, axis=0), atol=1e-5)

    def test_autolog_backtest_quantile(self, mlflow_tracking, autolog_context):
        """A quantile metric (mql) logs one key per quantile."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_qlr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.mql,
                    metric_kwargs={"q": [0.1, 0.5, 0.9]},
                    retrain=False,
                    start=-2,
                    num_samples=200,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (3,)  # (n_quantiles,)

        m = mlflow.get_run(run.info.run_id).data.metrics
        for idx, key in enumerate([
            "backtest_mql_q0.100",
            "backtest_mql_q0.500",
            "backtest_mql_q0.900",
        ]):
            assert key in m, f"Expected quantile key {key}"
            assert m[key] == pytest.approx(ref[idx])

    def test_autolog_backtest_mixed_degenerate_axes_keep_metric_names(
        self, mlflow_tracking, autolog_context
    ):
        """Metrics whose differing axes have size one keep their metric names."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                _ = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=[dm.mae, dm.ae],
                    retrain=False,
                    start=-2,
                )

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert "backtest_mae" in m
        assert "backtest_ae" in m
        assert not any(k.startswith("backtest_metrics_") for k in m)

    def test_autolog_backtest_classification_labels_in_data(
        self, mlflow_tracking, autolog_context
    ):
        """f1 with explicit labels present in the series logs finite per-label keys."""
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                _fit_lr(TS_BINARY).backtest(
                    TS_BINARY,
                    metric=dm.f1,
                    metric_kwargs={"label_reduction": None, "labels": [0, 1]},
                    retrain=False,
                    start=10,
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
                _fit_lr(TS_BINARY).backtest(
                    TS_BINARY,
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

    def test_log_backtest_metrics_component_count_mismatch_allowed_when_reduced(
        self, autolog_context, mlflow_tracking
    ):
        """Default mae reduces components to scalars, so mixed component counts
        aggregate normally. Calls _log_backtest_metrics directly."""

        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    metric=dm.mae,
                    metric_kwargs={},
                    series=[TS_UNIVARIATE, TS_MULTIVARIATE],
                    forecast_horizon=1,
                    reduction=np.mean,
                    last_points_only=True,
                    start=-2,
                    retrain=True,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (2,)  # (n_series,)

        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["backtest_mae"] == pytest.approx(np.mean(ref))

    def test_log_backtest_metrics_component_count_mismatch_raises(
        self, mlflow_tracking
    ):
        """When components are preserved, mixed component counts raise rather
        than taking names from the first series only.

        Calls _log_backtest_metrics directly so the raise is not swallowed by
        MLflow's safe_patch wrapper.
        """
        backtest_args = {
            "metric": dm.mae,
            "metric_kwargs": {"component_reduction": None},
            "series": [TS_UNIVARIATE, TS_MULTIVARIATE],
            "forecast_horizon": 1,
            "reduction": np.mean,
            "last_points_only": True,
        }
        result = [1.0, np.array([1.0, 2.0])]

        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            with pytest.raises(ValueError, match="same number of components"):
                _log_backtest_metrics(
                    client,
                    run.info.run_id,
                    result,
                    {**BT_REQUIRED_DEFAUTLS, **backtest_args},
                )

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
                    client,
                    run.info.run_id,
                    np.array([0.5]),
                    {**BT_REQUIRED_DEFAUTLS, **backtest_args},
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
                    client,
                    run.info.run_id,
                    fake_result,
                    {**BT_REQUIRED_DEFAUTLS, **backtest_args},
                )

        assert not mlflow.get_run(run.info.run_id).data.metrics

    def test_log_backtest_metrics_aligns_windows_by_end_date(
        self, autolog_context, mlflow_tracking
    ):
        """A shorter series' window axis aligns from the end, not the start,
        so its last windows overlap the tail of a longer series. Steps and
        ``window_index`` are ``0 .. max_w - 1``. Calls _log_backtest_metrics
        directly with a fabricated result so the window counts per series
        are exact."""
        model = _fit_lr()
        input_length = abs(min(model.lags["target"]))
        n_windows = 2
        series_two_windows = TS_UNIVARIATE[-(input_length + n_windows) :]
        # series 0 has 2 windows; series 1 (shorter, later-starting) has 1
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = model.backtest(
                    metric=dm.mae,
                    metric_kwargs={},
                    series=[series_two_windows, series_two_windows[1:]],
                    forecast_horizon=1,
                    reduction=None,
                    last_points_only=False,
                    start=-n_windows,
                    retrain=False,
                )
        expected_shapes = [(n_windows,), (n_windows - 1,)]
        for ref_i, shape in zip(ref, expected_shapes):
            assert ref_i.shape == shape

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({
            0: ref[0][0],
            1: np.mean([ref[0][1], ref[1][0]]),
        })

        rows = _read_per_series_table(run.info.run_id)
        for s_idx, shape in enumerate(expected_shapes):
            w_size = shape[0]
            for w_idx in range(w_size):
                row = rows[s_idx * n_windows + w_idx]
                window_index = w_idx + n_windows - w_size
                assert row["key"] == "backtest_mae"
                assert row["series_index"] == s_idx
                assert row["step"] == window_index
                assert row["window_index"] == window_index
                assert row["value"] == pytest.approx(ref[s_idx][w_idx])

    def test_log_backtest_metrics_aligns_last_points_only_time_axis(
        self, autolog_context, mlflow_tracking
    ):
        """last_points_only stitches windows into one series scored per real
        timestep. Shorter series align from the end, same as the window-axis
        case, with steps ``0 .. t_max - 1`` rather than end-relative indexes.
        """
        model = _fit_lr()
        input_length = abs(min(model.lags["target"]))
        n_windows = 2
        horizon = 3
        series_two_windows = TS_UNIVARIATE[-(input_length + horizon + n_windows - 1) :]
        # series 0 forecast has 2 timesteps; series 1 (shorter, later-starting) has 1
        with autolog_context(log_metrics=True):
            with mlflow.start_run() as run:
                ref = model.backtest(
                    metric=dm.ae,
                    series=[series_two_windows, series_two_windows[1:]],
                    forecast_horizon=horizon,
                    reduction=None,
                    last_points_only=True,
                    start=-(horizon + n_windows - 1),
                    retrain=False,
                )

        ref = [np.atleast_1d(ref_i) for ref_i in ref]
        expected_shapes = [(n_windows,), (n_windows - 1,)]
        for ref_i, shape in zip(ref, expected_shapes):
            assert ref_i.shape == shape

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({
            0: ref[0][0],
            1: np.mean([ref[0][1], ref[1][0]]),
        })

        rows = _read_per_series_table(run.info.run_id)
        for s_idx, shape in enumerate(expected_shapes):
            w_size = shape[0]
            for w_idx in range(w_size):
                row = rows[s_idx * n_windows + w_idx]
                window_index = w_idx + n_windows - w_size
                assert row["key"] == "backtest_ae"
                assert row["series_index"] == s_idx
                assert row["step"] == window_index
                assert np.isnan(row["window_index"])
                assert row["value"] == pytest.approx(ref[s_idx][w_idx])

    def test_log_backtest_metrics_aligns_windows_by_start(
        self, autolog_context, mlflow_tracking
    ):
        """A shorter series' window axis aligns from the start when
        ``series_align='start'``, so its first windows overlap the head of a
        longer series."""
        model = _fit_lr()
        input_length = abs(min(model.lags["target"]))
        n_windows = 2
        series_two_windows = TS_UNIVARIATE[-(input_length + n_windows) :]
        with autolog_context(log_metrics=True, series_align="start"):
            with mlflow.start_run() as run:
                ref = model.backtest(
                    metric=dm.mae,
                    metric_kwargs={},
                    series=[series_two_windows, series_two_windows[1:]],
                    forecast_horizon=1,
                    reduction=None,
                    last_points_only=False,
                    start=-n_windows,
                    retrain=False,
                )
        expected_shapes = [(n_windows,), (n_windows - 1,)]
        for ref_i, shape in zip(ref, expected_shapes):
            assert ref_i.shape == shape

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({
            0: np.mean([ref[0][0], ref[1][0]]),
            1: ref[0][1],
        })

        rows = _read_per_series_table(run.info.run_id)
        for s_idx, shape in enumerate(expected_shapes):
            w_size = shape[0]
            for w_idx in range(w_size):
                row = rows[s_idx * n_windows + w_idx]
                assert row["key"] == "backtest_mae"
                assert row["series_index"] == s_idx
                assert row["step"] == w_idx
                assert row["window_index"] == w_idx
                assert row["value"] == pytest.approx(ref[s_idx][w_idx])

    def test_log_backtest_metrics_aligns_last_points_only_by_start(
        self, autolog_context, mlflow_tracking
    ):
        """last_points_only with ``series_align='start'`` aligns shorter series
        at early timesteps."""
        model = _fit_lr()
        input_length = abs(min(model.lags["target"]))
        n_windows = 2
        horizon = 3
        series_two_windows = TS_UNIVARIATE[-(input_length + horizon + n_windows - 1) :]
        with autolog_context(log_metrics=True, series_align="start"):
            with mlflow.start_run() as run:
                ref = model.backtest(
                    metric=dm.ae,
                    series=[series_two_windows, series_two_windows[1:]],
                    forecast_horizon=horizon,
                    reduction=None,
                    last_points_only=True,
                    start=-(horizon + n_windows - 1),
                    retrain=False,
                )

        ref = [np.atleast_1d(ref_i) for ref_i in ref]
        expected_shapes = [(n_windows,), (n_windows - 1,)]
        for ref_i, shape in zip(ref, expected_shapes):
            assert ref_i.shape == shape

        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        logged = {m.step: m.value for m in history}
        assert logged == pytest.approx({
            0: np.mean([ref[0][0], ref[1][0]]),
            1: ref[0][1],
        })

        rows = _read_per_series_table(run.info.run_id)
        for s_idx, shape in enumerate(expected_shapes):
            w_size = shape[0]
            for w_idx in range(w_size):
                row = rows[s_idx * n_windows + w_idx]
                assert row["key"] == "backtest_ae"
                assert row["series_index"] == s_idx
                assert row["step"] == w_idx
                assert np.isnan(row["window_index"])
                assert row["value"] == pytest.approx(ref[s_idx][w_idx])

    @pytest.mark.parametrize("config", list(itertools.product([1, 3], [True, False])))
    def test_autolog_backtest_log_aggregate_scalar(
        self, autolog_context, mlflow_tracking, config
    ):
        """log_backtest_aggregate logs backtest_agg_* equal to scalar backtest_*."""
        horizon, lpo = config
        n_windows = 2
        with autolog_context(log_metrics=True, log_backtest_aggregate=True):
            with mlflow.start_run() as run:
                ref = _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.mae,
                    retrain=False,
                    start=-(horizon + n_windows - 1),
                    last_points_only=lpo,
                    forecast_horizon=horizon,
                    reduction=None,
                )
        ref = np.nanmean(ref)
        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["backtest_agg_mae"] == pytest.approx(ref)

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_log_aggregate_stepped(
        self, autolog_context, mlflow_tracking, horizon
    ):
        """log_backtest_aggregate collapses stepped backtest metrics via agg_func."""
        with autolog_context(
            log_metrics=True, log_backtest_aggregate=True, agg_func=np.nanmean
        ):
            with mlflow.start_run() as run:
                _fit_lr().backtest(
                    TS_UNIVARIATE,
                    metric=dm.ae,
                    retrain=False,
                    start=-(horizon + 1),
                    last_points_only=False,
                    forecast_horizon=horizon,
                    reduction=np.nanmean,
                )
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_ae")
        stepped_values = [m.value for m in sorted(history, key=lambda m: m.step)]
        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["backtest_agg_ae"] == pytest.approx(np.nanmean(stepped_values))

    @pytest.mark.parametrize("horizon", [1, 3])
    def test_autolog_backtest_log_aggregate_custom_agg_func(
        self, autolog_context, mlflow_tracking, horizon
    ):
        """log_backtest_aggregate uses autolog's agg_func over all logged steps."""
        series = TS_UNIVARIATE
        series = [series * 0.9, series, series * 1.1]
        with autolog_context(
            log_metrics=True, log_backtest_aggregate=True, agg_func=np.median
        ):
            with mlflow.start_run() as run:
                ref = _fit_lr(series[0]).backtest(
                    metric=dm.mae,
                    series=series,
                    last_points_only=True,
                    start=-(horizon + 1),
                    forecast_horizon=horizon,
                )
        ref = np.atleast_1d(ref)
        assert ref.shape == (len(series),)
        m = mlflow.get_run(run.info.run_id).data.metrics
        assert m["backtest_agg_mae"] == pytest.approx(np.median(ref))


class TestAutoLogMetricHelperFunctions:
    @pytest.mark.parametrize(
        "metric_name, metric_kwargs, expected",
        [
            ("mae", {}, dict(has_time_axis=False, has_comp_axis=False, axis_size=1)),
            ("ae", {}, dict(has_time_axis=True, has_comp_axis=False, axis_size=1)),
            ("mae", {"component_reduction": None}, dict(has_comp_axis=True)),
        ],
    )
    def test_infer_metric_axes_reductions(self, metric_name, metric_kwargs, expected):
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

    def test_infer_metric_axes_quantiles(self):
        _, _, axis_labels = _infer_metric_axes(dm.mql, {"q": [0.1, 0.5, 0.9]})
        assert axis_labels == ["_q0.100", "_q0.500", "_q0.900"]

    def test_infer_metric_axes_quantile_interval(self):
        has_time, _, axis_labels = _infer_metric_axes(dm.iw, {"q_interval": (0.1, 0.9)})
        assert axis_labels == ["_qi0.800"]
        assert has_time is True

    def test_infer_metric_axes_unknown_labels_raises(self):
        """label_reduction=None with no explicit labels cannot determine the number
        of output labels ahead of time, so this raises rather than falling back."""
        with pytest.raises(ValueError, match="requires explicit `labels`"):
            _infer_metric_axes(dm.f1, {"label_reduction": None})

    def test_build_metric_keys_components_and_quantiles(self):
        """Shared key builder expands components x quantile suffixes per metric."""
        metric_axes = [
            (False, True, ["_q0.100", "_q0.900"]),
            (False, True, ["_label0", "_label1"]),
        ]
        metric_keys = _build_metric_keys(
            ["mae", "f1"],
            ["temp", "hum"],
            has_comp_axis=True,
            metric_axes=metric_axes,
            prefix="backtest_",
        )
        c_size = len(metric_keys[0])
        assert c_size == 4
        assert metric_keys == [
            [
                "backtest_mae_temp_q0.100",
                "backtest_mae_temp_q0.900",
                "backtest_mae_hum_q0.100",
                "backtest_mae_hum_q0.900",
            ],
            [
                "backtest_f1_temp_label0",
                "backtest_f1_temp_label1",
                "backtest_f1_hum_label0",
                "backtest_f1_hum_label1",
            ],
        ]

    def test_build_metric_keys_no_components_no_prefix(self):
        """Without components, each metric gets one key per axis label."""
        metric_axes = [(False, False, ["_q0.500"])]
        metric_keys = _build_metric_keys(
            ["mql"],
            ["ignored"],
            has_comp_axis=False,
            metric_axes=metric_axes,
        )
        c_size = len(metric_keys[0])
        assert c_size == 1
        assert metric_keys == [["mql_q0.500"]]

    def test_flush_logged_metrics_aggregates_and_writes_table(self, mlflow_tracking):
        """Cells are aggregated with agg_func; supplied table rows are persisted."""
        agg = {
            ("mae", 0): [1.0, 3.0],
            ("mae", 1): [10.0, 30.0],
        }
        rows = [
            {"key": "mae", "series_index": 0, "step": 0, "value": 1.0},
            {"key": "mae", "series_index": 1, "step": 0, "value": 3.0},
            {"key": "mae", "series_index": 0, "step": 1, "value": 10.0},
            {"key": "mae", "series_index": 1, "step": 1, "value": 30.0},
        ]
        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _flush_logged_metrics(
                client, run.info.run_id, agg, agg_func=np.mean, table_rows=rows
            )
            client.flush(synchronous=True)

        history0 = mlflow_tracking.get_metric_history(run.info.run_id, "mae")
        logged = {m.step: m.value for m in history0}
        assert logged[0] == pytest.approx(2.0)
        assert logged[1] == pytest.approx(20.0)
        table = mlflow.load_table(
            artifact_file="metrics_per_series.json", run_ids=[run.info.run_id]
        )
        assert len(table) == 4

    def test_flush_logged_metrics_skips_table_without_rows(self, mlflow_tracking):
        """Omitting table rows logs the aggregate only."""
        agg = {("mae", 0): [1.5]}
        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _flush_logged_metrics(client, run.info.run_id, agg, agg_func=np.mean)
            client.flush(synchronous=True)

        assert mlflow.get_run(run.info.run_id).data.metrics["mae"] == pytest.approx(1.5)
        artifacts = mlflow_tracking.list_artifacts(run.info.run_id)
        assert not any(a.path == "metrics_per_series.json" for a in artifacts)

    def test_flush_logged_metrics_backtest_aggregate(self, mlflow_tracking):
        """log_backtest_aggregate logs backtest_agg_* as agg_func over all steps."""
        agg = {
            ("backtest_mae", 0): [1.0, 3.0],
            ("backtest_mae", 1): [10.0, 30.0],
        }
        with mlflow.start_run() as run:
            client = MlflowAutologgingQueueingClient()
            _flush_logged_metrics(
                client,
                run.info.run_id,
                agg,
                agg_func=np.mean,
                log_backtest_aggregate=True,
            )
            client.flush(synchronous=True)

        m = mlflow.get_run(run.info.run_id).data.metrics
        history = mlflow_tracking.get_metric_history(run.info.run_id, "backtest_mae")
        logged = {h.step: h.value for h in history}
        assert logged[0] == pytest.approx(2.0)
        assert logged[1] == pytest.approx(20.0)
        assert m["backtest_agg_mae"] == pytest.approx(np.mean([2.0, 20.0]))

    def test_resolve_backtest_layout_series_reduction_disables_windows(self):
        """series_reduction in metric_kwargs collapses the window axis."""
        backtest_kwargs = {
            "last_points_only": False,
            "reduction": None,
            "forecast_horizon": 3,
            "historical_forecasts": None,
        }
        has_windows, _ = _resolve_backtest_layout(
            backtest_kwargs=backtest_kwargs,
            metric=dm.mae,
            metric_kwargs={"series_reduction": np.mean},
            is_single_series=True,
        )
        assert has_windows is False

    def test_reshape_metric_result_invalid_scalar_shape_raises(self):
        """Mis-sized reduced results raise a descriptive ValueError."""
        metric_keys = _build_metric_keys(
            ["mae"],
            TS_UNIVARIATE.components.tolist(),
            has_comp_axis=False,
            metric_axes=[(False, False, [""])],
        )
        with pytest.raises(ValueError, match="expected a single scalar"):
            _reshape_metric_result(
                TS_UNIVARIATE,
                [1.0, 2.0],
                metric_keys=metric_keys,
                has_time_axis=False,
                has_windows=False,
                forecast_horizon=None,
                is_backtest=True,
            )

    def test_log_per_series_table_skips_empty_rows(self, mlflow_tracking):
        """An empty row list is a no-op and does not create an artifact."""
        with mlflow.start_run() as run:
            _log_per_series_table([])

        artifacts = mlflow_tracking.list_artifacts(run.info.run_id)
        assert not any(a.path == "metrics_per_series.json" for a in artifacts)

    def test_run_has_logged_model_handles_get_run_failure(self, monkeypatch):
        """_run_has_logged_model returns False when run lookup fails."""

        def _fail_get_run(run_id):
            raise RuntimeError("connection failed")

        monkeypatch.setattr(mlflow, "get_run", _fail_get_run)
        assert _run_has_logged_model("fake_run_id") is False

    def test_unregister_metric_callback_missing_is_noop(self):
        """unregister_metric_callback ignores callbacks that were never registered."""

        def _dummy_callback(func, result, args, kwargs):
            pass

        unregister_metric_callback(_dummy_callback)


def test_z_mlflow_import_without_dependency(monkeypatch):
    """Importing darts.utils.mlflow without mlflow installed raises ImportError."""
    import builtins
    import importlib

    mod_name = "darts.utils.mlflow"
    saved = sys.modules.pop(mod_name, None)
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="mlflow"):
        importlib.import_module(mod_name)

    sys.modules.pop(mod_name, None)
    if saved is not None:
        sys.modules[mod_name] = saved
    else:
        importlib.import_module(mod_name)
