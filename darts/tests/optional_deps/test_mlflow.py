import os

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.tests.conftest import MLFLOW_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev
from darts.utils.mlflow import _DartsModelWrapper

if not MLFLOW_AVAILABLE:
    pytest.skip(
        f"MLflow not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import mlflow

from darts.models import ExponentialSmoothing, LinearRegressionModel
from darts.utils.mlflow import (
    _deserialize_timeseries_from_pyfunc,
    _serialize_timeseries_for_pyfunc,
    autolog,
    infer_signature,
    load_model,
    log_model,
    prepare_pyfunc_input,
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

    def test_serialize_deserialize_single(self):
        """Test serialization of single TimeSeries"""
        json_str = _serialize_timeseries_for_pyfunc(self.ts_univariate)
        assert isinstance(json_str, str)
        assert json_str.startswith("{")

        ts_restored = _deserialize_timeseries_from_pyfunc(json_str)
        assert isinstance(ts_restored, TimeSeries)
        np.testing.assert_array_almost_equal(
            self.ts_univariate.values(), ts_restored.values(), decimal=6
        )

    def test_serialize_deserialize_list(self):
        """Test serialization of list of TimeSeries"""
        ts_list = [self.ts_univariate, self.ts_univariate * 2]

        json_str = _serialize_timeseries_for_pyfunc(ts_list)
        assert isinstance(json_str, str)
        assert json_str.startswith("[")

        ts_list_restored = _deserialize_timeseries_from_pyfunc(json_str)
        assert isinstance(ts_list_restored, list)
        assert len(ts_list_restored) == 2
        np.testing.assert_array_almost_equal(
            ts_list[0].values(), ts_list_restored[0].values(), decimal=6
        )

    def test_serialize_with_static_covariates(self):
        """Test that static covariates are preserved through serialization"""
        json_str = _serialize_timeseries_for_pyfunc(self.ts_with_static)
        ts_restored = _deserialize_timeseries_from_pyfunc(json_str)

        assert ts_restored.static_covariates is not None
        pd.testing.assert_frame_equal(
            self.ts_with_static.static_covariates,
            ts_restored.static_covariates,
            check_dtype=False,
        )

    def test_serialize_multivariate(self):
        """Test serialization of multivariate TimeSeries"""
        json_str = _serialize_timeseries_for_pyfunc(self.ts_multivariate)
        ts_restored = _deserialize_timeseries_from_pyfunc(json_str)

        assert ts_restored.n_components == 2
        np.testing.assert_array_almost_equal(
            self.ts_multivariate.values(), ts_restored.values(), decimal=6
        )

    def test_prepare_input_simple(self):
        """Test prepare_pyfunc_input with only n parameter"""
        input_df = prepare_pyfunc_input(n=10)

        assert isinstance(input_df, pd.DataFrame)
        assert list(input_df.columns) == ["n", "num_samples"]
        assert input_df["n"].iloc[0] == 10
        assert input_df["num_samples"].iloc[0] == 1

    def test_prepare_input_with_series(self):
        """Test prepare_pyfunc_input with series parameter"""
        input_df = prepare_pyfunc_input(n=10, series=self.ts_univariate)

        assert "_darts_series" in input_df.columns
        json_str = input_df["_darts_series"].iloc[0]
        assert json_str.startswith("{")

        ts_restored = _deserialize_timeseries_from_pyfunc(json_str)
        np.testing.assert_array_almost_equal(
            self.ts_univariate.values(), ts_restored.values(), decimal=6
        )

    def test_prepare_input_with_covariates(self):
        """Test prepare_pyfunc_input with all covariate types"""
        input_df = prepare_pyfunc_input(
            n=10,
            series=self.ts_univariate,
            past_covariates=self.ts_past_cov,
            future_covariates=self.ts_future_cov,
        )

        assert "_darts_series" in input_df.columns
        assert "_darts_past_covariates" in input_df.columns
        assert "_darts_future_covariates" in input_df.columns

        ts_restored = _deserialize_timeseries_from_pyfunc(
            input_df["_darts_series"].iloc[0]
        )
        past_cov_restored = _deserialize_timeseries_from_pyfunc(
            input_df["_darts_past_covariates"].iloc[0]
        )
        future_cov_restored = _deserialize_timeseries_from_pyfunc(
            input_df["_darts_future_covariates"].iloc[0]
        )

        np.testing.assert_array_almost_equal(
            self.ts_univariate.values(), ts_restored.values(), decimal=6
        )
        np.testing.assert_array_almost_equal(
            self.ts_past_cov.values(), past_cov_restored.values(), decimal=6
        )
        np.testing.assert_array_almost_equal(
            self.ts_future_cov.values(), future_cov_restored.values(), decimal=6
        )

    def test_prepare_input_list_series(self):
        """Test prepare_pyfunc_input with list of series"""
        series_list = [self.ts_univariate, self.ts_univariate * 2]
        input_df = prepare_pyfunc_input(n=5, series=series_list)

        assert "_darts_series" in input_df.columns
        json_str = input_df["_darts_series"].iloc[0]
        assert json_str.startswith("[")

        ts_list_restored = _deserialize_timeseries_from_pyfunc(json_str)
        assert isinstance(ts_list_restored, list)
        assert len(ts_list_restored) == 2
        np.testing.assert_array_almost_equal(
            series_list[0].values(), ts_list_restored[0].values(), decimal=6
        )
        np.testing.assert_array_almost_equal(
            series_list[1].values(), ts_list_restored[1].values(), decimal=6
        )

    def test_wrapper_simple_prediction(self):
        """Test PyFunc wrapper with simple statistical model"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        wrapper = _DartsModelWrapper(model)
        input_df = prepare_pyfunc_input(n=12)
        output_df = wrapper.predict(input_df)

        # Compare with direct model prediction
        expected_df = model.predict(n=12).to_dataframe()

        assert isinstance(output_df, pd.DataFrame)
        assert len(output_df) == 12
        assert list(output_df.columns) == list(expected_df.columns)
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )

    def test_wrapper_with_series(self):
        """Test PyFunc wrapper with global model and series parameter"""
        train, test = self.ts_univariate.split_before(0.7)

        model = LinearRegressionModel(lags=5)
        model.fit(train)

        wrapper = _DartsModelWrapper(model)
        input_df = prepare_pyfunc_input(n=5, series=test)
        output_df = wrapper.predict(input_df)

        # Compare with direct model prediction
        expected_df = model.predict(n=5, series=test).to_dataframe()

        assert len(output_df) == 5
        assert list(output_df.columns) == list(expected_df.columns)
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )

    def test_wrapper_params_override(self):
        """Test that params dict overrides DataFrame values"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        wrapper = _DartsModelWrapper(model)
        input_df = prepare_pyfunc_input(n=5)

        output_df = wrapper.predict(input_df, params={"n": 10})
        assert len(output_df) == 10

        # Verify the override actually produced the right prediction (n=10, not n=5)
        expected_df = model.predict(n=10).to_dataframe()
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )

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

    def test_save_with_signature(self, tmpdir_fn):
        """Test that signature is saved in MLmodel file"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        signature = infer_signature(model, n=5)
        model_path = os.path.join(tmpdir_fn, "test_model")
        save_model(model, model_path, signature=signature)

        mlmodel = mlflow.models.Model.load(os.path.join(model_path, "MLmodel"))
        loaded_signature = mlmodel.signature

        assert loaded_signature is not None
        assert loaded_signature.inputs is not None
        assert loaded_signature.outputs is not None

        input_names = [spec.name for spec in loaded_signature.inputs.inputs]
        assert input_names == ["n", "num_samples"]

        output_names = [spec.name for spec in loaded_signature.outputs.inputs]
        expected_output_names = list(model.predict(n=5).to_dataframe().columns)
        assert output_names == expected_output_names

    def test_log_model_basic(self, tmpdir_fn):
        """Test basic log_model functionality"""
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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

    def test_pyfunc_load_and_predict(self, tmpdir_fn):
        """Test loading model as pyfunc and making predictions"""
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
        mlflow.set_experiment("test_experiment")

        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        pyfunc_model = mlflow.pyfunc.load_model(log_info.model_uri)
        input_df = prepare_pyfunc_input(n=10)
        output_df = pyfunc_model.predict(input_df)

        # Compare with direct model prediction
        expected_df = model.predict(n=10).to_dataframe()

        assert isinstance(output_df, pd.DataFrame)
        assert len(output_df) == 10
        assert list(output_df.columns) == list(expected_df.columns)
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )

    def test_pyfunc_with_covariates(self, tmpdir_fn):
        """Test pyfunc with model that requires covariates"""
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
        mlflow.set_experiment("test_experiment")

        train = self.ts_univariate[:40]
        past_cov_train = self.ts_past_cov[:52]

        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(train, past_covariates=past_cov_train)

        with mlflow.start_run():
            signature = infer_signature(
                model, series=train, past_covariates=past_cov_train, n=5
            )
            log_info = log_model(model, name="model", signature=signature)

        pyfunc_model = mlflow.pyfunc.load_model(log_info.model_uri)
        test = self.ts_univariate[40:]
        past_cov_test = self.ts_past_cov[40:]

        input_df = prepare_pyfunc_input(n=5, series=test, past_covariates=past_cov_test)
        output_df = pyfunc_model.predict(input_df)

        # Compare with direct model prediction using same series/covariates
        expected_df = model.predict(
            n=5, series=test, past_covariates=past_cov_test
        ).to_dataframe()

        assert len(output_df) == 5
        assert list(output_df.columns) == list(expected_df.columns)
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )

    def test_infer_signature_simple(self):
        """Test signature inference without series"""
        model = ExponentialSmoothing()
        model.fit(self.ts_univariate)

        signature = infer_signature(model, n=5)

        assert signature.inputs is not None
        assert signature.outputs is not None

        input_cols = [spec.name for spec in signature.inputs.inputs]
        assert input_cols == ["n", "num_samples"]

        output_cols = [spec.name for spec in signature.outputs.inputs]
        expected_output_cols = list(model.predict(n=5).to_dataframe().columns)
        assert output_cols == expected_output_cols

    def test_infer_signature_with_covariates(self):
        """Test signature inference with covariates"""
        train = self.ts_univariate[:40]
        past_cov = self.ts_past_cov[:52]

        model = LinearRegressionModel(lags=5, lags_past_covariates=3)
        model.fit(train, past_covariates=past_cov)

        signature = infer_signature(model, series=train, past_covariates=past_cov, n=5)

        input_cols = [spec.name for spec in signature.inputs.inputs]
        assert "n" in input_cols
        assert "num_samples" in input_cols
        assert "_darts_series" in input_cols
        assert "_darts_past_covariates" in input_cols
        # future covariates were not provided, so should not appear
        assert "_darts_future_covariates" not in input_cols

        output_cols = [spec.name for spec in signature.outputs.inputs]
        expected_output_cols = list(
            model.predict(n=5, series=train, past_covariates=past_cov)
            .to_dataframe()
            .columns
        )
        assert output_cols == expected_output_cols

    def test_autolog_enable_disable(self, tmpdir_fn):
        """Test autolog can be enabled and disabled"""
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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
        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
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

    def test_serialize_invalid_type(self):
        """Test that serializing invalid type raises error"""
        with pytest.raises(TypeError):
            _serialize_timeseries_for_pyfunc("not_a_timeseries")

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

    @pytest.mark.parametrize("use_torch", [False, True])
    def test_pyfunc_multiple_models(self, tmpdir_fn, use_torch):
        """Test pyfunc for statistical and torch models"""
        if use_torch and not TORCH_AVAILABLE:
            pytest.skip("torch not available")

        mlflow.set_tracking_uri(f"file://{tmpdir_fn}")
        mlflow.set_experiment("test_experiment")

        if use_torch:
            model = NBEATSModel(
                input_chunk_length=4,
                output_chunk_length=2,
                n_epochs=1,
                **tfm_kwargs_dev,
            )
        else:
            model = ExponentialSmoothing()

        model.fit(self.ts_univariate)

        with mlflow.start_run():
            log_info = log_model(model, name="model")

        pyfunc_model = mlflow.pyfunc.load_model(log_info.model_uri)
        input_df = prepare_pyfunc_input(n=2)
        output_df = pyfunc_model.predict(input_df)

        # Compare with direct model prediction
        expected_df = model.predict(n=2).to_dataframe()

        assert len(output_df) == 2
        assert list(output_df.columns) == list(expected_df.columns)
        np.testing.assert_array_almost_equal(
            output_df.values, expected_df.values, decimal=4
        )
