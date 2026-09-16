import pytest

from darts.tests.conftest import ONNX_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev

if not (TORCH_AVAILABLE and ONNX_AVAILABLE):
    pytest.skip(
        f"Torch or Onnx not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import os.path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

import darts.utils.timeseries_generation as tg
from darts.models import (
    BlockRNNModel,
    NHiTSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TFTModel,
    TiDEModel,
)
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.utils.onnx.export import (
    OnnxExportBundle,
    _flatten_module_state,
    _save_onnx_export,
    _unflatten_module_state,
)
from darts.utils.onnx.inference import (
    ONNX_SPEC_METADATA_KEY,
    OnnxModelSpec,
    prepare_onnx_inputs,
    run_onnx_prediction,
)

torch_model_cls = [
    BlockRNNModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
]

tfm_kwargs_dev = {**tfm_kwargs_dev, **{"random_state": 42}}


def _onnx_files(directory: str) -> list[str]:
    return sorted(f for f in os.listdir(directory) if f.endswith(".onnx"))


class TestOnnx:
    ts_tg = tg.linear_timeseries(start_value=0, end_value=100, length=30).astype(
        "float32"
    )
    ts_tg_with_static = ts_tg.with_static_covariates(
        pd.Series(data=[12], index=["loc"])
    )
    ts_pc = tg.constant_timeseries(value=123.4, length=300).astype("float32")
    ts_fc = tg.sine_timeseries(length=40).astype("float32")

    def _make_model(self, model_cls, **extra):
        if model_cls is RNNModel:
            return RNNModel(
                input_chunk_length=4,
                training_length=6,
                n_epochs=1,
                **extra,
                **tfm_kwargs_dev,
            )
        return model_cls(
            input_chunk_length=4,
            output_chunk_length=2,
            n_epochs=1,
            **extra,
            **tfm_kwargs_dev,
        )

    def _series_for(self, model):
        return (
            self.ts_tg_with_static if model.supports_static_covariates else self.ts_tg
        )

    def _load_spec(self, onnx_filename, session=None):
        if session is None:
            session = ort.InferenceSession(onnx_filename)
        return OnnxModelSpec.from_session(session), session

    def _onnx_pred(self, onnx_filename, model, n, series=None, session=None):
        spec, session = self._load_spec(onnx_filename, session=session)
        return run_onnx_prediction(
            n=n,
            session=session,
            series=series if series is not None else self._series_for(model),
            past_covariates=self.ts_pc if model.uses_past_covariates else None,
            future_covariates=self.ts_fc if model.uses_future_covariates else None,
            verbose=False,
        )

    def _assert_forecasts_equal(
        self, actual, expected, *, decimal=4, rtol=None, atol=None
    ):
        assert actual.shape == expected.shape
        assert actual.time_index.equals(expected.time_index)
        assert actual.components.equals(expected.components)
        if rtol is not None:
            np.testing.assert_allclose(
                actual.all_values(), expected.all_values(), rtol=rtol, atol=atol
            )
        else:
            np.testing.assert_array_almost_equal(
                actual.all_values(), expected.all_values(), decimal=decimal
            )

    @pytest.mark.parametrize("model_cls", torch_model_cls)
    def test_onnx_save_load(self, tmpdir_fn, model_cls):
        model = self._make_model(model_cls)
        onnx_filename = f"test_onnx_{model.model_name}.onnx"

        with pytest.raises(ValueError) as msg:
            model.to_onnx("dummy_name.onnx")
        assert "`fit()` needs to be called before `to_onnx()`." in str(msg.value)

        series = self._series_for(model)
        model.fit(
            series=series,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        pred = model.predict(2)

        model.to_onnx(onnx_filename)
        assert os.path.exists(onnx_filename)
        assert not os.path.exists(f"{onnx_filename}.spec.json")

        n_onnx_files = len(_onnx_files(tmpdir_fn))
        model.to_onnx()
        assert len(_onnx_files(tmpdir_fn)) == n_onnx_files + 1

        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)
        metadata = {prop.key: prop.value for prop in onnx_model.metadata_props}
        assert ONNX_SPEC_METADATA_KEY in metadata
        onnx_pred = self._onnx_pred(onnx_filename, model, n=2)
        self._assert_forecasts_equal(onnx_pred, pred)

    @pytest.mark.parametrize("clean", [True, False])
    def test_onnx_from_ckpt(self, tmpdir_fn, clean):
        """Check that creating the onnx export from a model directly loaded from a checkpoint work as expected"""
        model_cls = BlockRNNModel
        model = self._make_model(model_cls)
        onnx_filename = f"test_onnx_{model.model_name}.onnx"
        onnx_filename2 = f"test_onnx_{model.model_name}_weights.onnx"
        ckpt_filename = f"test_ckpt_{model.model_name}.pt"

        series = self._series_for(model)
        model.fit(
            series=series,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        model.save(ckpt_filename, clean=clean)

        pl_trainer_kwargs = tfm_kwargs_dev["pl_trainer_kwargs"] if clean else None
        model_loaded = model_cls.load(
            ckpt_filename, pl_trainer_kwargs=pl_trainer_kwargs
        )
        pred = model_loaded.predict(
            n=2,
            series=self._series_for(model_loaded),
            past_covariates=self.ts_pc if model_loaded.uses_past_covariates else None,
            future_covariates=self.ts_fc
            if model_loaded.uses_future_covariates
            else None,
        )

        model_loaded.to_onnx(onnx_filename)
        onnx_pred = self._onnx_pred(onnx_filename, model_loaded, n=2)
        self._assert_forecasts_equal(onnx_pred, pred)

        model_weights = self._make_model(model_cls)
        model_weights.load_weights(ckpt_filename)
        pred_weights = model_weights.predict(
            n=2,
            series=self._series_for(model_weights),
            past_covariates=self.ts_pc if model_weights.uses_past_covariates else None,
            future_covariates=self.ts_fc
            if model_weights.uses_future_covariates
            else None,
        )

        model_weights.to_onnx(onnx_filename2)
        onnx_pred_weights = self._onnx_pred(onnx_filename2, model_weights, n=2)
        self._assert_forecasts_equal(onnx_pred_weights, pred_weights)

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM"])
    def test_onnx_rnn_state_io(self, tmpdir_fn, rnn_type):
        """RNN uses the same graph as other models, plus flattened state I/O."""
        model = self._make_model(RNNModel, model=rnn_type)
        model.fit(series=self.ts_tg)
        pred = model.predict(n=2)
        onnx_filename = f"test_onnx_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)

        onnx_model = onnx.load(onnx_filename)
        spec, _ = self._load_spec(onnx_filename)
        onnx.checker.check_model(onnx_model)
        output_names = [node.name for node in onnx_model.graph.output]

        assert "prediction" in output_names
        assert "past_target" in spec.feature_input_names
        assert spec.stepwise_state
        assert any(name.startswith("state_out_") for name in output_names)
        if rnn_type == "LSTM":
            assert len(spec.state_input_names) == 2
        else:
            assert len(spec.state_input_names) == 1
        ort.InferenceSession(onnx_filename)
        pred_onnx = self._onnx_pred(onnx_filename, model, n=2)
        self._assert_forecasts_equal(pred_onnx, pred)

    def test_onnx_roll_size_smaller_than_output_chunk_length(self, tmpdir_fn):
        """First autoregressive step must pass output_chunk_length future covariates."""
        model = BlockRNNModel(
            input_chunk_length=4,
            output_chunk_length=2,
            hidden_dim=4,
            n_epochs=1,
            **tfm_kwargs_dev,
        )
        series = self.ts_tg
        future_cov = self.ts_fc
        model.fit(series=series, future_covariates=future_cov)

        n = 5
        roll_size = 1
        pred = model.predict(
            n=n,
            series=series,
            future_covariates=future_cov,
            roll_size=roll_size,
        )
        onnx_filename = f"test_roll_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)
        onnx_pred = run_onnx_prediction(
            n=n,
            session=session,
            series=series,
            future_covariates=future_cov,
            roll_size=roll_size,
            verbose=False,
        )
        self._assert_forecasts_equal(onnx_pred, pred)

        with pytest.raises(ValueError) as msg:
            _ = run_onnx_prediction(
                n=n,
                session=session,
                series=series,
                future_covariates=future_cov,
                roll_size=model.output_chunk_length + 1,
                verbose=False,
            )
        assert (
            "`roll_size` must be an integer between 1 and `output_chunk_length`"
            in str(msg.value)
        )

    def test_onnx_icl_smaller_than_roll_size_with_covariates(self, tmpdir_fn):
        """Autoregression with `input_chunk_length < roll_size` must consume extra covariates."""
        model = NLinearModel(
            input_chunk_length=1,
            output_chunk_length=4,
            n_epochs=1,
            **tfm_kwargs_dev,
        )
        series = self.ts_tg
        past_cov = self.ts_pc
        future_cov = self.ts_fc
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        n = 8
        pred = model.predict(
            n=n,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        onnx_filename = f"test_icl_lt_roll_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)
        onnx_pred = run_onnx_prediction(
            n=n,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        self._assert_forecasts_equal(onnx_pred, pred)

    @pytest.mark.parametrize("model_cls", [RNNModel, TiDEModel])
    def test_onnx_autoregressive_horizon(self, tmpdir_fn, model_cls):
        """ONNX autoregression matches torch predict when n > output_chunk_length."""
        model = self._make_model(model_cls)
        series = self.ts_tg
        past_cov = self.ts_pc if model.supports_past_covariates else None
        future_cov = self.ts_fc if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        n = 5
        pred = model.predict(
            n, series=series, past_covariates=past_cov, future_covariates=future_cov
        )
        onnx_filename = f"test_ar_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)
        onnx_pred = run_onnx_prediction(
            n=n,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        self._assert_forecasts_equal(onnx_pred, pred)

    def test_onnx_io_schema(self, tmpdir_fn):
        model = self._make_model(TiDEModel)
        model.fit(
            series=self.ts_tg,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )
        onnx_filename = "test_schema.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)

        assert spec.input_names == [inp.name for inp in session.get_inputs()]
        assert spec.output_names == [out.name for out in session.get_outputs()]
        assert "past_target" in spec.feature_input_names
        assert "past_covariates" in spec.feature_input_names
        assert "future_covariates" in spec.feature_input_names

        inputs = prepare_onnx_inputs(
            series=self.ts_tg,
            spec=spec,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )
        ort_inputs = {
            name: arr
            for name, arr in inputs.items()
            if name in spec.feature_input_names
        }
        outputs = session.run(spec.output_names, ort_inputs)
        assert outputs[0].shape == (
            1,
            model.output_chunk_length,
            self.ts_tg.n_components,
            1,
        )

    @pytest.mark.parametrize("model_cls", [TiDEModel, RNNModel])
    def test_onnx_likelihood(self, tmpdir_fn, model_cls):
        """Likelihood models export raw params in component dimension."""
        quantiles = [0.1, 0.5, 0.9]
        model = self._make_model(model_cls, likelihood=QuantileRegression(quantiles))
        series = self.ts_tg.stack(self.ts_tg + 100.0)
        past_cov = self.ts_pc if model.supports_past_covariates else None
        future_cov = self.ts_fc if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        onnx_filename = f"test_ll_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)

        n_chunk = model.output_chunk_length
        shape_expected = (n_chunk, series.n_components * len(quantiles), 1)
        pred_params = model.predict(
            n=n_chunk,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            predict_likelihood_parameters=True,
        )
        assert pred_params.shape == shape_expected

        onnx_pred = run_onnx_prediction(
            n=n_chunk,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        assert onnx_pred.components.equals(pred_params.components)
        self._assert_forecasts_equal(onnx_pred, pred_params)

        # auto-regression not allowed for likelihood models
        with pytest.raises(
            ValueError, match="Cannot generate auto-regressive predictions"
        ):
            _ = run_onnx_prediction(
                n=n_chunk + 1,
                session=session,
                series=series,
                past_covariates=past_cov,
                future_covariates=future_cov,
            )

    @pytest.mark.parametrize("model_cls", [TCNModel, TFTModel])
    def test_onnx_multiseries(self, tmpdir_fn, model_cls):
        """ONNX inference returns a sequence of forecasts for multiple series."""
        model = self._make_model(model_cls, loss_fn=torch.nn.L1Loss())
        series = [self.ts_tg, self.ts_tg + 100.0]

        past_cov = [self.ts_pc] * 2 if model.supports_past_covariates else None
        future_cov = [self.ts_fc] * 2 if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        onnx_filename = f"test_ms_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)

        n_ar = model.output_chunk_length + 1
        pred = model.predict(
            n=n_ar,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        onnx_pred = run_onnx_prediction(
            n=n_ar,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        assert isinstance(onnx_pred, list)
        assert len(onnx_pred) == len(series)
        for onnx_ts, expected_ts in zip(onnx_pred, pred):
            self._assert_forecasts_equal(onnx_ts, expected_ts)

    def test_onnx_reversible_instance_norm(self, tmpdir_fn):
        """RINorm is in the exported graph: ONNX matches torch predict, including AR."""
        model = self._make_model(TiDEModel, use_reversible_instance_norm=True)
        series = self.ts_tg
        past_cov = self.ts_pc if model.supports_past_covariates else None
        future_cov = self.ts_fc if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        onnx_filename = f"test_rin_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec, session = self._load_spec(onnx_filename)

        n_chunk = model.output_chunk_length
        pred = model.predict(
            n_chunk,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        onnx_pred = run_onnx_prediction(
            n=n_chunk,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        # single window: norm + denorm must match torch (proves stats are not baked
        # from the dummy export batch and output is on the original scale)
        self._assert_forecasts_equal(onnx_pred, pred, rtol=1e-4, atol=1e-4)

        n_ar = n_chunk + 3
        pred_ar = model.predict(
            n_ar,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        onnx_ar = run_onnx_prediction(
            n=n_ar,
            session=session,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False,
        )
        # AR recomputes RINorm per window; untrained + RINorm can explode, so
        # compare relatively (float32 drift on O(1e6) values)
        self._assert_forecasts_equal(onnx_ar, pred_ar, rtol=1e-5, atol=1e-3)

    def test_flatten_unflatten_module_state(self):
        tensor = torch.ones(2, 3)
        tensors, spec = _flatten_module_state(tensor)
        restored = _unflatten_module_state(tensors, spec)
        assert torch.equal(restored, tensor)
        assert _unflatten_module_state([], None) is None

        with pytest.raises(ValueError, match="Unsupported module state type"):
            _flatten_module_state({"hidden": tensor})

    def test_onnx_export_with_dynamic_axes(self, tmpdir_fn):
        """Legacy (non-dynamo) export is used when `dynamic_axes` is set."""

        class _Tiny(nn.Module):
            def forward(self, x):
                return x

        spec = OnnxModelSpec(
            input_chunk_length=2,
            output_chunk_length=1,
            output_chunk_shift=0,
            uses_past_covariates=False,
            uses_future_covariates=False,
            uses_static_covariates=False,
            likelihood_parameter_names=None,
            feature_input_names=["past_target"],
            input_names=["past_target"],
            output_names=["prediction"],
        )
        bundle = OnnxExportBundle(
            wrapper=_Tiny(),
            example_inputs=(torch.zeros(1, 2, 1),),
            input_names=["past_target"],
            output_names=["prediction"],
            spec=spec,
            dynamic_axes={
                "past_target": {0: "batch"},
                "prediction": {0: "batch"},
            },
        )
        path = "tiny_dynamic.onnx"
        _save_onnx_export(bundle, path)
        assert os.path.exists(path)
        onnx.checker.check_model(onnx.load(path))
