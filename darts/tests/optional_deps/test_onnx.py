import os.path
from itertools import product

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts.tests.conftest import ONNX_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev
from darts.utils.onnx.inference import (
    OnnxModelSpec,
    extract_point_forecast,
    run_onnx_prediction,
)
from darts.utils.onnx_utils import prepare_onnx_inputs

if not (TORCH_AVAILABLE and ONNX_AVAILABLE):
    pytest.skip(
        f"Torch or Onnx not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import onnx
import onnxruntime as ort

from darts.models import (
    BlockRNNModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
)
from darts.utils.likelihood_models.torch import GaussianLikelihood

torch_model_cls = [
    BlockRNNModel,
    NHiTSModel,
    RNNModel,
    TiDEModel,
]


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

    def _onnx_pred(self, onnx_filename, spec, model, n, series=None):
        return run_onnx_prediction(
            n=n,
            session=ort.InferenceSession(onnx_filename),
            spec=spec,
            series=series if series is not None else self._series_for(model),
            past_covariates=self.ts_pc if model.uses_past_covariates else None,
            future_covariates=self.ts_fc if model.uses_future_covariates else None,
        )

    @pytest.mark.parametrize("model_cls", torch_model_cls)
    def test_onnx_save_load(self, tmpdir_fn, model_cls):
        model = self._make_model(model_cls)
        onnx_filename = f"test_onnx_{model.model_name}.onnx"
        spec_filename = f"{onnx_filename}.spec.json"

        with pytest.raises(ValueError):
            model.to_onnx("dummy_name.onnx")

        series = self._series_for(model)
        model.fit(
            series=series,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        pred = model.predict(2)

        model.to_onnx(onnx_filename)
        assert os.path.exists(onnx_filename)
        assert os.path.exists(spec_filename)

        n_onnx_files = len(_onnx_files(tmpdir_fn))
        model.to_onnx()
        assert len(_onnx_files(tmpdir_fn)) == n_onnx_files + 1

        onnx.checker.check_model(onnx.load(onnx_filename))
        spec = OnnxModelSpec.load_json(spec_filename)
        onnx_pred = self._onnx_pred(onnx_filename, spec, model, n=2)
        assert pred.shape == onnx_pred.shape, "forecasts don't have the same shape."
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

    @pytest.mark.parametrize(
        "params",
        list(product([BlockRNNModel, NHiTSModel, TiDEModel], [True, False])),
    )
    def test_onnx_from_ckpt(self, tmpdir_fn, params):
        """Check that creating the onnx export from a model directly loaded from a checkpoint work as expected"""
        model_cls, clean = params
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

        load_kwargs = tfm_kwargs_dev if clean else {}
        model_loaded = model_cls.load(ckpt_filename, **load_kwargs)
        pred = model_loaded.predict(
            n=2,
            series=self._series_for(model_loaded),
            past_covariates=self.ts_pc if model_loaded.uses_past_covariates else None,
            future_covariates=self.ts_fc
            if model_loaded.uses_future_covariates
            else None,
        )

        model_loaded.to_onnx(onnx_filename)
        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        onnx_pred = self._onnx_pred(onnx_filename, spec, model_loaded, n=2)
        assert pred.shape == onnx_pred.shape, "forecasts don't have the same shape."
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

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
        spec2 = OnnxModelSpec.load_json(f"{onnx_filename2}.spec.json")
        onnx_pred_weights = self._onnx_pred(onnx_filename2, spec2, model_weights, n=2)
        assert pred_weights.shape == onnx_pred_weights.shape, (
            "forecasts don't have the same shape."
        )
        np.testing.assert_array_almost_equal(
            onnx_pred_weights, pred_weights.all_values(), decimal=4
        )

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM"])
    def test_onnx_rnn_state_io(self, tmpdir_fn, rnn_type):
        """RNN uses the same graph as other models, plus flattened state I/O."""
        model = self._make_model(RNNModel, model=rnn_type)
        model.fit(series=self.ts_tg)
        onnx_filename = f"test_onnx_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)

        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        onnx_model = onnx.load(onnx_filename)
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
        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        onnx_pred = run_onnx_prediction(
            n=n,
            session=ort.InferenceSession(onnx_filename),
            spec=spec,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

    def test_onnx_io_schema(self, tmpdir_fn):
        model = self._make_model(TiDEModel)
        model.fit(
            series=self.ts_tg,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )
        onnx_filename = "test_schema.onnx"
        model.to_onnx(onnx_filename)
        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        session = ort.InferenceSession(onnx_filename)

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
        forecast = extract_point_forecast(outputs)
        assert forecast.ndim == 2

    def test_prepare_onnx_inputs_legacy_model_arg(self):
        """Legacy ``prepare_onnx_inputs(model=...)`` API remains supported."""
        model = self._make_model(NHiTSModel)
        model.fit(series=self.ts_tg)
        inputs = prepare_onnx_inputs(
            model=model,
            series=self.ts_tg,
        )
        assert "past_target" in inputs
        assert inputs["past_target"].shape[0] == 1

    @pytest.mark.parametrize("model_cls", [RNNModel, TiDEModel])
    def test_onnx_likelihood(self, tmpdir_fn, model_cls):
        """Likelihood models export raw params; ONNX point forecast is the first param."""
        model = self._make_model(model_cls, likelihood=GaussianLikelihood())
        series = self.ts_tg
        past_cov = self.ts_pc if model.supports_past_covariates else None
        future_cov = self.ts_fc if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        onnx_filename = f"test_ll_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        session = ort.InferenceSession(onnx_filename)

        n_chunk = model.output_chunk_length
        pred_params = model.predict(
            n=n_chunk,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            predict_likelihood_parameters=True,
        )
        onnx_pred = run_onnx_prediction(
            n=n_chunk,
            session=session,
            spec=spec,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        # first Gaussian parameter is μ; matches predict_likelihood_parameters
        np.testing.assert_array_almost_equal(
            onnx_pred, pred_params.all_values()[:, :1, :], decimal=4
        )

        inputs = prepare_onnx_inputs(
            series=series,
            spec=spec,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        if spec.stepwise_state:
            inputs = {name: arr[:, -1:, :] for name, arr in inputs.items()}
            for name, shape in zip(
                spec.state_input_names, spec.state_input_shapes or []
            ):
                inputs[name] = np.zeros(shape, dtype=series.dtype)
        session_names = {inp.name for inp in session.get_inputs()}
        raw = session.run(
            spec.output_names,
            {name: arr for name, arr in inputs.items() if name in session_names},
        )[0]
        assert raw.ndim == 4
        assert raw.shape[-1] == 2

        n_ar = n_chunk + 3
        onnx_ar = run_onnx_prediction(
            n=n_ar,
            session=session,
            spec=spec,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        assert onnx_ar.shape == (n_ar, series.n_components, 1)

    @pytest.mark.parametrize("model_cls", [NHiTSModel, TiDEModel])
    def test_onnx_reversible_instance_norm(self, tmpdir_fn, model_cls):
        """RINorm is in the exported graph: ONNX matches torch predict, including AR."""
        model = self._make_model(model_cls, use_reversible_instance_norm=True)
        series = self.ts_tg
        past_cov = self.ts_pc if model.supports_past_covariates else None
        future_cov = self.ts_fc if model.supports_future_covariates else None
        model.fit(series=series, past_covariates=past_cov, future_covariates=future_cov)

        onnx_filename = f"test_rin_{model.model_name}.onnx"
        model.to_onnx(onnx_filename)
        spec = OnnxModelSpec.load_json(f"{onnx_filename}.spec.json")
        session = ort.InferenceSession(onnx_filename)

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
            spec=spec,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        assert pred.shape == onnx_pred.shape
        # single window: norm + denorm must match torch (proves stats are not baked
        # from the dummy export batch and output is on the original scale)
        np.testing.assert_allclose(onnx_pred, pred.all_values(), rtol=1e-4, atol=1e-4)

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
            spec=spec,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        assert pred_ar.shape == onnx_ar.shape
        # AR recomputes RINorm per window; untrained + RINorm can explode, so
        # compare relatively (float32 drift on O(1e6) values)
        np.testing.assert_allclose(onnx_ar, pred_ar.all_values(), rtol=1e-5, atol=1e-3)
