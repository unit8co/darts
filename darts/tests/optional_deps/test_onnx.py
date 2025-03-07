from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import pytest

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.tests.conftest import ONNX_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs_dev
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
    TiDEModel,
)

# TODO: check how RINorm can be handled with respect to ONNX
torch_model_cls = [
    BlockRNNModel,
    NHiTSModel,
    TiDEModel,
]


class TestOnnx:
    ts_tg = tg.linear_timeseries(start_value=0, end_value=100, length=30).astype(
        "float32"
    )
    ts_tg_with_static = ts_tg.with_static_covariates(
        pd.Series(data=[12], index=["loc"])
    )
    ts_pc = tg.constant_timeseries(value=123.4, length=300).astype("float32")
    ts_fc = tg.sine_timeseries(length=32).astype("float32")

    @pytest.mark.parametrize("model_cls", torch_model_cls)
    def test_onnx_save_load(self, tmpdir_fn, model_cls):
        model = model_cls(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        onnx_filename = f"test_onnx_{model.model_name}.onnx"

        # exporting without fitting the model fails
        with pytest.raises(ValueError):
            model.to_onnx("dummy_name.onnx")

        model.fit(
            series=self.ts_tg_with_static
            if model.supports_static_covariates
            else self.ts_tg,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        # native inference
        pred = model.predict(2)

        # model export
        model.to_onnx(onnx_filename)

        # onnx model verification
        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)

        # onnx model loading and inference
        onnx_pred = self._helper_onnx_inference(
            model=model,
            onnx_filename=onnx_filename,
            series=self.ts_tg_with_static,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )[0][0]

        # check that the predictions are similar
        assert pred.shape == onnx_pred.shape, "forecasts don't have the same shape."
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

    @pytest.mark.parametrize(
        "params",
        product(
            torch_model_cls,
            [True, False],  # clean
        ),
    )
    def test_onnx_from_ckpt(self, tmpdir_fn, params):
        """Check that creating the onnx export from a model directly loaded from a checkpoint work as expected"""
        model_cls, clean = params
        model = model_cls(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        onnx_filename = f"test_onnx_{model.model_name}.onnx"
        onnx_filename2 = f"test_onnx_{model.model_name}_weights.onnx"
        ckpt_filename = f"test_ckpt_{model.model_name}.pt"

        model.fit(
            series=self.ts_tg_with_static
            if model.supports_static_covariates
            else self.ts_tg,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        model.save(ckpt_filename, clean=clean)

        # load the entire checkpoint
        model_loaded = model_cls.load(ckpt_filename)
        pred = model_loaded.predict(
            n=2,
            series=self.ts_tg_with_static
            if model_loaded.uses_static_covariates
            else self.ts_tg,
            past_covariates=self.ts_pc if model_loaded.uses_past_covariates else None,
            future_covariates=self.ts_fc
            if model_loaded.uses_future_covariates
            else None,
        )

        # export the loaded model
        model_loaded.to_onnx(onnx_filename)

        # onnx model loading and inference
        onnx_pred = self._helper_onnx_inference(
            model=model_loaded,
            onnx_filename=onnx_filename,
            series=self.ts_tg_with_static,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )[0][0]

        # check that the predictions are similar
        assert pred.shape == onnx_pred.shape, "forecasts don't have the same shape."
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

        # load only the weights
        model_weights = model_cls(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs_dev
        )
        model_weights.load_weights(ckpt_filename)
        pred_weights = model_weights.predict(
            n=2,
            series=self.ts_tg_with_static
            if model_weights.uses_static_covariates
            else self.ts_tg,
            past_covariates=self.ts_pc if model_weights.uses_past_covariates else None,
            future_covariates=self.ts_fc
            if model_weights.uses_future_covariates
            else None,
        )

        # export the loaded model
        model_weights.to_onnx(onnx_filename2)

        # onnx model loading and inference
        onnx_pred_weights = self._helper_onnx_inference(
            model=model_weights,
            onnx_filename=onnx_filename2,
            series=self.ts_tg_with_static,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )[0][0]

        assert pred_weights.shape == onnx_pred_weights.shape, (
            "forecasts don't have the same shape."
        )
        np.testing.assert_array_almost_equal(
            onnx_pred_weights, pred_weights.all_values(), decimal=4
        )

    def _helper_onnx_inference(
        self,
        model,
        onnx_filename: str,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
    ):
        """Darts model is only used to detect which covariates are supported by the weights."""
        ort_session = ort.InferenceSession(onnx_filename)

        # extract the input arrays from the series
        past_feats, future_feats, static_feats = prepare_onnx_inputs(
            model=model,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # extract only the features expected by the model
        ort_inputs = {}
        for name, arr in zip(
            ["x_past", "x_future", "x_static"], [past_feats, future_feats, static_feats]
        ):
            if name in [inp.name for inp in list(ort_session.get_inputs())]:
                ort_inputs[name] = arr

        # output has shape (batch, output_chunk_length, n components, 1 or n likelihood params)
        return ort_session.run(None, ort_inputs)
