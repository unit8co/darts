from typing import Optional

import numpy as np
import pytest

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.tests.conftest import ONNX_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not (TORCH_AVAILABLE and ONNX_AVAILABLE):
    pytest.skip(
        f"Torch or Onnx not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )
import onnx
import onnxruntime as ort
import torch

from darts.models import (
    BlockRNNModel,
    NLinearModel,
    TFTModel,
    TiDEModel,
)

# from darts.models.components.layer_norm_variants import RINorm


class TestOnnx:
    ts_tgt = tg.linear_timeseries(start_value=0, end_value=100, length=100)
    ts_pc = tg.constant_timeseries(value=123.4, length=100)
    ts_fc = tg.sine_timeseries(length=100)

    @pytest.mark.parametrize(
        "model_cls",
        [
            BlockRNNModel,
            NLinearModel,
            TFTModel,
            TiDEModel,
        ],
    )
    def test_onnx_save_load(self, tmpdir_fn, model_cls):
        model = model_cls(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs
        )
        onnx_filename = f"test_{model.name}"
        model.fit()
        # native inference
        pred = model.predict(2)

        # model export
        model.to_onnx(onnx_filename)

        # onnx model verification
        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)

        # manual feature extraction from the series
        past_feats, future_feats, static_feats = self._helper_prepare_onnx_inputs(
            model=model,
            series=self.ts_tgt,
            past_covariates=self.ts_pc,
            future_covariates=self.ts_fc,
        )

        # onnx model loading and inference
        onnx_pred = self._helper_onnx_inference(
            onnx_filename, past_feats, future_feats, static_feats
        )

        # check that the predictions are similar
        torch.testing.assert_close(onnx_pred, pred)

    def _helper_prepare_onnx_inputs(
        model,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper function to slice and concatenate the input features"""
        past_feats, future_feats, static_feats = None, None, None
        # get input & output windows
        past_start = series.end_time() - (model.input_chunk_length - 1) * series.freq
        past_end = series.end_time()
        future_start = past_end + 1 * series.freq
        future_end = past_end + model.output_chunk_length * series.freq
        # extract all historic and future features from target, past and future covariates
        past_feats = series[past_start:past_end].values()
        if past_covariates and model.uses_past_covariates:
            # extract past covariates
            past_feats = np.concatenate(
                [past_feats, past_covariates[past_start:past_end].values()], axis=1
            )
        if future_covariates and model.uses_future_covariates:
            # extract past part of future covariates
            past_feats = np.concatenate(
                [past_feats, future_covariates[past_start:past_end].values()], axis=1
            )
            # extract future part of future covariates
            future_feats = future_covariates[future_start:future_end].values()
        # add batch dimension -> (batch, n time steps, n components)
        past_feats = np.expand_dims(past_feats, axis=0).astype(series.dtype)
        future_feats = np.expand_dims(future_feats, axis=0).astype(series.dtype)
        # extract static covariates
        if series.has_static_covariates and model.uses_static_covariates:
            static_feats = np.expand_dims(
                series.static_covariates_values(), axis=0
            ).astype(series.dtype)
        return past_feats, future_feats, static_feats

    def _helper_onnx_inference(
        self,
        onnx_filename: str,
        past_feats: torch.Tensor,
        future_feats: torch.Tensor,
        static_feats: torch.Tensor,
    ):
        ort_session = ort.InferenceSession(onnx_filename)
        # extract only the features expected by the model
        ort_inputs = {}
        for name, arr in zip(
            ["x_past", "x_future", "x_static"], [past_feats, future_feats, static_feats]
        ):
            if name in [inp.name for inp in list(ort_session.get_inputs())]:
                ort_inputs[name] = arr

        # output has shape (batch, output_chunk_length, n components, 1 or n likelihood params)
        return ort_session.run(None, ort_inputs)
