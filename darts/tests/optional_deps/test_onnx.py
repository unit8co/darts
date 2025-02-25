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

from darts.models import (
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    TCNModel,
    TFTModel,
    TiDEModel,
    TransformerModel,
    TSMixerModel,
)

# TODO: check how RINorm can be handled with respect to ONNX


class TestOnnx:
    ts_tg = tg.linear_timeseries(start_value=0, end_value=100, length=30).astype(
        "float32"
    )
    ts_pc = tg.constant_timeseries(value=123.4, length=300).astype("float32")
    ts_fc = tg.sine_timeseries(length=32).astype("float32")

    @pytest.mark.parametrize(
        "model_cls",
        [
            NLinearModel,
            TFTModel,
            TiDEModel,
            TCNModel,
            NBEATSModel,
            NHiTSModel,
            TransformerModel,
            TSMixerModel,
        ],
    )
    def test_onnx_save_load(self, tmpdir_fn, model_cls):
        model = model_cls(
            input_chunk_length=4, output_chunk_length=2, n_epochs=1, **tfm_kwargs
        )
        onnx_filename = f"test_{model}"
        # model.model = model.model.to(torch.float)
        model.fit(
            series=self.ts_tg,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )
        # model.model = model.model.float()
        # native inference
        pred = model.predict(2)

        # model export
        # TODO: LSTM model should be exported with a batch size of 1, it seems to create prediction shape problems for
        # for TFT and TCN.

        model.to_onnx(onnx_filename)

        # onnx model verification
        onnx_model = onnx.load(onnx_filename)
        onnx.checker.check_model(onnx_model)

        # manual feature extraction from the series
        past_feats, future_feats, static_feats = self._helper_prepare_onnx_inputs(
            model=model,
            series=self.ts_tg,
            past_covariates=self.ts_pc if model.supports_past_covariates else None,
            future_covariates=self.ts_fc if model.supports_future_covariates else None,
        )

        # onnx model loading and inference
        onnx_pred = self._helper_onnx_inference(
            onnx_filename, past_feats, future_feats, static_feats
        )[0][0]

        # check that the predictions are similar
        assert pred.shape == onnx_pred.shape, "forecasts don't have the same shape."
        np.testing.assert_array_almost_equal(onnx_pred, pred.all_values(), decimal=4)

    def _helper_prepare_onnx_inputs(
        self,
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

        print(series.dtype)
        return past_feats, future_feats, static_feats

    def _helper_onnx_inference(
        self,
        onnx_filename: str,
        past_feats: np.ndarray,
        future_feats: np.ndarray,
        static_feats: np.ndarray,
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
