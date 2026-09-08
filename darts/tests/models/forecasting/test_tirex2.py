from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from darts.tests.conftest import TIREX2_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE or not TIREX2_AVAILABLE:
    pytest.skip("TiRex-2 and Torch are required.", allow_module_level=True)

import torch
from torch import nn

from darts.models import TiRex2Model
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import linear_timeseries

LOAD_MODEL = "darts.models.forecasting.tirex2_model.load_model"


class TiRex2Stub(nn.Module):
    """Encode target, horizon and quantile axes in predictions to test alignment."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("quantiles", torch.arange(1, 10) / 10)
        self.future_len = 32
        self.inputs = []

    def predict(self, timeseries, prediction_length):
        self.inputs.extend(timeseries)
        return [
            ts.target[:, -1, None, None]
            + torch.arange(prediction_length, device=ts.target.device)[None, None, :]
            + self.quantiles[None, :, None] * self.weight
            for ts in timeseries
        ]


@pytest.fixture
def pipeline():
    model = TiRex2Stub()
    with patch(LOAD_MODEL, return_value=SimpleNamespace(model=model)) as loader:
        yield model, loader


class TestTiRex2Model:
    series = linear_timeseries(length=60, dtype=np.float32, column_name="a")
    multi = series.stack((series + 10).with_columns_renamed("a", "b"))
    cov = linear_timeseries(length=100, dtype=np.float32)

    @pytest.mark.parametrize("past", [False, True])
    @pytest.mark.parametrize("future", [False, True])
    @pytest.mark.parametrize("shift", [0, 3])
    @pytest.mark.parametrize("probabilistic", [False, True])
    def test_alignment(self, pipeline, past, future, shift, probabilistic):
        backbone, _ = pipeline
        model = TiRex2Model(
            8,
            5,
            output_chunk_shift=shift,
            likelihood=QuantileRegression([0.1, 0.5, 0.9]) if probabilistic else None,
            **tfm_kwargs,
        )
        model.fit(
            self.multi,
            past_covariates=self.cov if past else None,
            future_covariates=self.cov + 100 if future else None,
        )
        pred = model.predict(5, predict_likelihood_parameters=probabilistic)
        quantiles = [0.1, 0.5, 0.9] if probabilistic else [0.5]
        expected = (
            self.multi.values()[-1][None, :, None]
            + np.arange(shift, shift + 5)[:, None, None]
            + np.array(quantiles)[None, None, :]
        )
        np.testing.assert_allclose(pred.values().reshape(5, 2, -1), expected, rtol=1e-6)
        assert (
            pred.start_time() == self.multi.end_time() + (shift + 1) * self.multi.freq
        )
        ts = backbone.inputs[-1]
        np.testing.assert_array_equal(ts.target.numpy(), self.multi.values()[-8:].T)
        if past:
            np.testing.assert_array_equal(
                ts.past_covariates.numpy(), self.cov.values()[52:60].T
            )
        else:
            assert ts.past_covariates is None
        if future:
            np.testing.assert_array_equal(
                ts.future_covariates[:, :8].numpy(), (self.cov.values()[52:60] + 100).T
            )
            assert torch.isnan(ts.future_covariates[:, 8 : 8 + shift]).all()
            np.testing.assert_array_equal(
                ts.future_covariates[:, 8 + shift :].numpy(),
                (self.cov.values()[60 + shift : 65 + shift] + 100).T,
            )
        else:
            assert ts.future_covariates is None
        assert not backbone.weight.requires_grad
        assert "tirex2.weight" in model.model.state_dict()

    def test_batches_autoregression_and_samples(self, pipeline):
        model = TiRex2Model(
            8,
            5,
            likelihood=QuantileRegression([0.1, 0.5, 0.9]),
            batch_size=2,
            **tfm_kwargs,
        )
        series = [self.multi, self.multi + 100, self.multi + 200]
        model.fit(
            series, past_covariates=[self.cov] * 3, future_covariates=[self.cov] * 3
        )
        preds = model.predict(
            12,
            series=series,
            past_covariates=[self.cov] * 3,
            future_covariates=[self.cov] * 3,
            num_samples=4,
        )
        assert len(preds) == 3
        assert all(p.all_values().shape == (12, 2, 4) for p in preds)
        assert preds[1].all_values().mean() > preds[0].all_values().mean() + 90

    def test_variable_context(self, pipeline):
        model = TiRex2Model((3, 8), 5, **tfm_kwargs)
        model.fit(self.series)
        preds = model.predict(4, series=[self.series[:4], self.series[:7]])
        assert all(len(p) == 4 for p in preds)

    def test_loading_options(self, pipeline):
        _, loader = pipeline
        options = {
            "hf_kwargs": {"local_files_only": True, "revision": "old"},
            "use_flex_attention": False,
        }
        model = TiRex2Model(
            8,
            5,
            hub_model_name="org/checkpoint",
            hub_model_revision="pinned",
            local_dir="/tmp/weights",
            tirex2_kwargs=options,
            **tfm_kwargs,
        )
        model.fit(self.series)
        loader.assert_called_once_with(
            ckpt_path="org/checkpoint",
            device="cpu",
            hf_kwargs={
                "local_files_only": True,
                "revision": "pinned",
                "local_dir": "/tmp/weights",
            },
            use_flex_attention=False,
        )
        assert options["hf_kwargs"]["revision"] == "old"
        assert "device" not in options

    def test_save_load(self, pipeline, tmp_path):
        model = TiRex2Model(8, 5, **tfm_kwargs)
        model.fit(self.multi, future_covariates=self.cov)
        expected = model.predict(5)
        path = str(tmp_path / "tirex2.pt")
        model.save(path)
        loaded = TiRex2Model.load(path, map_location="cpu")
        assert loaded.predict(5) == expected

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"enable_finetuning": True}, "does not support fine-tuning"),
            (
                {"enable_finetuning": {"unfreeze": ["*"]}},
                "does not support fine-tuning",
            ),
            ({"likelihood": GaussianLikelihood()}, "Only QuantileRegression"),
            ({"tirex2_kwargs": {"ckpt_path": "bad"}}, "via `hub_model_name`"),
        ],
    )
    def test_invalid_options(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TiRex2Model(8, 5, **kwargs, **tfm_kwargs)

    def test_checkpoint_quantiles(self, pipeline):
        model = TiRex2Model(
            8, 5, likelihood=QuantileRegression([0.05, 0.5, 0.95]), **tfm_kwargs
        )
        with pytest.raises(
            ValueError, match="does not support the requested quantiles"
        ):
            model.fit(self.series)

    def test_checkpoint_horizon(self, pipeline):
        model = TiRex2Model(8, 30, output_chunk_shift=3, **tfm_kwargs)
        with pytest.raises(ValueError, match="maximum prediction length 32"):
            model.fit(self.series)


@pytest.mark.slow
@pytest.mark.parametrize("with_covariates,shift", [(False, 0), (True, 2)])
def test_fidelity(with_covariates, shift):
    """Compare Darts forecasts with the public upstream API on the same checkpoint."""
    from tirex2 import TimeseriesType, load_model

    series = TestTiRex2Model.multi
    cov = TestTiRex2Model.cov
    pipeline = load_model("NX-AI/TiRex-2", device="cpu")
    future = None
    if with_covariates:
        future = torch.tensor(cov.values()[44 : 68 + shift].T.copy())
        future[:, 16 : 16 + shift] = float("nan")
    reference = pipeline.forecast(
        [
            TimeseriesType(
                target=torch.tensor(series.values()[-16:].T.copy()),
                past_covariates=torch.tensor(cov.values()[44:60].T.copy())
                if with_covariates
                else None,
                future_covariates=future,
            )
        ],
        prediction_length=8 + shift,
        output_type="numpy",
    )[0][:, [0, 4, 8], shift:].transpose(2, 0, 1)
    model = TiRex2Model(
        16,
        8,
        output_chunk_shift=shift,
        likelihood=QuantileRegression([0.1, 0.5, 0.9]),
        **tfm_kwargs,
    )
    with patch(LOAD_MODEL, return_value=pipeline):
        model.fit(
            series,
            past_covariates=cov if with_covariates else None,
            future_covariates=cov if with_covariates else None,
        )
    actual = model.predict(8, predict_likelihood_parameters=True)
    np.testing.assert_allclose(
        actual.values().reshape(8, 2, 3), reference, rtol=1e-5, atol=1e-5
    )
