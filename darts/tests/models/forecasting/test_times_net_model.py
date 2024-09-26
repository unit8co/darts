import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.models.forecasting.times_net_model import FFT_for_Period, TimesNetModel
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg
import torch

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )


class TestTimesNetModel:
    times = pd.date_range("20130101", "20130410")
    pd_series = pd.Series(range(100), index=times)
    series: TimeSeries = TimeSeries.from_series(pd_series)
    series_multivariate = series.stack(series * 2)

    def test_fit_and_predict(self):
        model = TimesNetModel(
            input_chunk_length=12,
            output_chunk_length=12,
            hidden_size=16,
            num_layers=1,
            num_kernels=2,
            top_k=3,
            n_epochs=2,
            random_state=42,
            **tfm_kwargs,
        )
        model.fit(self.series)
        pred = model.predict(n=2)
        assert len(pred) == 2
        assert isinstance(pred, TimeSeries)

    def test_multivariate(self):
        model = TimesNetModel(
            input_chunk_length=12,
            output_chunk_length=12,
            n_epochs=2,
            hidden_size=16,
            num_layers=1,
            num_kernels=2,
            top_k=3,
            **tfm_kwargs,
        )
        model.fit(self.series_multivariate)
        pred = model.predict(n=3)
        assert pred.width == 2
        assert len(pred) == 3

    def test_past_covariates(self):
        target = tg.sine_timeseries(length=100)
        covariates = tg.sine_timeseries(length=100, value_frequency=0.1)

        model = TimesNetModel(
            input_chunk_length=12,
            output_chunk_length=12,
            n_epochs=2,
            hidden_size=16,
            num_layers=1,
            num_kernels=2,
            top_k=3,
            **tfm_kwargs,
        )
        model.fit(target, past_covariates=covariates)
        pred = model.predict(n=3, past_covariates=covariates)
        assert len(pred) == 3

    def test_save_load(self, tmpdir_module):
        model = TimesNetModel(
            input_chunk_length=12,
            output_chunk_length=12,
            n_epochs=2,
            model_name="unittest-model-TimesNet",
            work_dir=tmpdir_module,
            save_checkpoints=True,
            force_reset=True,
            hidden_size=16,
            num_layers=1,
            num_kernels=2,
            top_k=3,
            **tfm_kwargs,
        )
        model.fit(self.series)
        model_loaded = model.load_from_checkpoint(
            model_name="unittest-model-TimesNet",
            work_dir=tmpdir_module,
            best=False,
            map_location="cpu",
        )
        pred1 = model.predict(n=1)
        pred2 = model_loaded.predict(n=1)

        # Two models with the same parameters should deterministically yield the same output
        np.testing.assert_array_equal(pred1.values(), pred2.values())

    def test_prediction_with_custom_encoders(self):
        target = tg.sine_timeseries(length=100, freq="H")
        model = TimesNetModel(
            input_chunk_length=12,
            output_chunk_length=12,
            add_encoders={
                "cyclic": {"future": ["hour"]},
                "datetime_attribute": {"future": ["dayofweek"]},
            },
            n_epochs=2,
            hidden_size=16,
            num_layers=1,
            num_kernels=2,
            top_k=3,
            **tfm_kwargs,
        )
        model.fit(target)
        pred = model.predict(n=12)
        assert len(pred) == 12


class TestT_FFT_for_Period:

    sample_input = (
        torch.sin(torch.linspace(0, 4 * torch.pi, 100)).unsqueeze(0).unsqueeze(-1)
    )

    def test_FFT_for_Period_output_shape(self):
        period, amplitudes = FFT_for_Period(self.sample_input)

        assert isinstance(period, torch.Tensor)
        assert isinstance(amplitudes, torch.Tensor)
        assert period.shape == (2,)  # Default k=2
        assert amplitudes.shape == (1, 2)  # (B, k)

    def test_FFT_for_Period_custom_k(self):
        k = 3
        period, amplitudes = FFT_for_Period(self.sample_input, k=k)

        assert period.shape == (k,)
        assert amplitudes.shape == (1, k)

    def test_FFT_for_Period_period_values(self):
        period, _ = FFT_for_Period(self.sample_input)

        # The main period should be close to 50 (half of the input length)
        assert torch.isclose(period[0], torch.tensor(50), rtol=0.1)

    def test_FFT_for_Period_amplitude_values(self):
        _, amplitudes = FFT_for_Period(self.sample_input)

        # Amplitudes should be non-negative
        assert torch.all(amplitudes >= 0)

    def test_FFT_for_Period_different_shapes(self):
        # Test with different input shapes
        x1 = torch.randn(2, 100, 3)  # [B, T, C] = [2, 100, 3]
        x2 = torch.randn(1, 200, 1)  # [B, T, C] = [1, 200, 1]

        period1, amplitudes1 = FFT_for_Period(x1)
        period2, amplitudes2 = FFT_for_Period(x2)

        assert period1.shape == (2,)
        assert amplitudes1.shape == (2, 2)
        assert period2.shape == (2,)
        assert amplitudes2.shape == (1, 2)

    def test_FFT_for_Period_zero_frequency_removal(self):
        x = torch.ones(1, 100, 1)  # Constant input
        _, amplitudes = FFT_for_Period(x)

        # The amplitude of the zero frequency should be zero
        assert torch.isclose(amplitudes[0, 0], torch.tensor(0.0))
