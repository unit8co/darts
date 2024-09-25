import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.models.forecasting.times_net_model import TimesNetModel
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

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
