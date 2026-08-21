from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch

from darts import TimeSeries
from darts.ad import FilteringAnomalyModel
from darts.ad.scorers import DifferenceScorer
from darts.models import TSPulseFilter


class _TSPulseStub(torch.nn.Module):
    def __init__(
        self,
        context_length: int = 4,
        width: int = 1,
        mask_type: str | None = None,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            context_length=context_length,
            num_input_channels=width,
        )
        if mask_type is not None:
            self.config.mask_type = mask_type
        self.calls = []

    def forward(self, past_values, past_observed_mask, return_loss, return_dict):
        self.calls.append({
            "past_values": past_values.detach().cpu().clone(),
            "past_observed_mask": past_observed_mask.detach().cpu().clone(),
            "return_loss": return_loss,
            "return_dict": return_dict,
        })
        return SimpleNamespace(reconstruction_outputs=past_values + 10.0)


class _InvalidOutputStub(_TSPulseStub):
    def forward(self, past_values, past_observed_mask, return_loss, return_dict):
        return SimpleNamespace(reconstruction_outputs=past_values[:, :-1])


class _NonTensorOutputStub(_TSPulseStub):
    def forward(self, past_values, past_observed_mask, return_loss, return_dict):
        return SimpleNamespace(
            reconstruction_outputs=np.zeros(tuple(past_values.shape))
        )


class _ReducedPrecisionStub(_TSPulseStub):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))

    def forward(self, past_values, past_observed_mask, return_loss, return_dict):
        assert past_values.dtype == self.weight.dtype
        return super().forward(
            past_values,
            past_observed_mask,
            return_loss,
            return_dict,
        )


class TestTSPulseFilter:
    def test_filter_reconstructs_batched_overlapping_windows(self):
        values = np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
                [8.0, 9.0],
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            dtype=np.float32,
        )
        series = TimeSeries.from_times_and_values(
            pd.date_range("2026-01-01", periods=len(values), freq="h"),
            values,
            columns=["sensor_a", "sensor_b"],
            static_covariates=pd.DataFrame({"site": ["north", "south"]}),
        )
        model = _TSPulseStub(context_length=4, width=2)
        tspulse = TSPulseFilter(model=model, batch_size=2, stride=2)

        filtered = tspulse.filter(series)

        np.testing.assert_allclose(filtered.values(), values + 10.0)
        assert len(model.calls) == 2
        assert model.calls[0]["past_values"].shape == (2, 4, 2)
        assert model.calls[1]["past_values"].shape == (1, 4, 2)
        assert all(call["return_loss"] is False for call in model.calls)
        assert all(call["return_dict"] is True for call in model.calls)
        assert filtered.time_index.equals(series.time_index)
        assert filtered.components.equals(series.components)
        assert filtered.static_covariates.equals(series.static_covariates)

    def test_filter_rejects_series_shorter_than_context(self):
        series = TimeSeries.from_values(np.array([[1.0], [2.0]], dtype=np.float32))
        model = _TSPulseStub(context_length=4)

        with pytest.raises(ValueError, match="at least the model context length"):
            TSPulseFilter(model=model).filter(series)
        assert model.calls == []

    def test_impute_replaces_only_missing_values(self):
        values = np.array([[1.0], [np.nan], [3.0], [np.nan]], dtype=np.float32)
        series = TimeSeries.from_values(values)
        original = series.copy()
        model = _TSPulseStub(context_length=4)

        imputed = TSPulseFilter(model=model).impute(series)

        np.testing.assert_allclose(imputed.values(), [[1.0], [10.0], [3.0], [10.0]])
        assert series == original
        np.testing.assert_array_equal(
            model.calls[0]["past_observed_mask"].numpy()[:, :, 0],
            [[True, False, True, False]],
        )

    def test_impute_is_deterministic_with_user_masking(self):
        values = np.array([[1.0], [np.nan], [3.0], [np.nan]], dtype=np.float32)
        series = TimeSeries.from_values(values)
        model = _TSPulseStub(context_length=4, mask_type="user")
        tspulse = TSPulseFilter(model=model)

        first = tspulse.impute(series)
        second = tspulse.impute(series)

        assert first == second
        assert len(model.calls) == 2

    def test_impute_multivariate_overlapping_windows_with_sparse_missing_values(self):
        values = np.array(
            [
                [0.0, 1.0],
                [np.nan, 3.0],
                [4.0, np.nan],
                [6.0, 7.0],
                [np.nan, 9.0],
                [10.0, np.nan],
            ],
            dtype=np.float32,
        )
        series = TimeSeries.from_values(values)
        model = _TSPulseStub(context_length=4, width=2, mask_type="user")

        imputed = TSPulseFilter(model=model, stride=2).impute(series)

        expected = np.where(np.isnan(values), 10.0, values)
        np.testing.assert_allclose(imputed.values(), expected)
        assert model.calls[0]["past_values"].shape == (2, 4, 2)

    def test_impute_rejects_entirely_missing_component(self):
        values = np.column_stack((np.arange(4, dtype=np.float32), np.full(4, np.nan)))
        series = TimeSeries.from_values(values, columns=["observed", "missing"])
        model = _TSPulseStub(context_length=4, width=2, mask_type="user")

        with pytest.raises(
            ValueError,
            match="'missing'.*no observed values.*each component/window",
        ):
            TSPulseFilter(model=model).impute(series)
        assert model.calls == []

    def test_impute_rejects_window_without_component_observations(self):
        values = np.column_stack((
            np.arange(6, dtype=np.float32),
            np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan]),
        ))
        series = TimeSeries.from_values(values, columns=["sensor_a", "sensor_b"])
        model = _TSPulseStub(context_length=4, width=2, mask_type="user")

        with pytest.raises(
            ValueError,
            match=r"'sensor_b'.*positions \[2, 6\)",
        ):
            TSPulseFilter(model=model, stride=2).impute(series)
        assert model.calls == []

    def test_impute_complete_series_does_not_run_model(self):
        series = TimeSeries.from_values(
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        )
        model = _TSPulseStub(context_length=4)

        imputed = TSPulseFilter(model=model).impute(series)

        assert imputed == series
        assert imputed is not series
        assert model.calls == []

    def test_filtering_anomaly_model_compatibility(self):
        series = TimeSeries.from_values(np.arange(6, dtype=np.float32).reshape(-1, 1))
        tspulse = TSPulseFilter(
            model=_TSPulseStub(context_length=4),
            stride=2,
        )
        anomaly_model = FilteringAnomalyModel(
            model=tspulse,
            scorer=DifferenceScorer(),
        )

        scores, prediction = anomaly_model.score(
            series,
            return_model_prediction=True,
        )

        np.testing.assert_allclose(prediction.values(), series.values() + 10.0)
        np.testing.assert_allclose(scores.values(), -10.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"batch_size": 0}, "batch_size"),
            ({"batch_size": 1.5}, "batch_size"),
            ({"batch_size": True}, "batch_size"),
            ({"stride": 0}, "stride"),
            ({"stride": 1.5}, "stride"),
            ({"stride": True}, "stride"),
            (
                {"model_kwargs": {"num_input_channels": 2}},
                "managed by TSPulseFilter",
            ),
            ({"model_kwargs": {"mask_type": "block"}}, "managed by TSPulseFilter"),
            ({"model_kwargs": {"return_dict": False}}, "managed by TSPulseFilter"),
            ({"model_kwargs": {"dtype": torch.float16}}, "managed by TSPulseFilter"),
            (
                {"model_kwargs": {"torch_dtype": torch.bfloat16}},
                "managed by TSPulseFilter",
            ),
        ],
    )
    def test_constructor_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TSPulseFilter(**kwargs)

    def test_filter_input_validation(self):
        model = _TSPulseStub(context_length=4)
        tspulse = TSPulseFilter(model=model)

        with pytest.raises(ValueError, match="deterministic"):
            tspulse.filter(TimeSeries.from_values(np.ones((4, 1, 2))))
        with pytest.raises(ValueError, match="infinite"):
            tspulse.filter(TimeSeries.from_values(np.array([[1.0], [np.inf]])))
        with pytest.raises(ValueError, match="finite float32 range"):
            tspulse.filter(
                TimeSeries.from_values(
                    np.array(
                        [[np.finfo(np.float64).max], [1.0], [2.0], [3.0]],
                        dtype=np.float64,
                    )
                )
            )
        with pytest.raises(ValueError, match="must not exceed"):
            TSPulseFilter(model=model, stride=5).filter(
                TimeSeries.from_values(np.ones((5, 1)))
            )

    def test_supplied_model_width_validation(self):
        model = _TSPulseStub(context_length=4, width=1)
        tspulse = TSPulseFilter(model=model)

        with pytest.raises(ValueError, match="expects 1 input channels"):
            tspulse.filter(TimeSeries.from_values(np.ones((4, 2))))

    def test_supplied_upstream_model_requires_user_masking(self):
        model = _TSPulseStub(context_length=4, mask_type="var_hybrid")

        with pytest.raises(
            ValueError,
            match=r"config.mask_type='user'.*from_pretrained",
        ):
            TSPulseFilter(model=model).impute(
                TimeSeries.from_values(
                    np.array([[1.0], [np.nan], [3.0], [4.0]], dtype=np.float32)
                )
            )
        assert model.calls == []

    def test_supplied_upstream_model_accepts_user_masking(self):
        model = _TSPulseStub(context_length=4, mask_type="user")
        series = TimeSeries.from_values(
            np.array([[1.0], [np.nan], [3.0], [4.0]], dtype=np.float32)
        )

        imputed = TSPulseFilter(model=model).impute(series)

        np.testing.assert_allclose(imputed.values(), [[1.0], [10.0], [3.0], [4.0]])

    def test_output_shape_validation(self):
        tspulse = TSPulseFilter(model=_InvalidOutputStub(context_length=4))

        with pytest.raises(ValueError, match="reconstruction shape"):
            tspulse.filter(TimeSeries.from_values(np.ones((4, 1))))

    def test_output_type_validation(self):
        tspulse = TSPulseFilter(model=_NonTensorOutputStub(context_length=4))

        with pytest.raises(ValueError, match="must be a `torch.Tensor`"):
            tspulse.filter(TimeSeries.from_values(np.ones((4, 1))))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_filter_rejects_reduced_precision_model(self, dtype):
        model = _ReducedPrecisionStub(dtype=dtype)
        series = TimeSeries.from_values(np.arange(4, dtype=np.float64).reshape(-1, 1))

        with pytest.raises(ValueError, match="only supports float32 models"):
            TSPulseFilter(model=model).filter(series)
        assert model.calls == []

    def test_context_length_rejects_boolean(self):
        model = _TSPulseStub(context_length=True)

        with pytest.raises(ValueError, match="positive integer"):
            TSPulseFilter(model=model).filter(TimeSeries.from_values(np.ones((4, 1))))
