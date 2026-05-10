from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries, concatenate
from darts.models import PatchTSTFMModel
from darts.tests.conftest import tfm_kwargs
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# quantiles used during PatchTST-FM pre-training (0.01 to 0.99 in steps of 0.01)
all_quantiles = [round(i / 100, 2) for i in range(1, 100)]


class TestPatchTSTFMModel:
    np.random.seed(42)

    # ---- Dummy Tests ---- #
    dummy_local_dir = (
        Path(__file__).parent / "artefacts" / "patchtst_fm" / "tiny_patchtst_fm"
    ).absolute()
    # tiny model: context_length=128, d_patch=16, num_quantile=9
    # max input = context_length - output_chunk_length
    dummy_context_length = 128
    dummy_max_prediction_length = 64  # can be up to context_length - input_chunk_length

    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test PatchTSTFMModel predictions against reference output.
        The reference was generated using the full Darts PatchTSTFMModel with the
        ibm-granite/granite-timeseries-patchtst-fm-r1 pretrained weights.

        Reference generation code:

        ```python
        import numpy as np
        from darts import TimeSeries
        from darts.models import PatchTSTFMModel
        from darts.utils.likelihood_models import QuantileRegression

        all_quantiles = [round(i/100, 2) for i in range(1, 100)]
        input_chunk_length = 512
        output_chunk_length = 64

        model = PatchTSTFMModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=QuantileRegression(quantiles=all_quantiles),
            pl_trainer_kwargs={'accelerator': 'cpu'},
        )

        np.random.seed(42)
        test_data = np.sin(np.linspace(0, 4*np.pi, 576)).astype(np.float32)
        ts = TimeSeries.from_values(test_data.reshape(-1, 1), columns=['A'])

        model.fit(series=ts)
        pred = model.predict(n=output_chunk_length, predict_likelihood_parameters=True)
        pred_np = pred.values()  # (64, 99)
        np.savez_compressed("patchtst_fm.npz", pred=pred_np, context=test_data)
        ```
        """
        path = (
            Path(__file__).parent
            / "artefacts"
            / "patchtst_fm"
            / "patchtst_fm_prediction"
            / "patchtst_fm.npz"
        )
        if not path.exists():
            pytest.skip("Reference prediction artefact not found")

        ref_data = np.load(path)
        original = ref_data["pred"]  # shape: (64, 99)
        context_data = ref_data["context"]  # shape: (576,)

        input_chunk_length = 512
        output_chunk_length = 64

        model = PatchTSTFMModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=(
                QuantileRegression(quantiles=all_quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )

        ts = TimeSeries.from_values(context_data.reshape(-1, 1), columns=["A"])
        model.fit(series=ts)

        pred = model.predict(
            n=output_chunk_length,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == output_chunk_length

        pred_np = pred.values()  # shape: (64, n_quantiles) or (64, 1)

        if probabilistic:
            expected = original  # (64, 99)
        else:
            expected = original[:, [49]]  # (64, 1) - median quantile

        np.testing.assert_allclose(pred_np, expected, rtol=1e-4, atol=1e-4)

    def test_creation(self):
        icl = 64
        ocl = 16
        # can use shorter input/output chunk length than max
        model = PatchTSTFMModel(
            input_chunk_length=icl,
            output_chunk_length=ocl,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10

        # cannot exceed context_length
        with pytest.raises(ValueError, match=r"cannot exceed model's context_length"):
            PatchTSTFMModel(
                input_chunk_length=100,
                output_chunk_length=50,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot exceed context_length with shift
        with pytest.raises(ValueError, match=r"cannot exceed model's context_length"):
            PatchTSTFMModel(
                input_chunk_length=100,
                output_chunk_length=20,
                output_chunk_shift=10,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use likelihood other than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            PatchTSTFMModel(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                likelihood=GaussianLikelihood(),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(
            ValueError, match="must be a subset of PatchTST-FM quantiles"
        ):
            PatchTSTFMModel(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

    def test_default(self):
        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(series=self.series)
            mock_fit.assert_not_called()
        assert model.model_created
        assert not model.supports_probabilistic_prediction

        # predictions should not be probabilistic
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

        # default model allows autoregressive predictions
        pred_ar = model.predict(n=20, series=self.series)
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 20
        assert pred_ar.n_components == 1

    def test_probabilistic(self):
        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(series=self.series)
            mock_fit.assert_not_called()
        assert model.model_created
        assert model.supports_probabilistic_prediction

        # predictions should be probabilistic
        pred = model.predict(
            n=10, series=self.series, predict_likelihood_parameters=True
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 3  # 3 quantiles

        # probabilistic model allows autoregressive predictions
        pred_ar = model.predict(
            n=20,
            series=self.series,
            num_samples=10,
        )
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 20
        assert pred_ar.n_components == 1
        assert pred_ar.n_samples == 10

    def test_multivariate(self):
        series_multi = concatenate(
            [
                linear_timeseries(length=200, dtype=np.float32, column_name="A"),
                sine_timeseries(length=200, dtype=np.float32, column_name="B"),
                gaussian_timeseries(length=200, dtype=np.float32, column_name="C"),
            ],
            axis=1,
        )
        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=series_multi)
        pred = model.predict(n=15, series=series_multi)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 15
        assert pred.n_components == 3

        # probabilistic multivariate
        model_prob = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model_prob.fit(series=series_multi)
        pred_prob = model_prob.predict(
            n=15, series=series_multi, predict_likelihood_parameters=True
        )
        assert isinstance(pred_prob, TimeSeries)
        assert len(pred_prob) == 15
        assert pred_prob.n_components == 9  # 3 variables x 3 quantiles

    def test_no_covariates(self):
        """PatchTST-FM does not support covariates."""
        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        assert not model.supports_past_covariates
        assert not model.supports_future_covariates

    def test_multiple_series(self):
        series1 = linear_timeseries(length=200, dtype=np.float32, column_name="A")
        series2 = sine_timeseries(length=150, dtype=np.float32, column_name="A")

        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=[series1, series2])
        pred = model.predict(n=10, series=[series1, series2])

        assert isinstance(pred, list) and len(pred) == 2
        assert all(isinstance(p, TimeSeries) for p in pred)
        assert all(len(p) == 10 for p in pred)

    @pytest.mark.slow
    def test_finetuning(self):
        model = PatchTSTFMModel(
            input_chunk_length=64,
            output_chunk_length=16,
            local_dir=self.dummy_local_dir,
            enable_finetuning=True,
            n_epochs=1,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        assert model.model_created

        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
