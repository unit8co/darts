import contextlib
import logging
import os
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from darts import TimeSeries, concatenate
from darts.tests.conftest import TIREX_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts.models import Chronos2Model, PatchTSTFMModel, TimesFM2p5Model, TiRexModel
from darts.tests.models.forecasting.foundation_test_utils import (
    CHRONOS2_TINY_DIR,
    HF_HUB_DOWNLOAD_PATCH_TARGET,
    PATCHTST_FM_TINY_DIR,
    TIREX_LOAD_MODEL_PATCH_TARGET,
    TiRexStub,
    mock_hf_hub_download,
    timesfm2p5_tiny_context,
)


def generate_series(n_variables: int, length: int, prefix: str):
    return concatenate(
        [
            linear_timeseries(
                length=length, dtype=np.float32, column_name=f"{prefix}_{i}"
            )
            for i in range(n_variables)
        ],
        axis=1,
    )


class TestFoundationModel:
    series = generate_series(n_variables=2, length=100, prefix="A")
    future_cov = generate_series(n_variables=3, length=200, prefix="C")

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_default(self, mock_method):
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        assert model.input_chunk_length == 12
        assert model.output_chunk_length == 6
        mock_method.assert_called()

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(
                series=self.series,
                future_covariates=self.future_cov,
            )
            mock_fit.assert_not_called()

        # foundation model should be deterministic
        assert model.model_created
        assert not model.supports_probabilistic_prediction

        # predictions should not be probabilistic
        pred = model.predict(n=10)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == self.series.n_components

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_invalid_params(self, mock_method):
        with pytest.raises(ValueError, match="Invalid model creation parameters"):
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                non_existent_param=None,
                **tfm_kwargs,
            )

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    @pytest.mark.parametrize(
        "user_rin, expected_rin",
        [
            (True, {"affine": False}),
            ({"eps": 1e-7}, {"affine": False, "eps": 1e-7}),
            ({"affine": True}, {"affine": False}),
            ({"eps": 1e-9, "affine": True}, {"affine": False, "eps": 1e-9}),
            ({"affine": False}, {"affine": False}),
            ({"eps": 1e-8, "affine": False}, {"eps": 1e-8, "affine": False}),
            (False, False),
        ],
    )
    def test_rinorm(self, mock_method, caplog, user_rin, expected_rin):
        """Checks that RINorm works, and that affine=True is overridden to affine=False."""
        # `affine=True` is overridden to `affine=False`
        affine_override = False
        if user_rin is True or (
            isinstance(user_rin, dict) and user_rin.get("affine", True)
        ):
            affine_override = True

        # `use_reversible_instance_norm` is overridden to `use_reversible_instance_norm={"affine": False}`
        with caplog.at_level(logging.WARNING):
            model = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                use_reversible_instance_norm=user_rin,
                **tfm_kwargs,
            )

        assert (
            "`use_reversible_instance_norm` is overridden to" in caplog.text
        ) is affine_override
        # RINorm affine transformation is disabled
        assert model.pl_module_params["use_reversible_instance_norm"] == expected_rin
        model.fit(series=self.series)

        if user_rin:
            assert model.model.rin.affine is False
        else:
            assert model.model.rin is None

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_local_dir(self, mock_method, caplog):
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            local_dir=CHRONOS2_TINY_DIR,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(
                series=self.series,
                future_covariates=self.future_cov,
            )
            mock_fit.assert_not_called()

        # foundation model should be deterministic
        assert model.model_created
        assert not model.supports_probabilistic_prediction

        # predictions should not be probabilistic
        pred = model.predict(n=10)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == self.series.n_components

        # create an empty directory
        empty_dir = CHRONOS2_TINY_DIR / "empty_dir"
        shutil.rmtree(empty_dir, ignore_errors=True)
        empty_dir.mkdir(exist_ok=True)
        # loading from an empty directory should trigger download
        with caplog.at_level(logging.WARNING):
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                local_dir=empty_dir,
                **tfm_kwargs,
            )
        assert "Attempting to download from HuggingFace Hub instead" in caplog.text
        mock_method.assert_called()
        # clean up
        shutil.rmtree(empty_dir)

        # cannot load from non-existent directory
        with pytest.raises(ValueError, match=r"directory .* does not exist"):
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                local_dir="/non_existent_dir_123456abc",
                **tfm_kwargs,
            )

        # cannot load from a file path
        with pytest.raises(ValueError, match=r"path .* is not a directory."):
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                local_dir=CHRONOS2_TINY_DIR / "config.json",
                **tfm_kwargs,
            )

        # cannot load from a directory named config.json
        test_local_dir = CHRONOS2_TINY_DIR / "test"
        test_local_dir.mkdir(exist_ok=True)
        config_path = test_local_dir / "config.json"
        config_path.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match=r"Path .* is not a file"):
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                local_dir=test_local_dir,
                **tfm_kwargs,
            )
        config_path.rmdir()
        test_local_dir.rmdir()

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_default_no_finetuning(self, mock_method):
        # Default behavior: enable_finetuning=False (no training)
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        # Check that the given parameters remain unchanged, but that enable_finetuning is False
        # (because if not specified, it is None, but we want it to be False by default for foundation models)
        assert model.input_chunk_length == 12
        assert model.output_chunk_length == 6
        assert model.model_params["enable_finetuning"] is False
        mock_method.assert_called()

        # calling `fit()` should NOT use `trainer.fit()` when finetuning is disabled
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(
                series=self.series,
                future_covariates=self.future_cov,
            )
            mock_fit.assert_not_called()

        # foundation model should be deterministic by default
        assert model.model_created

        # predictions should allow n > output_chunk_length (autoregressive)
        pred = model.predict(n=10)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == self.series.n_components

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_full_finetuning(self, mock_method, tmpdir):
        # 1. Enable Full Fine-tuning
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            n_epochs=1,
            **tfm_kwargs,
        )
        assert model.model_params["enable_finetuning"] is True

        # Initialize model (this will train for 1 epoch, but that's fine for verification)
        model.fit(self.series)

        # Verify all parameters require grad
        for n, p in model.model.named_parameters():
            assert p.requires_grad is True

        # 3. Persistence (Save/Load)
        save_path = os.path.join(tmpdir, "model_full_ft.pt")
        model.save(save_path)

        # Load back
        loaded_model = Chronos2Model.load(save_path)
        assert loaded_model.model_params["enable_finetuning"] is True

        # Check predictions match
        pred_orig = model.predict(n=6, series=self.series)
        pred_loaded = loaded_model.predict(n=6, series=self.series)
        # Relax tolerance slightly for floating point differences across save/load on CPU
        assert np.allclose(pred_orig.values(), pred_loaded.values(), atol=1e-6)

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_partial_finetuning_block_freeze(self, mock_method):
        # Test freezing specific layers (partial fine-tuning)
        # We freeze the encoder, so only other parts (like head/decoder) should be trainable

        # For this test, let's freeze 'encoder'
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning={"freeze": ["encoder.*"]},
            n_epochs=1,
            **tfm_kwargs,
        )

        # Initialize model
        model.fit(self.series)

        # Check requires_grad status
        frozen_found = False
        trainable_found = False

        for name, param in model.model.named_parameters():
            if "encoder" in name:
                assert param.requires_grad is False
                frozen_found = True
            elif param.requires_grad:
                trainable_found = True

        assert frozen_found
        assert trainable_found

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_partial_finetuning_unfreeze(self, mock_method):
        # Test unfreezing specific layers (partial fine-tuning)
        # Everything is frozen EXCEPT the specified patterns

        # Let's unfreeze only the 'encoder' (or part of it)
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning={"unfreeze": ["encoder.*"]},
            n_epochs=1,
            **tfm_kwargs,
        )

        # Initialize model
        model.fit(self.series)

        # Check requires_grad status
        unfrozen_found = False
        frozen_found = False

        for name, param in model.model.named_parameters():
            if "encoder" in name:
                assert param.requires_grad is True
                unfrozen_found = True
            else:
                assert param.requires_grad is False
                frozen_found = True

        assert unfrozen_found
        assert frozen_found

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_finetuning_misconfiguration(self, mock_method):
        # 1. Invalid dict key
        with pytest.raises(
            ValueError,
            match="If `enable_finetuning` is a dict, it must contain exactly one key: 'freeze' or 'unfreeze'.",
        ):
            model = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                enable_finetuning={"invalid_key": ["pattern"]},
                **tfm_kwargs,
            )

            model.fit(self.series)

        # 2. Invalid dict value type
        with pytest.raises(ValueError, match="must be a list of strings"):
            model = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                enable_finetuning={"freeze": "not_a_list"},
                **tfm_kwargs,
            )

            model.fit(self.series)

        # 3. Both keys (impossible due to dict construction, but multiple keys)
        with pytest.raises(ValueError, match="must contain exactly one key"):
            model = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                enable_finetuning={"freeze": ["p1"], "unfreeze": ["p2"]},
                **tfm_kwargs,
            )

            model.fit(self.series)

    @pytest.mark.parametrize(
        "config",
        [
            (
                TimesFM2p5Model,
                "output_projection_point.hidden_layer.weight",
                {"hub_model_name": "google/timesfm-2.5-200m-pytorch"},
            ),
            (
                Chronos2Model,
                "output_patch_embedding.*",
                {"hub_model_name": "autogluon/chronos-2-small"},
            ),
            (
                PatchTSTFMModel,
                "*out_layer.*",
                {"local_dir": PATCHTST_FM_TINY_DIR},
            ),
        ]
        + (
            [
                (
                    TiRexModel,
                    "*output_patch_embedding.*",
                    {"hub_model_name": "NX-AI/TiRex", "accept_license": True},
                )
            ]
            if TIREX_AVAILABLE
            else []
        ),
    )
    def test_finetuning_all_models(self, config):
        """Tests fine-tuning with user-quantiles that are different to the ones the model was trained on."""
        model_cls, pattern, kwargs = config
        quantiles = [0.1, 0.5, 0.9]

        icl, ocl = 12, 6
        model = model_cls(
            input_chunk_length=icl,
            output_chunk_length=ocl,
            enable_finetuning={"unfreeze": [pattern]},
            n_epochs=1,
            likelihood=QuantileRegression(quantiles),
            **kwargs,
            **tfm_kwargs,
        )

        # fit model with validation series (training quantile loss is different from evaluation quantile loss)
        series = self.series[: icl + ocl]
        model.fit(series, val_series=series)

        # Check requires_grad status
        unfrozen_found = False
        frozen_found = False

        for name, param in model.model.named_parameters():
            if pattern.replace("*", "") in name:
                assert param.requires_grad is True
                unfrozen_found = True
            else:
                assert param.requires_grad is False
                frozen_found = True

        assert unfrozen_found
        assert frozen_found

        preds = model.predict(n=6, predict_likelihood_parameters=True)
        assert preds.shape == (6, self.series.n_components * len(quantiles), 1)
        assert not np.isnan(preds.all_values(copy=False)).any()


class TestVariableInputChunkLength:
    """Tests for variable input chunk length support in foundation models."""

    series_long = generate_series(n_variables=2, length=20, prefix="A")
    series_short = generate_series(n_variables=2, length=8, prefix="B")
    series_very_short = generate_series(n_variables=2, length=3, prefix="C")
    future_cov = generate_series(n_variables=3, length=200, prefix="FC")

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_variable_icl_properties(self, mock_method):
        """Variable ICL model should report correct properties."""
        model = Chronos2Model(
            input_chunk_length=(1, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        assert model.min_input_chunk_length == 1
        assert model.input_chunk_length == 14

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_fixed_icl_properties(self, mock_method):
        """Fixed ICL model should not report variable input support."""
        model = Chronos2Model(
            input_chunk_length=14,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        assert model.min_input_chunk_length == 14

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_invalid_variable_icl(self, mock_method):
        """Invalid (min, max) tuples should raise."""
        msg_expected = "must be an integer or a"
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=(1, 10, 5),
                output_chunk_length=6,
                **tfm_kwargs,
            )
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=[1, 10],
                output_chunk_length=6,
                **tfm_kwargs,
            )
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=1.2,
                output_chunk_length=6,
                **tfm_kwargs,
            )

        msg_expected = "must be >= 1."
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=0,
                output_chunk_length=6,
                **tfm_kwargs,
            )

        msg_expected = "must satisfy `1 <= min_length <= max_length`"
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=(15, 10),
                output_chunk_length=6,
                **tfm_kwargs,
            )
        with pytest.raises(ValueError, match=msg_expected):
            Chronos2Model(
                input_chunk_length=(0, 10),
                output_chunk_length=6,
                **tfm_kwargs,
            )

    @pytest.mark.parametrize(
        "model_cls,extra_kwargs,mock_ctx_factory",
        [
            pytest.param(
                Chronos2Model,
                {},
                lambda: patch(
                    HF_HUB_DOWNLOAD_PATCH_TARGET,
                    side_effect=mock_hf_hub_download,
                ),
                id="Chronos2",
            ),
            pytest.param(
                PatchTSTFMModel,
                {"local_dir": PATCHTST_FM_TINY_DIR},
                contextlib.nullcontext,
                id="PatchTSTFM",
            ),
            pytest.param(
                TimesFM2p5Model,
                {},
                timesfm2p5_tiny_context,
                id="TimesFM2p5",
            ),
            pytest.param(
                TiRexModel,
                {"accept_license": True},
                lambda: patch(
                    TIREX_LOAD_MODEL_PATCH_TARGET,
                    return_value=TiRexStub(),
                ),
                id="TiRex",
            ),
        ],
    )
    @pytest.mark.parametrize("series_length", [16, 14, 8, 1])
    def test_variable_icl_fit_predict(
        self, model_cls, extra_kwargs, mock_ctx_factory, series_length
    ):
        """Fit and Predict with variable ICL should work on series longer or shorter than ICL."""
        with mock_ctx_factory():
            model = model_cls(
                input_chunk_length=(1, 14),
                output_chunk_length=6,
                **extra_kwargs,
                **tfm_kwargs,
            )
            with patch("pytorch_lightning.Trainer.fit") as mock_fit:
                model.fit(series=self.series_long[:series_length])
                mock_fit.assert_not_called()

            pred = model.predict(n=6, series=self.series_long[:series_length])
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 6
        assert pred.n_components == self.series_long.n_components
        assert not np.isnan(pred.all_values(copy=False)).any()

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    @pytest.mark.parametrize(
        "series_length",
        [
            # 16,
            # 14,
            8,
            # 1
        ],
    )
    def test_variable_icl_with_covariates(self, mock_method, series_length):
        """Variable ICL inference should work with future covariates on a long series."""
        model = Chronos2Model(
            input_chunk_length=(1, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        model.fit(
            series=self.series_long[:series_length], future_covariates=self.future_cov
        )

        # use a long enough series so padding doesn't extend the time index
        # beyond what covariates cover
        pred = model.predict(
            n=6,
            series=self.series_long[:series_length],
            future_covariates=self.future_cov,
        )
        assert len(pred) == 6
        assert not np.isnan(pred.all_values(copy=False)).any()

        # covariates extend further into the past; these values will be ignored; prediction is identical
        pred2 = model.predict(
            n=6,
            series=self.series_long[:series_length],
            future_covariates=self.future_cov.prepend_values([[0, 1, 2]]),
        )
        assert pred == pred2

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_variable_icl_with_covariates_invalid_time_span(self, mock_method):
        """Variable ICL inference should fail with future covariates starting too late or ending too early."""
        series_length = 2
        ocl = 6
        model = Chronos2Model(
            input_chunk_length=(1, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        model.fit(
            series=self.series_long[:series_length], future_covariates=self.future_cov
        )

        # covs start too late
        with pytest.raises(ValueError, match="must start at or before time step"):
            _ = model.predict(
                n=ocl,
                series=self.series_long[:series_length],
                future_covariates=self.future_cov[1 : series_length + ocl],
            )
        # covs end too soon
        with pytest.raises(ValueError, match="must end at or after time step"):
            _ = model.predict(
                n=ocl,
                series=self.series_long[:series_length],
                future_covariates=self.future_cov[: series_length + ocl - 1],
            )

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_variable_icl_inference_mixed_lengths(self, mock_method):
        """Predict should work on a mix of short and long series."""
        model = Chronos2Model(
            input_chunk_length=(1, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        model.fit(series=self.series_long)

        preds = model.predict(n=6, series=[self.series_long, self.series_short])
        assert isinstance(preds, list)
        assert len(preds) == 2
        for pred in preds:
            assert len(pred) == 6
            assert not np.isnan(pred.all_values(copy=False)).any()

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_variable_icl_finetuning_short_series(self, mock_method):
        """Fine-tuning with variable ICL should accept series shorter than ICL
        but at least min_input_chunk_length + output_chunk_length."""
        icl = 14
        ocl = 6
        model = Chronos2Model(
            input_chunk_length=(1, icl),
            output_chunk_length=ocl,
            enable_finetuning=True,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            **tfm_kwargs,
        )
        # series_short has length 8, which is >= min_icl(1) + ocl(6) = 7
        model.fit(series=self.series_short, epochs=1)

        pred = model.predict(n=6, series=self.series_short)
        assert len(pred) == 6
        assert not np.isnan(pred.all_values(copy=False)).any()

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_variable_icl_finetuning_too_short(self, mock_method):
        """Fine-tuning should fail if series < min_icl + ocl."""
        icl = 14
        ocl = 6
        model = Chronos2Model(
            input_chunk_length=(5, icl),
            output_chunk_length=ocl,
            enable_finetuning=True,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            **tfm_kwargs,
        )
        # series_very_short has length 3, which is < min_icl(5) + ocl(6) = 11
        with pytest.raises(ValueError, match="too short"):
            model.fit(series=self.series_very_short, epochs=1)

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_fixed_icl_still_rejects_short_series(self, mock_method):
        """Without a variable input chunk length tuple, short series should still fail."""
        model = Chronos2Model(
            input_chunk_length=14,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        model.fit(series=self.series_long)

        with pytest.raises(ValueError):
            model.predict(n=6, series=self.series_short)

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_extreme_lags_variable_icl(self, mock_method):
        """extreme_lags should use min_input_chunk_length for variable ICL."""
        model = Chronos2Model(
            input_chunk_length=(3, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        min_target_lag = model.extreme_lags[0]
        assert min_target_lag == -3

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_extreme_lags_fixed_icl(self, mock_method):
        """extreme_lags should use input_chunk_length for fixed ICL."""
        model = Chronos2Model(
            input_chunk_length=14,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        min_target_lag = model.extreme_lags[0]
        assert min_target_lag == -14

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_min_train_series_length_variable(self, mock_method):
        """min_train_series_length should use min_input_chunk_length."""
        model_var = Chronos2Model(
            input_chunk_length=(2, 14),
            output_chunk_length=6,
            **tfm_kwargs,
        )
        # min_train_series_length = min_icl + ocl
        assert model_var.min_train_series_length == 8

    @patch(
        HF_HUB_DOWNLOAD_PATCH_TARGET,
        side_effect=mock_hf_hub_download,
    )
    def test_min_train_series_length_fixed(self, mock_method):
        """min_train_series_length for fixed ICL should be standard (ICL only as pre-trained)."""
        model = Chronos2Model(
            input_chunk_length=14,
            output_chunk_length=6,
            **tfm_kwargs,
        )
        # min_train_series_length = icl + ocl
        assert model.min_train_series_length == 20

    @pytest.mark.parametrize(
        "model_cls,extra_kwargs,mock_ctx_factory,supports_ckpt",
        [
            pytest.param(
                Chronos2Model,
                {},
                lambda: patch(
                    HF_HUB_DOWNLOAD_PATCH_TARGET,
                    side_effect=mock_hf_hub_download,
                ),
                True,
                id="Chronos2",
            ),
            pytest.param(
                PatchTSTFMModel,
                {"local_dir": PATCHTST_FM_TINY_DIR},
                contextlib.nullcontext,
                True,
                id="PatchTSTFM",
            ),
            pytest.param(
                TimesFM2p5Model,
                {},
                timesfm2p5_tiny_context,
                True,
                id="TimesFM2p5",
            ),
            pytest.param(
                TiRexModel,
                {"accept_license": True},
                lambda: patch(
                    TIREX_LOAD_MODEL_PATCH_TARGET,
                    return_value=TiRexStub(),
                ),
                False,
                id="TiRex",
            ),
        ],
    )
    def test_variable_icl_save_load(
        self, model_cls, extra_kwargs, mock_ctx_factory, supports_ckpt, tmp_path
    ):
        """Saving and loading a model with variable ICL should preserve predictions.

        Tests save()/load(), load_weights(), and load_from_checkpoint().
        """
        icl = (1, 14)
        ocl = 6
        series = self.series_short  # length 8, within (1, 14)

        with mock_ctx_factory():
            # -- save() / load() --
            model = model_cls(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model.fit(series=series)
            pred_orig = model.predict(n=ocl, series=series)

            save_path = str(tmp_path / "var_icl_model.pt")
            model.save(save_path)

            loaded = model_cls.load(save_path)
            assert loaded.min_input_chunk_length == icl[0]
            assert loaded.input_chunk_length == icl[1]
            pred_loaded = loaded.predict(n=ocl, series=series)
            np.testing.assert_array_almost_equal(
                pred_orig.all_values(copy=False),
                pred_loaded.all_values(copy=False),
            )

            # -- load_weights() --
            model2 = model_cls(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model2.load_weights(save_path)
            assert model2.min_input_chunk_length == icl[0]
            assert model2.input_chunk_length == icl[1]
            pred_weights = model2.predict(n=ocl, series=series)
            np.testing.assert_array_almost_equal(
                pred_orig.all_values(copy=False),
                pred_weights.all_values(copy=False),
            )

            # -- load_from_checkpoint() (requires fine-tuning + save_checkpoints) --
            if not supports_ckpt:
                return

            model_name = "var_icl_ckpt"
            model_ft = model_cls(
                input_chunk_length=icl,
                output_chunk_length=ocl,
                enable_finetuning=True,
                n_epochs=1,
                save_checkpoints=True,
                model_name=model_name,
                work_dir=str(tmp_path),
                force_reset=True,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model_ft.fit(series=series)
            pred_ft = model_ft.predict(n=ocl, series=series)

            loaded_ckpt = model_cls.load_from_checkpoint(
                model_name=model_name,
                work_dir=str(tmp_path),
                best=False,
            )
            assert loaded_ckpt.min_input_chunk_length == icl[0]
            assert loaded_ckpt.input_chunk_length == icl[1]
            pred_ckpt = loaded_ckpt.predict(n=ocl, series=series)
            np.testing.assert_array_almost_equal(
                pred_ft.all_values(copy=False),
                pred_ckpt.all_values(copy=False),
            )

    @pytest.mark.parametrize(
        "model_cls,extra_kwargs,mock_ctx_factory",
        [
            pytest.param(
                Chronos2Model,
                {},
                lambda: patch(
                    HF_HUB_DOWNLOAD_PATCH_TARGET,
                    side_effect=mock_hf_hub_download,
                ),
                id="Chronos2",
            ),
            pytest.param(
                PatchTSTFMModel,
                {"local_dir": PATCHTST_FM_TINY_DIR},
                contextlib.nullcontext,
                id="PatchTSTFM",
            ),
            pytest.param(
                TimesFM2p5Model,
                {},
                timesfm2p5_tiny_context,
                id="TimesFM2p5",
            ),
            pytest.param(
                TiRexModel,
                {"accept_license": True},
                lambda: patch(
                    TIREX_LOAD_MODEL_PATCH_TARGET,
                    return_value=TiRexStub(),
                ),
                id="TiRex",
            ),
        ],
    )
    def test_variable_icl_matches_fixed_icl(
        self, model_cls, extra_kwargs, mock_ctx_factory
    ):
        """A single model with variable ICL should produce the same predictions
        as dedicated fixed-ICL models for each series length.

        With input_chunk_length=(1, len(longer_series)), the variable ICL model
        should match a model with input_chunk_length=len(series_i) on each
        series_i individually.
        """
        series_long = self.series_long  # length 20
        series_short = self.series_short  # length 8
        ocl = 6

        with mock_ctx_factory():
            # variable ICL model: covers both series lengths
            var_model = model_cls(
                input_chunk_length=(1, len(series_long)),
                output_chunk_length=ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            var_model.fit(series=series_long)

            pred_var_long, pred_var_short = var_model.predict(
                n=ocl, series=[series_long, series_short]
            )

        with mock_ctx_factory():
            # fixed ICL model for the long series
            fixed_long = model_cls(
                input_chunk_length=len(series_long),
                output_chunk_length=ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            fixed_long.fit(series=series_long)
            pred_fixed_long = fixed_long.predict(n=ocl, series=series_long)

        with mock_ctx_factory():
            # fixed ICL model for the short series
            fixed_short = model_cls(
                input_chunk_length=len(series_short),
                output_chunk_length=ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            fixed_short.fit(series=series_short)
            pred_fixed_short = fixed_short.predict(n=ocl, series=series_short)

        np.testing.assert_array_almost_equal(
            pred_var_long.all_values(copy=False),
            pred_fixed_long.all_values(copy=False),
        )
        np.testing.assert_array_almost_equal(
            pred_var_short.all_values(copy=False),
            pred_fixed_short.all_values(copy=False),
        )

    @pytest.mark.parametrize(
        "model_cls,extra_kwargs,mock_ctx_factory",
        [
            pytest.param(
                Chronos2Model,
                {},
                lambda: patch(
                    HF_HUB_DOWNLOAD_PATCH_TARGET,
                    side_effect=mock_hf_hub_download,
                ),
                id="Chronos2",
            ),
            pytest.param(
                PatchTSTFMModel,
                {"local_dir": PATCHTST_FM_TINY_DIR},
                contextlib.nullcontext,
                id="PatchTSTFM",
            ),
            pytest.param(
                TimesFM2p5Model,
                {},
                timesfm2p5_tiny_context,
                id="TimesFM2p5",
            ),
            pytest.param(
                TiRexModel,
                {"accept_license": True},
                lambda: patch(
                    TIREX_LOAD_MODEL_PATCH_TARGET,
                    return_value=TiRexStub(),
                ),
                id="TiRex",
            ),
        ],
    )
    def test_load_weights_different_chunk_params(
        self, model_cls, extra_kwargs, mock_ctx_factory, tmp_path
    ):
        """load_weights() should accept different ICL, OCL, and OCS for
        foundation models, since their weights are independent of these
        parameters."""
        series = self.series_long  # length 20
        save_icl, save_ocl = 14, 6

        with mock_ctx_factory():
            model_save = model_cls(
                input_chunk_length=save_icl,
                output_chunk_length=save_ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model_save.fit(series=series)
            pred_save = model_save.predict(n=save_ocl)
            save_path = str(tmp_path / "donor_model.pt")
            model_save.save(save_path)

            # same OCL but variable ICL: predictions must match
            model_same_ocl = model_cls(
                input_chunk_length=(1, save_icl),
                output_chunk_length=save_ocl,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model_same_ocl.load_weights(save_path)
            pred_same_ocl = model_same_ocl.predict(n=save_ocl, series=series)
            np.testing.assert_array_almost_equal(
                pred_save.all_values(copy=False),
                pred_same_ocl.all_values(copy=False),
            )

            # different OCL and OCS: weights load and predictions are valid
            # (values may differ for models where OCL affects the forward pass,
            # e.g. PatchTSTFM shifts patch boundaries based on forecast_length)
            model_diff = model_cls(
                input_chunk_length=(1, 20),
                output_chunk_length=4,
                output_chunk_shift=1,
                **extra_kwargs,
                **tfm_kwargs,
            )
            model_diff.load_weights(save_path)
            pred_diff = model_diff.predict(n=4, series=series)
            assert len(pred_diff) == 4
            assert not np.isnan(pred_diff.all_values(copy=False)).any()
