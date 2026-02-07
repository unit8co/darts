import logging
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from darts import TimeSeries, concatenate
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts.models import Chronos2Model
from darts.utils.callbacks.fine_tuning import LayerFreezeCallback, PeftCallback


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


dummy_local_dir = (
    Path(__file__).parent / "artefacts" / "chronos2" / "tiny_chronos2"
).absolute()


def mock_download(
    repo_id: str,
    filename: str,
    revision: str | None,
    local_dir: str | Path | None,
    **kwargs,
):
    path = dummy_local_dir / filename
    if local_dir is None:
        return str(path)
    else:
        dest_path = Path(local_dir) / filename
        shutil.copy(path, dest_path)
        return str(dest_path)


class TestFoundationModel:
    series = generate_series(n_variables=2, length=100, prefix="A")
    future_cov = generate_series(n_variables=3, length=200, prefix="C")

    @patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
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
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
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
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
    )
    def test_local_dir(self, mock_method, caplog):
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            local_dir=dummy_local_dir,
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
        empty_dir = dummy_local_dir / "empty_dir"
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
                local_dir=dummy_local_dir / "config.json",
                **tfm_kwargs,
            )

        # cannot load from a directory named config.json
        test_local_dir = dummy_local_dir / "test"
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
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
    )
    def test_full_finetuning(self, mock_method, tmpdir):
        # 1. Training activation
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            n_epochs=5,
            **tfm_kwargs,
        )
        assert model._requires_training is True

        # Capture initial weights
        model.fit(self.series)
        initial_params = {
            n: p.clone() for n, p in model.internal_model.named_parameters()
        }

        # 2. Weight update
        # We need to actually train for 1 epoch. tfm_kwargs usually has "accelerator": "cpu"
        model.fit(self.series, epochs=1)

        # Check if at least some weights changed
        any_changed = False
        for n, p in model.internal_model.named_parameters():
            if not torch.equal(initial_params[n], p):
                any_changed = True
                break
        assert any_changed, "The weights should be updated after fine-tuning"

        # 3. Persistence (Save/Load)
        save_path = os.path.join(tmpdir, "model.pt")
        model.save(save_path)
        loaded_model = Chronos2Model.load(save_path)

        pred_orig = model.predict(n=6, series=self.series)
        pred_loaded = loaded_model.predict(n=6, series=self.series)
        assert np.allclose(pred_orig.values(), pred_loaded.values()), (
            "Prediction of the fine-tuned model and the saved/loaded fine-tuned model should be the same"
        )

    @patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
    )
    def test_partial_finetuning(self, mock_method):
        # 1. Callback injection
        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            freeze_patterns=["encoder.block.0"],
            unfreeze_patterns=["encoder.block.0.layer.0"],  # Example unfreeze
            **tfm_kwargs,
        )
        assert any(
            isinstance(c, LayerFreezeCallback)
            for c in model.trainer_params["callbacks"]
        )

        # 2. Freezing logic
        # We call fit to initialize the model and trigger the callback setup automatically
        model.fit(self.series, epochs=5)

        # Check requires_grad status.
        found_any = False
        for name, param in model.internal_model.named_parameters():
            if name.startswith("encoder.block.0"):
                found_any = True
                if name.startswith("encoder.block.0.layer.0"):
                    assert param.requires_grad is True, (
                        f"Parameter {name} should be trainable"
                    )
                else:
                    assert param.requires_grad is False, (
                        f"Parameter {name} should be frozen"
                    )
        assert found_any, "No parameters matched the freeze patterns, test is invalid"

    @patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
    )
    def test_finetuning_misconfiguration(self, mock_method):
        # Warning if freeze_patterns assigned but enable_finetuning is False
        with patch(
            "darts.models.forecasting.foundation_model.logger.warning"
        ) as mock_warning:
            _ = Chronos2Model(
                input_chunk_length=12,
                output_chunk_length=6,
                enable_finetuning=False,
                freeze_patterns=["some_pattern"],
                **tfm_kwargs,
            )
            mock_warning.assert_called_once()
            assert "enable_finetuning` is False" in mock_warning.call_args[0][0]

    @patch(
        "darts.models.components.huggingface_connector.hf_hub_download",
        side_effect=mock_download,
    )
    def test_lora_callback(self, mock_method, tmpdir):
        pytest.importorskip("peft")
        from peft import LoraConfig, PeftModel

        lora_config = LoraConfig(target_modules=["q", "v"])
        callback = PeftCallback(peft_config=lora_config)

        # Avoid duplicate pl_trainer_kwargs
        kwargs = {k: v for k, v in tfm_kwargs.items() if k != "pl_trainer_kwargs"}
        pl_trainer_kwargs = tfm_kwargs.get("pl_trainer_kwargs", {}).copy()
        pl_trainer_kwargs["callbacks"] = [callback]

        model = Chronos2Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            pl_trainer_kwargs=pl_trainer_kwargs,
            **kwargs,
        )

        # 1. Initialize and fit
        model.fit(self.series, epochs=5)

        # Verify transformation happened
        assert isinstance(model.internal_model, PeftModel), (
            "Internal model should be a PeftModel after fit"
        )

        # 2. Checkpoint merging test (via save/load)
        save_path = os.path.join(tmpdir, "lora_model.pt")
        model.save(save_path)

        # Loading back should yield a standard model (weights merged)
        loaded_model = Chronos2Model.load(save_path)
        assert not isinstance(loaded_model.internal_model, PeftModel), (
            "Loaded model should have merged weights and not be a PeftModel"
        )

        # Verify predictions match
        pred_orig = model.predict(n=6, series=self.series)
        pred_loaded = loaded_model.predict(n=6, series=self.series)
        assert np.allclose(pred_orig.values(), pred_loaded.values()), (
            "Prediction of the fine-tuned model and the saved/loaded fine-tuned model should be the same"
        )
