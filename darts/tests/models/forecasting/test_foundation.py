import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from darts import TimeSeries, concatenate
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts.models import Chronos2Model


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


dummy_local_dir = (Path(__file__).parent / "dummy" / "chronos2").absolute()


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
