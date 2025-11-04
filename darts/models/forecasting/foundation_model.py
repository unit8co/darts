"""
Time Series Foundation Model (TSFM)
---------------------------------
"""

import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import TorchTrainingSample

logger = get_logger(__name__)


class HuggingFaceModelMixin:
    _repo_id: str
    _repo_commit: str
    _config_file: str = "config.json"
    _model_file: str = "model.safetensors"

    _local_dir: Optional[os.PathLike] = None

    @property
    def repo_id(self) -> str:
        """The HuggingFace repository ID where the model is stored."""
        return self._repo_id

    @property
    def repo_commit(self) -> str:
        """The commit ID of the model in the HuggingFace repository."""
        return self._repo_commit

    @property
    def config_file(self) -> str:
        """The name of the configuration file."""
        return self._config_file

    @property
    def model_file(self) -> str:
        """The name of the model weight file."""
        return self._model_file

    @property
    def local_dir(self) -> Optional[os.PathLike]:
        """The local directory where the pre-downloaded model files are stored."""
        return self._local_dir

    @local_dir.setter
    def local_dir(self, value: Optional[Union[str, os.PathLike]]) -> None:
        """Set the local directory where the pre-downloaded model files are stored.

        Parameters
        ----------
        value
            The local directory path.
        """
        if value is not None:
            path = Path(value)
            raise_if_not(path.exists(), f"Directory {value} does not exist.", logger)
            raise_if_not(path.is_dir(), f"Path {value} is not a directory.", logger)
            self._local_dir = path

    def _get_file_path(
        self,
        filename: str,
    ) -> os.PathLike:
        """Get the path to a file either from a local directory or by downloading it from HuggingFace.

        Parameters
        ----------
        filename
            The name of the file to retrieve.

        Returns
        -------
        os.PathLike
            The path to the requested file.
        """
        if self.local_dir is not None:
            path = Path(self.local_dir) / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"File {filename} not found in {self.local_dir}"
                )
            if not path.is_file():
                raise ValueError(f"Path {path} is not a file")
            return path
        else:
            repo_path = snapshot_download(
                repo_id=self.repo_id,
                revision=self.repo_commit,
            )
            return Path(repo_path) / filename

    def _load_config(
        self,
    ) -> dict:
        """Load the model configuration from a JSON file.

        Returns
        -------
        dict
            The model configuration.
        """
        config_path = self._get_file_path(self.config_file)
        with open(config_path) as f:
            config = json.load(f)
        return config

    def _load_model_weights(
        self,
        module: nn.Module,
    ) -> None:
        """Load the model weights from a safetensors file.

        Parameters
        ----------
        module
            The PyTorch module to load the weights into.
        """
        module_path = self._get_file_path(self.model_file)
        state_dict = load_file(module_path)
        module.load_state_dict(state_dict)

    def _load_model(
        self,
        module_class: type[PLForecastingModule],
    ) -> PLForecastingModule:
        """Load the model by creating an instance of the given module class and loading
        the weights.

        Parameters
        ----------
        module_class
            The class of the PyTorch Lightning module to instantiate.

        Returns
        -------
        PLForecastingModule
            The loaded PyTorch Lightning module.
        """
        config = self._load_config()
        module = module_class(**config)
        self._load_model_weights(module)
        return module


class FoundationModel(MixedCovariatesTorchModel):
    @abstractmethod
    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        """Just like in `TorchForecastingModel`, subclasses must implement this method."""

    # TODO: implement `fit()` but bypass training since the model is already trained.
