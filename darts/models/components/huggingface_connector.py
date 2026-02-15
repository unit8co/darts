"""
Hugging Face Connector
----------------------
"""

import inspect
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
)

logger = get_logger(__name__)


class HuggingFaceConnector:
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        config_file: str = "config.json",
        model_file: str = "model.safetensors",
    ):
        """HuggingFaceConnector enables loading a model configuration and weights from HuggingFace Hub.

        This class provides methods to download the model configuration and weights from a specified HuggingFace
        repository, or to load them from a local directory, if provided.

        Optionally, the local directory where pre-downloaded model files are stored can be set using the `local_dir`
        parameter.

        This class provides methods to load the model configuration and weights:
        - load_config() : Load the model configuration from a JSON file.
        - load_model_weights(module: PLForecastingModule) : Load the model weights into the given PyTorch module.
        - load_model(module_class: type[PLForecastingModule], pl_module_params: dict) :
          Load the model by creating an instance of the given module class and loading the weights.

        Parameters
        ----------
        model_name
             The HuggingFace repository name where the model is stored, e.g., "amazon/chronos-2".
        model_revision
             The revision of the model in the HuggingFace repository. Must be a branch name, tag name, or commit hash.
                If not provided, the default branch and the latest commit will be used.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead.
        config_file
             The name of the configuration file. Default is "config.json".
        model_file
             The name of the model weight file. Default is "model.safetensors".
        """
        if local_dir is not None:
            local_dir_path = Path(local_dir)
            if not local_dir_path.exists():
                raise_log(
                    ValueError(f"`local_dir` directory `{local_dir}` does not exist."),
                    logger,
                )
            if not local_dir_path.is_dir():
                raise_log(
                    ValueError(f"`local_dir` path `{local_dir}` is not a directory."),
                    logger,
                )
            local_dir = local_dir_path

        self.model_name = model_name
        self.model_revision = model_revision
        self.local_dir = local_dir
        self.config_file = config_file
        self.model_file = model_file

    def load_config(self) -> dict:
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

    def load_model_weights(
        self,
        module: PLForecastingModule,
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

    def load_model(
        self,
        module_class: type[PLForecastingModule],
        pl_module_params: dict,
        additional_params: dict | None = None,
    ) -> PLForecastingModule:
        """Load the model by creating an instance of the given module class and loading
        the weights. Some configuration files might contain external parameters that
        are not part of the module class constructor like `architectures`. They are filtered
        out before instantiating the module.

        Parameters
        ----------
        module_class
            The class of the PyTorch Lightning module to instantiate.
        pl_module_params
            The parameters of the PyTorch Lightning module to instantiate.
        additional_params
            Additional parameters to pass to the `module_class` constructor when instantiating.

        Returns
        -------
        PLForecastingModule
            The loaded PyTorch Lightning module.
        """
        additional_params = additional_params or {}
        config = self.load_config()
        module_params = self._extract_module_params(module_class, config)
        module = module_class(
            **module_params,
            **pl_module_params,
            **additional_params,
        )
        self.load_model_weights(module)
        return module

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
                logger.warning(
                    f"File {filename} not found in `local_dir` {self.local_dir}. "
                    f"Attempting to download from HuggingFace Hub instead and "
                    f"save it to the `local_dir`."
                )
            elif not path.is_file():
                raise_log(
                    ValueError(f"Path {path} is not a file"),
                    logger,
                )
            else:
                return path

        # Download the file from HuggingFace Hub and download to `local_dir` if specified
        # Otherwise, it will be downloaded to a cache directory managed by `huggingface_hub`
        file_path = hf_hub_download(
            repo_id=self.model_name,
            filename=filename,
            revision=self.model_revision,
            local_dir=self.local_dir,
        )
        return Path(file_path)

    @staticmethod
    def _extract_module_params(
        module_class: type[PLForecastingModule],
        config: dict,
    ):
        """Extract params from `config` to set up the given `module_class`."""
        get_params = list(inspect.signature(module_class.__init__).parameters.keys())
        get_params.remove("self")
        return {kwarg: config.get(kwarg) for kwarg in get_params if kwarg in config}
