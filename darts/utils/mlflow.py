"""
MLflow Integration for Darts
-----------------------------

Custom MLflow model flavor for darts forecasting models. Supports saving, loading,
logging and autolog for any darts ``ForecastingModel`` (statistical, ML-based, and PyTorch-based)
to MLflow.
"""

import importlib
import json
import os
from typing import Optional, Union

import mlflow
import yaml
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import TempDir, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

import darts
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.utils import PL_AVAILABLE

logger = get_logger(__name__)

FLAVOR_NAME = "darts"
_MODEL_DATA_SUBFOLDER = "data"


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"
_MODEL_FILE_TORCH_CKPT = "model.pt.ckpt"


def save_model(
    model,
    path: str,
    conda_env: Optional[Union[dict, str]] = None,
    code_paths: Optional[list[str]] = None,
    pip_requirements: Optional[list[str]] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    signature=None,
    input_example=None,
    metadata: Optional[dict] = None,
    mlflow_model: Optional[Model] = None,
) -> None:
    """Save a darts forecasting model in MLflow format.

    Produces an MLflow model directory at ``path`` containing:

    * The serialised darts model (delegated to the model's own ``save()`` method).
    * An ``MLmodel`` YAML file with flavor metadata.
    * ``conda.yaml`` and ``requirements.txt`` environment files.

    Parameters
    ----------
    model
        A fitted darts ``ForecastingModel`` instance.
    path
        Local filesystem path where the model directory will be created.
    conda_env
        A conda environment specification (dict or path to a ``conda.yaml``).
        If ``None``, a default environment is generated.
    code_paths
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are prepended to the system path
        when the model is loaded.
    pip_requirements
        A list of pip requirement strings. Overrides ``conda_env`` pip section
        when provided.
    extra_pip_requirements
        A list of additional pip requirement strings to add to the model's environment,
        in addition to the default requirements.
    signature
        An ``mlflow.models.ModelSignature`` instance describing model input/output.
        Use :func:`infer_signature` to automatically generate from example inputs.
    input_example
        An example input for the model (used by MLflow UI). Should be a DataFrame
        created with :func:`prepare_pyfunc_input`.
    metadata
        Optional dictionary of custom metadata to store in the ``MLmodel`` file.
    mlflow_model
        Optional MLflow Model object to use for saving. When provided (typically by
        ``Model.log()``), this model instance is used instead of creating a new one.
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    data_dir = os.path.join(path, _MODEL_DATA_SUBFOLDER)
    is_torch = _is_torch_model(model)

    os.makedirs(data_dir, exist_ok=True)

    if is_torch:
        model_file = _MODEL_FILE_TORCH
        model.save(os.path.join(data_dir, model_file))
    else:
        model_file = _MODEL_FILE_STAT
        model.save(os.path.join(data_dir, model_file))

    module_path, class_name = _get_model_class_path(model)

    darts_flavor_conf = {
        "darts_version": darts.__version__,
        "model_class_module": module_path,
        "model_class_name": class_name,
        "model_file": model_file,
        "is_torch_model": is_torch,
        "data": _MODEL_DATA_SUBFOLDER,
    }

    if code_dir_subpath is not None:
        darts_flavor_conf["code"] = code_dir_subpath

    default_reqs = None if pip_requirements else get_default_pip_requirements(is_torch)
    conda_env, pip_requirements, pip_constraints = (
        _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
        if conda_env is None
        else _process_conda_env(conda_env)
    )

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    if metadata is not None:
        mlflow_model.metadata = metadata

    mlflow_model.add_flavor(FLAVOR_NAME, **darts_flavor_conf)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(
    model_uri: str,
    dst_path: Optional[str] = None,
    **kwargs,
):
    """Load a darts model from an MLflow model URI.

    Parameters
    ----------
    model_uri
        An MLflow model URI, e.g. ``"runs:/<run_id>/model"``,
        ``"models:/<name>/<version>"``, or a local ``file:///...`` path.
    dst_path
        Optional local path for downloading remote artifacts.
    **kwargs
        Additional keyword arguments forwarded to the model's ``load()`` method
        (e.g. ``map_location`` for Torch models).

    Returns
    -------
    ForecastingModel
        The loaded darts forecasting model.
    """
    local_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )

    flavor_conf = _get_flavor_configuration(
        model_path=local_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_path, flavor_conf)

    model_cls = _import_model_class(
        flavor_conf["model_class_module"], flavor_conf["model_class_name"]
    )

    model_path = os.path.join(
        local_path,
        flavor_conf.get("data", _MODEL_DATA_SUBFOLDER),
        flavor_conf["model_file"],
    )

    if flavor_conf.get("is_torch_model", False):
        return model_cls.load(model_path, **kwargs)
    else:
        return model_cls.load(model_path)


def log_model(
    model,
    artifact_path: Optional[str] = None,
    name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    conda_env: Optional[Union[dict, str]] = None,
    code_paths: Optional[list[str]] = None,
    pip_requirements: Optional[list[str]] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    signature=None,
    input_example=None,
    metadata: Optional[dict] = None,
    log_params: bool = True,
):
    """Log a darts model to the current MLflow run.

    Parameters
    ----------
    model
        A fitted darts ``ForecastingModel`` instance.
    artifact_path
        The run-relative artifact path under which to log the model.
        Defaults to ``"model"``. Deprecated in favour of ``name``.
    name
        The name for the model artifact. If provided, takes precedence over
        ``artifact_path``.
    registered_model_name
        If provided, the model is registered in the MLflow Model Registry
        under this name.
    conda_env
        Conda environment specification (dict or path).
    code_paths
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are prepended to the system path
        when the model is loaded.
    pip_requirements
        Pip requirements list.
    extra_pip_requirements
        A list of additional pip requirement strings to add to the model's environment,
        in addition to the default requirements.
    signature
        An ``mlflow.models.ModelSignature``. Use :func:`infer_signature`
        to automatically generate from example inputs.
    input_example
        An example model input. Should be a DataFrame created with
        :func:`prepare_pyfunc_input`.
    metadata
        Optional dict of custom metadata.
    log_params
        If ``True`` (default), log the model's creation parameters via
        ``mlflow.log_params()``.

    Returns
    -------
    ModelInfo
        MLflow ModelInfo object containing model_uri, run_id, artifact_path,
        model_id, timestamps, and other metadata about the logged model.
    """
    # import required as Model.log will call flavor.save_model() internally
    import darts.utils.mlflow as darts_mlflow

    if log_params:
        _log_model_params(model)
        _log_covariate_info(model)

    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=darts_mlflow,
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        signature=signature,
        input_example=input_example,
        metadata=metadata,
    )


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models: bool = True,
    log_params: bool = True,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Enable (or disable) automatic MLflow logging for darts models.

    When enabled, every call to ``model.fit()`` on any darts forecasting model
    will automatically:

    1. Start an MLflow run (or reuse the currently active one).
    2. Log model creation parameters (``model.model_params``).
    3. For PyTorch-based models: inject a callback that logs per-epoch
       ``train_loss`` / ``val_loss`` metrics.
    4. Log the trained model artifact at the end of training.

    Parameters
    ----------
    log_models
        If ``True`` (default), log the trained model artifact after ``fit()``.
    log_params
        If ``True`` (default), log model creation parameters.
    disable
        If ``True``, restore the original ``fit()`` methods and stop
        autologging.
    silent
        If ``True`` (default ``False``), suppress all event logging and warnings from
        MLflow during autologging.
    """
    safe_patch(
        FLAVOR_NAME,
        ForecastingModel,
        "fit",
        _patched_fit,
        manage_run=True,
    )

    try:
        from darts.models.forecasting.torch_forecasting_model import (
            TorchForecastingModel,
        )

        safe_patch(
            FLAVOR_NAME,
            TorchForecastingModel,
            "fit",
            _patched_fit,
            manage_run=True,
        )
    except ImportError:
        pass


def get_default_pip_requirements(is_torch: bool = False) -> list[str]:
    """Return the default pip requirements for logging a darts model.

    Parameters
    ----------
    is_torch
        Whether the model is a PyTorch-based model. If ``True``, adds
        ``torch`` and ``pytorch-lightning`` to the requirements.

    Returns
    -------
    list[str]
        A list of pip requirement strings.
    """
    reqs = [_get_pinned_requirement("darts")]
    if is_torch:
        reqs.extend([
            _get_pinned_requirement("torch"),
            _get_pinned_requirement("pytorch-lightning"),
        ])
    return reqs


def get_default_conda_env(is_torch: bool = False) -> dict:
    """Return a default conda environment dict for a darts model.

    Parameters
    ----------
    is_torch
        Whether the model is a PyTorch-based model.

    Returns
    -------
    dict
        A conda environment specification dictionary.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(is_torch),
        additional_conda_channels=["conda-forge"],
    )


def _is_torch_model(model) -> bool:
    """Check if a model is a TorchForecastingModel.

    Parameters
    ----------
    model
        A Darts forecasting model instance.

    Returns
    -------
    bool
        True if the model is a TorchForecastingModel, False otherwise.
    """
    try:
        from darts.models.forecasting.torch_forecasting_model import (
            TorchForecastingModel,
        )

        return isinstance(model, TorchForecastingModel)
    except ImportError:
        return False


def _get_model_class_path(model) -> tuple[str, str]:
    """Extract the module path and class name from a model instance.

    Parameters
    ----------
    model
        A Darts forecasting model instance.

    Returns
    -------
    tuple[str, str]
        A tuple containing (module_path, class_name).
    """
    cls = type(model)
    return cls.__module__, cls.__name__


def _import_model_class(module_path: str, class_name: str):
    """Dynamically import and return a model class.

    Parameters
    ----------
    module_path : str
        The fully qualified module path (e.g., "darts.models.exponential_smoothing").
    class_name : str
        The name of the class to import (e.g., "ExponentialSmoothing").

    Returns
    -------
    type
        The imported model class.
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _log_model_params(model) -> None:
    """Log model creation parameters to MLflow.

    Extracts model parameters from ``model.model_params`` and logs them to the active
    MLflow run. Logs non-serializable values as "<non-serialisable>".

    Parameters
    ----------
    model
        A Darts forecasting model instance with a ``model_params`` attribute.
    """
    try:
        params = model.model_params
    except AttributeError:
        logger.debug("Model has no model_params attribute; skipping parameter logging.")
        return

    safe_params = {}
    for key, value in params.items():
        try:
            safe_params[key] = str(value)
        except Exception:
            safe_params[key] = "<non-serialisable>"

    # mlflow validates and truncates the param values internally
    if safe_params:
        mlflow.log_params(safe_params)


def _log_covariate_info(model) -> None:
    """Log covariate usage information to MLflow.

    Extracts information about past, future, and static covariates used during
    training and logs them as tags, parameters, and a JSON artifact for easy
    filtering, comparison, and documentation.

    Logs three types of information:
    - Tags: Boolean flags for filtering (e.g., "uses_past_covariates")
    - Parameters: Feature counts and names (truncated to MAX_PARAM_VAL_LENGTH chars)
    - Artifact: Complete covariate metadata as JSON file

    Parameters
    ----------
    model
        A fitted Darts forecasting model instance.
    """
    covariate_types = [
        (
            "past_covariates",
            "_uses_past_covariates",
            "past_covariate_series",
            "components",
        ),
        (
            "future_covariates",
            "_uses_future_covariates",
            "future_covariate_series",
            "components",
        ),
        (
            "static_covariates",
            "_uses_static_covariates",
            "static_covariates",
            "columns",
        ),
    ]

    covariate_info = {}

    for cov_key, uses_attr, series_attr, names_attr in covariate_types:
        info = {"used": False, "count": 0, "names": []}

        if getattr(model, uses_attr, False):
            info["used"] = True
            series = getattr(model, series_attr, None)
            if series is not None:
                names = getattr(series, names_attr).tolist()
                info["names"] = names
                info["count"] = len(names)

        covariate_info[cov_key] = info

        mlflow.set_tag(f"uses_{cov_key}", str(info["used"]).lower())
        mlflow.log_param(f"n_{cov_key}", info["count"])

        if info["names"]:
            names_str = ",".join(info["names"])
            mlflow.log_param(f"{cov_key.split('_')[0]}_cov_names", names_str)

    # log complete information as JSON artifact
    with TempDir() as tmp:
        covariates_path = tmp.path("covariates.json")
        with open(covariates_path, "w") as f:
            json.dump(covariate_info, f, indent=2)
        mlflow.log_artifact(covariates_path)


def _patched_fit(original, self, *args, **kwargs):
    """Patch function for ForecastingModel.fit() autologging.

    Handles both statistical and PyTorch-based models. For PyTorch models,
    automatically injects MLflow callback for per-epoch metrics logging.

    Parameters
    ----------
    original
        The original fit method being patched.
    self
        The model instance (ForecastingModel or TorchForecastingModel).
    args
        Positional arguments passed to fit.
    kwargs
        Keyword arguments passed to fit.

    Returns
    -------
        The result of calling the original fit method.
    """
    log_models = get_autologging_config(FLAVOR_NAME, "log_models", True)
    log_params = get_autologging_config(FLAVOR_NAME, "log_params", True)

    mlflow.set_tag("darts.model_class", type(self).__name__)

    if log_params:
        _log_model_params(self)

    # Inject callback for torch models (no-op for non-torch models)
    _inject_mlflow_callback(self)

    result = original(self, *args, **kwargs)

    if log_params:
        _log_covariate_info(self)

    if log_models:
        try:
            log_model(self, name="model", log_params=False)
        except Exception as e:
            logger.warning(f"Failed to autolog model artifact: {e}")

    return result


if PL_AVAILABLE:
    import pytorch_lightning as pl

    class _DartsMlflowCallback(pl.Callback):
        """PyTorch Lightning callback that logs epoch-level metrics to MLflow.

        This callback automatically logs training and validation metrics (such as
        loss values) to the active MLflow run at the end of each epoch.

        Notes
        -----
        The callback is automatically injected into TorchForecastingModel instances
        when ``autolog()`` is enabled with PyTorch-based models.
        """

        def on_train_epoch_end(self, trainer, pl_module):
            """Log training metrics at the end of each training epoch."""
            self._log_epoch_metrics(trainer)

        def on_validation_epoch_end(self, trainer, pl_module):
            """Log validation metrics at the end of each validation epoch."""
            self._log_epoch_metrics(trainer)

        def _log_epoch_metrics(self, trainer) -> None:
            """Extract and log metrics from the trainer to MLflow.

            Parameters
            ----------
            trainer
                PyTorch Lightning Trainer instance containing metrics.
            """
            if mlflow.active_run() is None:
                return

            epoch = trainer.current_epoch
            metrics: dict[str, float] = {}

            for source in (trainer.callback_metrics, trainer.logged_metrics):
                for key, value in source.items():
                    try:
                        metrics[key] = float(value)
                    except (TypeError, ValueError):
                        pass

            if metrics:
                mlflow.log_metrics(metrics, step=epoch)

else:
    _DartsMlflowCallback = None


def _get_mlflow_callback():
    """Create and return a ``_DartsMlflowCallback`` instance.

    Returns
    -------
    _DartsMlflowCallback or None
        A callback instance if PyTorch Lightning is available, None otherwise.
    """
    if not PL_AVAILABLE:
        return None

    return _DartsMlflowCallback()


def _inject_mlflow_callback(model) -> None:
    """Inject the MLflow callback into a ``TorchForecastingModel``'s trainer params.

    Adds a ``_DartsMlflowCallback`` to the model's PyTorch Lightning trainer callbacks
    if not already present. This enables automatic logging of training metrics to MLflow.

    Parameters
    ----------
    model
        A TorchForecastingModel instance with a ``trainer_params`` attribute.

    Notes
    -----
    This is a no-op if the callback is already present or PyTorch Lightning is unavailable.
    """
    callback = _get_mlflow_callback()
    if callback is None:
        return

    if not hasattr(model, "trainer_params"):
        return

    existing_callbacks = model.trainer_params.get("callbacks", [])

    if any(isinstance(cb, _DartsMlflowCallback) for cb in existing_callbacks):
        logger.debug("MLflow callback already present, skipping injection")
        return

    if not isinstance(existing_callbacks, list):
        existing_callbacks = list(existing_callbacks)

    existing_callbacks.append(callback)
    model.trainer_params["callbacks"] = existing_callbacks
    logger.debug(f"Injected MLflow callback into {type(model).__name__}")
