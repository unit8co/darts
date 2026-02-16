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
import tempfile
from functools import wraps
from typing import Any, Optional, Union

import mlflow
import yaml
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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


class _ModelLogInfo:
    """Lightweight container returned by :func:`log_model` with essential metadata.

    Attributes
    ----------
    model_uri : str
        The full MLflow model URI (e.g., "runs:/run_id/model").
    run_id : str
        The MLflow run ID that logged the model.
    artifact_path : str
        The artifact path where the model was logged within the run.
    """

    def __init__(self, model_uri: str, run_id: str, artifact_path: str):
        self.model_uri = model_uri
        self.run_id = run_id
        self.artifact_path = artifact_path


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"
_MODEL_FILE_TORCH_CKPT = "model.pt.ckpt"


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


def _create_mlmodel_file(
    path: str,
    darts_flavor_conf: dict,
    signature,
    input_example,
    metadata: Optional[dict],
) -> None:
    """Create and save MLmodel file with metadata.

    Creates the following files in the model directory:
    * ``MLmodel`` - MLmodel file with flavor metadata.

    Parameters
    ----------
    path: str
        Root directory of the MLflow model where the MLmodel file will be saved.
    darts_flavor_conf: dict
        Dictionary containing the flavor configuration for the darts model.
    signature: mlflow.models.ModelSignature
        An ``mlflow.models.ModelSignature`` instance describing model input/output.
        Use :func:`infer_signature` to automatically generate from example inputs.
    input_example: DataFrame
        An example input for the model (used by MLflow UI). Should be a DataFrame
        created with :func:`prepare_pyfunc_input`.
    metadata: Optional[dict]
        Optional dictionary of custom metadata to store in the ``MLmodel`` file.
    """
    mlmodel = Model()

    if signature is not None:
        mlmodel.signature = signature

    if input_example is not None:
        _save_example(mlmodel, input_example, path)

    if metadata is not None:
        mlmodel.metadata = metadata

    mlmodel.add_flavor(FLAVOR_NAME, **darts_flavor_conf)
    mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))


def _write_environment_files(
    path: str,
    conda_env: dict,
    pip_requirements: list[str],
    pip_constraints: Optional[list[str]],
) -> None:
    """Write Python environment specification files for model reproducibility.

    Creates the following files in the model directory:

    * ``conda.yaml`` - Conda environment specification including Python version and pip dependencies
    * ``requirements.txt`` - Pip requirements list for non-conda environments
    * ``python_env.yaml`` - MLflow Python environment specification
    * ``constraints.txt`` (optional) - Pip version constraints if specified

    Parameters
    ----------
    path: str
        Root directory of the MLflow model where environment files will be written.
    conda_env: dict
        Processed conda environment dictionary containing 'name', 'channels',
        'dependencies' keys.
    pip_requirements: list[str]
        List of pip requirement strings (e.g., ['numpy>=1.20.0', 'pandas']).
    pip_constraints: Optional[list[str]]
        Optional list of pip constraint strings for pinning transitive dependencies.
        Only written if provided.
    """
    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


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

    _write_environment_files(path, conda_env, pip_requirements, pip_constraints)
    _create_mlmodel_file(path, darts_flavor_conf, signature, input_example, metadata)


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
    _ModelLogInfo
        A lightweight object with ``model_uri``, ``run_id``, and
        ``artifact_path`` attributes.
    """
    artifact_name = name or artifact_path or "model"

    if log_params:
        _log_model_params(model)
        _log_covariate_info(model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = os.path.join(tmp_dir, artifact_name)
        save_model(
            model=model,
            path=model_dir,
            conda_env=conda_env,
            code_paths=code_paths,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            signature=signature,
            input_example=input_example,
            metadata=metadata,
        )
        mlflow.log_artifacts(model_dir, artifact_path=artifact_name)

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_name}"

    if registered_model_name is not None:
        mlflow.register_model(model_uri, registered_model_name)

    return _ModelLogInfo(
        model_uri=model_uri,
        run_id=run_id,
        artifact_path=artifact_name,
    )


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


# stores original (unpatched) `fit` methods so they can be restored.
_ORIGINAL_FIT_METHODS: dict[type, Any] = {}

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

    cb_type = type(callback)
    for existing in existing_callbacks:
        if type(existing).__name__ == cb_type.__name__:
            return

    existing_callbacks.append(callback)
    model.trainer_params["callbacks"] = existing_callbacks


def autolog(
    log_models: bool = True,
    log_params: bool = True,
    disable: bool = False,
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
    """
    if disable:
        _restore_original_fit_methods()
        return

    _patch_fit(ForecastingModel, log_models=log_models, log_params=log_params)

    try:
        from darts.models.forecasting.torch_forecasting_model import (
            TorchForecastingModel,
        )

        _patch_fit(
            TorchForecastingModel,
            log_models=log_models,
            log_params=log_params,
            inject_callback=True,
        )
    except ImportError:
        pass


def _patch_fit(
    cls,
    *,
    log_models: bool,
    log_params: bool,
    inject_callback: bool = False,
) -> None:
    """Replace a model class's ``fit`` method with an MLflow logging wrapper.

    Parameters
    ----------
    cls : type
        The model class to patch (e.g., ForecastingModel, TorchForecastingModel).
    log_models : bool
        Whether to log the trained model artifact after fitting.
    log_params : bool
        Whether to log model creation parameters.
    inject_callback : bool, optional
        Whether to inject the MLflow callback for PyTorch Lightning models.
        Default is False.
    """
    if cls in _ORIGINAL_FIT_METHODS:
        return

    original_fit = cls.fit
    _ORIGINAL_FIT_METHODS[cls] = original_fit

    @wraps(original_fit)
    def _patched_fit(self, *args, **kwargs):
        run_started_here = False
        if mlflow.active_run() is None:
            mlflow.start_run()
            run_started_here = True

        try:
            mlflow.set_tag("darts.model_class", type(self).__name__)

            if log_params:
                _log_model_params(self)

            if inject_callback:
                _inject_mlflow_callback(self)

            result = original_fit(self, *args, **kwargs)

            if log_params:
                _log_covariate_info(self)

            if log_models:
                try:
                    log_model(self, name="model", log_params=False)
                except Exception as e:
                    logger.warning(f"Failed to autolog model artifact: {e}")

            return result

        except Exception:
            raise
        finally:
            if run_started_here:
                mlflow.end_run()

    cls.fit = _patched_fit


def _restore_original_fit_methods() -> None:
    """Restore all patched ``fit()`` methods to their original implementations.

    This function is called when ``autolog(disable=True)`` is invoked to remove
    all MLflow logging functionality added by autologging.
    """
    for cls, original_fit in _ORIGINAL_FIT_METHODS.items():
        cls.fit = original_fit
    _ORIGINAL_FIT_METHODS.clear()
