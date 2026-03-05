"""
MLflow Integration for Darts
-----------------------------

Custom MLflow model flavor for darts forecasting models. Supports saving, loading,
logging and autolog for any darts ``ForecastingModel`` (statistical, ML-based, and PyTorch-based)
to MLflow.

This module is partly adapted from and inspired by the open-source
implementation of SKtime's MLflow integration, with modifications to support
autologging and handle Darts-specific model and covariate metadata.

References:
https://github.com/sktime/sktime/blob/main/sktime/utils/mlflow_sktime.py
"""

import importlib
import json
import os
import re
from collections.abc import Callable

import mlflow
import numpy as np
import yaml
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _inspect_original_var_name
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.safety import safe_patch
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
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import ForecastingModel

logger = get_logger(__name__)

FLAVOR_NAME = "darts"
_MODEL_DATA_SUBFOLDER = "data"


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"


def save_model(
    model,
    path: str,
    conda_env: dict | str | None = None,
    code_paths: list[str] | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    signature=None,
    input_example=None,
    metadata: dict | None = None,
    mlflow_model: Model | None = None,
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
        *Unsupported, see notes.* An ``mlflow.models.ModelSignature`` instance describing model input/output.
        Use ``mlflow.models.infer_signature()`` to automatically generate from example inputs.
    input_example
        *Unsupported, see notes.* An example input for the model (used by MLflow UI).
    metadata
        Optional dictionary of custom metadata to store in the ``MLmodel`` file.
    mlflow_model
        Optional MLflow Model object to use for saving. When provided (typically by
        ``Model.log()``), this model instance is used instead of creating a new one.

    Notes
    -----
    Signature and input_example params are currently not supported, as they
    are used to support serving and input validation in the MLflow pyfunc flavor,
    which is not implemented for darts models. They are accepted as params for
    simplifying potential future extensibility, and to keep in line with MLflow API
    conventions.
    """
    if not isinstance(model, ForecastingModel):
        raise_log(
            ValueError(
                "Model must be an instance of darts.models.forecasting.ForecastingModel."
            ),
            logger,
        )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    is_torch = _is_torch_model(model)

    # clean=True excludes any timeseries or callbacks from the model file
    model_file = _MODEL_FILE_TORCH if is_torch else _MODEL_FILE_STAT
    model.save(os.path.join(path, model_file), clean=True)

    module_path, class_name = _get_model_class_path(model)

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

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        darts_version=darts.__version__,
        data=model_file,
        model_class=f"{module_path}.{class_name}",
        code=code_dir_subpath,
        is_torch_model=is_torch,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(
    model_uri: str,
    dst_path: str | None = None,
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

    model_path = os.path.join(local_path, flavor_conf["data"])

    if flavor_conf.get("is_torch_model", False):
        return model_cls.load(model_path, **kwargs)
    else:
        return model_cls.load(model_path)


def log_model(
    model,
    artifact_path: str | None = None,
    name: str | None = None,
    registered_model_name: str | None = None,
    conda_env: dict | str | None = None,
    code_paths: list[str] | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    signature=None,
    input_example=None,
    metadata: dict | None = None,
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
       *Unsupported, see notes.* An ``mlflow.models.ModelSignature``. Use ``mlflow.models.infer_signature()``
        to automatically generate from example inputs.
    input_example
        *Unsupported, see notes.* An example model input.
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

    Notes
    -----
    Signature and input_example params are currently not supported, as they
    are used to support serving and input validation in the MLflow pyfunc flavor,
    which is not implemented for darts models. They are accepted as params for
    simplifying potential future extensibility, and to keep in line with MLflow API
    conventions.
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


def autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_metrics: bool = True,
    inject_per_epoch_callbacks: bool = True,
    disable: bool = False,
    silent: bool = False,
    manage_run: bool = True,
) -> None:
    """Enable (or disable) automatic MLflow logging for darts models.

    When enabled, every call to ``model.fit()`` on any darts forecasting model
    will automatically:

    1. Start an MLflow run (or reuse the currently active one).
    2. Log model creation parameters (``model.model_params``).
    3. Log covariate usage information (past, future, and static covariates).
    4. Patch all darts metric functions so that any call made inside an active
       MLflow run automatically logs the result.  Repeated calls overwrite
       the previous value.
    5. For PyTorch-based models: leverage ``mlflow.pytorch.autolog()`` to
       automatically log per-epoch training and validation metrics.
    6. Log the trained model artifact at the end of training.

    .. important::

        ``autolog()`` must be called **before** importing metric functions from
        ``darts.metrics``.  Metric functions imported before ``autolog()`` is
        enabled will **not** log to MLflow.

    .. note::

        Logged metric keys depend on the result shape:

        * **Scalar** → ``{metric_name}``
        * **Per-component** (1-D, single series) →
          ``{metric_name}_{component_name}``
        * **Per-series** (1-D, list of series) →
          ``{metric_name}_{series_idx}``
        * **Per-series × per-component** (2-D) →
          ``{metric_name}_{component_name}_{series_idx}``

        When a dataset variable name can be captured via frame inspection,
        it is inserted after the metric name (e.g.
        ``{metric_name}_{dataset_name}_{component_name}``).

    Parameters
    ----------
    log_models
        If ``True`` (default), log the trained model artifact after ``fit()``.
    log_params
        If ``True`` (default), log model creation parameters.
    log_metrics
        If ``True`` (default), patch all darts metric functions so that any
        call made inside an active MLflow run is automatically logged.
    inject_per_epoch_callbacks
        If ``True`` (default), enable ``mlflow.pytorch.autolog(log_models=False)``
        around PyTorch-based model training to automatically log per-epoch
        training and validation metrics. Only effective for PyTorch-based models.
    extra_metrics
        An optional list of additional Darts metric functions to log on top of
        the defaults (``mae``, ``mse``, ``rmse``, ``mape``).  Each function
        must follow the standard Darts metric signature
        ``metric(actual_series, pred_series)``.
    disable
        If ``True``, restore the original ``fit()`` methods and stop
        autologging.
    silent
        If ``True`` (default ``False``), suppress all event logging and warnings from
        MLflow during autologging.
    manage_run
        If `True`, applies the `with_managed_run` wrapper to the specified
        `patch_function`, which automatically creates & terminates an MLflow
        active run during patch code execution if necessary. If `False`,
        does not apply the `with_managed_run` wrapper to the specified
        `patch_function`.
    """
    # Enable/disable mlflow.pytorch.autolog for per-epoch metrics on torch models.
    # This must happen outside the @autologging_integration-decorated _autolog()
    # because the decorator short-circuits on disable=True before the function
    # body executes, and MLflow's session manager suppresses nested autolog
    # patches if called from within a safe_patch context.
    if inject_per_epoch_callbacks and not disable:
        try:
            import mlflow.pytorch

            mlflow.pytorch.autolog(log_models=False, log_datasets=False, silent=silent)
        except ImportError:
            logger.info(
                "mlflow.pytorch not available; skipping per-epoch metrics logging."
            )
    elif disable:
        try:
            import mlflow.pytorch

            mlflow.pytorch.autolog(disable=True)
        except (ImportError, Exception):
            logger.info(
                "mlflow.pytorch not available; skipping per-epoch metrics logging."
            )

    _autolog(
        log_models=log_models,
        log_params=log_params,
        log_metrics=log_metrics,
        inject_per_epoch_callbacks=inject_per_epoch_callbacks,
        disable=disable,
        silent=silent,
        manage_run=manage_run,
    )


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_metrics: bool = True,
    inject_per_epoch_callbacks: bool = True,
    disable: bool = False,
    silent: bool = False,
    manage_run: bool = True,
) -> None:
    """Internal autolog implementation decorated with ``@autologging_integration``.

    Handles patching of darts ``ForecastingModel.fit()`` and metric functions.
    The ``mlflow.pytorch.autolog`` coordination is handled by the public
    ``autolog()`` wrapper because the decorator short-circuits on
    ``disable=True``.
    """

    # recursively get all subclasses of ForecastingModel that override fit()
    def get_all_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses

    classes_to_patch = [ForecastingModel]

    for subclass in get_all_subclasses(ForecastingModel):
        if "fit" in subclass.__dict__:
            classes_to_patch.append(subclass)

    for cls in classes_to_patch:
        try:
            safe_patch(
                FLAVOR_NAME,
                cls,
                "fit",
                _patched_fit,
                manage_run=manage_run,
            )
        except Exception as e:
            logger.info(f"Failed to patch {cls.__name__}.fit() for autologging: {e}")

    if log_metrics:
        import darts.metrics as _darts_metrics

        for metric_name in _darts_metrics.__all__:
            try:
                # metrics should not create their own runs;
                # they log into the run started by fit(), so manage_run=False here
                safe_patch(
                    FLAVOR_NAME,
                    _darts_metrics,
                    metric_name,
                    _make_metric_patch(metric_name),
                    manage_run=False,
                )
            except Exception as e:
                logger.info(
                    f"Failed to patch metric '{metric_name}' on darts.metrics: {e}"
                )


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


def _log_model_params(model) -> None:
    """Log model creation parameters to MLflow.

    Extracts model parameters from ``model.model_params`` and logs them to the active
    MLflow run.

    Parameters
    ----------
    model
        A Darts forecasting model instance with a ``model_params`` attribute.
    """
    try:
        params = model.model_params
    except AttributeError:
        logger.info("Model has no model_params attribute; skipping parameter logging.")
        return

    if params:
        mlflow.log_params(params)


def _log_covariate_info(model) -> None:
    """Log covariate usage information to MLflow.

    Extracts information about past, future, and static covariates used during
    training and logs them as tags, parameters, and a JSON artifact for easy
    filtering, comparison, and documentation.

    Logs three types of information:
    - Tags: Boolean flags for filtering (e.g., "uses_past_covariates")
    - Parameters: Feature counts and names (truncated by MLflow)
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

    covariate_info = {
        cov_key: _extract_covariate_metadata(model, uses_attr, series_attr, names_attr)
        for cov_key, uses_attr, series_attr, names_attr in covariate_types
    }

    for cov_key, info in covariate_info.items():
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
        logger.info("TorchForecastingModel not available; treating model as non-torch")
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
    cls = getattr(module, class_name, None)
    if cls is None:
        raise_log(
            ImportError(f"Class `{class_name}` not found in module `{module_path}`"),
            logger,
        )
    return cls


def _extract_covariate_metadata(
    model, uses_attr: str, series_attr: str, names_attr: str
) -> dict:
    """Extract metadata for a single covariate type.

    Parameters
    ----------
    model
        A Darts forecasting model instance.
    uses_attr : str
        Model attribute name indicating covariate usage.
    series_attr : str
        Model attribute name for the covariate series.
    names_attr : str
        Series attribute name for feature names ("components" or "columns").

    Returns
    -------
    dict
        Dictionary with keys: "used" (bool), "count" (int), "names" (list).
    """
    info = {"used": False, "count": 0, "names": []}

    if getattr(model, uses_attr, False):
        info["used"] = True
        series = getattr(model, series_attr, None)
        if series is not None:
            names = getattr(series, names_attr).tolist()
            info["names"] = names
            info["count"] = len(names)

    return info


def _patched_fit(original, self, *args, **kwargs):
    """Patch function for ForecastingModel.fit() autologging.

    Logs model parameters, class, covariates and the model itself.
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

    result = original(self, *args, **kwargs)

    if log_params:
        _log_covariate_info(self)

    if log_models:
        try:
            log_model(self, name="model", log_params=False)
        except Exception:
            logger.info(
                f"Failed to autolog model artifact for {type(self).__name__}.",
                exc_info=True,
            )

    return result


def _sanitize_mlflow_key(name: str) -> str:
    """Sanitize a string for use as an MLflow metric key.

    Replaces any character that is not alphanumeric, a hyphen, or an
    underscore with an underscore, so component names become valid
    MLflow keys.

    Parameters
    ----------
    name
        The raw name to sanitize.

    Returns
    -------
    str
        A string safe for use as an MLflow metric key.
    """
    return re.sub(r"[^\w-]", "_", name)


def _log_metric_result(
    metric_name: str,
    result,
    dataset_name: str | None = None,
    component_names: list[str] | None = None,
    input_is_list: bool = False,
) -> None:
    """Log a metric result to the active MLflow run.

    Handles Python scalars, numpy scalars, 1-D arrays, and 2-D arrays.

    The logged MLflow key follows the pattern::

        {metric_name}_{dataset_name}_{component}_{series_index}

    Specifically:

    * **Scalar** (0-d) → ``{metric}_{dataset}``
    * **1-D, single TimeSeries input** (per-component) →
      ``{metric}_{dataset}_{component_name_or_idx}``
    * **1-D, list input** (per-series, component already reduced) →
      ``{metric}_{dataset}_{series_idx}``
    * **2-D, list input** (per-series × per-component) →
      ``{metric}_{dataset}_{component_name_or_idx}_{series_idx}``

    All optional parts are omitted when not available (e.g. no dataset name
    when the variable name could not be inspected).

    Parameters
    ----------
    metric_name
        Base metric name used as the MLflow key.
    result
        The metric result to log.
    dataset_name
        Sanitized variable name of ``actual_series`` in the caller's frame.
        Omitted from key when ``None``.
    component_names
        Component name strings to use as the component part of the key.
        For single-series input these come from ``series.components``; for
        list input they come from the first series in the list.
        Falls back to integer indices when ``None`` or length mismatches.
    input_is_list
        ``True`` when ``actual_series`` was a ``Sequence[TimeSeries]``.  Drives
        whether the first result axis is treated as *series* or *components*.
    """
    result_arr = np.asarray(result)

    if mlflow.active_run() is None:
        return

    base_key = f"{metric_name}_{dataset_name}" if dataset_name else metric_name

    def _comp_suffix(idx: int) -> str:
        if component_names is not None and idx < len(component_names):
            return _sanitize_mlflow_key(component_names[idx])
        return str(idx)

    if result_arr.ndim == 0:
        # scalar result
        mlflow.log_metric(base_key, float(result_arr))

    elif result_arr.ndim == 1:
        if not input_is_list:
            # single series: log per-component
            for c_i, val in enumerate(result_arr):
                mlflow.log_metric(f"{base_key}_{_comp_suffix(c_i)}", float(val))
        else:
            # list input, components already reduced: log per-series
            for s_i, val in enumerate(result_arr):
                mlflow.log_metric(f"{base_key}_{s_i}", float(val))

    elif result_arr.ndim == 2:
        # list input: log per-series and per-component
        n_series, n_components = result_arr.shape
        for s_i in range(n_series):
            for c_i in range(n_components):
                mlflow.log_metric(
                    f"{base_key}_{_comp_suffix(c_i)}_{s_i}",
                    float(result_arr[s_i, c_i]),
                )

    else:
        # unexpected shape — flatten with integer indices
        for i, val in enumerate(result_arr.flatten()):
            mlflow.log_metric(f"{base_key}_{i}", float(val))


def _make_metric_patch(metric_name: str) -> Callable:
    """Create a ``safe_patch``-compatible patch function for a darts metric.

    The returned patch calls the original metric and, when an active MLflow
    run exists, logs the result under a key built as::

        {metric_name}_{dataset_name}_{component}_{series_index}

    where:

    * ``dataset_name`` – Python variable name of the first argument in the
      caller's frame (captured via frame inspection, omitted if not found).
    * ``component`` – component label from the ``TimeSeries`` if the result is
      per-component, otherwise an integer index.
    * ``series_index`` – integer index appended when the input is a
      ``Sequence[TimeSeries]`` and the result has a series axis.

    The original return value is always forwarded unchanged.

    Parameters
    ----------
    metric_name
        The darts metric function name used as the MLflow metric key.
    """

    def _patched_metric(original, *args, **kwargs):
        result = original(*args, **kwargs)

        if mlflow.active_run() is None:
            return result

        series = args[0]

        # capture the variable name of actual_series for metric key
        raw = _inspect_original_var_name(series, fallback_name=None)
        dataset_name = _sanitize_mlflow_key(raw) if raw else None

        # handling multi_series input
        input_is_list = not hasattr(series, "components")

        # extract component names from the series (or first element if list)
        component_names = None
        try:
            if input_is_list:
                # we assume that subsequent series have same component order
                component_names = series[0].components.tolist()
            else:
                component_names = series.components.tolist()
        except Exception:
            logger.info("Could not extract component names from series.")

        try:
            _log_metric_result(
                metric_name,
                result,
                dataset_name=dataset_name,
                component_names=component_names,
                input_is_list=input_is_list,
            )
        except Exception:
            logger.info(
                f"Failed to log metric '{metric_name}' to MLflow.", exc_info=True
            )

        return result

    return _patched_metric
