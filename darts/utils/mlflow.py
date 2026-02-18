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
from typing import Callable, Optional, Union

import mlflow
import numpy as np
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
from darts.logging import get_logger, raise_if, raise_if_not
from darts.metrics import mae, mape, mse, rmse
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.utils import PL_AVAILABLE

logger = get_logger(__name__)

FLAVOR_NAME = "darts"
_MODEL_DATA_SUBFOLDER = "data"


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"


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
    raise_if_not(
        isinstance(model, ForecastingModel),
        "model must be an instance of darts.models.forecasting.ForecastingModel",
        logger,
    )
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    data_dir = os.path.join(path, _MODEL_DATA_SUBFOLDER)
    is_torch = _is_torch_model(model)

    os.makedirs(data_dir, exist_ok=True)

    # pass in clean=True to not include any timeseries or callbacks within the model file
    if is_torch:
        model_file = _MODEL_FILE_TORCH
        model.save(os.path.join(data_dir, model_file), clean=True)
    else:
        model_file = _MODEL_FILE_STAT
        model.save(os.path.join(data_dir, model_file), clean=True)

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


_DEFAULT_METRICS = [mae, mse, rmse, mape]


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_training_metrics: bool = True,
    log_validation_metrics: bool = True,
    inject_per_epoch_callbacks: bool = True,
    extra_metrics: Optional[list[Callable]] = None,
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
    4. For PyTorch-based models: inject a callback that logs per-epoch metrics.
    5. Log the trained model artifact at the end of training.
    6. Optionally compute and log forecasting metrics on training and/or
       validation data.

    Parameters
    ----------
    log_models
        If ``True`` (default), log the trained model artifact after ``fit()``.
    log_params
        If ``True`` (default), log model creation parameters.
    log_training_metrics
        If ``True``, compute in-sample forecasting metrics on the training data
        after ``fit()`` completes.
        Default ``True``.
    log_validation_metrics
        If ``True``, compute forecasting metrics on the validation series
        (``val_series``) passed to ``fit()``.  Only effective for models whose
        ``fit()`` accepts a ``val_series`` argument (e.g. PyTorch-based models).
        Default ``True``.
    inject_per_epoch_callbacks
        If ``True`` (default), inject a PyTorch Lightning callback to log training and validation
        metrics at the end of each epoch. Only effective for PyTorch-based models. To provide
        additional callbacks use ``torch_metrics`` parameter while initializing the model.
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
    raise_if(
        cls is None,
        f"Class '{class_name}' not found in module '{module_path}'",
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

    Handles both statistical and PyTorch-based models. For PyTorch models,
    automatically injects MLflow callback for per-epoch metrics logging.

    Logs the trained model artifact if configured.

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
    log_training_metrics = get_autologging_config(
        FLAVOR_NAME, "log_training_metrics", False
    )
    log_validation_metrics = get_autologging_config(
        FLAVOR_NAME, "log_validation_metrics", False
    )
    inject_per_epoch_callbacks = get_autologging_config(
        FLAVOR_NAME, "inject_per_epoch_callbacks", True
    )
    extra_metrics = get_autologging_config(FLAVOR_NAME, "extra_metrics", None)

    mlflow.set_tag("darts.model_class", type(self).__name__)

    if log_params:
        _log_model_params(self)

    # Inject per-epoch callbacks for torch models (no-op for non-torch models)
    if inject_per_epoch_callbacks:
        _inject_mlflow_callback(self)

    result = original(self, *args, **kwargs)

    if log_params:
        _log_covariate_info(self)

    if log_training_metrics or log_validation_metrics:
        _log_forecasting_metrics(
            model=self,
            fit_args=args,
            fit_kwargs=kwargs,
            log_training=log_training_metrics,
            log_validation=log_validation_metrics,
            extra_metrics=extra_metrics,
        )

    if log_models:
        try:
            log_model(self, name="model", log_params=False)
        except Exception as e:
            logger.warning(
                f"Failed to autolog model artifact for {type(self).__name__}: {e}"
            )

    return result


def _log_forecasting_metrics(
    model,
    fit_args: tuple,
    fit_kwargs: dict,
    log_training: bool,
    log_validation: bool,
    extra_metrics: Optional[list[Callable]] = None,
) -> None:
    """Compute and log training and/or validation forecasting metrics to MLflow.

    After a model has been fitted this function optionally:

    * Runs ``model.backtest()`` on the training series (``retrain=False``)
      and logs metrics with a ``train_`` prefix.
    * Runs ``model.backtest()`` on the validation series and logs metrics
      with a ``val_`` prefix.

    For multiple series the metrics are averaged to produce a single value per metric.

    Parameters
    ----------
    model
        A fitted Darts forecasting model.
    fit_args
        Positional arguments originally passed to ``fit()``.
    fit_kwargs
        Keyword arguments originally passed to ``fit()``.
    log_training
        Whether to compute and log in-sample training metrics.
    log_validation
        Whether to compute and log validation metrics.
    extra_metrics
        Optional extra metric functions in addition to the defaults.
    """
    metrics_list = _get_metrics_list(extra_metrics)

    # determine forecast horizon from model attributes
    forecast_horizon = getattr(model, "output_chunk_length", None) or 1

    # extract series and covariates from fit() call
    train_series = fit_args[0] if fit_args else fit_kwargs.get("series")
    past_covariates = fit_kwargs.get("past_covariates")
    future_covariates = fit_kwargs.get("future_covariates")
    val_series = fit_kwargs.get("val_series")
    val_past_covariates = fit_kwargs.get("val_past_covariates")
    val_future_covariates = fit_kwargs.get("val_future_covariates")

    if log_training and train_series is not None:
        try:
            _backtest_and_log(
                model,
                series=train_series,
                metrics_list=metrics_list,
                prefix="train",
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                forecast_horizon=forecast_horizon,
            )
        except Exception:
            logger.info(
                "Could not compute training forecasting metrics for "
                f"{type(model).__name__}.",
                exc_info=True,
            )

    if log_validation and val_series is not None:
        try:
            _backtest_and_log(
                model,
                series=val_series,
                metrics_list=metrics_list,
                prefix="val",
                past_covariates=val_past_covariates or past_covariates,
                future_covariates=val_future_covariates or future_covariates,
                forecast_horizon=forecast_horizon,
            )
        except Exception:
            logger.info(
                "Could not compute validation forecasting metrics for "
                f"{type(model).__name__}.",
                exc_info=True,
            )


def _backtest_and_log(
    model,
    series: Union[TimeSeries, list[TimeSeries]],
    metrics_list: list[Callable],
    prefix: str,
    past_covariates=None,
    future_covariates=None,
    forecast_horizon: int = 1,
) -> None:
    """Run ``model.backtest()`` and log the resulting scores to MLflow.

    Parameters
    ----------
    model
        A fitted Darts forecasting model.
    series
        One or more target series to evaluate on.
    metrics_list
        List of Darts metric functions to evaluate.
    prefix
        Prefix for the logged metric names (e.g. ``"train"`` or ``"val"``).
    past_covariates
        Optional past covariates matching ``series``.
    future_covariates
        Optional future covariates matching ``series``.
    forecast_horizon
        Number of steps to forecast at each backtest step.
    """
    backtest_kwargs = dict(
        series=series,
        forecast_horizon=forecast_horizon,
        retrain=False,
        overlap_end=False,
        last_points_only=True,
        reduction=None,
        verbose=False,
        show_warnings=False,
    )
    if past_covariates is not None:
        backtest_kwargs["past_covariates"] = past_covariates
    if future_covariates is not None:
        backtest_kwargs["future_covariates"] = future_covariates

    logged = {}
    for metric_fn in metrics_list:
        try:
            score = model.backtest(**backtest_kwargs, metric=metric_fn)
            # backtest returns a float, np.ndarray, or list depending on the
            # input.  We want a single scalar per metric so we take the mean.

            logged[f"{prefix}_{metric_fn.__name__}"] = float(np.nanmean(score))
        except Exception:
            logger.debug(
                f"Backtest metric {metric_fn.__name__} failed for "
                f"{type(model).__name__}, skipping.",
                exc_info=True,
            )

    if logged:
        mlflow.log_metrics(logged)


def _get_metrics_list(
    extra_metrics: Optional[list[Callable]] = None,
) -> list[Callable]:
    """Return the combined list of default and extra metric functions.

    Parameters
    ----------
    extra_metrics
        Optional additional metric functions to append to the defaults.

    Returns
    -------
    list[Callable]
        A list of metric functions (``mae``, ``mse``, ``rmse``, ``mape``, plus extras).
    """
    metrics = list(_DEFAULT_METRICS)
    if extra_metrics:
        seen_names = {m.__name__ for m in metrics}
        for m in extra_metrics:
            if m.__name__ not in seen_names:
                metrics.append(m)
                seen_names.add(m.__name__)
    return metrics


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
        logger.info("MLflow callback already present, skipping injection")
        return

    if not isinstance(existing_callbacks, list):
        existing_callbacks = list(existing_callbacks)

    existing_callbacks.append(callback)
    model.trainer_params["callbacks"] = existing_callbacks
    logger.info(f"Injected MLflow callback into {type(model).__name__}")
