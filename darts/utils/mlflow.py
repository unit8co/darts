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

import inspect
import json
import os
import re
import sys
import threading
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import mlflow
import numpy as np
import yaml
from mlflow.entities import LoggedModel
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils import _get_fully_qualified_class_name, _inspect_original_var_name
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient
from mlflow.utils.autologging_utils.safety import safe_patch
from mlflow.utils.class_utils import _get_class_from_string
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
from darts.metrics import CLASSIFICATION_METRICS
from darts.metrics.utils import _LabelReduction
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)

logger = get_logger(__name__)

FLAVOR_NAME = "darts"


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"

_covariate_types = [
    (
        "past_covariates",
        "uses_past_covariates",
        "past_covariate_series",
        "components",
    ),
    (
        "future_covariates",
        "uses_future_covariates",
        "future_covariate_series",
        "components",
    ),
    (
        "static_covariates",
        "uses_static_covariates",
        "static_covariates",
        "columns",
    ),
]

# Thread-local flag set during historical_forecasts to suppress per-iteration
# autologging in _patched_fit (which would otherwise spawn one run per iteration).
_autolog_state = threading.local()


def save_model(
    model,
    path: str,
    conda_env: dict | str | None = None,
    code_paths: list[str] | None = None,
    mlflow_model: Model | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
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
    mlflow_model
        Optional MLflow Model object to use for saving. When provided (typically by
        ``Model.log()``), this model instance is used instead of creating a new one.
    signature
        *Unsupported, see notes.* An ``mlflow.models.ModelSignature`` instance describing model input/output.
        Use ``mlflow.models.infer_signature()`` to automatically generate from example inputs.
    input_example
        *Unsupported, see notes.* An example input for the model (used by MLflow UI).
    pip_requirements
        A list of pip requirement strings. Overrides ``conda_env`` pip section
        when provided.
    extra_pip_requirements
        A list of additional pip requirement strings to add to the model's environment,
        in addition to the default requirements.
    metadata
        Optional dictionary of custom metadata to store in the ``MLmodel`` file.

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

    model_class = _get_fully_qualified_class_name(model)

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
        model_class=model_class,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if pip_requirements is None:
        default_reqs = get_default_pip_requirements()
        # TODO: `infer_pip_requirements` requires `pyfunc` flavor to be implemented.
        # inferred_reqs = infer_pip_requirements(path, FLAVOR_NAME, fallback=default_reqs)
        # default_reqs = sorted(set(inferred_reqs).union(default_reqs))
    else:
        default_reqs = None
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

    model_cls_str = flavor_conf.get("model_class", None)
    model_cls = _get_class_from_string(model_cls_str)

    if not issubclass(model_cls, ForecastingModel):
        raise_log(
            ValueError(
                f"Cannot load model: class `{model_cls_str}` is not a subclass of `ForecastingModel`."
            ),
            logger,
        )

    model_path = os.path.join(local_path, flavor_conf["data"])

    return model_cls.load(model_path, **kwargs)


def log_model(
    model,
    artifact_path: str | None = None,
    conda_env: dict | str | None = None,
    code_paths: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    await_registration_for: int = DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs,
):
    """Log a darts model to the current MLflow run.

    Parameters
    ----------
    model
        A fitted darts ``ForecastingModel`` instance.
    artifact_path
        The run-relative artifact path under which to log the model.
        Defaults to ``"model"``. Deprecated in favour of ``name``.
    conda_env
        Conda environment specification (dict or path).
    code_paths
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are prepended to the system path
        when the model is loaded.
    registered_model_name
        If provided, the model is registered in the MLflow Model Registry
        under this name.
    signature
       *Unsupported, see notes.* An ``mlflow.models.ModelSignature``. Use ``mlflow.models.infer_signature()``
        to automatically generate from example inputs.
    input_example
        *Unsupported, see notes.* An example model input.
    await_registration_for
        Number of seconds to wait for the model version to finish being created and is in ``READY`` status.
        By default, the function waits for five minutes. Specify 0 to skip waiting.
    pip_requirements
        Pip requirements list.
    extra_pip_requirements
        A list of additional pip requirement strings to add to the model's environment,
        in addition to the default requirements.
    metadata
        Optional dict of custom metadata.
    name
        The name for the model artifact. If provided, takes precedence over
        ``artifact_path``.
    params
        Optional dictionary of parameters to log alongside the model.
    tags
        Optional dictionary of tags to log alongside the model.
    model_type
        Optional string for the model type.
    step
        Optional step value to log with the model's metrics. Defaults to 0.
    model_id
        Optional string for the model ID.

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

    return Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        registered_model_name=registered_model_name,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        metadata=metadata,
        name=name,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_metrics: bool = True,
    log_torch_metrics: bool = True,
    disable: bool = False,
    silent: bool = False,
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

        Metric functions imported with ``from darts.metrics import <name>``
        before ``autolog()`` is enabled will **not** log to MLflow (function
        reference is copied out of the module before the patch is applied).

        To avoid this restriction, use ``import darts.metrics as dm`` and call
        ``dm.<metric>()`` instead (module attribute lookups are resolved at call
        time and will always see the patched version).

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
    log_torch_metrics
        If ``True`` (default), enable ``mlflow.pytorch.autolog(log_models=False)``
        around PyTorch-based model training to automatically log per-epoch
        training and validation metrics. Only effective for PyTorch-based models.
    disable
        If ``True``, restore the original ``fit()`` methods and stop
        autologging.
    silent
        If ``True`` (default ``False``), suppress all event logging and warnings from
        MLflow during autologging.
    """
    # Enable/disable mlflow.pytorch.autolog for per-epoch metrics on torch models.
    # This must happen outside the @autologging_integration-decorated _autolog()
    # because the decorator short-circuits on disable=True before the function
    # body executes, and MLflow's session manager suppresses nested autolog
    # patches if called from within a safe_patch context.
    if log_torch_metrics and not disable:
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
        disable=disable,
        silent=silent,
    )


def _get_forecasting_models():
    """
    Returns:
        A list of (name, class) tuples for all forecasting models in the Darts library.
    """
    import darts.models

    classes = inspect.getmembers(darts.models, inspect.isclass)

    classes = [
        (name, cls) for name, cls in classes if issubclass(cls, ForecastingModel)
    ]

    return sorted(set(classes), key=itemgetter(0))


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_metrics: bool = True,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Internal autolog implementation decorated with ``@autologging_integration``.

    Handles patching of darts ``ForecastingModel.fit()`` and metric functions.
    The ``mlflow.pytorch.autolog`` coordination is handled by the public
    ``autolog()`` wrapper because the decorator short-circuits on
    ``disable=True``.
    """

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
        # Create a training session to track the training process and log information
        autologging_client = MlflowAutologgingQueueingClient()

        if getattr(_autolog_state, "in_historical_forecasts", False):
            return original(self, *args, **kwargs)

        # Track which model is active so metric patches can prefix their keys
        _autolog_state.current_model_name = type(self).__name__

        result = original(self, *args, **kwargs)

        active_run = mlflow.active_run()
        if active_run is None:
            return result
        run_id = active_run.info.run_id

        # Set tags to identify the model class and relevant information
        autologging_client.set_tags(run_id=run_id, tags=_get_model_info_tags(self))

        if log_params:
            # Log the parameters for model creation
            autologging_client.log_params(run_id=run_id, params=self.model_params)
            _log_covariate_info(self)

        param_logging_ops = autologging_client.flush(synchronous=False)

        if log_models:
            model_name = type(self).__name__
            model: LoggedModel = _initialize_logged_model(
                name=model_name, flavor=FLAVOR_NAME
            )
            try:
                registered_model_name = get_autologging_config(
                    FLAVOR_NAME, "registered_model_name", None
                )
                log_model(
                    result,
                    name=model_name,
                    registered_model_name=registered_model_name,
                    model_id=model.model_id,
                )
            except Exception:
                logger.info(
                    f"Failed to autolog model artifact for {type(self).__name__}.",
                    exc_info=True,
                )

        param_logging_ops.await_completion()

        return result

    def _patched_historical_forecasts(original, self, *args, **kwargs):
        """Suppress per-iteration fit() autologging during historical_forecasts.

        Sets a thread-local flag so _patched_fit skips autologging for the
        internal fit() calls. The outer safe_patch() on
        historical_forecasts itself provides a single managed run.
        """
        _autolog_state.in_historical_forecasts = True
        try:
            return original(self, *args, **kwargs)
        finally:
            _autolog_state.in_historical_forecasts = False

    def _patched_backtest(original, self, *args, **kwargs):
        """Wrap ``backtest`` to log metric result(s) to the active MLflow run.

        Delegates to ``_log_backtest_metrics``, which infers result shape from
        the metric signature and logs every cell under a descriptive key.
        """
        _autolog_state.in_backtest = True
        try:
            result = original(self, *args, **kwargs)
        finally:
            _autolog_state.in_backtest = False

        active_run = mlflow.active_run()
        if not log_metrics or active_run is None:
            return result

        bound = inspect.signature(ForecastingModel.backtest).bind(self, *args, **kwargs)
        bound.apply_defaults()
        backtest_args = bound.arguments

        autologging_client = MlflowAutologgingQueueingClient()
        _log_backtest_metrics(
            autologging_client, active_run.info.run_id, result, backtest_args
        )
        autologging_client.flush(synchronous=False).await_completion()
        return result

    # patch `fit()` for all forecasting models
    for _, cls in _get_forecasting_models():
        safe_patch(
            FLAVOR_NAME,
            cls,
            "fit",
            _patched_fit,
        )

    # patch `historical_forecasts()` for all forecasting models so that the
    # N internal fit() calls don't each spawn their own MLflow run
    for _, cls in _get_forecasting_models():
        safe_patch(
            FLAVOR_NAME,
            cls,
            "historical_forecasts",
            _patched_historical_forecasts,
        )

    # patch `backtest()` for all forecasting models to log metric results
    for _, cls in _get_forecasting_models():
        safe_patch(
            FLAVOR_NAME,
            cls,
            "backtest",
            _patched_backtest,
        )

    if log_metrics:
        import darts.metrics

        # TODO: To log metrics post-fitting, only patching the metric methods may not be enough.
        # This is becuase `model.fit()` method would terminate the active MLflow run at the end of training,
        # so any metric calls made after that would not be logged.
        # To address this, we need to implement three things:
        # 1. Implement a singleton `_AutologgingMetricsManager` (`mlflow.sklearn`) to maintain a mapping between fitted
        # models and their prediction outputs.
        # 2. Patch the `model.predict()` method to create a mapping between the model (run_id) and prediction output.
        # 3. Patch the metric functions to find the model (run_id) from the input series, then log the metrics to
        # the corresponding run.
        # This way, even if the metric calls are made after `fit()` has terminated the active run, we can still log
        # the metrics to the correct run.

        # patch all metric functions to log results
        for metric_name in darts.metrics.__all__:
            safe_patch(
                FLAVOR_NAME,
                darts.metrics,
                metric_name,
                _make_metric_patch(metric_name),
            )


def get_default_pip_requirements():
    """Return the default pip requirements for logging a darts model.

    Returns
    -------
    list[str]
        A list of pip requirement strings.
    """
    reqs = [_get_pinned_requirement("darts[all]")]
    return reqs


def get_default_conda_env():
    """Return a default conda environment dict for a darts model.

    Returns
    -------
    dict
        A conda environment specification dictionary.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(),
    )


def _get_model_info_tags(model: ForecastingModel) -> dict[str, Any]:
    """
    Returns:
        A dictionary of MLflow run tag keys and values describing the specified model.
    """
    return {
        "model_name": model.__class__.__name__,
        "model_class": (model.__class__.__module__ + "." + model.__class__.__name__),
        "model_likelihood": (
            model.likelihood.__class__.__name__
            if model.likelihood is not None
            else None
        ),
        "model_uses_past_covariates": model.uses_past_covariates,
        "model_uses_future_covariates": model.uses_future_covariates,
        "model_uses_static_covariates": model.uses_static_covariates,
    }


def _log_covariate_info(model: ForecastingModel) -> None:
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

    covariate_info = {
        cov_key: _extract_covariate_metadata(model, uses_attr, series_attr, names_attr)
        for cov_key, uses_attr, series_attr, names_attr in _covariate_types
    }

    if model.uses_static_covariates and model.static_covariates is not None:
        covariate_info["static_covariates"]["is_global"] = len(
            model.static_covariates
        ) != len(model.static_covariates.columns)

    # log complete information as JSON artifact
    with TempDir() as tmp:
        covariates_path = tmp.path("covariates.json")
        with open(covariates_path, "w") as f:
            json.dump(covariate_info, f, indent=2)
        mlflow.log_artifact(covariates_path)


def _is_torch_model(model) -> bool:
    """Check if a model is a `TorchForecastingModel`.

    Parameters
    ----------
    model
        A Darts forecasting model instance.

    Returns
    -------
    bool
        True if the model is a `TorchForecastingModel`, False otherwise.
    """
    method = getattr(model, "predict_from_dataset", None)
    return callable(method)


def _extract_covariate_metadata(
    model: ForecastingModel, uses_attr: str, series_attr: str, names_attr: str
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


def _log_backtest_metrics(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    result,
    backtest_args: dict,
) -> None:
    """Log backtest metric result(s) to MLflow.

    Reshapes each per-series result to a canonical ``(W, T, C, M)`` layout
    (windows, timesteps, components × quantiles, metrics) inferred from the
    metric signatures and ``backtest_args``, logging every cell under a
    descriptive key with the time axis (or window axis when time is reduced)
    mapped to the MLflow ``step``.

    Shape inference respects all kwargs that affect output dimensions:

    * ``time_reduction`` – collapses the time axis (``T=1``).
    * ``component_reduction`` – collapses the component axis (``C=1``).
    * ``series_reduction`` – if non-``None``, windows are already aggregated
      inside the metric, so ``W=1`` regardless of ``backtest.reduction``.
    * ``q`` / ``q_interval`` – expand the component axis with one entry per
      quantile / interval.
    * ``label_reduction`` / ``labels`` – expand the component axis for
      classification metrics.
    * ``reduction=None`` – no aggregation across windows → one value per window.
    * ``last_points_only`` – collapses all windows into one TimeSeries before scoring,
      so there is effectively only one window regardless of reduction.

    When ``label_reduction=None`` is used without explicit ``labels``, the
    unique class values are inferred from ``series`` at runtime so structured
    per-label keys are still produced. When two metrics have incompatible axis
    layouts (different ``time_reduction`` / ``component_reduction`` / quantile
    count), each series result is flattened to integer-indexed keys instead.

    Parameters
    ----------
    autologging_client
        MLflow autologging client used to queue metric writes.
    run_id
        ID of the active MLflow run.
    result
        Return value of ``backtest()``.
    backtest_args
        Bound arguments of the ``backtest()`` call (from
        ``inspect.BoundArguments.arguments`` after ``apply_defaults``).
    """
    metric = backtest_args.get("metric")
    metric = metric if isinstance(metric, list) else [metric]
    metric_kwargs = backtest_args.get("metric_kwargs") or {}
    metric_kwargs = (
        metric_kwargs if isinstance(metric_kwargs, list) else [metric_kwargs]
    )
    # backtest accepts a single dict that applies to all metrics; broadcast it
    if len(metric_kwargs) != len(metric):
        metric_kwargs = [metric_kwargs[0]] * len(metric)
    metric_names = [
        _sanitize_mlflow_key(getattr(m, "__name__", f"metric_{i}"))
        for i, m in enumerate(metric)
    ]
    n_metrics = len(metric)

    # reduction=None means no aggregation across windows → one value per window.
    # last_points_only collapses all windows into one TimeSeries before scoring,
    # so there is effectively only one window regardless of reduction.
    has_windows = backtest_args.get("reduction") is None and not backtest_args.get(
        "last_points_only", False
    )

    # series_reduction inside the metric itself already aggregates across windows,
    # so the result has no window axis even when backtest.reduction is None.
    metric_0_params = inspect.signature(metric[0]).parameters
    if "series_reduction" in metric_0_params:
        effective_sr = metric_kwargs[0].get(
            "series_reduction", metric_0_params["series_reduction"].default
        )
        if effective_sr is not None:
            has_windows = False

    # check the dim axes from the metric kwargs for each
    metric_axes = [_infer_metric_axes(m, kw) for m, kw in zip(metric, metric_kwargs)]
    has_time_axis, has_comp_axis, quantiles_num, _ = metric_axes[0]

    # Inconsistent axes across metrics (different has_time_axis, has_comp_axis, or
    # quantiles_num) means the result can't be reshaped into a single canonical array.
    # Fall back to flat integer-indexed keys for this case.
    axes_inconsistent = any(ax[:3] != metric_axes[0][:3] for ax in metric_axes[1:])

    # quantiles_num=None means label_reduction=None was requested with labels=None,
    labels_unknown = quantiles_num is None

    series = backtest_args.get("series")
    forecast_horizon = backtest_args.get("forecast_horizon")

    series_seq = series2seq(series)
    results = [result] if get_series_seq_type(series) == SeriesType.SINGLE else result

    metrics_by_step: dict[int, dict[str, float]] = {}
    for series_index, (s, r) in enumerate(zip(series_seq, results)):
        series_suffix = f"_s{series_index}" if len(series_seq) > 1 else ""

        if axes_inconsistent:
            name_prefix = metric_names[0] if len(metric_names) == 1 else "metrics"
            flat = np.asarray(r, dtype=float).flatten()
            for i, val in enumerate(flat):
                key = _sanitize_mlflow_key(f"backtest_{name_prefix}{series_suffix}_{i}")
                metrics_by_step.setdefault(0, {})[key] = float(val)
            continue

        # resolve label names from series data when not provided explicitly.
        # mirrors np.unique(np.concatenate([y_true, y_pred])) inside _confusion_matrix.
        # NOTE: importantly this checks the series for labels, not just the windows, if
        # this is an issue, then I'd suggest falling back to flat integer-indexed keys
        # and enforcing explicit labels.
        if labels_unknown:
            inferred_labels = np.unique(s.values())
            quantiles_num = len(inferred_labels)
            metric_axes = [
                (
                    has_time_axis,
                    has_comp_axis,
                    quantiles_num,
                    [f"_label{x:g}" for x in inferred_labels],
                )
            ] + list(metric_axes[1:])

        comps = s.components.tolist()
        # c_size = components × quantiles/intervals/labels per component
        c_size = (s.n_components if has_comp_axis else 1) * quantiles_num
        arr = np.asarray(r, dtype=float)
        # after stripping C and M axes, rest = W*T (or W or T alone)
        rest, extra = divmod(arr.size, c_size * n_metrics)
        if extra:
            logger.warning(
                "Backtest metric logging skipped: result size (%d) is not "
                "divisible by c_size * n_metrics (%d * %d = %d). "
                "The metric output shape does not match the inferred axes.",
                arr.size,
                c_size,
                n_metrics,
                c_size * n_metrics,
            )
            return

        # both time and window axes present: backtest returns (W*T*C*M,) in C order so we can
        # recover W and T only if forecast_horizon is known (T = forecast_horizon)
        if has_time_axis and has_windows:
            if not forecast_horizon or rest % forecast_horizon:
                logger.warning(
                    "Backtest metric logging skipped: cannot split window/time "
                    "axes — %d elements remain after stripping component and "
                    "metric axes, but forecast_horizon=%r does not divide "
                    "evenly. Pass an explicit forecast_horizon to backtest().",
                    rest,
                    forecast_horizon,
                )
                return
            t_size, w_size = forecast_horizon, rest // forecast_horizon
        elif has_time_axis:
            t_size, w_size = rest, 1
        elif has_windows:
            t_size, w_size = 1, rest
        else:
            if rest != 1:
                logger.warning(
                    "Backtest metric logging skipped: expected a single scalar "
                    "per component/metric after reduction, but got %d elements. "
                    "Check time_reduction and component_reduction defaults.",
                    rest,
                )
                return
            t_size, w_size = 1, 1

        canonical = arr.reshape(w_size, t_size, c_size, n_metrics)
        for m, metric_name in enumerate(metric_names):
            quantiles_labels = metric_axes[m][3]
            for w in range(w_size):
                for c in range(c_size):
                    # c is a flat index into the (n_components × quantiles_num) C axis:
                    # c = comp_i * quantiles_num + q_i
                    component_index, quantile_index = divmod(c, quantiles_num)
                    comp_part = (
                        "_" + _sanitize_mlflow_key(comps[component_index])
                        if has_comp_axis
                        else ""
                    )
                    key = f"backtest_{metric_name}{comp_part}{quantiles_labels[quantile_index]}"
                    if has_time_axis and has_windows:
                        key += f"_w{w}"
                    key = _sanitize_mlflow_key(key + series_suffix)
                    for t in range(t_size):
                        # MLflow step maps to the axis the UI should chart:
                        # time when present, otherwise window index
                        step = t if has_time_axis else w
                        metrics_by_step.setdefault(step, {})[key] = float(
                            canonical[w, t, c, m]
                        )

    for step, metrics in metrics_by_step.items():
        autologging_client.log_metrics(run_id=run_id, metrics=metrics, step=step)


def _infer_metric_axes(metric: Callable, metric_kwargs: dict) -> tuple:
    """Infer a metric's output axes from its signature and ``metric_kwargs``.

    Covers ``time_reduction``, ``component_reduction``, ``q``, ``q_interval``,
    and ``label_reduction`` / ``labels`` for classification metrics.
    ``series_reduction`` is handled at the ``_log_backtest_metrics`` level.

    Parameters
    ----------
    metric
        A darts metric callable.
    metric_kwargs
        Keyword arguments that will be forwarded to ``metric``.

    Returns
    -------
    tuple
        ``(has_time_axis, has_comp_axis, quantiles_num, quantiles_labels)`` where

        - ``has_time_axis`` – ``True`` when ``time_reduction`` is ``None`` (i.e. a
          per-timestep axis is present in the output).
        - ``has_comp_axis`` – ``True`` when components are expanded (not collapsed to a scalar).
        - ``quantiles_num`` – number of quantile/interval/label entries; ``None`` when it
          cannot be determined (e.g. ``label_reduction=None`` without explicit
          ``labels``), signalling the caller to fall back to flat logging.
        - ``quantiles_labels`` – one key suffix per ``quantiles_num`` entry (empty list when
          ``quantiles_num`` is ``None``).
    """
    params = inspect.signature(metric).parameters

    def effective(param_name: str) -> Any:
        """Return metric_kwargs value if present, else the signature default."""
        if param_name in metric_kwargs:
            return metric_kwargs[param_name]
        return params[param_name].default if param_name in params else None

    has_time_axis = "time_reduction" in params and effective("time_reduction") is None
    has_comp_axis = (
        "component_reduction" in params and effective("component_reduction") is None
    )

    q_interval, q = metric_kwargs.get("q_interval"), metric_kwargs.get("q")
    if "q_interval" in params and q_interval is not None:
        intervals = np.atleast_2d(np.array(q_interval, dtype=float))
        quantiles_labels = [f"_qi{lo:g}_{hi:g}" for lo, hi in intervals]
    elif "q" in params and q is not None:
        quantiles_labels = [f"_q{v:g}" for v in np.atleast_1d(np.array(q, dtype=float))]
    elif "label_reduction" in params and getattr(metric, "__name__", "") in {
        m.__name__ for m in CLASSIFICATION_METRICS
    }:
        label_reduction = effective("label_reduction")
        if isinstance(label_reduction, _LabelReduction):
            label_reduction = label_reduction.value
        labels = metric_kwargs.get("labels")
        # label_reduction=None means one output per label, but without explicit
        # labels we can't know how many — signal the caller to fall back
        if label_reduction is None and labels is None:
            return (has_time_axis, has_comp_axis, None, [])
        quantiles_labels = (
            [f"_label{x}" for x in np.atleast_1d(labels)]
            if label_reduction is None
            else [""]
        )
    else:
        quantiles_labels = [""]

    return (has_time_axis, has_comp_axis, len(quantiles_labels), quantiles_labels)


def _log_metric_result(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
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

    base_key = f"{dataset_name}_{metric_name}" if dataset_name else metric_name

    def _comp_suffix(idx: int) -> str:
        if component_names is not None and idx < len(component_names):
            return _sanitize_mlflow_key(component_names[idx])
        return str(idx)

    metrics = {}

    if result_arr.ndim == 0:
        # scalar result
        metrics[base_key] = float(result_arr)

    elif result_arr.ndim == 1:
        if not input_is_list:
            # single series: log per-component
            for c_i, val in enumerate(result_arr):
                metrics[f"{base_key}_{_comp_suffix(c_i)}"] = float(val)
        else:
            # list input, components already reduced: log per-series
            for s_i, val in enumerate(result_arr):
                metrics[f"{base_key}_{s_i}"] = float(val)

    elif result_arr.ndim == 2:
        # list input: log per-series and per-component
        n_series, n_components = result_arr.shape
        for s_i in range(n_series):
            for c_i in range(n_components):
                metrics[f"{base_key}_{_comp_suffix(c_i)}_{s_i}"] = float(
                    result_arr[s_i, c_i]
                )

    else:
        # unexpected shape — flatten with integer indices
        for i, val in enumerate(result_arr.flatten()):
            metrics[f"{base_key}_{i}"] = float(val)

    autologging_client.log_metrics(run_id=run_id, metrics=metrics)
    operation = autologging_client.flush(synchronous=False)
    operation.await_completion()


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

        active_run = mlflow.active_run()
        if active_run is None:
            return result

        # backtest() calls metric functions internally; _patched_backtest
        # handles logging the aggregated result, so skip here to avoid
        # generating one flat key per window (series_gen_mape_0, _1, …).
        if getattr(_autolog_state, "in_backtest", False):
            return result

        autologging_client = MlflowAutologgingQueueingClient()
        run_id = active_run.info.run_id

        if len(args) > 0:
            series = args[0]
        else:
            series = kwargs.get("actual_series", None)

        # capture the variable name of actual_series for metric key
        raw = _inspect_original_var_name(series, fallback_name=None)
        dataset_name = _sanitize_mlflow_key(raw) if raw else None

        # handling multi_series input
        input_is_list = not hasattr(series, "components")

        # extract component names from the series (or first element if list)
        single_series = get_single_series(series)
        component_names = (
            single_series.components.tolist() if single_series is not None else None
        )

        _log_metric_result(
            autologging_client,
            run_id,
            metric_name,
            result,
            dataset_name=dataset_name,
            component_names=component_names,
            input_is_list=input_is_list,
        )

        return result

    return _patched_metric
