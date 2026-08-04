"""
MLflow Integration
------------------

Custom MLflow model flavor for Darts forecasting models. Supports saving, loading,
and logging any Darts ``ForecastingModel`` (statistical, ML-based, and PyTorch-based)
to MLflow, as well as automatic logging (``autolog()``) of:

- Model parameters and data metadata: The model creation parameters, target series, and covariate usage information.
- Model storage: The trained model artifact after each ``fit()`` call when ``log_models=True`` (default ``False``).
- Metrics:
  - The score for each metric called inside an active MLflow run.
  - The score(s) from a ``backtest()`` call.
  - Per-epoch training/validation metrics for PyTorch-based models.

See the `MLflow quickstart example <https://github.com/unit8co/darts/blob/master/examples/29-MLflow-quickstart.ipynb>`_
for an end-to-end walkthrough.

To keep auto-logged metrics comparable across runs, use the same evaluation
time frame, forecast horizon, and evaluation start date for every
``backtest()`` / metric call you intend to compare.
"""

import inspect
import re
import sys
import threading
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import Any

from darts.logging import raise_log
from darts.typing import TimeSeriesLike

try:
    import mlflow
except ImportError:
    raise_log(
        ImportError(
            "The `mlflow` module could not be imported. To enable MLflow support "
            "in Darts, follow the detailed instructions in the installation guide: "
            "https://github.com/unit8co/darts/blob/master/INSTALL.md"
        )
    )

import numpy as np
import pandas as pd
import yaml
from mlflow.entities import LoggedModel
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils import _get_fully_qualified_class_name
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

import darts
from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.metrics.utils import (
    _LabelReduction,
    register_metric_callback,
    unregister_metric_callback,
)
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)
from darts.utils.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import (
        TorchForecastingModel,
    )

logger = get_logger(__name__)

FLAVOR_NAME = "darts"


_MODEL_FILE_STAT = "model.pkl"
_MODEL_FILE_TORCH = "model.pt"

# Thread-local flags used by _patched_fit to suppress nested/re-entrant
# autologging: in_historical_forecasts covers historical_forecasts' internal
# fit() calls, in_fit covers nested fit() calls (e.g. ensembles, super()),
# so only the outermost call logs.
_autolog_state = threading.local()


def save_model(
    model: ForecastingModel,
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
    """Save a Darts forecasting model in MLflow format.

    Produces an MLflow model directory at ``path`` containing:

    - The serialized Darts model (delegated to the model's own ``save()`` method).
    - An ``MLmodel`` YAML file with flavor metadata.
    - ``conda.yaml`` and ``requirements.txt`` environment files.

    Parameters
    ----------
    model
        A fitted Darts ``ForecastingModel`` instance.
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
    which is not implemented for Darts models. They are accepted as params for
    simplifying potential future extensibility, and to keep in line with MLflow API
    conventions.
    """
    if not isinstance(model, ForecastingModel):
        raise_log(
            ValueError(
                "Model must be an instance of darts.models.forecasting.ForecastingModel."
            )
        )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = Path(path).resolve()
    _validate_and_prepare_target_save_path(str(path))
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, str(path))

    is_torch = _is_torch_model(model)

    # clean=True excludes any timeseries or callbacks from the model file
    model_file = _MODEL_FILE_TORCH if is_torch else _MODEL_FILE_STAT
    model.save(str(path / model_file), clean=True)

    model_class = _get_fully_qualified_class_name(model)

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, str(path))

    if metadata is not None:
        mlflow_model.metadata = metadata

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        darts_version=darts.__version__,
        data=model_file,
        model_class=model_class,
        code=code_dir_subpath,
    )
    mlflow_model.save(str(path / MLMODEL_FILE_NAME))

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

    with open(path / _CONDA_ENV_FILE_NAME, "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(str(path / _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(str(path / _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))
    _PythonEnv.current().to_yaml(str(path / _PYTHON_ENV_FILE_NAME))


def load_model(
    model_uri: str,
    dst_path: str | None = None,
    **kwargs,
) -> ForecastingModel:
    """Load a Darts model from an MLflow model URI.

    Parameters
    ----------
    model_uri
        An MLflow model URI, e.g. ``"runs:/<run_id>/model"``,
        ``"models:/<name>/<version>"``, or a local ``file:///...`` path.
    dst_path
        Optional local path for downloading remote artifacts.
    **kwargs
        Additional keyword arguments forwarded to the model's ``load()`` method
        (e.g. ``map_location`` for a `TorchForecastingModel`).

    Returns
    -------
    ForecastingModel
        The loaded Darts forecasting model.
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
            )
        )

    model_path = Path(local_path) / flavor_conf["data"]

    return model_cls.load(str(model_path), **kwargs)


def log_model(model: ForecastingModel, **kwargs):
    """Log a Darts model to the current MLflow run, using the Darts MLflow flavor.

    This is a thin wrapper around ``mlflow.models.Model.log()`` that supplies
    the Darts flavor for saving/loading; every other argument is forwarded
    as-is. See the `MLflow documentation
    <https://mlflow.org/docs/latest/api_reference/python_api/mlflow.models.html#mlflow.models.Model.log>`_
    for the full list of accepted parameters (e.g. ``name``,
    ``registered_model_name``, ``conda_env``, ``pip_requirements``,
    ``metadata``, ``tags``, ...).

    Parameters
    ----------
    model
        A fitted Darts ``ForecastingModel`` instance.
    **kwargs
        Forwarded to ``mlflow.models.Model.log()``. Use ``name`` to set the
        run-relative artifact path. ``artifact_path`` parameter is deprecated
        by MLflow and not exposed here.

    Returns
    -------
    ModelInfo
        MLflow ModelInfo object containing model_uri, run_id, artifact_path,
        model_id, timestamps, and other metadata about the logged model.

    Notes
    -----
    ``signature`` and ``input_example`` are currently not supported, as they
    are used to support serving and input validation in the MLflow pyfunc
    flavor, which is not implemented for Darts models.
    """
    # MLflow still requires "artifact_path" to be provided (it has no default),
    # but it is deprecated in favour of "name". Accept it via kwargs for
    # compatibility, defaulting to None so callers can use "name" alone.
    artifact_path = kwargs.pop("artifact_path", None)
    return Model.log(
        artifact_path,
        flavor=sys.modules[__name__],
        model=model,
        **kwargs,
    )


def autolog(
    log_models: bool = False,
    log_params: bool = True,
    log_metrics: bool = True,
    log_torch_metrics: bool = True,
    agg_func: Callable = np.mean,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Enable (or disable) automatic MLflow logging for Darts.

    When enabled, the following functionalities emit detailed logs:

    - Calling ``ForecastingModel.fit()`` inside an active MLflow run (e.g. within
      ``with mlflow.start_run():``); does nothing if no run is active:
      - Logs model creation parameters (``model.model_params``), both as MLflow
        params and as a ``model_params.json`` artifact.
      - Logs target series info and covariate usage information (past, future,
        and static covariates) as a ``series_info.json`` artifact.
      - Stores the trained model artifact when ``log_models=True`` (default:
        ``False``).
      - Logs per-epoch training and validation metrics for PyTorch-based models.
    - Calling ``ForecastingModel.historical_forecasts(retrain=True)`` inside an
      active MLflow run; does nothing if no run is active or ``retrain`` is not
      ``True``:
      - Logs the same model creation parameters and ``series_info.json`` as
        ``fit()`` (overwriting any prior ``fit()`` artifacts in the same run).
      - Does not log the trained model artifact; call ``log_model()`` manually
        if needed.
    - Calling any Darts metric inside an active MLflow run; does nothing if no
      run is active:
      - Logs the result of that metric call as an MLflow metric. More information
        in the notes below.
    - Calling ``ForecastingModel.backtest()`` inside an active MLflow run; does
      nothing if no run is active:
      - Logs all evaluation metrics under ``backtest_*`` keys. More information
        in the notes below.

    .. note::

        Logged metric keys follow the pattern
        ``{metric_name}{component}{quantile_or_label}``, where each part is
        included only when the corresponding axis is present:

        - ``metric_name`` – the metric function name, or the ``name`` metric
          keyword argument when provided.
        - ``component`` – the component name when ``component_reduction=None``.
        - ``quantile_or_label``, e.g.:
          – ``_q0.500`` for quantile metrics with keyword argument ``q=[0.5]``
          - ``_qi_80.000`` for quantile interval metrics with keyword argument
            ``q_interval=[(0.1, 0.9)]`` (80% interval between quantiles 0.1 and
            0.9).
          - ``_label1`` for classification metrics with keyword argument
            ``labels`` when ``label_reduction=None``.

        Per-timestep metrics (``time_reduction=None``) are charted across the
        MLflow ``step``.

        When ``series_reduction`` is set on a metric call, results are already
        aggregated across series inside the metric itself, so the
        cross-series aggregation described below does not apply.

        For a list of series, the logged metric is aggregated over all series
        using ``agg_func``. The detailed per-series metrics / backtest metrics
        are logged under a single ``metrics_per_series.json`` table
        artifact.

        When components are preserved (``component_reduction=None``), all
        series scored together must have the same number of components; names
        are taken from the first series.

        Metric values are only comparable across runs when the evaluation
        settings match. Use the same evaluation time frame, forecast horizon,
        and evaluation start date for every ``backtest()`` / metric call you
        intend to compare.

    Parameters
    ----------
    log_models
        If ``True``, log the trained model artifact after ``fit()``. Defaults to
        ``False``.
    log_params
        If ``True`` (default), log model creation parameters.
    log_metrics
        If ``True`` (default), log the result of any Darts metric call made
        inside an active MLflow run.
    log_torch_metrics
        If ``True`` (default), enable ``mlflow.pytorch.autolog(log_models=False)``
        around PyTorch-based model training to automatically log per-epoch
        training and validation metrics. Only effective for PyTorch-based models.
    agg_func
        Function used to aggregate a metric's per-series values into the
        single value logged for a list of series (e.g. ``np.mean``, the
        default, or ``np.median``). Called as ``agg_func(values)`` on a list
        of floats.
    disable
        If ``True``, restore the original ``fit()`` methods and stop
        autologging.
    silent
        If ``True`` (default ``False``), suppress all event logging and warnings from
        MLflow during autologging.
    """
    # Enable/disable mlflow.pytorch.autolog for per-epoch metrics on torch models.
    # This must happen outside the @autologging_integration-decorated _autolog()
    # because that decorator short-circuits _autolog()'s body entirely when
    # disable=True, so a call placed inside it would never run. Unlike
    # mlflow.sklearn, which exposes a private, undecorated _autolog(flavor_name=...)
    # that other flavors (e.g. xgboost) call to tag its patches under their own
    # integration name for cleanup, mlflow.pytorch has no such hook: its autolog()
    # hardcodes its own patches under "pytorch", so Darts can't fold pytorch's
    # patch lifecycle into its own and must call mlflow.pytorch.autolog() directly.
    if log_torch_metrics and not disable:
        try:
            import mlflow.pytorch

            mlflow.pytorch.autolog(
                log_models=False,
                log_datasets=False,
                checkpoint=False,
                silent=silent,
            )
        except ImportError:
            pass
    elif disable:
        try:
            import mlflow.pytorch

            mlflow.pytorch.autolog(disable=True)
        except (ImportError, Exception):
            pass

    # Register/unregister the metric-logging callback with darts.metrics.utils
    # directly, rather than via mlflow's safe_patch on each darts.metrics
    # attribute (which is import-order sensitive)
    unregister_metric_callback(_mlflow_metric_callback)
    if log_metrics and not disable:
        register_metric_callback(_mlflow_metric_callback)

    _autolog(
        log_models=log_models,
        log_params=log_params,
        log_metrics=log_metrics,
        agg_func=agg_func,
        disable=disable,
        silent=silent,
    )


def _get_forecasting_models():
    """Find all ``ForecastingModel`` subclasses currently loaded in memory.

    Traverses ``__subclasses__()``, avoiding force-importing all of the forecasting
    models.

    Returns:
        A list of (name, class) tuples for all matching classes.
    """
    seen: set[type] = set()
    stack = [ForecastingModel]
    while stack:
        current = stack.pop()
        for sub in current.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)

    classes = [(cls.__name__, cls) for cls in seen]
    return sorted(classes, key=itemgetter(0))


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_models: bool = True,
    log_params: bool = True,
    log_metrics: bool = True,
    agg_func: Callable = np.mean,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Internal autolog implementation decorated with ``@autologging_integration``.

    Handles patching of Darts ``ForecastingModel.fit()`` and metric functions.
    The ``mlflow.pytorch.autolog`` coordination is handled by the public
    ``autolog()`` wrapper because the decorator short-circuits on
    ``disable=True``.
    """

    def _patched_fit(original, self, *args, **kwargs):
        """Patch function for ForecastingModel.fit() autologging.

        Logs model parameters, class, and covariates; optionally logs the
        model artifact when ``log_models=True``.

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
        ForecastingModel
            The result of calling the original fit method.
        """
        # Create a training session to track the training process and log information
        autologging_client = MlflowAutologgingQueueingClient()

        if getattr(_autolog_state, "in_historical_forecasts", False):
            return original(self, *args, **kwargs)

        # handle nested fit() calls
        if getattr(_autolog_state, "in_fit", False):
            return original(self, *args, **kwargs)

        # Track which model is active so metric patches can prefix their keys
        _autolog_state.current_model_name = type(self).__name__

        _autolog_state.in_fit = True
        try:
            result = original(self, *args, **kwargs)
        finally:
            _autolog_state.in_fit = False

        active_run = mlflow.active_run()
        if active_run is None:
            return result
        run_id = active_run.info.run_id

        fit_args = inspect.signature(original).bind(self, *args, **kwargs).arguments
        _log_model_setup(
            self,
            autologging_client,
            run_id,
            series=fit_args["series"],
            past_covariates=fit_args.get("past_covariates"),
            future_covariates=fit_args.get("future_covariates"),
            log_params=log_params,
        )

        param_logging_ops = autologging_client.flush(synchronous=False)

        if log_models:
            model_name = type(self).__name__
            model: LoggedModel = _initialize_logged_model(
                name=model_name, flavor=FLAVOR_NAME
            )
            try:
                registered_model_name = get_autologging_config(
                    flavor_name=FLAVOR_NAME,
                    config_key="registered_model_name",
                    default_value=None,
                )
                log_model(
                    result,
                    name=model_name,
                    registered_model_name=registered_model_name,
                    model_id=model.model_id,
                )
            except Exception:
                raise_log(
                    ValueError(
                        f"Failed to autolog model artifact for {type(self).__name__}."
                    )
                )

        param_logging_ops.await_completion()

        return result

    def _patched_historical_forecasts(original, self, *args, **kwargs):
        """Suppress per-iteration fit() autologging; log model setup once when
        ``retrain=True``.

        Sets a thread-local flag so ``_patched_fit`` skips autologging for the
        internal ``fit()`` calls. When ``retrain is True`` and an MLflow run is
        active, logs model tags, creation parameters, and series info once after
        the call (overwriting any prior ``fit()`` artifacts in the same run).
        Does not start a run and does not log the trained model artifact.
        """
        _autolog_state.in_historical_forecasts = True
        try:
            result = original(self, *args, **kwargs)
        finally:
            _autolog_state.in_historical_forecasts = False

        active_run = mlflow.active_run()
        if active_run is None:
            return result

        bound = inspect.signature(ForecastingModel.historical_forecasts).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        if bound.arguments["retrain"] is not True:
            return result

        autologging_client = MlflowAutologgingQueueingClient()
        _log_model_setup(
            self,
            autologging_client,
            active_run.info.run_id,
            series=bound.arguments["series"],
            past_covariates=bound.arguments.get("past_covariates"),
            future_covariates=bound.arguments.get("future_covariates"),
            log_params=log_params,
        )
        autologging_client.flush(synchronous=False).await_completion()
        return result

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
            autologging_client=autologging_client,
            run_id=active_run.info.run_id,
            result=result,
            backtest_args=backtest_args,
            agg_func=agg_func,
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
    # N internal fit() calls don't each log, and so that retrain=True calls
    # log model setup once
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


def get_default_pip_requirements():
    """Return the default pip requirements for logging a Darts model.

    Returns
    -------
    list[str]
        A list of pip requirement strings.
    """
    reqs = [_get_pinned_requirement("darts")]
    return reqs


def get_default_conda_env():
    """Return a default conda environment dict for a Darts model.

    Returns
    -------
    dict
        A conda environment specification dictionary.
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(),
    )


def _infer_covariate_usage(
    model: ForecastingModel,
    series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None,
    future_covariates: TimeSeriesLike | None,
) -> tuple[bool, bool, bool]:
    """Infer past/future/static covariate usage from model state and call args.

    After ``historical_forecasts(retrain=True)`` the outer model is still
    unfitted (training happens on internal copies), so ``model.uses_*`` stays
    ``False``. Fall back to call args / ``add_encoders`` / static covariates on
    ``series``, gated by ``supports_*`` / ``considers_static_covariates``.
    """
    # encoder keys like "datetime_attribute" map to {"past": ..., "future": ...};
    # non-dict values ("tz", "transformer") are ignored by the isinstance check
    enc_types = {
        cov
        for val in (model.add_encoders or {}).values()
        if isinstance(val, dict)
        for cov in ("past", "future")
        if cov in val
    }
    first_series = get_single_series(series)
    uses_past = model.uses_past_covariates or (
        model.supports_past_covariates
        and (past_covariates is not None or "past" in enc_types)
    )
    uses_future = model.uses_future_covariates or (
        model.supports_future_covariates
        and (future_covariates is not None or "future" in enc_types)
    )
    uses_static = model.uses_static_covariates or (
        first_series is not None
        and first_series.static_covariates is not None
        and model.supports_static_covariates
        and model.considers_static_covariates
    )
    return uses_past, uses_future, uses_static


def _get_model_info_tags(
    model: ForecastingModel,
    series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
) -> dict[str, Any]:
    """
    Returns:
        A dictionary of MLflow run tag keys and values describing the specified model.
    """
    uses_past, uses_future, uses_static = _infer_covariate_usage(
        model, series, past_covariates, future_covariates
    )
    return {
        "model_class": model.__class__.__name__,
        "model_reference": (
            model.__class__.__module__ + "." + model.__class__.__name__
        ),
        "model_likelihood": (
            model.likelihood.__class__.__name__
            if model.likelihood is not None
            else None
        ),
        "model_uses_past_covariates": uses_past,
        "model_uses_future_covariates": uses_future,
        "model_uses_static_covariates": uses_static,
    }


def _log_model_setup(
    model: ForecastingModel,
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    *,
    log_params: bool = True,
) -> None:
    """Log model tags, creation parameters, and series info to an active run.

    Shared by ``fit()`` and ``historical_forecasts(retrain=True)`` autologging.
    Does not log the trained model artifact.
    """
    autologging_client.set_tags(
        run_id=run_id,
        tags=_get_model_info_tags(
            model,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        ),
    )
    if log_params:
        autologging_client.log_params(run_id=run_id, params=model.model_params)
        mlflow.log_dict(model.model_params, "model_params.json")
        _log_series_info(
            model,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )


def _log_series_info(
    model: ForecastingModel,
    series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None,
    future_covariates: TimeSeriesLike | None,
) -> None:
    """Log target series and covariate usage information to MLflow.

    Extracts information about the target series, and about past, future,
    and static covariates used during training and logs them as a JSON
    artifact for easy filtering, comparison, and documentation.

    Logs:
    - Target series: component count and names
    - Past / future covariates: usage, count, and names, including both
      explicitly-passed covariates and any generated by ``add_encoders``
    - Static covariates: usage, count, names, and whether they are global
    - Artifact: complete metadata as ``series_info.json``

    Parameters
    ----------
    model
        A fitted Darts forecasting model instance.
    series
        The ``series`` argument passed to ``fit()``: a single ``TimeSeries``
        or a ``Sequence[TimeSeries]``.
    past_covariates
        The past covariate argument passed to ``fit()``, or
        ``None``.
    future_covariates
        The future covariate covariate argument passed to ``fit()``, or
        ``None``.
    """
    first_series = get_single_series(series)
    uses_past, uses_future, uses_static = _infer_covariate_usage(
        model, series, past_covariates, future_covariates
    )
    series_info = {
        "series": {
            "count": first_series.n_components,
            "names": first_series.components.tolist(),
        },
        "past_covariates": _extract_covariate_metadata(
            uses_past,
            get_single_series(past_covariates),
            "components",
            encoded_names=model.encoders.past_components,
        ),
        "future_covariates": _extract_covariate_metadata(
            uses_future,
            get_single_series(future_covariates),
            "components",
            encoded_names=model.encoders.future_components,
        ),
    }

    static_covariates = (
        first_series.static_covariates if first_series is not None else None
    )
    series_info["static_covariates"] = _extract_covariate_metadata(
        uses_static, static_covariates, "columns"
    )
    if uses_static and static_covariates is not None:
        # static covariates are global (one shared row) unless there is one row
        # per series component, in which case they are component-specific
        series_info["static_covariates"]["is_global"] = (
            len(static_covariates) != first_series.n_components
        )

    # log complete information as JSON artifact
    mlflow.log_dict(series_info, "series_info.json")


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
    return TORCH_AVAILABLE and isinstance(model, TorchForecastingModel)


def _extract_covariate_metadata(
    uses: bool,
    single_cov: TimeSeries | pd.DataFrame | None,
    names_attr: str,
    encoded_names: pd.Index | list[str] | None = None,
) -> dict:
    """Extract metadata for a single covariate type from its (already
    singular) value.

    Parameters
    ----------
    uses
        Whether the model uses this covariate type.
    single_cov
        The covariate's value for one series: a ``TimeSeries`` (past/future
        covariates) or a static-covariates ``DataFrame``, or ``None``.
    names_attr : str
        Attribute holding the feature names ("components" for a
        ``TimeSeries``, "columns" for a static-covariates ``DataFrame``).
    encoded_names
        Additional covariate names generated by encoders (``add_encoders``),
        appended to the names extracted from ``single_cov``. Ignored when
        ``uses`` is ``False``.

    Returns
    -------
    dict
        Dictionary with keys: "used" (bool), "count" (int), "names" (list).
    """
    info = {"used": False, "count": 0, "names": []}

    if uses:
        info["used"] = True
        names = list(getattr(single_cov, names_attr)) if single_cov is not None else []
        if encoded_names is not None:
            names = names + list(encoded_names)
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


def _log_per_series_table(rows: list[dict]) -> None:
    """Append the granular per-series metric breakdown to a single, run-wide
    table artifact.

    Each row is a single metric cell for one series, with columns ``key`` (the
    aggregate MLflow key, without any series suffix), ``series_index``, ``step``
    (the time or window index charted by MLflow), and ``value``. All calls
    within a run append to the same ``metrics_per_series.json`` artifact.
    Used when more than one series is scored, since the logged metric keys
    only carry the aggregate over series.

    Parameters
    ----------
    rows
        One dict per metric cell with keys ``key``, ``series_index``, ``step``,
        and ``value``.
    """
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["key", "series_index", "step"])
    mlflow.log_table(data=df, artifact_file="metrics_per_series.json")


def _log_backtest_metrics(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    result,
    backtest_args: dict,
    agg_func: Callable = np.mean,
) -> None:
    """Log backtest metric result(s) to MLflow.

    Reshapes each per-series result to a canonical ``(W, T, C, M)`` layout
    (windows, timesteps, components x quantiles, metrics) inferred from the
    metric signatures and ``backtest_args``, logging every cell under a
    descriptive key with the time axis (or window axis when time is reduced)
    mapped to the MLflow ``step``. A metric's ``name`` entry in
    ``metric_kwargs`` overrides the metric-name token in the key (the
    ``backtest_`` prefix and axis suffixes are preserved).

    Shape inference respects all kwargs that affect output dimensions:

    - ``time_reduction`` – collapses the time axis (``T=1``).
    - ``component_reduction`` – collapses the component axis (``C=1``).
    - ``series_reduction`` – if other than ``None``, windows are already aggregated
      inside the metric, so ``W=1`` regardless of ``backtest.reduction``.
    - ``q`` / ``q_interval`` – expand the component axis with one entry per
      quantile / interval.
    - ``labels`` - expand each component with one entry per label along
      the component axis.
    - ``label_reduction`` – collapses the labels along the component axis.
      value; ``labels`` only restricts which classes are scored.
    - ``reduction=None`` – no aggregation across windows -> one value per window.
    - ``last_points_only`` – collapses all windows into one TimeSeries before scoring,
      so there is effectively only one window regardless of reduction.


    When more than one series is scored, the logged value is ``agg_func``
    applied over series for each cell, and the granular per-series breakdown
    is appended to the run's ``metrics_per_series.json`` table artifact
    (shared with ``_log_metric_result``). For a single series the aggregate
    is just the value itself and no artifact is written. When components
    are preserved (``component_reduction=None``), all series scored together
    must have the same number of components; names are taken from the first
    series.

    Series of different lengths are assumed to share the same end date, so any
    axis mapping to real dates (the window axis, or the per-timestep axis
    when ``last_points_only`` stitches windows into one series) is aligned
    from the end rather than the start. The per-horizon-step axis is left
    as-is, since it means "steps ahead" rather than a real date.

    Raises
    ------
    ValueError
        On a shape/size mismatch between the metric result and the inferred
        axes, when ``component_reduction=None`` and series in a sequence have
        different numbers of components, or when ``label_reduction=None`` is
        requested without explicit ``labels``.

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
    agg_func
        Function used to aggregate a metric's per-series values into the
        single value logged for a list of series. Called as
        ``agg_func(values)`` on a list of floats.
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
    # the `name` entry in metric_kwargs overrides the metric-name token in the key
    metric_names = [
        _sanitize_mlflow_key(
            metric_kwargs[i].get("name") or getattr(m, "__name__", f"metric_{i}")
        )
        for i, m in enumerate(metric)
    ]
    n_metrics = len(metric)

    # reduction=None means no aggregation across windows -> one value per window.
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
    has_time_axis, has_comp_axis, axis_labels_0 = metric_axes[0]
    axis_size = len(axis_labels_0)

    series = backtest_args.get("series")
    forecast_horizon = backtest_args.get("forecast_horizon")
    historical_forecasts = backtest_args.get("historical_forecasts")
    last_points_only = backtest_args.get("last_points_only", False)

    # if last_points_only is True, has_windows will be False, so fc_hzn is not needed
    if historical_forecasts is not None and not last_points_only:
        first_series_hf = (
            historical_forecasts
            if get_series_seq_type(series) == SeriesType.SINGLE
            else historical_forecasts[0]
        )
        forecast_horizon = len(first_series_hf[0])

    series_seq = series2seq(series)
    results = [result] if get_series_seq_type(series) == SeriesType.SINGLE else result
    # component names are only used when the metric preserves components
    if has_comp_axis:
        n_components = {s.n_components for s in series_seq}
        if len(n_components) > 1:
            raise_log(
                ValueError(
                    "Backtest metric logging failed: all series must have the same "
                    f"number of components, got {sorted(n_components)}. Consider "
                    f"setting a metric `component_reduction`, or make sure all series "
                    f"have the same number of components."
                )
            )

    # agg maps (key, step) -> per-series values, aggregated into the logged metric.
    agg: dict[tuple[str, int], list[float]] = {}
    rows: list[dict] = []

    # component names/count from the first series (all series share n_components)
    comps = series_seq[0].components.tolist()
    # c_size = components x quantiles/intervals/labels per component
    c_size = (len(comps) if has_comp_axis else 1) * axis_size
    # base_keys[m][c]: sanitized key without the optional window suffix
    base_keys = []
    for m, metric_name in enumerate(metric_names):
        axis_labels = metric_axes[m][2]
        keys_m = []
        for c in range(c_size):
            # c is a flat index into the (n_components x axis_size) C axis:
            # c = comp_i * axis_size + axis_idx
            component_index, axis_idx = divmod(c, axis_size)
            comp_part = (
                "_" + _sanitize_mlflow_key(comps[component_index])
                if has_comp_axis
                else ""
            )
            keys_m.append(
                _sanitize_mlflow_key(
                    f"backtest_{metric_name}{comp_part}{axis_labels[axis_idx]}"
                )
            )
        base_keys.append(keys_m)

    # first pass: reshape each series' result into a canonical (W, T, C, M)
    # array, recording its window-axis length for the alignment pass below.
    series_shapes = []
    for r in results:
        arr = np.asarray(r, dtype=float)
        # after stripping C and M axes, rest = W*T (or W or T alone)
        rest, extra = divmod(arr.size, c_size * n_metrics)
        if extra:
            raise_log(
                ValueError(
                    f"Backtest metric logging failed: result size ({arr.size}) "
                    f"is not divisible by c_size * n_metrics ({c_size} * "
                    f"{n_metrics} = {c_size * n_metrics}). The metric output "
                    "shape does not match the inferred axes."
                )
            )

        # both time and window axes present: backtest returns (W*T*C*M,) in C order so we can
        # recover W and T only if forecast_horizon is known (T = forecast_horizon)
        if has_time_axis and has_windows:
            t_size, w_size = forecast_horizon, rest // forecast_horizon
        elif has_time_axis:
            t_size, w_size = rest, 1
        elif has_windows:
            t_size, w_size = 1, rest
        else:
            if rest != 1:
                raise_log(
                    ValueError(
                        f"Backtest metric logging failed: expected a single "
                        f"scalar per component/metric after reduction, but got "
                        f"{rest} elements. Check time_reduction and "
                        "component_reduction defaults."
                    )
                )
            t_size, w_size = 1, 1

        series_shapes.append((
            t_size,
            w_size,
            arr.reshape(w_size, t_size, c_size, n_metrics),
        ))

    # align the calendar-relative axes from the end
    max_w_size = max((w_size for _, w_size, _ in series_shapes), default=0)
    t_axis_is_calendar = has_time_axis and not has_windows and last_points_only
    max_t_size = (
        max((t_size for t_size, _, _ in series_shapes), default=0)
        if t_axis_is_calendar
        else 0
    )

    for series_index, (t_size, w_size, canonical) in enumerate(series_shapes):
        w_offset = max_w_size - w_size if has_windows else 0
        t_offset = max_t_size - t_size if t_axis_is_calendar else 0
        for m in range(n_metrics):
            for w in range(w_size):
                aligned_w = w + w_offset
                for c in range(c_size):
                    key = base_keys[m][c]
                    if has_time_axis and has_windows:
                        key = f"{key}_w{aligned_w}"
                    for t in range(t_size):
                        # MLflow step maps to the axis the UI should chart:
                        # time when present, otherwise window index
                        step = t + t_offset if has_time_axis else aligned_w
                        value = float(canonical[w, t, c, m])
                        agg.setdefault((key, step), []).append(value)
                        rows.append({
                            "key": key,
                            "series_index": series_index,
                            "step": step,
                            "value": value,
                        })

    # aggregate across series for each (key, step); for a single series this
    # is just the value itself.
    metrics_by_step: dict[int, dict[str, float]] = {}
    for (key, step), values in agg.items():
        metrics_by_step.setdefault(step, {})[key] = float(agg_func(values))
    for step, metrics in metrics_by_step.items():
        autologging_client.log_metrics(run_id=run_id, metrics=metrics, step=step)

    # append the granular per-series breakdown to the run's table artifact
    # (multi-series only)
    if len(series_seq) > 1:
        _log_per_series_table(rows)


def _infer_metric_axes(metric: Callable, metric_kwargs: dict) -> tuple:
    """Infer a metric's output axes from its signature and ``metric_kwargs``.

    Covers ``time_reduction``, ``component_reduction``, ``q``, ``q_interval``,
    and ``label_reduction`` / ``labels`` for classification metrics.
    ``series_reduction`` is handled at the ``_log_backtest_metrics`` level.

    Parameters
    ----------
    metric
        A Darts metric callable.
    metric_kwargs
        Keyword arguments that will be forwarded to ``metric``.

    Returns
    -------
    tuple
        ``(has_time_axis, has_comp_axis, axis_labels)`` where

        - ``has_time_axis`` – ``True`` when ``time_reduction`` is ``None`` (i.e. a
          per-timestep axis is present in the output).
        - ``has_comp_axis`` – ``True`` when components are expanded (not collapsed to a scalar).
        - ``axis_labels`` – one key suffix per quantile/interval/label entry.

    Raises
    ------
    ValueError
        If ``label_reduction=None`` is requested without explicit ``labels``.
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
        axis_labels = [f"_qi_{100 * (hi - lo):.3f}" for lo, hi in intervals]
    elif "q" in params and q is not None:
        axis_labels = [f"_q{v:.3f}" for v in np.atleast_1d(np.array(q, dtype=float))]
    elif "label_reduction" in params and getattr(metric, "__name__", ""):
        label_reduction = effective("label_reduction")
        if isinstance(label_reduction, _LabelReduction):
            label_reduction = label_reduction.value
        labels = metric_kwargs.get("labels")
        # label_reduction=None means one output per label, but without explicit
        # labels we can't know how many ahead of time
        if label_reduction is None and labels is None:
            raise_log(
                ValueError(
                    "`label_reduction=None` requires explicit `labels` to be "
                    "passed for MLflow autologging (the number of output "
                    "labels cannot be determined ahead of time otherwise)."
                )
            )
        axis_labels = (
            [f"_label{x}" for x in np.atleast_1d(labels)]
            if label_reduction is None
            else [""]
        )
    else:
        axis_labels = [""]

    return (has_time_axis, has_comp_axis, axis_labels)


def _log_metric_result(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    metric_name: str,
    result,
    series,
    has_time_axis: bool,
    has_comp_axis: bool,
    axis_labels: list[str],
    series_reduced: bool = False,
    agg_func: Callable = np.mean,
) -> None:
    """Log a metric result to the active MLflow run.

    Reshapes each per-series result into a canonical ``(T, C)`` layout
    (timesteps, components x quantiles/intervals/labels) inferred from the
    metric signature and call kwargs by ``_infer_metric_axes``, logging every
    cell under a descriptive key with the time axis mapped to the MLflow
    ``step``. This mirrors ``_log_backtest_metrics`` (without the
    window/``forecast_horizon`` split, since ``multi_ts_support`` returns a
    clean per-series list).

    The logged MLflow key follows the pattern::

        {metric_name}{component}{quantile_or_label}

    where each optional part is included only when the corresponding axis is
    present:

    - ``component`` – ``_{component_name}`` when ``has_comp_axis``.
    - ``quantile_or_label`` – e.g. ``_q0.500`` / ``_qi_80.000`` / ``_label1``.

    When more than one series is scored, the logged value is ``agg_func``
    applied over series for each cell, and the granular per-series breakdown
    is appended to the run's ``metrics_per_series.json`` table artifact
    (shared with ``_log_backtest_metrics``). For a single series the
    aggregate is just the value itself and no artifact is written.

    TODO: improve this, it's about predictions of different or different
      intersection lengths between actual and pred (from the `Raises`
      below, it sounds even that this wouldn't be supported?)
    Series of different lengths are assumed to share the same end date, so the
    time axis is aligned from the end rather than the start: a shorter series
    lines up on its last value instead of its first.

    Raises
    ------
    ValueError
        On a shape/size mismatch between the metric result and the inferred
        axes, or when ``has_comp_axis`` is ``True`` and series in a sequence
        have different numbers of components.

    Parameters
    ----------
    metric_name
        Metric name used as the MLflow key (the metric's ``name`` keyword
        argument when provided, otherwise the metric function name).
    result
        The metric result to log.
    series
        The ``actual_series`` argument passed to the metric (single series or
        ``Sequence[TimeSeries]``); used for component names and series count.
        When ``has_comp_axis`` is ``True``, all series in a sequence must have
        the same number of components; names are taken from the first series.
    has_time_axis
        ``True`` when the result carries a per-timestep axis (``time_reduction=None``).
    has_comp_axis
        ``True`` when components are expanded (``component_reduction=None``).
    axis_labels
        One key suffix per quantile/interval/label entry.
    series_reduced
        ``True`` when ``series_reduction`` collapsed the series axis inside the
        metric, so the result has no leading series axis even for list input.
    agg_func
        Function used to aggregate a metric's per-series values into the
        single value logged for a list of series. Called as
        ``agg_func(values)`` on a list of floats.
    """
    axis_size = len(axis_labels)

    if series_reduced:
        # series_reduction aggregated across series -> single result, no series axis
        series_seq = [get_single_series(series)]
        results = [result]
    else:
        series_seq = series2seq(series)
        results = (
            [result] if get_series_seq_type(series) == SeriesType.SINGLE else result
        )
        # component names are only used when the metric preserves components
        if has_comp_axis:
            n_components = {s.n_components for s in series_seq}
            if len(n_components) > 1:
                raise_log(
                    ValueError(
                        f"Metric logging failed for `{metric_name}`: all series must "
                        f"have the same number of components, got "
                        f"{sorted(n_components)}."
                    )
                )

    # TODO: a lot of this seems duplicated from the backtest logic; improve
    # component names/count from the first series (all series share n_components)
    comps = series_seq[0].components.tolist()
    # c_size = components x quantiles/intervals/labels per component
    c_size = (len(comps) if has_comp_axis else 1) * axis_size
    keys = []
    for c in range(c_size):
        # c is a flat index into the (n_components x axis_size) C axis:
        # c = comp_i * axis_size + axis_idx
        component_index, axis_idx = divmod(c, axis_size)
        comp_part = (
            "_" + _sanitize_mlflow_key(comps[component_index]) if has_comp_axis else ""
        )
        keys.append(
            _sanitize_mlflow_key(metric_name + comp_part + axis_labels[axis_idx])
        )

    # first pass: reshape each series' result into a canonical (T, C) array,
    # recording its time-axis length for the alignment pass below.
    series_shapes = []
    for r in results:
        arr = np.asarray(r, dtype=float)
        # after stripping the C axis, the remainder is the time axis (or scalar)
        n_times, extra = divmod(arr.size, c_size)
        if extra:
            raise_log(
                ValueError(
                    f"Metric logging failed for `{metric_name}`: result size "
                    f"({arr.size}) is not divisible by the inferred "
                    f"component/quantile size ({c_size}). The metric output "
                    "shape does not match the inferred axes."
                )
            )

        if has_time_axis:
            t_size = n_times
        elif n_times != 1:
            raise_log(
                ValueError(
                    f"Metric logging failed for `{metric_name}`: expected a "
                    f"single value per component/quantile after reduction, "
                    f"but got {n_times} elements. Check time_reduction and "
                    "component_reduction."
                )
            )
        else:
            t_size = 1

        series_shapes.append((t_size, arr.reshape(t_size, c_size)))

    # align the time axis from the end (see docstring)
    max_t_size = max((t_size for t_size, _ in series_shapes), default=0)

    # agg maps (key, step) -> per-series values, aggregated into the logged metric.
    agg: dict[tuple[str, int], list[float]] = {}
    rows: list[dict] = []
    for series_index, (t_size, canonical) in enumerate(series_shapes):
        step_offset = max_t_size - t_size if has_time_axis else 0
        for c, key in enumerate(keys):
            for t in range(t_size):
                # MLflow step maps to the time axis when present
                step = t + step_offset if has_time_axis else 0
                value = float(canonical[t, c])
                agg.setdefault((key, step), []).append(value)
                rows.append({
                    "key": key,
                    "series_index": series_index,
                    "step": step,
                    "value": value,
                })

    # aggregate across series for each (key, step); for a single series this
    # is just the value itself.
    metrics_by_step: dict[int, dict[str, float]] = {}
    for (key, step), values in agg.items():
        metrics_by_step.setdefault(step, {})[key] = float(agg_func(values))
    for step, metrics in metrics_by_step.items():
        autologging_client.log_metrics(run_id=run_id, metrics=metrics, step=step)
    autologging_client.flush(synchronous=False).await_completion()

    # append the granular per-series breakdown to the run's table artifact
    # (multi-series only)
    if len(series_seq) > 1:
        _log_per_series_table(rows)


def _mlflow_metric_callback(func, result, args, kwargs) -> None:
    """Metric callback registered with ``darts.metrics.utils`` for autologging.

    Invoked by ``multi_ts_support`` (the outermost decorator on every Darts
    metric) after every top-level metric call, so it fires regardless of how
    the metric was imported. It is not invoked for internal metric-to-metric
    calls (e.g. ``rmse`` calling ``mse`` internally via ``_get_wrapped_metric``),
    since those bypass ``multi_ts_support`` entirely.

    When an active MLflow run exists, infers the output axes from the metric
    signature and call kwargs (via ``_infer_metric_axes``) and delegates to
    ``_log_metric_result``, which logs each cell under a key built as::

        {metric_name}{component}{quantile_or_label}

    where:

    - ``metric_name`` – the metric function name, or the ``name`` keyword
      argument when provided (it overrides only this token).
    - ``component`` – ``_{component_name}`` when ``component_reduction=None``.
    - ``quantile_or_label`` – quantile/interval/label suffix (e.g. ``_q0.500``,
      ``_qi_80.000``, ``_label1``) when applicable.

    When the input is a ``Sequence[TimeSeries]`` with more than one series, the
    logged value is ``autolog()``'s ``agg_func`` applied over series, and the
    per-series breakdown is appended to the run's ``metrics_per_series.json``
    table artifact instead of per-series keys.

    The per-timestep axis (``time_reduction=None``) is mapped to the MLflow
    ``step``.

    Parameters
    ----------
    func
        The Darts metric function that was called (used for its name and
        signature).
    result
        The metric's return value.
    args
        Positional arguments the metric was called with.
    kwargs
        Keyword arguments the metric was called with.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        return

    # backtest() calls metric functions internally; _patched_backtest
    # handles logging the aggregated result, so skip here to avoid
    # generating one flat key per window (series_gen_mape_0, _1, …).
    if getattr(_autolog_state, "in_backtest", False):
        return

    series = args[0] if len(args) > 0 else kwargs["actual_series"]

    autologging_client = MlflowAutologgingQueueingClient()
    run_id = active_run.info.run_id

    # the `name` kwarg overrides the metric-name token in the logged key
    key_name = _sanitize_mlflow_key(kwargs.get("name") or func.__name__)

    # infer output axes from the metric signature + call kwargs
    has_time_axis, has_comp_axis, axis_labels = _infer_metric_axes(func, kwargs)

    # series_reduction collapses the series axis inside the metric, so the
    # result has no leading series axis even for list input.
    params = inspect.signature(func).parameters
    series_reduced = False
    if "series_reduction" in params:
        effective_sr = kwargs.get(
            "series_reduction", params["series_reduction"].default
        )
        series_reduced = effective_sr is not None

    # _mlflow_metric_callback is a bare registered callback, not a closure over
    # autolog()'s call kwargs, so agg_func is read back from the autologging
    # config store that autolog() populated.
    agg_func = get_autologging_config(
        flavor_name=FLAVOR_NAME, config_key="agg_func", default_value=np.mean
    )
    _log_metric_result(
        autologging_client,
        run_id,
        key_name,
        result,
        series,
        has_time_axis,
        has_comp_axis,
        axis_labels,
        series_reduced=series_reduced,
        agg_func=agg_func,
    )
