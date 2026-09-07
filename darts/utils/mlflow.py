"""
MLflow Integration
------------------

.. _darts-mlflow-autolog-logging:

Custom MLflow model flavor for Darts forecasting models. Supports saving, loading,
and logging any Darts ``ForecastingModel`` (statistical, ML-based, and PyTorch-based)
to MLflow, as well as automatic logging  via ``autolog()``.

See the `MLflow quickstart <https://github.com/unit8co/darts/blob/master/examples/29-MLflow-examples.ipynb>`_
for an end-to-end walkthrough.

.. dropdown:: Here's a quick start example

    .. highlight:: python
    .. code-block:: python

        import os
        import tempfile

        import mlflow

        import darts.metrics as metrics
        from darts.datasets import AirPassengersDataset
        from darts.models import LinearRegressionModel
        from darts.utils.mlflow import autolog

        # dummy temporary directory for local MLflow tracking;
        # use a permanent location for real use cases
        tmpdir = tempfile.mkdtemp()
        mlflow_db = os.path.join(tmpdir, "mlflow.db")
        artifact_root = os.path.join(tmpdir, "mlruns")

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        # SQLite stores run metadata; artifacts default to ./mlruns unless we set a location
        mlflow.set_experiment(
            experiment_id=mlflow.create_experiment(
                "darts-quickstart",
                artifact_location=artifact_root,
            )
        )

        # load series and create train and val splits
        series = AirPassengersDataset().load()
        train, val = series[:-36], series[-36:]

        # two models to compare with different lookback windows
        model_lags, names = [12, 24], ["one-year", "two-years"]
        horizon = 12

        # activate autologging and try out the models
        autolog()
        for lags, name in zip(model_lags, names):
            with mlflow.start_run(run_name=name) as run:
                model = LinearRegressionModel(lags=lags)
                model.fit(train)  # autolog logs params and covariate metadata

                # log standalone metrics from manual predictions
                pred = model.predict(n=horizon)
                metrics.mae(val, pred)  # time-aggregated logged as scaler metric
                metrics.ae(val, pred)  # time-dependent logged as stepped metric

                # log backtest metrics from historical forecasts over val set
                bt = model.backtest(
                    series=series,
                    start=val.start_time(),
                    retrain=False,
                    forecast_horizon=horizon,
                    reduction=None,  # `None` logs as stepped metric, scalar otherwise
                    metric=[metrics.mae, metrics.rmse],
                )
        autolog(disable=True)

        # you can launch the MLflow UI with the command below from your terminal;
        # then open the returned address for example in a browser (similar to http://localhost:5000)
        print(f"mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")

When ``autolog()`` is enabled, the following functionalities emit detailed logs when inside
an active MLflow run (e.g. within ``with mlflow.start_run():``):

- Calling ``ForecastingModel.fit()``:

  - Logs model creation parameters (``model.model_params``), both as MLflow
    params and as a ``model_params.json`` artifact.
  - Logs target series info and covariate usage information (past, future,
    and static covariates) as a ``series_info.json`` artifact.
  - Stores the trained model artifact when ``log_models=True`` (default:
    ``False``), and sets ``darts.model_is_pretrained`` to ``True``.
  - Logs per-epoch training and validation metrics for PyTorch-based models.
- Calling ``ForecastingModel.historical_forecasts(retrain=True)``:

  - Logs the same model creation parameters and ``series_info.json`` as
    ``fit()`` (overwriting any prior ``fit()`` artifacts in the same run).
  - Stores an untrained model template (via ``untrained_model()``) when
    ``log_models=True`` and no model was logged yet in the run. Sets the run
    tag and model metadata ``darts.model_is_pretrained`` to ``False`` so
    downstream code knows the artifact must be refit before inference.
    If a model was already logged (e.g. via ``fit()`` or manual
    ``log_model()``), the existing artifact is kept unchanged.
- Calling any Darts metric:

  - Logs the result of that metric call as an MLflow metric. More information
    in the notes below.
- Calling ``ForecastingModel.backtest()``:

  - Logs all evaluation metrics under ``backtest_*`` keys. More information
    in the notes below.

.. important::

    **Cross-Model Run Comparability**: Metric values are only comparable
    across runs when the evaluation settings match. Use the same evaluation
    time frame, forecast horizon, and evaluation start date for every
    ``backtest()`` / metric call you intend to compare.

.. note::

    **Metric Naming Convention**: Logged metric keys follow the pattern
    ``{metric_name}{component}{quantile_or_label}`` (e.g.,
    ``"mae_target0_q0_500"``), where each part is included only when the
    corresponding axis is present:

    - ``metric_name`` - the metric function name, or the ``name`` metric
      keyword argument when provided (e.g., ``"mae"``).
    - ``component`` - the component name: e.g., ``"_target0"`` when
      ``component_reduction=None``.
    - ``quantile_or_label``:

      - the quantile label: e.g., ``"_q0.500"`` for quantile metrics with
        keyword argument ``q=[0.5]``
      - the quantile interval label: e.g., ``"_qi0.800"`` for quantile
        interval metrics with keyword argument ``q_interval=[(0.1, 0.9)]``
        (80% interval between quantiles 0.1 and 0.9).
      - the class label: e.g., ``"_label1"`` for classification metrics with
        keyword argument ``labels`` when ``label_reduction=None``.

    **Backtest metrics** are prefixed by ``"backtest_"``.

.. note::

    **Metric Logging and Display**: Darts offers many ways of evaluating
    forecasting models. To offer the highest value, we adapt what is logged
    to MLflow based on the use case scenario.

    - Metric type:

      - Time-aggregated metrics (e.g. ``mae()``): Logged as
        scaler values.
      - Per-time step metrics (e.g. ``ae()`` where ``time_reduction=None``):
        Logged as stepped metrics (one value per step in the forecast horizon).
    - Single or Multi-series:

      - Single series: Logged as explained above.
      - Multiple (a list of) series: When the metric's
        ``series_reduction=None``, the logged metric is aggregated over all
        series using autolog's ``agg_func``. The detailed per-series metrics /
        backtest metrics are logged under a single ``metrics_per_series.json``
        table run artifact.

    - Standalone or backtest metric:

      - Standalone metric: Logged as explained above.
      - Backtest metric: If backtest's ``reduction`` is other than ``None``,
        the windowed forecast metrics are aggregated and logged as explained
        above. If ``None``, metrics are logged as stepped MLflow metrics.

        - Time-aggregated metrics: each step represents a specifc forecast
          (window) metric. Steps represent the historical forecast windows
          (0, 1, ..., n_windows - 1).
        - Per-time step metrics: each step represents a step in the forecast
          horizon aggregated over all windows. Steps represent the steps in
          the forecast horizon (0, 1, ..., horizon - 1).

    - Multi-series alignment (``series_align`` autolog option):

      - ``"end"`` (default): shorter series skip early steps/windows so their
        last points overlap the tail of longer series.
      - ``"start"``: shorter series contribute at early steps/windows; longer
        series have fewer contributing series at tail steps.

    - Backtest aggregate (``log_backtest_aggregate`` autolog option):

      - When enabled, logs an additional scalar per backtest metric key under
        ``backtest_agg_{metric_key}`` (e.g. ``backtest_agg_mae``), computed
        as ``agg_func`` over all logged steps for that key.

    When components are preserved (``component_reduction=None``), all
    series scored together must have the same number of components; names
    are taken from the first series.
"""

import inspect
import re
import sys
import threading
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import Any, Literal

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
from mlflow.models import Model, ModelSignature
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
from darts.dataprocessing.encoders.encoders import SequentialEncoder
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
_MODEL_IS_PRETRAINED_TAG = "darts.model_is_pretrained"

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
    input_example: Any | None = None,
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
    ``signature`` and ``input_example`` params are currently not supported, as they
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
    agg_func: Callable = np.nanmean,
    series_align: Literal["start", "end"] = "end",
    log_backtest_aggregate: bool = False,
    disable: bool = False,
    silent: bool = False,
) -> None:
    """Enable (or disable) automatic MLflow logging for Darts.

    For a detailed overview of logged params, metrics, and artifacts, see
    :ref:`the detailed documentation <darts-mlflow-autolog-logging>`.

    Parameters
    ----------
    log_models
        If ``True``, log a model artifact when calling ``fit()`` (trained
        model) or ``historical_forecasts(retrain=True)`` (untrained template
        via ``untrained_model()``). Defaults to ``False``. If a model was
        already logged in the run (including via manual ``log_model()``),
        autolog does not add another artifact.
    log_params
        If ``True`` (default), log model creation parameters when calling
        ``fit()`` or ``historical_forecasts()``.
    log_metrics
        If ``True`` (default), log the result of any Darts metric call made
        inside an active MLflow run, including standalone metric calls or via
        backtest.
    log_torch_metrics
        If ``True`` (default), enable ``mlflow.pytorch.autolog(log_models=False)``
        around PyTorch-based model training to automatically log per-epoch
        training and validation metrics. Only effective for PyTorch-based models.
    agg_func
        Function used to aggregate a metric's per-series values into the
        single value logged for a list of series (e.g. ``np.nanmean``, the
        default, or ``np.median``). Called as ``agg_func(values)`` on a list
        of floats. Also used when ``log_backtest_aggregate=True`` to collapse
        all logged steps of a backtest metric into a single scalar.
    series_align
        How to align shorter series when multiple series differ in window or
        time length. ``"end"`` (default) aligns from the end so shorter series
        skip early steps/windows; ``"start"`` aligns from the start so shorter
        series contribute at early steps/windows.
    log_backtest_aggregate
        If ``True``, log an additional scalar per backtest metric key under
        ``backtest_agg_{metric_key}`` (e.g. ``backtest_agg_mae``), computed as
        ``agg_func`` over all logged steps for that key. Defaults to ``False``.
    disable
        If ``True``, restore the original ``fit()`` methods and stop
        autologging.
    silent
        If ``True`` (default ``False``), suppress all event logging and warnings from
        MLflow during autologging.
    """
    if series_align not in ("start", "end"):
        raise_log(
            ValueError(
                f"`series_align` must be 'start' or 'end', got {series_align!r}."
            )
        )

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
        series_align=series_align,
        log_backtest_aggregate=log_backtest_aggregate,
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
    agg_func: Callable = np.nanmean,
    series_align: Literal["start", "end"] = "end",
    log_backtest_aggregate: bool = False,
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

        fit_kwargs = inspect.signature(original).bind(self, *args, **kwargs).arguments
        _log_model_setup(
            model=self,
            autologging_client=autologging_client,
            run_id=run_id,
            series=fit_kwargs["series"],
            past_covariates=fit_kwargs.get("past_covariates"),
            future_covariates=fit_kwargs.get("future_covariates"),
            log_params=log_params,
        )
        _log_model_artifact(
            model=result,
            run_id=run_id,
            is_pretrained=True,
            log_models=log_models,
            autologging_client=autologging_client,
        )
        autologging_client.flush(synchronous=False).await_completion()

        return result

    def _patched_historical_forecasts(original, self, *args, **kwargs):
        """Suppress per-iteration fit() autologging; log model setup once when
        ``retrain=True``.

        Sets a thread-local flag so ``_patched_fit`` skips autologging for the
        internal ``fit()`` calls. When retraining is enabled and an MLflow run is
        active, logs model tags, creation parameters, and series info once after
        the call (overwriting any prior ``fit()`` artifacts in the same run).
        When ``log_models=True`` and no model exists yet in the run, also logs
        an untrained model template with ``darts.model_is_pretrained=False``.
        """
        _autolog_state.in_historical_forecasts = True
        try:
            result = original(self, *args, **kwargs)
        finally:
            _autolog_state.in_historical_forecasts = False

        active_run = mlflow.active_run()
        if active_run is None:
            return result

        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()

        retrain = bound.arguments["retrain"]
        if retrain is False or retrain == 0:
            return result

        run_id = active_run.info.run_id
        autologging_client = MlflowAutologgingQueueingClient()
        _log_model_setup(
            model=self,
            autologging_client=autologging_client,
            run_id=run_id,
            series=bound.arguments["series"],
            past_covariates=bound.arguments.get("past_covariates"),
            future_covariates=bound.arguments.get("future_covariates"),
            log_params=log_params,
        )
        _log_model_artifact(
            model=self.untrained_model(),
            run_id=run_id,
            is_pretrained=False,
            log_models=log_models,
            autologging_client=autologging_client,
        )
        autologging_client.flush(synchronous=False).await_completion()
        return result

    def _patched_backtest(original, self, *args, **kwargs):
        """Wrap ``backtest`` to log metric result(s) to the active MLflow run.

        Suppresses per-window metric logging. Delegates to ``_log_backtest_metrics``,
        which infers result shape from the metric signature and logs every cell under
        a descriptive key.
        """
        _autolog_state.in_backtest = True
        try:
            result = original(self, *args, **kwargs)
        finally:
            _autolog_state.in_backtest = False

        active_run = mlflow.active_run()
        if not log_metrics or active_run is None:
            return result

        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        backtest_kwargs = bound.arguments

        autologging_client = MlflowAutologgingQueueingClient()
        _log_backtest_metrics(
            autologging_client=autologging_client,
            run_id=active_run.info.run_id,
            result=result,
            backtest_kwargs=backtest_kwargs,
            agg_func=agg_func,
            series_align=series_align,
            log_backtest_aggregate=log_backtest_aggregate,
        )
        autologging_client.flush(synchronous=False).await_completion()
        return result

    # patch `fit()` to log model setup and model artifact
    for _, cls in _get_forecasting_models():
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=cls,
            function_name="fit",
            patch_function=_patched_fit,
        )

    # patch `historical_forecasts()` to log model setup only once (when retrain=True);
    # suppresses internal fit() call logging
    for _, cls in _get_forecasting_models():
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=cls,
            function_name="historical_forecasts",
            patch_function=_patched_historical_forecasts,
        )

    # patch `backtest()` to log backtest metrics once while suppressing per-window
    # metric logging
    for _, cls in _get_forecasting_models():
        safe_patch(
            autologging_integration=FLAVOR_NAME,
            destination=cls,
            function_name="backtest",
            patch_function=_patched_backtest,
        )


def _mlflow_metric_callback(func, result, args, kwargs) -> None:
    """Metric callback registered with ``darts.metrics.utils`` for autologging.

    Invoked by ``multi_ts_support`` (the outermost decorator on every Darts
    metric) after every top-level metric call, so it fires regardless of how
    the metric was imported. It is not invoked for internal metric-to-metric
    calls (e.g. ``rmse`` calling ``mse`` internally via ``_get_wrapped_metric``),
    since those bypass ``multi_ts_support`` entirely.

    When an active MLflow run exists, infers the output axes from the metric
    signature and call kwargs (via ``_infer_metric_axes``) and delegates to
    ``_log_standalone_metric``, which logs each cell under a key built as::

        {metric_name}{component}{quantile_or_label}

    where:

    - ``metric_name`` - the metric function name, or the ``name`` keyword
      argument when provided (it overrides only this token).
    - ``component`` - ``_{component_name}`` when ``component_reduction=None``.
    - ``quantile_or_label`` - quantile/interval/label suffix (e.g. ``_q0.500``,
      ``_qi0.800``, ``_label1``) when applicable.

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

    func_signature = inspect.signature(func)
    bound = func_signature.bind(
        *args, **{k: v for k, v in kwargs.items() if k in func_signature.parameters}
    )
    bound.apply_defaults()
    metric_kwargs = bound.arguments

    # _mlflow_metric_callback is a bare registered callback, not a closure over
    # autolog()'s call kwargs, so agg_func is read back from the autologging
    # config store that autolog() populated.
    agg_func = get_autologging_config(
        flavor_name=FLAVOR_NAME, config_key="agg_func", default_value=np.nanmean
    )
    series_align = get_autologging_config(
        flavor_name=FLAVOR_NAME, config_key="series_align", default_value="end"
    )

    autologging_client = MlflowAutologgingQueueingClient()
    _log_metric_results(
        autologging_client=autologging_client,
        run_id=active_run.info.run_id,
        result=result,
        metrics=func,
        metric_kwargs=metric_kwargs,
        agg_func=agg_func,
        series_align=series_align,
    )
    autologging_client.flush(synchronous=False).await_completion()


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
    # encoders can add past and future covariates
    encoders = _get_model_encoders(model)

    first_series = get_single_series(series)
    uses_past = model.uses_past_covariates or (
        model.supports_past_covariates
        and (past_covariates is not None or len(encoders.past_encoders) > 0)
    )
    uses_future = model.uses_future_covariates or (
        model.supports_future_covariates
        and (future_covariates is not None or len(encoders.future_encoders) > 0)
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


def _run_has_logged_model(run_id: str) -> bool:
    """Return whether the given run already has a logged model artifact."""
    if getattr(_autolog_state, "model_logged_run_id", None) == run_id:
        return True
    try:
        run = mlflow.get_run(run_id)
        return bool(run.outputs and run.outputs.model_outputs)
    except Exception:
        return False


def _log_model_artifact(
    *,
    model: ForecastingModel,
    run_id: str,
    is_pretrained: bool,
    log_models: bool,
    autologging_client: MlflowAutologgingQueueingClient,
) -> None:
    """Log a Darts model artifact to the active run unless one already exists.

    When autolog skips because the user already logged a model manually, the
    existing artifact and any user-provided metadata/tags are left unchanged.
    """
    if not log_models or _run_has_logged_model(run_id):
        return

    model_name = type(model).__name__
    logged_model: LoggedModel = _initialize_logged_model(
        name=model_name, flavor=FLAVOR_NAME
    )
    metadata = {_MODEL_IS_PRETRAINED_TAG: str(is_pretrained)}
    registered_model_name = get_autologging_config(
        flavor_name=FLAVOR_NAME,
        config_key="registered_model_name",
        default_value=None,
    )
    log_model(
        model,
        name=model_name,
        registered_model_name=registered_model_name,
        model_id=logged_model.model_id,
        metadata=metadata,
    )

    autologging_client.set_tags(
        run_id=run_id,
        tags={_MODEL_IS_PRETRAINED_TAG: str(is_pretrained)},
    )
    _autolog_state.model_logged_run_id = run_id


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
            model=model,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        ),
    )
    if log_params:
        autologging_client.log_params(run_id=run_id, params=model.model_params)
        mlflow.log_dict(model.model_params, "model_params.json")
        _log_series_info(
            model=model,
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

    Extracts information about the target series, past, future, and
    static covariates used during training and logs them as a JSON
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
    first_past_covariates = get_single_series(past_covariates)
    first_future_covariates = get_single_series(future_covariates)
    first_static_covariates = (
        first_series.static_covariates if first_series is not None else None
    )
    uses_past, uses_future, uses_static = _infer_covariate_usage(
        model=model,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )

    # if available, train encoders to get past and future encodings
    encoders = _get_model_encoders(model)
    if encoders.encoding_available:
        _ = encoders.encode_train(
            target=first_series,
            past_covariates=first_past_covariates,
            future_covariates=first_future_covariates,
        )

    series_info = {
        "series": _extract_data_metadata(
            uses=True,
            data=first_series,
            names_attr="components",
        ),
        "past_covariates": _extract_data_metadata(
            uses=uses_past,
            data=first_past_covariates,
            names_attr="components",
            encodings=encoders.past_components,
        ),
        "future_covariates": _extract_data_metadata(
            uses=uses_future,
            data=first_future_covariates,
            names_attr="components",
            encodings=encoders.future_components,
        ),
        "static_covariates": _extract_data_metadata(
            uses=uses_static,
            data=first_static_covariates,
            names_attr="columns",
        ),
    }

    # log complete information as JSON artifact
    mlflow.log_dict(series_info, "series_info.json")


def _is_torch_model(model) -> bool:
    """Check if a model is a ``TorchForecastingModel``.

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


def _extract_data_metadata(
    uses: bool,
    data: TimeSeries | pd.DataFrame | None,
    names_attr: str,
    encodings: pd.Index | None = None,
) -> dict:
    """Extract data metadata.

    Encodings are logged under the "encodings" key since they are not part
    of the input series.

    Parameters
    ----------
    uses
        Whether the model uses this data type.
    data
        The data from a single series: a ``TimeSeries`` (target or past/future covariates)
        or a static-covariates ``DataFrame``, or ``None``.
    names_attr
        Attribute holding the feature names ("components" for a
        ``TimeSeries``, "columns" for a static-covariates ``DataFrame``).
    encodings
        The encodings for the data: a ``pd.Index`` of encoding names, or ``None``.

    Returns
    -------
    dict
        Dictionary with keys: "used" (bool), "count" (int), "names" (list).
    """
    info = {"used": False, "count": 0, "names": [], "encodings": []}

    if uses:
        info["used"] = True
        names = list(getattr(data, names_attr)) if data is not None else []
        info["names"] = names
        info["count"] = len(names)
        if encodings is not None:
            info["encodings"] = list(encodings)

    return info


def _sanitize_mlflow_key(name: str) -> str:
    """Sanitize a string for use as an MLflow metric key.

    Replaces any invalid character from the metric key. Valid keys may
    only contain slashes, alphanumerics, underscores, periods, dashes,
    and spaces.

    Parameters
    ----------
    name
        The raw name to sanitize.

    Returns
    -------
    str
        A string safe for use as an MLflow metric key.
    """
    return re.sub(r"[^/\w.\- ]", "_", name)


def _get_model_encoders(model: ForecastingModel) -> SequentialEncoder:
    """Get the encoders for a model.

    Gets either the trained encoder of a fitted model or the
    fresh encoders of a model that has not been fitted yet.

    Parameters
    ----------
    model
        A Darts forecasting model instance.

    Returns
    -------
    SequentialEncoder
        The encoders for the model.
    """
    if model.add_encoders and not model.encoders.encoding_available:
        # model has encoders but it was not fitted yet
        return model.initialize_encoders()
    # model has been fitted, or does not have encodings available
    return model.encoders


def _log_backtest_metrics(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    result,
    backtest_kwargs: dict,
    agg_func: Callable = np.nanmean,
    series_align: Literal["start", "end"] = "end",
    log_backtest_aggregate: bool = False,
) -> None:
    """Log backtest metric result(s) to MLflow.

    Helper function for ``autolog()`` to log backtest metric results to MLflow.
    Details are documented in ``_log_metric_results()``.
    """
    _log_metric_results(
        autologging_client=autologging_client,
        run_id=run_id,
        result=result,
        metrics=backtest_kwargs["metric"],
        metric_kwargs=backtest_kwargs["metric_kwargs"] or dict(),
        agg_func=agg_func,
        series_align=series_align,
        log_backtest_aggregate=log_backtest_aggregate,
        backtest_kwargs=backtest_kwargs,
    )


def _log_metric_results(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    result,
    metrics: Callable | list[Callable],
    metric_kwargs: dict[str, Any] | list[dict[str, Any]],
    backtest_kwargs: dict[str, Any] | None = None,
    agg_func: Callable = np.nanmean,
    series_align: Literal["start", "end"] = "end",
    log_backtest_aggregate: bool = False,
) -> None:
    """Log backtest or standalone metric result(s) to MLflow.

    Shared implementation used by ``_log_backtest_metrics`` and
    ``_log_standalone_metric``. The two entry points differ only in how they
    populate the arguments below; all reshaping, key construction, step
    assignment, multi-series aggregation, and artifact writing happen here.

    Entry modes
    -----------
    **Backtest** (``backtest_kwargs`` is not ``None``):

    - ``metrics`` / ``metric_kwargs`` come from backtesting.
    - MLflow keys are prefixed with ``backtest_``.
    - A window axis ``W`` may be present (see *MLflow keys and steps*).

    **Standalone** (``backtest_kwargs`` is ``None``):

    - ``metrics`` / ``metric_kwargs`` describe a direct ``metric()`` call.
    - MLflow keys have no prefix.
    - There is never a backtest window axis (``W=1``); only the metric's own
      time / component / quantile / label axes apply.

    Canonical layout
    ----------------
    Each per-series result is reshaped to ``(W, T, C, M)``:

    - ``W`` - backtest windows (``1`` for standalone, or when windows were
      aggregated before scoring).
    - ``T`` - timesteps (``1`` when ``time_reduction`` is set).
    - ``C`` - sub-metrics per component (components * quantiles / intervals /
      labels; ``1`` when ``component_reduction`` is set).
    - ``M`` - number of metrics when several are logged at once.

    Axis sizes are inferred from each metric's signature and the corresponding
    ``metric_kwargs``.

    Kwargs that affect output dimensions (both modes unless noted):

    - ``time_reduction`` - collapses the time axis (``T=1``).
    - ``component_reduction`` - collapses the component axis (``C=1``).
    - ``q`` / ``q_interval`` - one sub-metric per quantile / interval.
    - ``labels`` - when ``label_reduction=None``, one sub-metric per label;
      otherwise ``labels`` only restricts which classes are scored.
    - ``label_reduction`` - collapses label outputs to a scalar per component.
    - ``series_reduction`` - collapses the series axis inside the metric, so
      the caller's ``result`` is already aggregated and treated as a single
      series (``W=1`` for backtest regardless of ``reduction``).

    Backtest-only kwargs:

    - ``reduction=None`` - no aggregation across windows → one value per
      window (``W > 1``).
    - ``last_points_only`` - concatenates all windows into one ``TimeSeries``
      before scoring, so there is effectively one window regardless of
      ``reduction``.
    - ``forecast_horizon`` / ``historical_forecasts`` - set the time-axis
      length for time-dependent metrics when windows are preserved.

    MLflow keys and steps
    ---------------------
    Every scalar cell is logged under a descriptive key built from the metric
    name, optional component suffix, and optional quantile / interval / label
    suffix (see ``_build_metric_keys``). A metric's ``name`` entry in
    ``metric_kwargs`` overrides the function-name token; the ``backtest_``
    prefix (when present) and axis suffixes are preserved.

    The MLflow ``step`` is the axis the UI should chart:

    - **Time-dependent metrics** (``time_reduction=None``): steps index the
      forecast horizon (``0 .. T-1``). When both time and window axes are
      present (backtest, no reduction, not ``last_points_only``), each step
      holds the ``agg_func`` aggregation over windows at that horizon index.
    - **Time-aggregated metrics**: steps index backtest windows aligned across
      series (``0 .. max_W-1``; shorter series skip early or late steps depending
      on ``series_align``). Standalone calls with a time axis but no window axis
      align timesteps the same way across series of different lengths.

    Series can differ in length and overlap in time. By default
    (``series_align="end"``), we align from the end: a shorter series contributes
    at the last steps/windows, not the first. With ``series_align="start"``,
    shorter series contribute at the first steps/windows instead. This matches
    the usual backtest layout but is an assumption — series may also end at
    different calendar times.

    Multi-series aggregation and artifacts
    ----------------------------------------
    When more than one series is scored (and ``series_reduction`` did not
    already collapse them), the logged value at each ``(key, step)`` is
    ``agg_func`` applied over series. The per-series breakdown is appended to
    the run's ``metrics_per_series.json`` table artifact when:

    - several series were scored together, or
    - (backtest only) a time-dependent metric retains both time and window axes,
      so the aggregate steps average over windows but the table keeps every
      ``(window, horizon)`` cell.

    For a single series, the aggregate is the value itself and no table is
    written unless the backtest time+window case above applies. When components
    are preserved (``component_reduction=None``), all series in a batch must
    have the same number of components; component names are taken from the first
    series.

    When ``log_backtest_aggregate`` is enabled (backtest only), an additional
    scalar per metric key is logged under ``backtest_agg_{metric_key}`` (e.g.
    ``backtest_agg_mae``), computed as ``agg_func`` over all logged steps for
    that key.

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
        Return value of ``backtest()`` or ``metric()``. Either a scalar/array
        for one series, a list of per-series results, or a single aggregate when
        ``series_reduction`` is set.
    metrics
        One metric callable or a list of callables logged together.
    metric_kwargs
        Keyword arguments forwarded to ``metrics``, either one dict shared by
        all metrics or one dict per metric. For standalone logging this is the
        bound ``metric()`` arguments; for backtest logging this is
        ``backtest_kwargs["metric_kwargs"]``.
    backtest_kwargs
        Bound arguments of the ``backtest()`` call (from
        ``inspect.BoundArguments.arguments`` after ``apply_defaults``). When
        ``None``, the call is treated as standalone metric logging.
    agg_func
        Function used to aggregate per-series values at each ``(key, step)``.
        Called as ``agg_func(values)`` on a list of floats. Also used to
        collapse all logged steps into a single scalar when
        ``log_backtest_aggregate`` is enabled.
    series_align
        How to align shorter series when multiple series differ in window or
        time length. ``"end"`` (default) or ``"start"``.
    log_backtest_aggregate
        If ``True`` (backtest only), log an additional scalar per metric key
        under ``backtest_agg_{metric_key}``.
    """
    metrics, metric_kwargs, metric_names = _normalize_metrics(
        metrics=metrics,
        metric_kwargs=metric_kwargs,
    )
    metric_axes = [_infer_metric_axes(m, kw) for m, kw in zip(metrics, metric_kwargs)]
    has_time_axis, has_comp_axis, _ = metric_axes[0]

    series: TimeSeriesLike = (
        metric_kwargs[0]["actual_series"]
        if backtest_kwargs is None
        else backtest_kwargs["series"]
    )
    is_single_series = get_series_seq_type(series) == SeriesType.SINGLE
    series = series2seq(series)

    if backtest_kwargs is None:
        # standalone metric
        is_backtest = False
        prefix = ""
        # `series_reduction` collapses the series axis inside the metric, so the
        # result has no leading series axis even for list input.
        series_reduced = metric_kwargs[0]["series_reduction"] is not None
        last_points_only = False
        has_windows = False
        forecast_horizon = 0
    else:
        # backtest metric
        is_backtest = True
        prefix = "backtest_"
        series_reduced = False
        last_points_only = backtest_kwargs["last_points_only"]
        has_windows, forecast_horizon = _resolve_backtest_layout(
            backtest_kwargs=backtest_kwargs,
            metric=metrics[0],
            metric_kwargs=metric_kwargs[0],
            is_single_series=is_single_series,
        )

    if series_reduced:
        results = [result]
    else:
        results = [result] if is_single_series else result

    if has_comp_axis and not series_reduced:
        n_components = {s.n_components for s in series}
        if len(n_components) > 1:
            raise_log(
                ValueError(
                    "Backtest metric logging failed: all series must have the same "
                    f"number of components, got {sorted(n_components)}. Consider "
                    f"setting a metric `component_reduction`, or make sure all series "
                    f"have the same number of components."
                )
            )

    metric_keys = _build_metric_keys(
        metric_names=metric_names,
        components=series[0].components.tolist(),
        has_comp_axis=has_comp_axis,
        metric_axes=metric_axes,
        prefix=prefix,
    )
    series_metrics = [
        _reshape_metric_result(
            series=series_,
            result=r,
            metric_keys=metric_keys,
            has_time_axis=has_time_axis,
            has_windows=has_windows,
            forecast_horizon=forecast_horizon,
            is_backtest=is_backtest,
        )
        for r, series_ in zip(results, series)
    ]
    stepped_metrics, detailed_metrics = _collect_stepped_and_detailed_metrics(
        series_metrics=series_metrics,
        metric_keys=metric_keys,
        has_time_axis=has_time_axis,
        has_windows=has_windows,
        last_points_only=last_points_only,
        series_align=series_align,
    )

    write_table = (not series_reduced and not is_single_series) or (
        has_time_axis and has_windows
    )
    _flush_logged_metrics(
        autologging_client,
        run_id,
        stepped_metrics,
        agg_func=agg_func,
        log_backtest_aggregate=log_backtest_aggregate and is_backtest,
        table_rows=detailed_metrics if write_table else None,
    )


def _normalize_metrics(
    metrics: Callable | list[Callable],
    metric_kwargs: dict[str, Any] | list[dict[str, Any]],
) -> tuple[list, list[dict], list[str]]:
    """Normalize ``metric`` / ``metric_kwargs`` to parallel lists and key names.

    ``backtest()`` accepts a single metric or a list, and a single kwargs
    dict (broadcast to all metrics) or a list of dicts. A ``name`` entry in
    ``metric_kwargs`` overrides the metric-name token in the MLflow key.
    """
    metrics = metrics if isinstance(metrics, list) else [metrics]
    metric_kwargs = (
        metric_kwargs if isinstance(metric_kwargs, list) else [metric_kwargs]
    )
    if len(metric_kwargs) != len(metrics):
        metric_kwargs = [metric_kwargs[0]] * len(metrics)

    metric_names = [
        _sanitize_mlflow_key(
            metric_kwargs[i].get("name") or getattr(m, "__name__", f"metric_{i}")
        )
        for i, m in enumerate(metrics)
    ]
    return metrics, metric_kwargs, metric_names


def _resolve_backtest_layout(
    backtest_kwargs: dict,
    metric: Callable,
    metric_kwargs: dict,
    is_single_series: bool,
) -> tuple[bool, int | None]:
    """Infer whether a window axis is present and the forecast horizon.

    A window axis is present only when backtest did not aggregate windows
    (``reduction is None``), forecasts were not concatenated first
    (``last_points_only`` is false), and the metric's own
    ``series_reduction`` did not already collapse windows.

    When the caller passed ``historical_forecasts``, ``backtest()`` ignores
    ``forecast_horizon``, so the horizon is read from the first forecast.

    Returns
    -------
    tuple
        ``(has_windows, forecast_horizon)``.
    """
    last_points_only = backtest_kwargs["last_points_only"]
    has_windows = backtest_kwargs["reduction"] is None and not last_points_only

    params = inspect.signature(metric).parameters
    if has_windows and "series_reduction" in params:
        series_reduction = metric_kwargs.get(
            "series_reduction", params["series_reduction"].default
        )
        if series_reduction is not None:
            has_windows = False

    forecast_horizon = backtest_kwargs["forecast_horizon"]
    historical_forecasts = backtest_kwargs["historical_forecasts"]
    if historical_forecasts is not None and not last_points_only:
        first_forecast = (
            historical_forecasts[0] if is_single_series else historical_forecasts[0][0]
        )
        forecast_horizon = len(first_forecast)

    return has_windows, forecast_horizon


def _infer_metric_axes(
    metric: Callable, metric_kwargs: dict
) -> tuple[bool, bool, list[str]]:
    """Infer a metric's output axes from its signature and ``metric_kwargs``.

    Covers ``time_reduction``, ``component_reduction``, ``q``, ``q_interval``,
    and ``label_reduction`` / ``labels`` for classification metrics.
    ``series_reduction`` is handled at the ``_log_metric_results`` level.

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

        - ``has_time_axis`` - ``True`` when ``time_reduction`` is ``None`` (i.e. a
          per-timestep axis is present in the output).
        - ``has_comp_axis`` - ``True`` when components are expanded (not collapsed to a scalar).
        - ``axis_labels`` - one key suffix per quantile/interval/label entry.

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
        axis_labels = [f"_qi{(hi - lo):.3f}" for lo, hi in intervals]
    elif "q" in params and q is not None:
        axis_labels = [f"_q{v:.3f}" for v in np.atleast_1d(np.array(q, dtype=float))]
    elif "label_reduction" in params:
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

    return has_time_axis, has_comp_axis, axis_labels


def _build_metric_keys(
    metric_names: list[str],
    components: list[str],
    has_comp_axis: bool,
    metric_axes: list[tuple[bool, bool, list[str]]],
    *,
    prefix: str = "",
) -> list[list[str]]:
    """Build sanitized MLflow metric keys for each metric x component x axis label.

    Parameters
    ----------
    metric_names
        One sanitized metric-name token per metric.
    components
        Component names from the first series (used only when ``has_comp_axis``).
    has_comp_axis
        Whether components are preserved in the metric output.
    metric_axes
        Per-metric ``(has_time_axis, has_comp_axis, axis_labels)`` tuples from
        ``_infer_metric_axes``. Axis size is taken from the first entry.
    prefix
        Optional key prefix (e.g. ``"backtest_"``).

    Returns
    -------
    list[list[str]]
        A list of lists of metric keys, where each inner list contains the sub-metric
        keys for a single metric. ``keys[m][i]`` is the sanitized key for metric index
        ``m`` and sub-metric index ``i``. Sub-metric index ``i`` is the component-label
        index walking axis labels innermost, then components (e.g. ``[_q0.100_comp1,
        _q0.900_comp1, _q0.100_comp2, _q0.900_comp2]``).
    """
    component_names = components if has_comp_axis else [None]

    metric_keys: list[list[str]] = []
    for metric_idx, metric_name in enumerate(metric_names):
        labels = metric_axes[metric_idx][2]
        sub_metric_keys: list[str] = []
        for component in component_names:
            component_suffix = (
                f"_{_sanitize_mlflow_key(component)}" if component is not None else ""
            )
            for label in labels:
                sub_metric_keys.append(
                    _sanitize_mlflow_key(
                        f"{prefix}{metric_name}{component_suffix}{label}"
                    )
                )
        metric_keys.append(sub_metric_keys)
    return metric_keys


def _reshape_metric_result(
    series: TimeSeries,
    result,
    *,
    metric_keys: list[list[str]],
    has_time_axis: bool,
    has_windows: bool,
    forecast_horizon: int | None,
    is_backtest: bool,
) -> np.ndarray:
    """Reshape one series' backtest result to canonical ``(W, T, C, M)``.

    - ``C`` is ``n_sub_metrics`` (n_components * n_quantiles/n_labels)
    - ``M`` is ``n_metrics``
    - ``T`` is ``1`` if not present, and otherwise the ``forecast_horizon`` or
      the length of the series (with ``last_points_only=True``).
    - ``W`` is ``1`` if not present, and otherwise the number of windows

    After stripping the ``C`` and ``M`` axes, the leftover element count is split into
    ``(t_size, w_size)``:

    - time axis and windows: ``T = forecast_horizon``, ``W = rest / T``
    - time axis only: ``T = rest``, ``W = 1``
    - windows only: ``T = 1``, ``W = rest``
    - neither: ``T = W = 1`` (``rest`` must be 1)

    Returns
    -------
    np.ndarray
        Series metric values with shape ``(W, T, C, M)``.
    """
    arr = np.asarray(result, dtype=series.dtype)
    n_metrics = len(metric_keys)
    n_sub_metrics = len(metric_keys[0])
    n_round_multiples, remainder = divmod(arr.size, n_sub_metrics * n_metrics)
    if remainder:
        raise_log(
            ValueError(
                f"{'Backtest metric' if is_backtest else 'Metric'} logging failed: "
                f"result size ({arr.size}) is not divisible by n_sub_metrics * "
                f"n_metrics ({n_sub_metrics} * {n_metrics} = {n_sub_metrics * n_metrics}). "
                f"The metric output shape does not match the inferred axes."
            )
        )

    if has_time_axis and has_windows:
        # T is the forecast horizon; W is recovered from the leftover length;
        # time-dependent metrics + last_points_only=False + no reduction
        assert forecast_horizon is not None
        t_size, w_size = forecast_horizon, n_round_multiples // forecast_horizon
    elif has_time_axis:
        # time-dependent metrics + last_points_only=False + reduction
        t_size, w_size = n_round_multiples, 1
    elif has_windows:
        # time-aggregated metrics + last_points_only=False + no reduction
        t_size, w_size = 1, n_round_multiples
    elif n_round_multiples == 1:
        # time-aggregated metrics + reduction
        t_size, w_size = 1, 1
    else:
        raise_log(
            ValueError(
                f"{'Backtest metric' if is_backtest else 'Metric'} logging failed: "
                f"expected a single scalar per component/metric after reduction, but got "
                f"{n_round_multiples} elements. Check time_reduction and "
                "component_reduction defaults."
            )
        )

    return arr.reshape(w_size, t_size, n_sub_metrics, n_metrics)


def _collect_stepped_and_detailed_metrics(
    series_metrics: list[np.ndarray],
    metric_keys: list[list[str]],
    *,
    has_time_axis: bool,
    has_windows: bool,
    last_points_only: bool,
    series_align: Literal["start", "end"] = "end",
) -> tuple[dict[tuple[str, int], list[float]], list[dict]]:
    """Parse series metrics and build inputs for MLflow stepped metrics and detailed metric table.

    Each ``series_metrics`` array has shape ``(w_size, t_size, n_metrics * n_sub_metrics)``.
    Shorter series are aligned from the start or end of the longest remaining axis
    (windows, or time when there is no window axis), controlled by ``series_align``.

    The MLflow ``step`` is the axis the UI should chart, always ``0 .. n-1``:

    - forecast-horizon index when a time axis is present (time-dependent metrics)
      - if windows are present: each horizon step is aggregated over the windows
      - if multi-series: each horizon step is aggregated over the series
    - window index otherwise (time-aggregated metrics), aligned across series
      via ``series_align``
      - if multi-series: each window is aggregated over the series
    """
    stepped_metrics: dict[tuple[str, int], list[float]] = {}
    detailed_metrics: list[dict] = []

    w_axis, t_axis = 0, 1
    max_w_size = max((values.shape[w_axis] for values in series_metrics), default=0)
    max_t_size = max((values.shape[t_axis] for values in series_metrics), default=0)

    for series_index, values in enumerate(series_metrics):
        w_size = values.shape[w_axis]
        t_size = values.shape[t_axis]

        # pad shorter series so their last (end) or first (start) point lines up
        if series_align == "end":
            w_offset = max_w_size - w_size if has_windows else 0
            t_offset = max_t_size - t_size if has_time_axis and not has_windows else 0
        else:
            w_offset = 0
            t_offset = 0

        for metric_idx, sub_metric_keys in enumerate(metric_keys):
            for sub_metric_idx, key in enumerate(sub_metric_keys):
                for w in range(w_size):
                    window_index = w + w_offset if has_windows else None
                    for t in range(t_size):
                        if has_time_axis and (has_windows or not last_points_only):
                            # bt: backtest, sm: standalone metric
                            # (bt) time-dependent metrics + last_points_only=False + no reduction
                            # (bt) time-dependent metrics + last_points_only=False + reduction
                            # (sm) time-dependent metrics
                            step = t
                        elif has_time_axis:
                            # (bt) time-dependent metrics + last_points_only=True + no reduction
                            step = t + t_offset
                        else:
                            # (bt) last_points_only=False + reduction (window index is None)
                            # (bt) last_points_only=False + no reduction (window index is not None)
                            # (sm) time-aggregated metrics (window index is None)
                            step = window_index

                        value = float(values[w, t, sub_metric_idx, metric_idx])
                        # table: one value per window and horizon
                        detailed_metrics.append({
                            "key": key,
                            "series_index": series_index,
                            "step": step,
                            "window_index": window_index,
                            "value": value,
                        })
                        # stepped metric chart: one value per window or horizon
                        stepped_metrics.setdefault((key, step), []).append(value)

    return stepped_metrics, detailed_metrics


def _flush_logged_metrics(
    autologging_client: MlflowAutologgingQueueingClient,
    run_id: str,
    agg: dict[tuple[str, int], list[float]],
    agg_func: Callable,
    log_backtest_aggregate: bool = False,
    table_rows: list[dict] | None = None,
) -> None:
    """Aggregate per-series cells, log MLflow metrics, and optionally write the
    per-series table artifact.

    Parameters
    ----------
    autologging_client
        MLflow autologging client used to queue metric writes.
    run_id
        ID of the active MLflow run.
    agg
        Map of ``(key, step) -> list of per-series float values``.
    agg_func
        Aggregation over the per-series values for each ``(key, step)``.
    log_backtest_aggregate
        If ``True``, log an additional scalar per backtest key under
        ``backtest_agg_{metric_key}`` as ``agg_func`` over all logged steps.
    table_rows
        Granular cells for ``metrics_per_series.json``. ``None`` skips writing
        the table artifact.
    """
    metrics_by_step: dict[int, dict[str, float]] = {}
    backtest_agg: dict[str, list[float]] = {}
    for (key, step), values in agg.items():
        step_agg = float(agg_func(values))
        metrics_by_step.setdefault(step, {})[key] = step_agg

        if log_backtest_aggregate and key.startswith("backtest_"):
            agg_key = key.replace("backtest_", "backtest_agg_", 1)
            backtest_agg.setdefault(agg_key, []).append(step_agg)

    for step, metrics in metrics_by_step.items():
        autologging_client.log_metrics(run_id=run_id, metrics=metrics, step=step)

    if backtest_agg:
        aggregate_metrics = {
            agg_key: float(agg_func(step_values))
            for agg_key, step_values in backtest_agg.items()
        }
        autologging_client.log_metrics(run_id=run_id, metrics=aggregate_metrics, step=0)

    if table_rows is not None:
        _log_per_series_table(table_rows)


def _log_per_series_table(rows: list[dict]) -> None:
    """Append the granular per-series metric breakdown to a single, run-wide
    table artifact.

    Each row is a single metric cell for one series, with columns ``key`` (the
    aggregate MLflow key, without any series suffix), ``series_index``, ``step``
    (the time or window index charted by MLflow), ``window_index`` (the
    source backtest window ``0 .. max_w - 1`` aligned per ``series_align``, or
    ``None`` when there is no window axis), and ``value``. All calls within a run
    append to the same ``metrics_per_series.json`` artifact.
    Used when more than one series is scored, since the logged metric keys
    only carry the aggregate over series.

    Parameters
    ----------
    rows
        One dict per metric cell with keys ``key``, ``series_index``, ``step``,
        ``window_index``, and ``value``.
    """
    if not rows:
        return

    df = pd.DataFrame(rows)
    sort_by = ["key", "series_index", "step"]
    if "window_index" in df.columns:
        sort_by.append("window_index")
    df = df.sort_values(sort_by)
    mlflow.log_table(data=df, artifact_file="metrics_per_series.json")
