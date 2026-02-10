"""
MLflow Integration for Darts
-----------------------------

Custom MLflow model flavor for darts forecasting models. Supports saving, loading,
logging and autolog for any darts ``ForecastingModel`` (statistical, ML-based, and PyTorch-based)
to MLflow, including a pyfunc wrapper for MLflow's generic inference API.
"""

import importlib
import os
import sys
import tempfile
from functools import wraps
from typing import Any, Optional, Union

import mlflow
import pandas as pd
import yaml
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

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
    reqs = [f"darts=={darts.__version__}"]
    if is_torch:
        reqs.extend(["torch>=2.0.0", "pytorch-lightning>=2.0.0"])
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
    return {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python",
            "pip",
            {
                "pip": get_default_pip_requirements(is_torch=is_torch),
            },
        ],
        "name": "darts_env",
    }


def save_model(
    model,
    path: str,
    conda_env: Optional[Union[dict, str]] = None,
    pip_requirements: Optional[list[str]] = None,
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
    pip_requirements
        A list of pip requirement strings. Overrides ``conda_env`` pip section
        when provided.
    signature
        An ``mlflow.models.ModelSignature`` instance describing model input/output.
    input_example
        An example input for the model (used by MLflow UI).
    metadata
        Optional dictionary of custom metadata to store in the ``MLmodel`` file.
    """
    data_dir = os.path.join(path, _MODEL_DATA_SUBFOLDER)

    is_torch = _is_torch_model(model)

    os.makedirs(path, exist_ok=True)
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

    if pip_requirements is not None:
        pip_reqs = pip_requirements
    else:
        pip_reqs = get_default_pip_requirements(is_torch=is_torch)

    if conda_env is not None:
        if isinstance(conda_env, str):
            with open(conda_env) as f:
                conda_env_dict = yaml.safe_load(f)
        else:
            conda_env_dict = conda_env
    else:
        conda_env_dict = get_default_conda_env(is_torch=is_torch)

    conda_path = os.path.join(path, "conda.yaml")
    with open(conda_path, "w") as f:
        yaml.dump(conda_env_dict, f, default_flow_style=False)

    reqs_path = os.path.join(path, "requirements.txt")
    with open(reqs_path, "w") as f:
        f.write("\n".join(pip_reqs) + "\n")

    python_env_path = os.path.join(path, "python_env.yaml")
    python_env = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "build_dependencies": ["pip"],
        "dependencies": ["requirements.txt"],
    }
    with open(python_env_path, "w") as f:
        yaml.dump(python_env, f, default_flow_style=False)
    pyfunc_conf = {
        "loader_module": "darts.utils.mlflow",
        "python_model": None,
        "data": _MODEL_DATA_SUBFOLDER,
        "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
    }

    mlmodel = Model()
    mlmodel.add_flavor(FLAVOR_NAME, **darts_flavor_conf)
    mlmodel.add_flavor("python_function", **pyfunc_conf)

    if signature is not None:
        mlmodel.signature = signature

    if input_example is not None:
        mlmodel.input_example = input_example

    if metadata is not None:
        mlmodel.metadata = metadata

    mlmodel.save(os.path.join(path, "MLmodel"))


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
    mlmodel_path = os.path.join(local_path, "MLmodel")
    mlmodel = Model.load(mlmodel_path)

    if FLAVOR_NAME not in mlmodel.flavors:
        raise ValueError(
            f"The MLflow model at '{model_uri}' does not have a '{FLAVOR_NAME}' flavor. "
            f"Available flavors: {list(mlmodel.flavors.keys())}"
        )

    flavor_conf = mlmodel.flavors[FLAVOR_NAME]
    module_path = flavor_conf["model_class_module"]
    class_name = flavor_conf["model_class_name"]
    model_file = flavor_conf["model_file"]
    is_torch = flavor_conf.get("is_torch_model", False)

    model_cls = _import_model_class(module_path, class_name)

    data_dir = os.path.join(local_path, flavor_conf.get("data", _MODEL_DATA_SUBFOLDER))
    model_path = os.path.join(data_dir, model_file)

    if is_torch:
        loaded_model = model_cls.load(model_path, **kwargs)
    else:
        loaded_model = model_cls.load(model_path)

    return loaded_model


def log_model(
    model,
    artifact_path: Optional[str] = None,
    name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    conda_env: Optional[Union[dict, str]] = None,
    pip_requirements: Optional[list[str]] = None,
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
    pip_requirements
        Pip requirements list.
    signature
        An ``mlflow.models.ModelSignature``.
    input_example
        An example model input.
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = os.path.join(tmp_dir, artifact_name)
        save_model(
            model=model,
            path=model_dir,
            conda_env=conda_env,
            pip_requirements=pip_requirements,
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
    MLflow run. Logs non-serializable values as "<non-serialisable>" and truncates long parameter values
    to fit MLflow's 500-character limit.

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
            str_val = str(value)
            if len(str_val) > 500:  # MLflow param value limit
                str_val = str_val[:497] + "..."
            safe_params[key] = str_val
        except Exception:
            safe_params[key] = "<non-serialisable>"

    if safe_params:
        mlflow.log_params(safe_params)


# ---------------------------------------------------------------------------
# PyFunc interface
# ---------------------------------------------------------------------------


class _DartsModelWrapper:
    """A pyfunc-compatible wrapper around a darts ``ForecastingModel``.

    MLflow's generic pyfunc inference API requires a ``predict(model_input)``
    method that accepts a ``pandas.DataFrame``.  This wrapper translates that
    call into a darts ``model.predict(n=...)`` call and returns the forecast
    as a ``pandas.DataFrame``.

    The forecast horizon ``n`` is determined from (in order of precedence):

    1. A key ``"n"`` in the ``params`` dict, or
    2. A column named ``"n"`` in ``model_input`` (first value is used), or
    3. A default of ``1``.

    Parameters
    ----------
    model : ForecastingModel
        A fitted Darts forecasting model instance.

    Attributes
    ----------
    model : ForecastingModel
        The wrapped Darts forecasting model.
    """

    def __init__(self, model):
        self.model = model

    def predict(
        self,
        model_input: pd.DataFrame,
        params: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Parameters
        ----------
        model_input
            A ``pandas.DataFrame``. If it contains a column ``"n"``, the first
            value is used as the forecast horizon.  Additional columns
            (``"num_samples"``) are forwarded if present.
        params
            Optional dict; ``params["n"]`` overrides the horizon from the
            DataFrame.

        Returns
        -------
        pd.DataFrame
            Forecasted values with a ``DatetimeIndex`` (or ``RangeIndex``)
            and one column per component of the target series.
        """
        n = 1
        if params and "n" in params:
            n = int(params["n"])
        elif "n" in model_input.columns:
            n = int(model_input["n"].iloc[0])

        num_samples = 1
        if params and "num_samples" in params:
            num_samples = int(params["num_samples"])
        elif "num_samples" in model_input.columns:
            num_samples = int(model_input["num_samples"].iloc[0])

        prediction = self.model.predict(n=n, num_samples=num_samples)
        return prediction.to_dataframe()


def _load_pyfunc(path: str):
    """Load a darts model as a pyfunc-compatible wrapper.

    This function is called by ``mlflow.pyfunc.load_model()`` when loading a model
    saved with the darts flavor. It reconstructs the original Darts model and wraps
    it in a ``_DartsModelWrapper`` for MLflow's generic inference API.

    Parameters
    ----------
    path : str
        Path to the model directory containing the MLmodel file.

    Returns
    -------
    _DartsModelWrapper
        A wrapper with a ``predict()`` method compatible with MLflow's
        pyfunc inference API.

    Raises
    ------
    FileNotFoundError
        If the MLmodel file cannot be found at the specified path.
    """
    mlmodel_path = os.path.join(path, "..", "MLmodel")
    if os.path.exists(mlmodel_path):
        parent_path = os.path.dirname(os.path.abspath(mlmodel_path))
    else:
        parent_path = path

    mlmodel_file = os.path.join(parent_path, "MLmodel")
    if not os.path.exists(mlmodel_file):
        raise FileNotFoundError(
            f"Cannot find MLmodel file. Searched at: {mlmodel_file}"
        )

    with open(mlmodel_file) as f:
        mlmodel_dict = yaml.safe_load(f)

    flavor_conf = mlmodel_dict["flavors"][FLAVOR_NAME]
    module_path = flavor_conf["model_class_module"]
    class_name = flavor_conf["model_class_name"]
    model_file = flavor_conf["model_file"]
    is_torch = flavor_conf.get("is_torch_model", False)

    model_cls = _import_model_class(module_path, class_name)

    data_dir = os.path.join(parent_path, flavor_conf.get("data", _MODEL_DATA_SUBFOLDER))
    model_path = os.path.join(data_dir, model_file)

    if is_torch:
        loaded_model = model_cls.load(model_path)
    else:
        loaded_model = model_cls.load(model_path)

    return _DartsModelWrapper(loaded_model)


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

    Examples
    --------
    .. code-block:: python

        from darts.utils.mlflow import autolog
        from darts.models import ExponentialSmoothing
        from darts.datasets import AirPassengersDataset

        autolog()  # enable

        series = AirPassengersDataset().load()
        model = ExponentialSmoothing()
        model.fit(series)  # automatically logged to MLflow

        autolog(disable=True)  # disable
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
