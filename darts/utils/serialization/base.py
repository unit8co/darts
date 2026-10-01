"""
Model Serialization Utilities (core)
------------------------------------

Shared helpers for Darts model persistence.

Security model:

- Prefer **scoped** allow-lists applied only around a single load call, not process-wide
  registration at import time.
- When an inspection API exists, derive the allow-list **from the specific artifact** being
  loaded (checkpoint-driven registration) rather than pre-registering entire libraries.
- Use class-hierarchy-based allowlisting: allow classes that subclass known-safe bases
  rather than blanket-trusting entire package prefixes.
- For non-torch models, use a :class:`RestrictedUnpickler` with ``find_class`` filtering
  that shares the same allowlist logic.
"""

import enum
import importlib
import inspect
import pickle
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeVar

from darts.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Prefixes for trusted packages when filtering checkpoint-referenced globals.
TrustedPrefixPolicy = tuple[str, ...]

# ---------------------------------------------------------------------------
# Denylist (defense-in-depth)
# ---------------------------------------------------------------------------
# Modules / prefixes that must never be allow-listed during safe loading.
BLOCKED_PREFIXES: tuple[str, ...] = (
    "os.",
    "posix.",
    "nt.",
    "posixpath.",
    "ntpath.",
    "subprocess.",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.__import__",
    "builtins.globals",
    "pty.",
    "socket.",
    "importlib.",
    "code.",
    "runpy.",
    "ctypes.",
    "webbrowser.",
    "shutil.",
    "signal.",
    "multiprocessing.",
    "pickle.",
    "_pickle.",
)

# ---------------------------------------------------------------------------
# Allowlist: exact class / function names
# ---------------------------------------------------------------------------
# Benign data-container classes that are always allowed regardless of hierarchy.
# Pandas offsets and numpy scalar types are not listed here (covered via safe
# bases below).
EXACT_ALLOWED_CLASSES: frozenset[str] = frozenset({
    # numpy containers and RNGs.  Scalar dtypes are covered by numpy.generic.
    # ``ndarray`` subclasses stay pinned: ``numpy.memmap`` opens a file.
    "numpy.dtype",
    "numpy.ndarray",
    "numpy.random.mtrand.RandomState",
    "numpy.random._mt19937.MT19937",
    # pandas containers. Date offsets are covered by BaseOffset.
    "pandas.DataFrame",
    "pandas.Series",
    "pandas.Index",
    "pandas.RangeIndex",
    "pandas.DatetimeIndex",
    "pandas.PeriodIndex",
    "pandas.TimedeltaIndex",
    "pandas.CategoricalIndex",
    "pandas.MultiIndex",
    "pandas.Timestamp",
    "pandas.Timedelta",
    "pandas.StringDtype",
    "pandas.arrays.ArrowStringArray",
    "pandas.arrays.DatetimeArray",
    "pandas.arrays.TimedeltaArray",
    "pandas.arrays.PeriodArray",
    "pandas.arrays.CategoricalArray",
    # stdlib
    "datetime.datetime",
    "datetime.timedelta",
    "datetime.date",
    "collections.OrderedDict",
    "collections.defaultdict",
    "builtins.set",
    "builtins.frozenset",
    "builtins.complex",
    "builtins.bytes",
    "builtins.bytearray",
    "builtins.range",
    "builtins.slice",
    "types.SimpleNamespace",
    "pathlib.PosixPath",
    "pathlib.WindowsPath",
    "pathlib.Path",
    # sklearn helper
    "sklearn.utils._random.MTRandState",
    # torch; dtype singletons are covered by safe instances
    "torch.dtype",
    "torch.device",
    "torch.Size",
})

# Pickle reconstruction helpers referenced by serialized model files.
# Allow-listed by exact name only. ``builtins.getattr`` is intentionally
# absent: it is a restricted-unpickler bypass when combined with any Python
# function whose module dict contains ``os``.
SAFE_PICKLE_FUNCTIONS: frozenset[str] = frozenset({
    "numpy.random._pickle.__randomstate_ctor",
    "numpy.random._pickle.__bit_generator_ctor",
    "numpy._core.multiarray._reconstruct",
    "numpy.core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.core.multiarray.scalar",
    "pandas.core.indexes.base._new_Index",
    "pandas.core.indexes.datetimes._new_DatetimeIndex",
    "pandas._libs.internals._unpickle_block",
    "pandas._libs.arrays.__pyx_unpickle_NDArrayBacked",
    "pandas._libs.tslibs.timestamps._unpickle_timestamp",
    "pandas._libs.tslibs.timedeltas._timedelta_unpickle",
    "pyarrow.lib._restore_array",
    "pyarrow.lib.py_buffer",
    "pyarrow.lib.type_for_alias",
    "sklearn.metrics._dist_metrics.newObj",
    "sklearn.neighbors._kd_tree.newObj",
    "torch._utils._rebuild_tensor_v2",
    "copyreg._reconstructor",
    "_codecs.encode",
    "torchmetrics.utilities.data.dim_zero_cat",
    "torchmetrics.utilities.data.dim_zero_sum",
    "torchmetrics.utilities.data.dim_zero_mean",
    "torchmetrics.utilities.data.dim_zero_max",
    "torchmetrics.utilities.data.dim_zero_min",
    "torchmetrics.metric.jit_distributed_available",
})

# ---------------------------------------------------------------------------
# Allowlist: class namespaces (classes only) + exact state classes
# ---------------------------------------------------------------------------
# A *class* under one of these prefixes is allowed if it subclasses a known-safe
# base (see :func:`safe_base_classes`) or is an ``enum.Enum``.  Callables never
# pass this check.  This is not a blanket trust of the package.
TRUSTED_PREFIXES: TrustedPrefixPolicy = (
    "darts.",
    "lightgbm.",
    "lightning.",
    "lightning_fabric.",
    "neuralforecast.",
    "numpy.",
    "pandas.",
    "pytorch_lightning.",
    "sklearn.",
    "statsforecast.",
    "statsmodels.",
    "torch.",
    "torchmetrics.",
    "xgboost.",
)


# Classes that do not subclass a safe base, but are persisted model state.
# Instantiating them does not compile code or spawn a process. CmdStan types
# are intentionally absent: ``CmdStanModel.__init__`` compiles a Stan file.
EXACT_STATE_CLASSES: frozenset[str] = frozenset({
    # boosters and non-BaseEstimator estimators (estimators covered via safe base below)
    "lightgbm.basic.Booster",
    "lightgbm.callback._EarlyStoppingCallback",
    "xgboost.core.Booster",
    "catboost.core.CatBoostRegressor",
    "catboost.core.CatBoostClassifier",
    # sklearn; C-extension nodes (models covered via safe base below)
    "sklearn.tree._tree.Tree",
    "sklearn.neighbors._kd_tree.KDTree",
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    # scipy; optimize results stored by Holt-Winters / Theta
    "scipy.optimize._optimize.OptimizeResult",
    "scipy.optimize._lbfgsb_py.LbfgsInvHessProduct",
    "numpy.poly1d",
    # pandas; block managers
    "pandas.core.internals.managers.BlockManager",
    "pandas.core.internals.managers.SingleBlockManager",
    # lightning; callback state (dict subclass; weights_only=True cannot rebuild it)
    "lightning_fabric.utilities.data.AttributeDict",
    # prophet; forecast state; Stan backend is stripped before pickling
    "prophet.forecaster.Prophet",
    # nfoursid; kalman forecaster internals
    "nfoursid.kalman.Kalman",
    "nfoursid.state_space.StateSpace",
    # statsforecast; models covered via safe base below
    "statsforecast.mfles.MFLES",
    "statsforecast.utils.results",
    # statsmodels; general models and results covered via safe bases below
    "statsmodels.base.data.ModelData",
    "statsmodels.tools.tools.Bunch",
    "statsmodels.tsa.arima.params.SARIMAXParams",
    "statsmodels.tsa.arima.model.ARIMAResultsWrapper",
    "statsmodels.tsa.arima.specification.SARIMAXSpecification",
    "statsmodels.tsa.statespace.kalman_smoother.SmootherResults",
    "statsmodels.tsa.statespace.simulation_smoother.SimulationSmoothResults",
    "statsmodels.tsa.statespace.simulation_smoother.SimulationSmoother",
    "statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper",
    "statsmodels.tsa.statespace.initialization.Initialization",
    "statsmodels.tsa.statespace.varmax.VARMAXResultsWrapper",
    # statsmodels; dtype-specific structs persisted by fitted statsmodels statespace models
    "statsmodels.tsa.statespace._initialization.cInitialization",
    "statsmodels.tsa.statespace._initialization.dInitialization",
    "statsmodels.tsa.statespace._initialization.sInitialization",
    "statsmodels.tsa.statespace._initialization.zInitialization",
    "statsmodels.tsa.statespace._kalman_filter.cKalmanFilter",
    "statsmodels.tsa.statespace._kalman_filter.dKalmanFilter",
    "statsmodels.tsa.statespace._kalman_filter.sKalmanFilter",
    "statsmodels.tsa.statespace._kalman_filter.zKalmanFilter",
    "statsmodels.tsa.statespace._kalman_smoother.cKalmanSmoother",
    "statsmodels.tsa.statespace._kalman_smoother.dKalmanSmoother",
    "statsmodels.tsa.statespace._kalman_smoother.sKalmanSmoother",
    "statsmodels.tsa.statespace._kalman_smoother.zKalmanSmoother",
    "statsmodels.tsa.statespace._representation.cStatespace",
    "statsmodels.tsa.statespace._representation.dStatespace",
    "statsmodels.tsa.statespace._representation.sStatespace",
    "statsmodels.tsa.statespace._representation.zStatespace",
    "statsmodels.tsa.statespace._simulation_smoother.cSimulationSmoother",
    "statsmodels.tsa.statespace._simulation_smoother.dSimulationSmoother",
    "statsmodels.tsa.statespace._simulation_smoother.sSimulationSmoother",
    "statsmodels.tsa.statespace._simulation_smoother.zSimulationSmoother",
})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _qualname_set(
    trusted_classes: list[type] | None,
) -> frozenset[str] | None:
    if not trusted_classes:
        return None
    return frozenset(f"{cls.__module__}.{cls.__qualname__}" for cls in trusted_classes)


def resolve_reference(reference):
    """Try to resolve a reference."""
    mod_name, _, attr = reference.rpartition(".")
    try:
        return getattr(importlib.import_module(mod_name), attr, None)
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not resolve reference {reference}: {e}")
        return None


def dedupe_by_identity(objects: Sequence[T]) -> list[T]:
    """Return ``objects`` with duplicate entries removed (by ``id``, preserving order)."""
    seen: set[int] = set()
    out: list[T] = []
    for obj in objects:
        obj_id = id(obj)
        if obj_id not in seen:
            seen.add(obj_id)
            out.append(obj)
    return out


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------
def is_blocked_global(qualname: str) -> bool:
    """Return ``True`` if ``qualname`` matches a hard-blocked module prefix."""
    return qualname.startswith(BLOCKED_PREFIXES)


@lru_cache(maxsize=1)
def _pandas_allowed_bases() -> type | None:
    """Return pandas ``BaseOffset``, or ``None`` when pandas cannot be imported."""
    try:
        from pandas._libs.tslibs.offsets import BaseOffset
    except Exception:  # pragma: no cover
        return None
    return BaseOffset


@lru_cache(maxsize=1)
def _numpy_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from numpy import generic
    except Exception:  # pragma: no cover
        return None
    return generic


@lru_cache(maxsize=1)
def _statsforecast_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from statsforecast.models import _TS
    except Exception:  # pragma: no cover
        return None
    return _TS


@lru_cache(maxsize=1)
def _neuralforecast_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from neuralforecast.common._base_model import BaseModel
    except Exception:  # pragma: no cover
        return None
    return BaseModel


@lru_cache(maxsize=1)
def _statsmodels_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from statsmodels.base.model import Results
        from statsmodels.tsa.base.tsa_model import TimeSeriesModel
    except Exception:  # pragma: no cover
        return None
    return TimeSeriesModel, Results


@lru_cache(maxsize=1)
def _sklearn_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from sklearn.base import BaseEstimator
    except Exception:  # pragma: no cover
        return None
    return BaseEstimator


@lru_cache(maxsize=1)
def _torch_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from torch.nn import Module
        from torch.optim import Optimizer, lr_scheduler

        lr_scheduler_cls = getattr(
            lr_scheduler,
            "LRScheduler",
            getattr(lr_scheduler, "_LRScheduler", Module),
        )
    except Exception:  # pragma: no cover
        return None
    return Module, Optimizer, lr_scheduler_cls


@lru_cache(maxsize=1)
def _torchmetrics_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from torchmetrics import Metric, MetricCollection
    except Exception:  # pragma: no cover
        return None
    return Metric, MetricCollection


@lru_cache(maxsize=1)
def _lightning_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from pytorch_lightning import Callback
    except Exception:  # pragma: no cover
        return None
    return Callback


@lru_cache(maxsize=1)
def _darts_core_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from darts import TimeSeries
        from darts.dataprocessing.encoders.encoder_base import (
            CovariatesIndexGenerator,
            Encoder,
            SequentialEncoderTransformer,
        )
        from darts.dataprocessing.transformers.base_data_transformer import (
            BaseDataTransformer,
        )
        from darts.models.filtering.filtering_model import FilteringModel
        from darts.models.forecasting.forecasting_model import ForecastingModel
        from darts.utils.likelihood_models.base import Likelihood, LikelihoodType
    except Exception:  # pragma: no cover
        return None
    return (
        TimeSeries,
        CovariatesIndexGenerator,
        Encoder,
        SequentialEncoderTransformer,
        BaseDataTransformer,
        FilteringModel,
        ForecastingModel,
        Likelihood,
        LikelihoodType,
    )


@lru_cache(maxsize=1)
def _darts_torch_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from darts.utils.data.torch_datasets.utils import TorchSample
        from darts.utils.likelihood_models.torch import TorchLikelihood
    except Exception:  # pragma: no cover
        return None
    return TorchSample, TorchLikelihood


@lru_cache(maxsize=1)
def _darts_foundation_allowed_bases() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from darts.models.components.huggingface_connector import HuggingFaceConnector
    except Exception:  # pragma: no cover
        return None
    return HuggingFaceConnector


def _is_fenced_allowed_subclass(qualname: str, obj: type) -> bool:
    """Allow in-package (sub)classes that share an audited base.

    For example ``pandas.`` classes that subclass ``BaseOffset`` A subclass
    defined outside those packages does not pass: its ``__setstate__`` is
    not part of the audit.
    """
    if qualname.startswith("darts.models.components.huggingface_connector."):
        base = _darts_foundation_allowed_bases()
    elif qualname.startswith((
        "darts.utils.data.torch_datasets",
        "darts.utils.likelihood_models.torch",
    )):
        base = _darts_torch_allowed_bases()
    elif qualname.startswith("darts."):
        base = _darts_core_allowed_bases()
    elif qualname.startswith("pandas."):
        base = _pandas_allowed_bases()
    elif qualname.startswith("numpy."):
        base = _numpy_allowed_bases()
    elif qualname.startswith(("sklearn.", "lightgbm.", "xgboost.", "catboost.")):
        base = _sklearn_allowed_bases()
    elif qualname.startswith("statsforecast."):
        base = _statsforecast_allowed_bases()
    elif qualname.startswith("statsmodels."):
        base = _statsmodels_allowed_bases()
    elif qualname.startswith("torch."):
        base = _torch_allowed_bases()
    elif qualname.startswith("torchmetrics."):
        base = _torchmetrics_allowed_bases()
    ## TODO: remove callbacks
    elif qualname.startswith(("pytorch_lightning.", "lightning.")):
        base = _lightning_allowed_bases()
    elif qualname.startswith("neuralforecast."):
        base = _neuralforecast_allowed_bases()
    else:
        base = None
    return base is not None and issubclass(obj, base)


@lru_cache(maxsize=1)
def _torch_allowed_instances() -> type | tuple[type, ...] | None:
    """Return ``numpy.generic``, or ``None`` when numpy cannot be imported."""
    try:
        from torch import dtype
    except Exception:  # pragma: no cover
        return None
    return dtype


def _is_fenced_allowed_instance(qualname: str, obj: type) -> bool:
    if qualname.startswith("torch."):
        base = _torch_allowed_instances()
    else:
        base = None
    return base is not None and isinstance(obj, base)


def _normalize_qualname(qualname: str) -> str:
    """Map torch's Python-2 ``__builtin__`` module name onto ``builtins``."""
    if qualname.startswith("__builtin__."):
        return "builtins." + qualname.removeprefix("__builtin__.")
    return qualname


def is_allowed_global(
    qualname: str,
    obj: Any,
    *,
    safe_bases: tuple[type, ...],
    extra_safe_callables: frozenset[str] | None = None,
) -> bool:
    """Return whether ``qualname`` / ``obj`` may be allow-listed for safe loading.

    Checks (in order):

    1. **Deny** if it matches :data:`BLOCKED_PREFIXES`.
    2. **Allow** if it is in :data:`EXACT_ALLOWED_CLASSES`.
    3. **Allow** if it is in :data:`SAFE_PICKLE_FUNCTIONS` or ``extra_safe_callables``.
    4. **Allow** if it is in :data:`EXACT_STATE_CLASSES` and is a class.
    5. **Allow** ``torch.*`` module-level :class:`torch.dtype` singletons.
    6. **Allow classes** under trusted prefixes if the class subclasses one of the safe bases.
    7. **Allow classes** under :data:`TRUSTED_PREFIXES` that subclass a ``safe_bases``
       entry or are an ``enum.Enum``.  Callables never pass this step.
    8. **Deny** everything else.
    """
    qualname = _normalize_qualname(qualname)
    if is_blocked_global(qualname):
        return False

    if qualname in EXACT_ALLOWED_CLASSES:
        return obj is not None

    if qualname in SAFE_PICKLE_FUNCTIONS:
        return obj is not None and callable(obj)

    if extra_safe_callables and qualname in extra_safe_callables:
        return obj is not None and callable(obj)

    if qualname in EXACT_STATE_CLASSES:
        return obj is not None and inspect.isclass(obj)

    if obj is None or not qualname.startswith(TRUSTED_PREFIXES):
        return False

    if not inspect.isclass(obj):
        if callable(obj):
            return False
        return _is_fenced_allowed_instance(qualname, obj)

    if _is_fenced_allowed_subclass(qualname, obj):
        return True

    if safe_bases and issubclass(obj, safe_bases):
        return True

    if issubclass(obj, enum.Enum):
        return True
    return False


def format_unpickling_error(qualname: str, path: str) -> str:
    """Build an actionable error message for a blocked global during safe loading."""
    return (
        f"Global `{qualname}` referenced by model file `{path}` is not allow-listed "
        f"for safe loading.  If you trust this file, reload with `weights_only=False` "
        f"(torch models) or `trusted=True` (non-torch models).  If you used custom "
        f"encoders, callbacks, or other third-party classes, pass them via "
        f"`trusted_classes` or use the opt-out flag."
    )


# TODO: remove
def safe_base_classes() -> tuple[type, ...]:
    """Collect known-safe base classes for allowlist validation.

    A class that subclasses one of these bases is considered safe for
    deserialization.  Covers the Darts ecosystem: models, encoders,
    transformers, sklearn estimators, PyTorch modules, Lightning callbacks, etc.
    """
    bases: list[type] = [OrderedDict]

    # TODO: remove callbacks
    # Lightning
    try:
        from pytorch_lightning.callbacks import Callback

        bases.append(Callback)
    except Exception:  # pragma: no cover
        pass

    return tuple(b for b in bases if inspect.isclass(b))


# ---------------------------------------------------------------------------
# Restricted Unpickler (for non-torch .pkl files)
# ---------------------------------------------------------------------------
class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that restricts which globals can be loaded.

    Only classes / functions that pass :func:`is_allowed_global` are permitted.
    This prevents arbitrary code execution from maliciously crafted pickle files
    (CWE-502).
    """

    def __init__(
        self,
        file,
        *,
        safe_bases: tuple[type, ...] | None = None,
        trusted_classes: list[type] | None = None,
        **kwargs,
    ):
        super().__init__(file, **kwargs)
        self._safe_bases = safe_bases if safe_bases is not None else safe_base_classes()
        self._extra_allowed = _qualname_set(trusted_classes)

    def find_class(self, module: str, name: str):
        # Normalize Python-2-era module name emitted by torch.save's pickle
        # protocol so that lookups hit the ``builtins.*`` entries in our
        # allowlists.
        if module == "__builtin__":
            module = "builtins"
        qualname = f"{module}.{name}"

        if is_blocked_global(qualname):
            raise pickle.UnpicklingError(
                f"Blocked unsafe global `{qualname}` during model loading. "
                "If you trust this file, reload with `trusted=True`."
            )

        obj = resolve_reference(qualname)
        if is_allowed_global(
            qualname,
            obj,
            safe_bases=self._safe_bases,
            extra_safe_callables=self._extra_allowed,
        ):
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Global `{qualname}` is not allow-listed for safe model loading. "
            "If you trust this file, reload with `trusted=True`. "
            "If you used custom classes, ensure they subclass a known Darts or "
            "sklearn base class, or pass `trusted=True`."
        )


def restricted_pickle_load(
    file,
    *,
    trusted: bool = False,
    trusted_classes: list[type] | None = None,
    **kwargs,
):
    """Load a pickle file with restricted deserialization by default.

    Parameters
    ----------
    file
        A readable binary file object.
    trusted
        If ``True``, falls back to unrestricted ``pickle.load``.  Only use for
        files from trusted sources.  Default: ``False``.
    trusted_classes
        Optional list of additional classes to allow during restricted loading.
        Each class must be importable by its module path.
    """
    if trusted:
        return pickle.load(file, **kwargs)

    return RestrictedUnpickler(
        file,
        trusted_classes=trusted_classes,
        **kwargs,
    ).load()


class UnpicklingError(RuntimeError):
    """Raised when a global referenced by a serialized file is not allow-listed."""
