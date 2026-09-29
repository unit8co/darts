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
from collections.abc import Sequence
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
EXACT_ALLOWED_CLASSES: frozenset[str] = frozenset({
    # numpy
    "numpy.dtype",
    "numpy.ndarray",
    "numpy.float64",
    "numpy.float32",
    "numpy.float16",
    "numpy.int64",
    "numpy.int32",
    "numpy.int16",
    "numpy.int8",
    "numpy.uint64",
    "numpy.uint32",
    "numpy.bool_",
    "numpy.complex128",
    "numpy.complex64",
    "numpy.str_",
    "numpy.bytes_",
    "numpy.random.mtrand.RandomState",
    "numpy.random._mt19937.MT19937",
    # pandas
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
    "pandas.core.indexes.base.Index",
    "pandas.core.indexes.datetimes.DatetimeIndex",
    "pandas.core.indexes.range.RangeIndex",
    "pandas.core.indexes.period.PeriodIndex",
    "pandas.core.indexes.timedeltas.TimedeltaIndex",
    "pandas._libs.tslibs.timestamps.Timestamp",
    "pandas._libs.tslibs.timedeltas.Timedelta",
    "pandas._libs.tslibs.offsets.MonthEnd",
    "pandas._libs.tslibs.offsets.YearEnd",
    "pandas._libs.tslibs.offsets.QuarterEnd",
    "pandas._libs.tslibs.offsets.Week",
    "pandas._libs.tslibs.offsets.Day",
    "pandas._libs.tslibs.offsets.Hour",
    "pandas._libs.tslibs.offsets.Minute",
    "pandas._libs.tslibs.offsets.Second",
    "pandas._libs.tslibs.offsets.Milli",
    "pandas._libs.tslibs.offsets.Micro",
    "pandas._libs.tslibs.offsets.Nano",
    "pandas._libs.tslibs.offsets.BusinessDay",
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
    # sklearn helper
    "sklearn.utils._random.MTRandState",
})

# Pickle reconstruction helpers referenced by numpy/pandas/pyarrow in serialized
# files.  Allow-listed by exact name only; never a blanket module scan.
SAFE_PICKLE_FUNCTIONS: frozenset[str] = frozenset({
    "numpy.random._pickle.__randomstate_ctor",
    "numpy.random._pickle.__bit_generator_ctor",
    "numpy._core.multiarray._reconstruct",
    "numpy.core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "numpy.core.multiarray.scalar",
    "pandas.core.indexes.base._new_Index",
    "pandas._libs.internals._unpickle_block",
    "pandas.core.internals.blocks.new_block_2d",
    "pandas.core.internals.blocks.new_block",
    "pandas._libs.tslibs.offsets._unpickle_offset",
    "pyarrow.lib._restore_array",
    "pyarrow.lib.py_buffer",
    "pyarrow.lib.type_for_alias",
    "builtins.getattr",
    "copyreg._reconstructor",
    "_codecs.encode",
})

# ---------------------------------------------------------------------------
# Allowlist: trusted package prefixes + class hierarchy
# ---------------------------------------------------------------------------
# A class under one of these prefixes is allowed if it subclasses a known-safe
# base (see :func:`safe_base_classes`). This is NOT a blanket trust — the
# class must pass the hierarchy check.
TRUSTED_PREFIXES: TrustedPrefixPolicy = (
    "torch.",
    "torchmetrics.",
    "darts.",
    "neuralforecast.",
    "pytorch_lightning.",
    "lightning.",
    "lightning_fabric.",
    "sklearn.",
    "numpy.",
    "pandas.",
    "pyarrow.",
    "fsspec.",
    "statsmodels.",
    "statsforecast.",
    "scipy.",
)

# Packages whose internal classes and callables are considered safe for
# deserialization.  These are either pure data packages (numpy, pandas,
# pyarrow) or training-infrastructure packages (lightning) that contain only
# data containers, loggers, and framework helpers — no code-execution
# primitives.  All classes and callables under these prefixes are trusted
# without requiring a safe-base hierarchy check.
_SAFE_PACKAGE_PREFIXES: tuple[str, ...] = (
    "numpy.",
    "pandas.",
    "pyarrow.",
    "torch.",
    "torchmetrics.",
    "lightning.",
    "lightning_fabric.",
    "pytorch_lightning.",
    "statsmodels.",
    "statsforecast.",
    "neuralforecast.",
    "scipy.",
    "fsspec.",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def all_imported_subclasses(cls: type) -> set[type]:
    """Gives all currently imported subclasses for `cls` (including `cls`)."""
    found = {cls}
    for sub in cls.__subclasses__():
        found |= all_imported_subclasses(sub)
    return found


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


def safe_base_classes() -> tuple[type, ...]:
    """Collect known-safe base classes for allowlist validation.

    A class that subclasses one of these bases is considered safe for
    deserialization.  Covers the Darts ecosystem: models, encoders,
    transformers, sklearn estimators, PyTorch modules, Lightning callbacks, etc.
    """
    bases: list[type] = []

    # sklearn
    try:
        from sklearn.base import BaseEstimator

        bases.append(BaseEstimator)
    except Exception:  # pragma: no cover
        pass

    # darts core
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
        from darts.models.forecasting.forecasting_model import ForecastingModel
        from darts.utils.data.torch_datasets.utils import TorchSample

        bases += [
            TimeSeries,
            Encoder,
            CovariatesIndexGenerator,
            SequentialEncoderTransformer,
            BaseDataTransformer,
            ForecastingModel,
            TorchSample,
        ]
    except Exception:  # pragma: no cover
        pass

    # PyTorch
    try:
        from torch.nn import Module
        from torch.optim import Optimizer, lr_scheduler

        bases += [Module, Optimizer]
        bases.append(
            getattr(
                lr_scheduler,
                "LRScheduler",
                getattr(lr_scheduler, "_LRScheduler", Module),
            )
        )
    except Exception:  # pragma: no cover
        pass

    # torchmetrics
    try:
        import torchmetrics

        bases += [torchmetrics.Metric, torchmetrics.MetricCollection]
    except Exception:  # pragma: no cover
        pass

    # Lightning
    try:
        from pytorch_lightning.callbacks import Callback

        bases.append(Callback)
    except Exception:  # pragma: no cover
        pass

    # Darts likelihoods
    try:
        from darts.utils.likelihood_models.base import LikelihoodType
        from darts.utils.likelihood_models.torch import TorchLikelihood

        bases += [TorchLikelihood, LikelihoodType]
    except Exception:  # pragma: no cover
        pass

    return tuple(b for b in bases if inspect.isclass(b))


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
    4. **Reject** if it is not under a :data:`TRUSTED_PREFIXES`.
    5. **Allow** all classes and callables from safe infrastructure packages
       (:data:`_SAFE_PACKAGE_PREFIXES` — numpy, pandas, pyarrow, lightning).
    6. **Allow classes** that subclass a ``safe_bases`` entry or are an ``enum.Enum``.
    7. **Deny** everything else.
    """
    if is_blocked_global(qualname):
        return False

    if qualname in EXACT_ALLOWED_CLASSES:
        return obj is not None

    if qualname in SAFE_PICKLE_FUNCTIONS:
        return obj is not None and callable(obj)

    if extra_safe_callables and qualname in extra_safe_callables:
        return obj is not None and callable(obj)

    if not qualname.startswith(TRUSTED_PREFIXES):
        return False

    if obj is None:
        return False

    # Safe infrastructure packages (data + training framework) contain only
    # data containers, reconstruction helpers, and framework internals — no
    # code-execution primitives.  Trust all classes and callables.
    if qualname.startswith(_SAFE_PACKAGE_PREFIXES):
        return True

    if inspect.isclass(obj):
        if safe_bases and issubclass(obj, safe_bases):
            return True
        if issubclass(obj, enum.Enum):
            return True
        return False

    if callable(obj) and not inspect.isclass(obj):
        return False

    return False


def format_unpickling_error(qualname: str, path: str) -> str:
    """Build an actionable error message for a blocked global during safe loading."""
    return (
        f"Global `{qualname}` referenced by model file `{path}` is not allow-listed "
        f"for safe loading.  If you trust this file, reload with `weights_only=False` "
        f"(torch models) or `trusted=True` (non-torch models).  If you used custom "
        f"encoders, callbacks, or other third-party classes, ensure they subclass a "
        f"known base class (e.g. BaseEstimator, nn.Module) or use the opt-out flag."
    )


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
        extra_allowed: frozenset[str] | None = None,
        **kwargs,
    ):
        super().__init__(file, **kwargs)
        self._safe_bases = safe_bases if safe_bases is not None else safe_base_classes()
        self._extra_allowed = extra_allowed

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

    extra_allowed = None
    if trusted_classes:
        extra_allowed = frozenset(
            f"{cls.__module__}.{cls.__qualname__}" for cls in trusted_classes
        )

    return RestrictedUnpickler(
        file,
        extra_allowed=extra_allowed,
        **kwargs,
    ).load()


class UnpicklingError(RuntimeError):
    """Raised when a global referenced by a serialized file is not allow-listed."""
