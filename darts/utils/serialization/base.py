"""
Model Serialization Utilities (core)
------------------------------------

Shared helpers for Darts model persistence.

Security model:

- Prefer **scoped** allow-lists applied only around a single load call, not process-wide
  registration at import time.
- When an inspection API exists, derive the allow-list **from the specific artifact** being
  loaded (checkpoint-driven registration) rather than pre-registering entire libraries.
"""

import importlib
import inspect
from collections.abc import Sequence
from typing import Any, Literal, TypeVar

from darts.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

LoadScope = Literal["checkpoint", "wrapper"]

# Prefixes for trusted packages when filtering checkpoint-referenced globals.
TrustedPrefixPolicy = tuple[str, ...]

CHECKPOINT_TRUSTED_PREFIXES: TrustedPrefixPolicy = (
    "torch.",
    "torchmetrics.",
    "darts.",
    "neuralforecast.",
)

WRAPPER_TRUSTED_PREFIXES: TrustedPrefixPolicy = CHECKPOINT_TRUSTED_PREFIXES + (
    "pytorch_lightning.",
    "lightning.",
    "lightning_fabric.",
    "pandas.",
    "numpy.",
    "sklearn.",
    "pyarrow.",
    "fsspec.",
)

# Pickle helper callables from the data stack (wrapper scope only; exact-prefix guard).
DATA_STACK_FUNCTION_PREFIXES: tuple[str, ...] = (
    "numpy.",
    "pandas.",
    "pyarrow.",
    "fsspec.",
)

# Domain packages whose classes are allowed without subclass checks in wrapper scope.
WRAPPER_DOMAIN_CLASS_PREFIXES: tuple[str, ...] = (
    "darts.",
    "neuralforecast.",
    "pytorch_lightning.",
    "lightning.",
    "lightning_fabric.",
)

TRUSTED_PREFIXES_BY_SCOPE: dict[LoadScope, TrustedPrefixPolicy] = {
    "checkpoint": CHECKPOINT_TRUSTED_PREFIXES,
    "wrapper": WRAPPER_TRUSTED_PREFIXES,
}

# Modules / prefixes that must never be allow-listed during safe loading.
BLOCKED_PREFIXES: tuple[str, ...] = (
    "os.",
    "subprocess.",
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "pty.",
    "socket.",
    "pickle.",
    "importlib.",
    "code.",
    "runpy.",
)

# Exact qualified names for benign data-container classes that do not subclass Darts bases.
EXACT_ALLOWED_CLASSES: frozenset[str] = frozenset({
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
    "pandas.core.indexes.base.Index",
    "pandas.core.indexes.datetimes.DatetimeIndex",
    "pandas.core.indexes.range.RangeIndex",
    "pandas.core.indexes.period.PeriodIndex",
    "pandas.core.indexes.timedeltas.TimedeltaIndex",
    "pandas._libs.tslibs.timestamps.Timestamp",
    "pandas.Index",
    "pandas.RangeIndex",
    "pandas.StringDtype",
    "pandas.arrays.ArrowStringArray",
    "numpy.dtype",
    "numpy.ndarray",
    "numpy.float64",
    "numpy.float32",
    "numpy.int64",
    "numpy.random._mt19937.MT19937",
    "datetime.datetime",
    "datetime.timedelta",
    "datetime.date",
    "sklearn.utils._random.MTRandState",
})

# Pickle reconstruction helpers referenced by numpy/pandas/pyarrow in wrapper files.
# Allow-listed by exact name only (wrapper scope); never a blanket module scan.
WRAPPER_SAFE_FUNCTIONS: frozenset[str] = frozenset({
    "numpy.random._pickle.__randomstate_ctor",
    "numpy.random._pickle.__bit_generator_ctor",
    "numpy._core.multiarray._reconstruct",
    "pandas.core.indexes.base._new_Index",
    "pyarrow.lib._restore_array",
    "pyarrow.lib.py_buffer",
    "pyarrow.lib.type_for_alias",
    # Common stdlib pickle reducer helpers.
    "builtins.getattr",
    "builtins.slice",
})

# Prefixes for which classes are allowed without subclass checks (data-stack containers).
DATA_STACK_CLASS_PREFIXES: tuple[str, ...] = (
    "numpy.",
    "pandas.",
    "pyarrow.",
    "sklearn.",
    "fsspec.",
    "datetime.",
)


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


def sanitize_for_wrapper_save(value: Any) -> Any:
    """Return a ``weights_only``-friendly copy of ``value`` for wrapper (``.pt``) saves.

    Converts Lightning ``AttributeDict`` instances (and other mapping types that break
    PyTorch's ``weights_only`` unpickler) into plain ``dict`` objects recursively.
    """
    if isinstance(value, dict):
        return {k: sanitize_for_wrapper_save(v) for k, v in value.items()}

    if isinstance(value, list):
        return [sanitize_for_wrapper_save(v) for v in value]

    if isinstance(value, tuple):
        return tuple(sanitize_for_wrapper_save(v) for v in value)

    type_name = type(value).__name__
    if type_name == "AttributeDict" and hasattr(value, "items"):
        return {k: sanitize_for_wrapper_save(v) for k, v in value.items()}

    return value


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


def is_blocked_global(qualname: str) -> bool:
    """Return ``True`` if ``qualname`` matches a hard-blocked module prefix."""
    return qualname.startswith(BLOCKED_PREFIXES)


def format_unpickling_error(qualname: str, path: str, *, scope: LoadScope) -> str:
    """Build an actionable error message for a blocked global during safe loading."""
    artifact = "checkpoint" if scope == "checkpoint" else "model wrapper"
    return (
        f"Global `{qualname}` referenced by {artifact} `{path}` is not allow-listed for "
        f"safe loading. If you trust this file, reload with `weights_only=False`. If you "
        f"used custom encoders, callbacks, or other third-party classes, ensure they live "
        f"under an allow-listed package or pass `weights_only=False`."
    )


def is_allowed_global(
    qualname: str,
    obj: Any,
    *,
    scope: LoadScope,
    safe_bases: tuple[type, ...],
    checkpoint_safe_functions: frozenset[str] | None = None,
    wrapper_safe_functions: frozenset[str] | None = None,
) -> bool:
    """Return whether ``qualname`` / ``obj`` may be allow-listed for the given ``scope``.

    Parameters
    ----------
    qualname
        Fully-qualified global name as reported by the torch inspection API.
    obj
        Resolved object, or ``None`` if resolution failed.
    scope
        ``"checkpoint"`` for Lightning ``.ckpt`` files; ``"wrapper"`` for Darts ``.pt`` files.
    safe_bases
        Known-safe base classes; a candidate class must subclass one of these unless it is
        listed in :data:`EXACT_ALLOWED_CLASSES` or falls under :data:`DATA_STACK_CLASS_PREFIXES`
        in wrapper scope.
    checkpoint_safe_functions
        Exact qualified names of callables allowed only in ``checkpoint`` scope (e.g.
        torchmetrics reduction helpers).
    wrapper_safe_functions
        Exact qualified names of callables allowed only in ``wrapper`` scope (e.g. numpy/pandas
        pickle reconstruction helpers).
    """
    if is_blocked_global(qualname):
        return False

    if qualname in EXACT_ALLOWED_CLASSES:
        return obj is not None

    trusted_prefixes = TRUSTED_PREFIXES_BY_SCOPE[scope]

    if checkpoint_safe_functions and qualname in checkpoint_safe_functions:
        return callable(obj)

    if scope == "wrapper" and wrapper_safe_functions:
        if qualname in wrapper_safe_functions:
            return callable(obj)

    if callable(obj) and not inspect.isclass(obj):
        if scope == "wrapper" and qualname.startswith(
            DATA_STACK_FUNCTION_PREFIXES + ("neuralforecast.",)
        ):
            return True
        return False

    if not qualname.startswith(trusted_prefixes):
        return False

    if obj is None:
        return False

    if inspect.isclass(obj):
        if safe_bases and issubclass(obj, safe_bases):
            return True
        if scope == "wrapper" and qualname.startswith(DATA_STACK_CLASS_PREFIXES):
            return True
        if scope == "wrapper" and qualname.startswith(WRAPPER_DOMAIN_CLASS_PREFIXES):
            # Production Darts/PL/neuralforecast classes only — not test modules.
            if qualname.startswith("darts.") and (
                ".tests." in qualname or qualname.startswith("darts.tests.")
            ):
                return False
            return True
        return False

    return False
