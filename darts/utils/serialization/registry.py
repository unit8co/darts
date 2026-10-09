"""
Safe-global Registration for Model Loading
------------------------------------------

Process-wide and context-scoped safe-global registration for model loading.
"""

import importlib
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeAlias

from darts.logging import get_logger, raise_log

logger = get_logger(__name__)

SafeGlobal: TypeAlias = Callable[..., object] | tuple[Callable[..., object], str]
UserSafeGlobals: TypeAlias = dict[str, Callable]

_process_globals: list[SafeGlobal] = []
_process_lock = threading.Lock()
_context_stack: ContextVar[list[SafeGlobal]] = ContextVar(
    "_darts_safe_globals_context", default=[]
)


def add_safe_globals(safe_globals: list[SafeGlobal]) -> None:
    """Register globals as safe for loading.

    For example, functions added to this list can be called during unpickling, classes could be instantiated and have
    state set.

    Each item in the list can either be a function/class or a tuple of the form (function/class, string) where string
    is the full path of the function/class.

    Within the serialized format, each function is identified with its full path as ``{__module__}.{__qualname__}``.
    When calling this API, you can provide this full path that should match the one in the checkpoint otherwise the
    default ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Parameters
    ----------
    safe_globals
        List of globals to mark as safe. The elements must either be callables, or a tuple of `(callable, full path)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from darts.datasets import AirPassengersDataset
    >>> from darts.models import LinearRegressionModel
    >>> from darts.utils.serialization import add_safe_globals
    >>>
    >>> def my_encoder(idx: pd.DatetimeIndex):
    >>>     # custom encoder example adding the series month value
    >>>     return idx.month
    >>>
    >>> # add encoder as a future covariate, fit and save the model
    >>> model = LinearRegressionModel(
    >>>     lags=12,
    >>>     lags_future_covariates=[0],
    >>>     add_encoders={"custom": {"future": [my_encoder]}}
    >>> )
    >>> series = AirPassengersDataset().load()
    >>> model.fit(series)
    >>> model.save("model.pkl")
    >>>
    >>> # calling `load()` directly will fail since `my_encoder` is not an allow-listed global;
    >>> # you can globally add it to the safe globals and loading will succeed
    >>> add_safe_globals([my_encoder])
    >>> model = LinearRegressionModel.load("model.pkl")
    """
    for entry in safe_globals:
        _parse_safe_global(entry)
    with _process_lock:
        _process_globals.extend(safe_globals)


@contextmanager
def safe_globals(safe_globals: list[SafeGlobal]) -> Generator[None, None, None]:
    """Temporarily add safe globals for loads within the ``with`` block.

    Parameters
    ----------
    safe_globals
        List of globals to mark as safe. The elements must either be callables, or a tuples of `(callable, full path)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from darts.datasets import AirPassengersDataset
    >>> from darts.models import LinearRegressionModel
    >>> from darts.utils.serialization import safe_globals
    >>>
    >>> def my_encoder(idx: pd.DatetimeIndex):
    >>>     # custom encoder example adding the series month value
    >>>     return idx.month
    >>>
    >>> # add encoder as a future covariate, fit and save the model
    >>> model = LinearRegressionModel(
    >>>     lags=12,
    >>>     lags_future_covariates=[0],
    >>>     add_encoders={"custom": {"future": [my_encoder]}}
    >>> )
    >>> series = AirPassengersDataset().load()
    >>> model.fit(series)
    >>> model.save("model.pkl")
    >>>
    >>> # calling `load()` directly will fail since `my_encoder` is not an allow-listed global;
    >>> # you can temporarily add it to the safe globals and loading will succeed
    >>> with safe_globals([my_encoder]):
    >>>     model = LinearRegressionModel.load("model.pkl")
    """
    for entry in safe_globals:
        _parse_safe_global(entry)
    token = _context_stack.set(_context_stack.get() + safe_globals)
    try:
        yield
    finally:
        _context_stack.reset(token)


def get_safe_globals() -> list[SafeGlobal]:
    """Returns a list of the current user-added safe globals.

    Examples
    --------
    >>> from darts.utils.serialization import add_safe_globals, get_safe_globals
    >>>
    >>> class MyClass:
    >>>     pass
    >>>
    >>> add_safe_globals([MyClass])
    >>> get_safe_globals()
    [<class '__main__.MyClass'>]
    """
    with _process_lock:
        process = list(_process_globals)
    context = list(_context_stack.get())
    return process + context


def clear_safe_globals() -> None:
    """Clear all user-added safe globals.

    Examples
    --------
    >>> from darts.utils.serialization import add_safe_globals, clear_safe_globals, get_safe_globals
    >>>
    >>> class MyClass:
    >>>     pass
    >>>
    >>> add_safe_globals([MyClass])
    >>> clear_safe_globals()
    >>> get_safe_globals()
    []
    """
    with _process_lock:
        _process_globals.clear()


def _parse_safe_global(entry: SafeGlobal) -> tuple[Callable[..., object], str]:
    """Return ``(callable, qualname)`` for a PyTorch-compatible safe-global entry."""
    if isinstance(entry, tuple):
        if len(entry) != 2:
            raise_log(
                ValueError(
                    "Safe-global tuples must be (callable, qualname_str), "
                    f"got length {len(entry)}."
                ),
            )
        obj, qualname = entry
        if not callable(obj):
            raise_log(
                ValueError(
                    "Safe globals must be callables (classes or functions), "
                    f"got {type(obj).__name__}."
                ),
            )
        if not isinstance(qualname, str):
            raise_log(
                ValueError(
                    "Safe-global tuple second element must be str, "
                    f"got {type(qualname).__name__}."
                ),
            )
    else:
        obj = entry
        if not callable(obj):
            raise_log(
                ValueError(
                    "Safe globals must be callables (classes or functions), "
                    f"got {type(obj).__name__}."
                ),
            )
        qualname = f"{obj.__module__}.{obj.__qualname__}"

    qualname = _normalize_qualname(qualname)
    return obj, qualname


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _get_user_safe_globals() -> UserSafeGlobals:
    """Merged user allow-list as qualname → callable (context overrides process)."""
    user_entries = get_safe_globals()
    user_safe_globals: UserSafeGlobals = {}
    for entry in user_entries:
        obj, qualname = _parse_safe_global(entry)
        user_safe_globals[qualname] = obj
    return user_safe_globals


def _normalize_qualname(qualname: str) -> str:
    """Map torch's Python-2 ``__builtin__`` module name onto ``builtins``."""
    if qualname.startswith("__builtin__."):  # pragma: no cover
        return "builtins." + qualname.removeprefix("__builtin__.")
    return qualname


def _resolve_allowed_global(
    qualname: str,
    user_globals: UserSafeGlobals | None = None,
) -> object | None:
    """Return a global by qualname, preferring user-registered objects over import."""
    qualname = _normalize_qualname(qualname)
    if user_globals and qualname in user_globals:
        return user_globals[qualname]

    mod_name, _, attr = qualname.rpartition(".")
    try:
        return getattr(importlib.import_module(mod_name), attr, None)
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not resolve reference {qualname}: {e}")
        return None
