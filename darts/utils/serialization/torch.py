"""
Torch Checkpoint Serialization
------------------------------

Safe-by-default loading for PyTorch Lightning ``.ckpt`` files and Darts ``.pt`` wrapper
files used by ``TorchForecastingModel``.

Two scopes share the same :func:`~darts.utils.serialization.base.is_allowed_global` filter:

* **checkpoint** — Lightning ``.ckpt`` checkpoints (modules, optimizers, metrics).
  Uses ``torch.serialization.get_unsafe_globals_in_checkpoint`` to discover referenced
  globals, then filters them.
* **wrapper** — Darts ``.pt`` model wrapper files (the full ``TorchForecastingModel``
  object graph including encoders, transformers, sklearn estimators, etc.).
  Same discovery + filter approach, with a broader set of known-safe base classes.
"""

import enum
import inspect
from collections.abc import Callable
from typing import Any, Literal

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    SAFE_PICKLE_FUNCTIONS,
    UnpicklingError,
    all_imported_subclasses,
    dedupe_by_identity,
    format_unpickling_error,
    is_allowed_global,
    is_blocked_global,
    resolve_reference,
    safe_base_classes,
)

logger = get_logger(__name__)

LoadScope = Literal["checkpoint", "wrapper"]

# ``torchmetrics`` metrics store references to a few of their own tensor-reduction /
# distributed helper *functions* in their pickled state (e.g. ``dim_zero_sum``), so a
# ``weights_only=True`` load of a metric-bearing checkpoint needs these specific
# callables allow-listed.  Listed by EXACT qualified name (never a blanket scan) and
# only ever added when a given checkpoint actually references them.
TORCHMETRICS_SAFE_FUNCTIONS: frozenset[str] = frozenset({
    "torchmetrics.utilities.data.dim_zero_cat",
    "torchmetrics.utilities.data.dim_zero_sum",
    "torchmetrics.utilities.data.dim_zero_mean",
    "torchmetrics.utilities.data.dim_zero_max",
    "torchmetrics.utilities.data.dim_zero_min",
    "torchmetrics.metric.jit_distributed_available",
})

# All extra safe callables (torchmetrics + data-stack reconstruction helpers).
_ALL_SAFE_CALLABLES: frozenset[str] = (
    TORCHMETRICS_SAFE_FUNCTIONS | SAFE_PICKLE_FUNCTIONS
)


class DartsCheckpointIO(TorchCheckpointIO):
    """Custom CheckpointIO that defaults ``weights_only`` to ``True`` (safe-by-default).

    PyTorch >= 2.6 changed ``torch.load`` to default to ``weights_only=True`` as a
    mitigation against CWE-502 (arbitrary code execution when unpickling an untrusted
    checkpoint).  Darts ``.ckpt`` files contain a few non-tensor objects (likelihoods,
    optimizer / scheduler classes, hparams, …); the classes required to deserialize a
    *legitimate* Darts checkpoint under ``weights_only=True`` are allow-listed **at load
    time** and scoped to the load itself (see :func:`load_torch_safely`), not registered
    process-wide at import.

    By injecting this plugin into the Trainer, internal checkpoint-loading paths that go
    through the CheckpointIO plugin inherit the safe default.  Callers that trust a
    checkpoint and need full unpickling can still pass ``weights_only=False`` explicitly.
    """

    def load_checkpoint(self, path, map_location=None, weights_only=None, **kwargs):
        return load_torch_safely(
            load_fn=super().load_checkpoint,
            path=path,
            scope="checkpoint",
            weights_only=weights_only,
            map_location=map_location,
            **kwargs,
        )


def likelihood_safe_globals() -> list:
    """Return Darts' own (data-only) classes to allow-list for a ``weights_only=True`` load.

    Only the ``TorchLikelihood`` subclasses and the ``LikelihoodType`` enum that Darts
    stores in a checkpoint's ``hyper_parameters`` are listed.
    """
    out: list = []
    try:
        import darts.utils.likelihood_models.torch  # noqa: F401 — register subclasses
        from darts.utils.likelihood_models.base import LikelihoodType
        from darts.utils.likelihood_models.torch import TorchLikelihood

        out = [LikelihoodType, *all_imported_subclasses(TorchLikelihood)]
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect Darts likelihood safe globals: {e}")
    return out


# ---------------------------------------------------------------------------
# Safe-base helpers by scope
# ---------------------------------------------------------------------------
def _checkpoint_safe_bases() -> tuple[type, ...]:
    """Known-safe base classes for Lightning ``.ckpt`` deserialization."""
    bases: list = []
    try:
        from torch.nn.modules.module import Module as _Mod
        from torch.optim import Optimizer as _Opt
        from torch.optim import lr_scheduler as _lrs

        bases += [_Mod, _Opt]
        bases.append(getattr(_lrs, "LRScheduler", getattr(_lrs, "_LRScheduler", _Mod)))
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect PyTorch safe globals: {e}")

    try:
        import torchmetrics

        bases += [torchmetrics.Metric, torchmetrics.MetricCollection]
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect TorchMetrics safe globals: {e}")

    bases += likelihood_safe_globals()
    return tuple(b for b in bases if inspect.isclass(b))


def _wrapper_safe_bases() -> tuple[type, ...]:
    """Known-safe base classes for Darts ``.pt`` wrapper deserialization.

    Extends the checkpoint bases with the full Darts / sklearn / Lightning class
    hierarchy.  Uses :func:`safe_base_classes` from ``base.py``, which already
    collects sklearn ``BaseEstimator``, Darts ``Encoder`` / ``BaseDataTransformer``,
    Lightning ``Callback``, etc.
    """
    return safe_base_classes()


def _safe_bases_for_scope(scope: LoadScope) -> tuple[type, ...]:
    """Return the appropriate safe-base tuple for ``scope``."""
    if scope == "wrapper":
        return _wrapper_safe_bases()
    return _checkpoint_safe_bases()


# ---------------------------------------------------------------------------
# Seed globals (types the inspection API may miss)
# ---------------------------------------------------------------------------
def _wrapper_seed_globals() -> list:
    """Globals not always reported by ``get_unsafe_globals_in_checkpoint``.

    Nested numpy / pandas / sklearn objects inside a ``.pt`` file can reference
    dtype classes, ``enum.Enum``, or random-state types that the inspection API
    omits.  Seeded for every wrapper load so legitimate Darts saves succeed.
    """
    out: list = [enum.Enum]
    try:
        import numpy.dtypes as _dtypes
        import numpy.random.mtrand as _mtrand

        out += [
            getattr(_dtypes, name)
            for name in dir(_dtypes)
            if name.endswith("DType") and not name.startswith("_")
        ]
        out.append(_mtrand.RandomState)
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect wrapper seed globals: {e}")

    try:
        from lightning_fabric.utilities.data import AttributeDict

        out.append(AttributeDict)
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect Lightning Fabric seed globals: {e}")

    return out


# ---------------------------------------------------------------------------
# File-inspection-driven safe globals (per scope)
# ---------------------------------------------------------------------------
def safe_globals_for_torch_file(path, *, scope: LoadScope = "checkpoint") -> list:
    """Inspect ``path`` and return referenced globals safe to allow-list for ``scope``.

    Uses ``torch.serialization.get_unsafe_globals_in_checkpoint`` (torch >= 2.6) to
    discover which globals the specific file needs, then keeps only those that pass
    :func:`is_allowed_global`.

    Anything else is left blocked so the load fails loudly instead of silently
    trusting an attacker-chosen global.  Returns ``[]`` when the file has no
    unsafe globals.
    """
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if not unsafe_globals:
        return []

    safe_bases = _safe_bases_for_scope(scope)
    extra_safe_callables = _ALL_SAFE_CALLABLES

    resolved: list = []
    blocked: list[str] = []
    for name in unsafe_globals:
        if not isinstance(name, str):
            continue

        if is_blocked_global(name):
            raise UnpicklingError(format_unpickling_error(name, path))

        obj = resolve_reference(name)
        if is_allowed_global(
            name,
            obj,
            safe_bases=safe_bases,
            extra_safe_callables=extra_safe_callables,
        ):
            resolved.append(obj)
        else:
            blocked.append(name)

    if blocked:
        raise UnpicklingError(format_unpickling_error(blocked[0], path))

    return resolved


def safe_globals_for_checkpoint(path) -> list:
    """Backward-compatible alias: checkpoint scope."""
    return safe_globals_for_torch_file(path, scope="checkpoint")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_torch_safely(
    load_fn: Callable[..., Any],
    path,
    *,
    extra_globals: list | None = None,
    scope: LoadScope = "checkpoint",
    weights_only: bool | None = True,
    **kwargs,
):
    """Run ``load_fn`` inside a *scoped* ``torch.serialization.safe_globals`` context
    seeded with the checkpoint-driven safe subset for ``path``.

    Scoped (a context manager around this one load) rather than a process-wide,
    import-time ``add_safe_globals`` registration: it only affects this call and
    auto-reverts, and nothing extra is imported / registered unless a file is
    actually loaded.

    Parameters
    ----------
    load_fn
        A ``torch.load``-backed callable.
    path
        Filesystem path to the ``.pt`` or ``.ckpt`` file.
    extra_globals
        Additional globals to allow-list for this load only.
    scope
        ``"checkpoint"`` for Lightning ``.ckpt`` files; ``"wrapper"`` for Darts
        ``.pt`` model wrapper files.
    weights_only
        Security-relevant flag forwarded to ``load_fn``.  ``True`` by default.
    **kwargs
        Passed through to ``load_fn``.
    """
    weights_only = True if weights_only is None else weights_only
    if not weights_only:
        return load_fn(path, weights_only=weights_only, **kwargs)

    seed_globals = _wrapper_seed_globals() if scope == "wrapper" else []
    allow = dedupe_by_identity(
        (extra_globals or [])
        + seed_globals
        + safe_globals_for_torch_file(path, scope=scope)
    )
    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=weights_only, **kwargs)
