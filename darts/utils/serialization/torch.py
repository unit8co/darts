"""
Torch Checkpoint Serialization
------------------------------

Safe-by-default loading for PyTorch Lightning ``.ckpt`` files and Darts ``.pt`` wrapper
files used by ``TorchForecastingModel``.

Two loading strategies, both guarded by the same
:func:`~darts.utils.serialization.base.is_allowed_global` filter:

* **checkpoint** (``.ckpt``) — Uses ``torch.load(weights_only=True)`` with a *scoped*
  ``safe_globals`` context seeded from the file's own referenced globals.
* **wrapper** (``.pt``) — Uses ``torch.load(weights_only=False,
  pickle_module=<restricted>)`` so that every ``find_class`` call flows through
  :class:`~darts.utils.serialization.base.RestrictedUnpickler`.  This avoids the
  ``SETITEMS`` limitation of ``weights_only=True`` (which cannot reconstruct
  ``AttributeDict`` and other dict subclasses used by Lightning callbacks) while
  still blocking arbitrary-code-execution (CWE-502).
"""

import inspect
from collections.abc import Callable
from typing import Any, Literal

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    SAFE_PICKLE_FUNCTIONS,
    RestrictedUnpickler,
    UnpicklingError,
    all_imported_subclasses,
    dedupe_by_identity,
    format_unpickling_error,
    is_allowed_global,
    is_blocked_global,
    resolve_reference,
)

logger = get_logger(__name__)

LoadScope = Literal["checkpoint"]

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
    checkpoint). Darts ``.ckpt`` files contain a few non-tensor objects (likelihoods,
    optimizer / scheduler classes, hparams, ...); the classes required to deserialize a
    *legitimate* Darts checkpoint under ``weights_only=True`` are allow-listed **at load
    time** and scoped to the load itself (see :func:`load_torch_safely`), not registered
    process-wide at import.

    By injecting this plugin into the Trainer, internal checkpoint-loading paths that go
    through the CheckpointIO plugin inherit the safe default. Callers that trust a
    checkpoint and need full unpickling can still pass ``weights_only=False`` explicitly.
    """

    def load_checkpoint(self, path, map_location=None, weights_only=None, **kwargs):
        return load_torch_safely(
            load_fn=super().load_checkpoint,
            path=path,
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


# ---------------------------------------------------------------------------
# File-inspection-driven safe globals (per scope)
# ---------------------------------------------------------------------------
def safe_globals_for_torch_file(path) -> list:
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

    safe_bases = _checkpoint_safe_bases()
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_torch_safely(
    load_fn: Callable[..., Any],
    path,
    *,
    extra_globals: list | None = None,
    weights_only: bool | None = True,
    **kwargs,
):
    """Run ``load_fn`` inside a *scoped* ``torch.serialization.safe_globals`` context
    seeded with the checkpoint-driven safe subset for ``path``.

    This function is used for Lightning ``.ckpt`` files (checkpoint scope) which
    contain actual tensor storages and benefit from ``torch.load``'s
    ``weights_only=True`` mode.

    For Darts ``.pt`` wrapper files use :func:`load_wrapper_safely` instead.

    Parameters
    ----------
    load_fn
        A ``torch.load``-backed callable.
    path
        Filesystem path to the ``.ckpt`` file.
    extra_globals
        Additional globals to allow-list for this load only.
    weights_only
        Security-relevant flag forwarded to ``load_fn``.  ``True`` by default.
    **kwargs
        Passed through to ``load_fn``.
    """
    weights_only = True if weights_only is None else weights_only
    if not weights_only:
        return load_fn(path, weights_only=weights_only, **kwargs)

    allow = dedupe_by_identity(
        (extra_globals or []) + safe_globals_for_torch_file(path)
    )
    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=weights_only, **kwargs)


# ---------------------------------------------------------------------------
# Wrapper (.pt) loading via pickle_module=RestrictedUnpickler
# ---------------------------------------------------------------------------


class _RestrictedPickleModule:
    """Module-like object injected as ``pickle_module`` into ``torch.load``.

    ``torch.load(weights_only=False, pickle_module=...)`` expects the module to
    expose an ``Unpickler`` class.  Its internal ``UnpicklerWrapper`` subclasses
    that class and delegates ``find_class`` via ``super()``, so every global in
    the pickle stream passes through our :class:`RestrictedUnpickler` filter.
    """

    Unpickler = RestrictedUnpickler


def load_wrapper_safely(
    path,
    *,
    trusted: bool = False,
    **kwargs,
):
    """Load a Darts ``.pt`` wrapper file with CWE-502 protection.

    Uses ``torch.load(weights_only=False, pickle_module=<restricted>)`` so that
    every ``find_class`` call flows through
    :class:`~darts.utils.serialization.base.RestrictedUnpickler`.  This avoids
    the ``SETITEMS`` limitation of ``weights_only=True`` (which cannot
    reconstruct ``AttributeDict`` and other dict subclasses used internally by
    Lightning callbacks) while still blocking arbitrary-code-execution.

    Parameters
    ----------
    path
        Filesystem path to the ``.pt`` wrapper file.
    trusted
        If ``True``, loads with unrestricted ``pickle`` (no ``find_class``
        filtering).  Only use for files from trusted sources.
    **kwargs
        Passed through to ``torch.load`` (e.g. ``map_location``).
    """
    if trusted:
        return torch.load(path, weights_only=False, **kwargs)

    return torch.load(
        path,
        weights_only=False,
        pickle_module=_RestrictedPickleModule,
        **kwargs,
    )
