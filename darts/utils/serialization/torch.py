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

from collections.abc import Callable
from typing import Any

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    RestrictedUnpickler,
    UnpicklingError,
    _normalize_qualname,
    _qualname_set,
    dedupe_by_identity,
    format_unpickling_error,
    is_allowed_global,
    is_blocked_global,
    resolve_reference,
)

logger = get_logger(__name__)


class _DartsCheckpointIO(TorchCheckpointIO):
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


# ---------------------------------------------------------------------------
# File-inspection-driven safe globals (per scope)
# ---------------------------------------------------------------------------
def safe_globals_for_torch_file(
    path, *, extra_allowed: frozenset[str] | None = None
) -> list:
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

    resolved: list = []
    blocked: list[str] = []
    for name in unsafe_globals:
        if not isinstance(name, str):
            continue
        name = _normalize_qualname(name)

        if is_blocked_global(name):
            raise UnpicklingError(format_unpickling_error(name, path))

        obj = resolve_reference(name)
        if is_allowed_global(
            name,
            obj,
            safe_bases=tuple(),
            extra_safe_callables=extra_allowed,
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
    trusted_classes: list[type] | None = None,
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
    trusted_classes
        Optional classes to allow for this load only, in addition to the
        checkpoint-driven allow-list.  Ignored when ``weights_only`` is ``False``.
    weights_only
        Security-relevant flag forwarded to ``load_fn``.  ``True`` by default.
    **kwargs
        Passed through to ``load_fn``.
    """
    weights_only = True if weights_only is None else weights_only
    if not weights_only:
        return load_fn(path, weights_only=weights_only, **kwargs)

    allow = dedupe_by_identity(
        list(trusted_classes or [])
        + safe_globals_for_torch_file(
            path, extra_allowed=_qualname_set(trusted_classes)
        )
    )
    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=weights_only, **kwargs)


# ---------------------------------------------------------------------------
# Wrapper (.pt) loading via pickle_module=RestrictedUnpickler
# ---------------------------------------------------------------------------
def load_wrapper_safely(
    path,
    *,
    trusted: bool = False,
    trusted_classes: list[type] | None = None,
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
    trusted_classes
        Optional classes to allow during restricted loading.  Ignored when
        ``trusted`` is ``True``.
    **kwargs
        Passed through to ``torch.load`` (e.g. ``map_location``).
    """
    if trusted:
        return torch.load(path, weights_only=False, **kwargs)

    # make ``torch.load()`` use Darts' ``RestrictedUnpickler`` for safe-loading
    class _RestrictedPickleModule:
        class Unpickler(RestrictedUnpickler):
            def __init__(self, file, **unpickler_kwargs):
                super().__init__(
                    file,
                    trusted_classes=trusted_classes,
                    **unpickler_kwargs,
                )

    return torch.load(
        path,
        weights_only=False,
        pickle_module=_RestrictedPickleModule,
        **kwargs,
    )
