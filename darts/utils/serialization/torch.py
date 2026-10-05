"""
Torch Checkpoint Serialization
------------------------------

Safe-by-default loading for PyTorch Lightning ``.ckpt`` files and Darts ``.pt`` wrapper files used by
``TorchForecastingModel``.

Two loading strategies, both guarded by the same :func:`~darts.utils.serialization.base.is_allowed_global` filter:

* **checkpoint** (``.ckpt``) — Uses ``torch.load(weights_only=True)`` with a *scoped* ``safe_globals`` context seeded
  from the file's own referenced globals.
* **wrapper** (``.pt``) — Uses ``torch.load(weights_only=False, pickle_module=<restricted>)`` so that every
  ``find_class`` call flows through :class:`~darts.utils.serialization.base.RestrictedUnpickler`.  This avoids the
  ``SETITEMS`` limitation of ``weights_only=True`` (which cannot reconstruct ``AttributeDict`` and other dict
  subclasses used by Lightning callbacks) while still blocking arbitrary-code-execution (CWE-502).
"""

from collections.abc import Callable
from typing import Any

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    RestrictedUnpickler,
    _raise_unsafe_global,
    is_allowed_global,
)
from darts.utils.serialization.registry import (
    UserSafeGlobals,
    _get_user_safe_globals,
    _resolve_allowed_global,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wrapper (.pt) loading via pickle_module=RestrictedUnpickler
# ---------------------------------------------------------------------------
def load_wrapper_safely(
    path,
    *,
    trusted: bool = False,
    **kwargs,
):
    """Load a Darts ``.pt`` wrapper file with CWE-502 protection.

    Uses ``torch.load(weights_only=False, pickle_module=<restricted>)`` so that every ``find_class`` call flows through
    :class:`~darts.utils.serialization.base.RestrictedUnpickler`. This avoids the ``SETITEMS`` limitation of
    ``weights_only=True`` (which cannot reconstruct ``AttributeDict`` and other dict subclasses used internally by
    Lightning callbacks) while still blocking arbitrary-code-execution.

    Parameters
    ----------
    path
        Filesystem path to the ``.pt`` wrapper file.
    trusted
        If ``True``, disables safe-loading restrictions and fully unpickles the file (CWE-502 opt-out) with
        `weights_only=False`. Only use for files from trusted sources. Default: ``False``.
    **kwargs
        Passed through to ``torch.load`` (e.g. ``map_location``).
    """
    if trusted:
        return torch.load(path, weights_only=False, **kwargs)

    user_globals = _get_user_safe_globals()

    class _RestrictedPickleModule:
        class Unpickler(RestrictedUnpickler):
            def __init__(self, file, **unpickler_kwargs):
                super().__init__(
                    file,
                    extra_user_globals=user_globals,
                    **unpickler_kwargs,
                )

    return torch.load(
        path,
        weights_only=False,
        pickle_module=_RestrictedPickleModule,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Checkpoint (.pt.ckpt) loading via torch-side loading + safe globals
# ---------------------------------------------------------------------------
def load_ckpt_safely(
    load_fn: Callable[..., Any],
    path,
    *,
    trusted: bool = False,
    weights_only: bool | None = None,
    **kwargs,
):
    """Run ``load_fn`` inside a *scoped* ``torch.serialization.safe_globals`` context.

    This function is used for Lightning ``.ckpt`` files (checkpoint scope) which contain actual tensor storages and
    benefit from ``torch.load``'s ``weights_only=True`` mode.

    Parameters
    ----------
    load_fn
        A ``torch.load``-backed callable.
    path
        Filesystem path to the ``.ckpt`` file.
    trusted
        If ``True``, disables safe-loading restrictions and fully unpickles the file (CWE-502 opt-out) with
        `weights_only=False`. Only use for files from trusted sources. Default: ``False``.
    weights_only
        Internal override forwarded to ``load_fn`` when set explicitly by PyTorch Lightning (``None`` means safe
        loading unless ``trusted`` is ``True``).
    **kwargs
        Passed through to ``load_fn``.
    """
    if trusted:
        return load_fn(path, weights_only=False, **kwargs)

    use_weights_only = True if weights_only is None else weights_only
    if not use_weights_only:
        return load_fn(path, weights_only=False, **kwargs)

    registry = _get_user_safe_globals()
    from_file = _safe_globals_for_torch_file(path, extra_user_globals=registry)
    merged: UserSafeGlobals = {**from_file, **registry}
    allow = list(zip(merged.values(), merged.keys()))

    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=True, **kwargs)


class _DartsCheckpointIO(TorchCheckpointIO):
    """Custom CheckpointIO that defaults to safe ``weights_only`` loading.

    PyTorch >= 2.6 changed ``torch.load`` to default to ``weights_only=True`` as a mitigation against CWE-502
    (arbitrary code execution when unpickling an untrusted checkpoint). Darts ``.ckpt`` files contain a few non-tensor
    objects (likelihoods, optimizer / scheduler classes, hparams, ...); the classes required to deserialize a
    *legitimate* Darts checkpoint under ``weights_only=True`` are allow-listed **at load time** and scoped to the load
    itself (see :func:`load_ckpt_safely`), not registered process-wide at import.

    By injecting this plugin into the Trainer, internal checkpoint-loading paths that go through the CheckpointIO
    plugin inherit the safe default. Unsafe full unpickling is only used when ``load_ckpt_safely`` is called with
    ``trusted=True``.
    """

    def load_checkpoint(self, path, map_location=None, weights_only=None, **kwargs):
        trusted = False if weights_only is None else not weights_only
        return load_ckpt_safely(
            load_fn=super().load_checkpoint,
            path=path,
            trusted=trusted,
            weights_only=weights_only,
            map_location=map_location,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# File-inspection-driven safe globals (per scope)
# ---------------------------------------------------------------------------
def _safe_globals_for_torch_file(
    path,
    *,
    extra_user_globals: UserSafeGlobals | None = None,
) -> UserSafeGlobals:
    """Inspect ``path`` and return referenced globals safe to allow-list for ``scope``.

    Uses ``torch.serialization.get_unsafe_globals_in_checkpoint`` (torch >= 2.6) to discover which globals the specific
    file needs, then keeps only those that pass :func:`is_allowed_global`.

    Anything else is left blocked so the load fails loudly instead of silently trusting an attacker-chosen global.
    Returns an empty mapping when the file has no unsafe globals.
    """
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if not unsafe_globals:
        return {}

    resolved: UserSafeGlobals = {}
    for qualname in unsafe_globals:
        if is_allowed_global(
            qualname=qualname,
            extra_user_globals=extra_user_globals,
        ):
            obj = _resolve_allowed_global(qualname, extra_user_globals)
            resolved[qualname] = obj
        else:
            _raise_unsafe_global(qualname)

    return resolved
