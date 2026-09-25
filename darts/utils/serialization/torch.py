"""
Torch Checkpoint Serialization
----------------------------

Safe-by-default loading for PyTorch Lightning ``.ckpt`` files used by Darts torch models.
"""

import inspect
from collections.abc import Callable
from typing import Any

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    TrustedPrefixPolicy,
    all_imported_subclasses,
    dedupe_by_identity,
    resolve_reference,
)

logger = get_logger(__name__)

# Trusted package prefixes used by the checkpoint-driven allow-list below. A global
# referenced by a checkpoint is only auto-allow-listed for a ``weights_only=True`` load if it
# is a CLASS (never a function/callable) that lives under one of these packages AND subclasses
# one of the known-safe bases. This lets a legitimate Darts ``.ckpt`` load without
# pre-registering all of torch/torchmetrics at import time, while still refusing arbitrary,
# attacker-chosen globals.
TRUSTED_PREFIXES: TrustedPrefixPolicy = (
    "torch.",
    "torchmetrics.",
    "darts.",
    "neuralforecast.",
)

# ``torchmetrics`` metrics store references to a few of their own tensor-reduction / distributed
# helper *functions* in their pickled state (e.g. ``dim_zero_sum``), so a ``weights_only=True``
# load of a metric-bearing checkpoint needs these specific callables allow-listed. We list them
# by EXACT qualified name (never a blanket module scan) and only ever add the ones a given
# checkpoint actually references. HONESTY NOTE: allow-listing a callable is not zero-risk --
# PyTorch may invoke a registered function while unpickling -- so this is deliberately a tiny,
# audited set of pure tensor-reduction helpers that operate only on the metric's own state.
TORCHMETRICS_SAFE_FUNCTIONS = frozenset({
    "torchmetrics.utilities.data.dim_zero_cat",
    "torchmetrics.utilities.data.dim_zero_sum",
    "torchmetrics.utilities.data.dim_zero_mean",
    "torchmetrics.utilities.data.dim_zero_max",
    "torchmetrics.utilities.data.dim_zero_min",
    "torchmetrics.metric.jit_distributed_available",
})


class DartsCheckpointIO(TorchCheckpointIO):
    """Custom CheckpointIO that defaults ``weights_only`` to ``True`` (safe-by-default).

    PyTorch >= 2.6 changed ``torch.load`` to default to ``weights_only=True`` as a mitigation
    against CWE-502 (arbitrary code execution when unpickling an untrusted checkpoint). Darts
    ``.ckpt`` files contain a few non-tensor objects (likelihoods, optimizer/scheduler classes,
    hparams, ...); the classes required to deserialize a *legitimate* Darts checkpoint under
    ``weights_only=True`` are allow-listed **at load time** and scoped to the load itself (see
    :func:`load_torch_safely`), not registered process-wide at import.

    By injecting this plugin into the Trainer, internal checkpoint-loading paths that go through
    the CheckpointIO plugin inherit the safe default. Callers that trust a checkpoint and need
    full unpickling can still pass ``weights_only=False`` explicitly (e.g. the training-resume
    path, which needs full optimizer state).
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

    Only the ``TorchLikelihood`` subclasses and the ``LikelihoodType`` enum that Darts stores
    in a checkpoint's ``hyper_parameters`` are listed. These are plain ``type`` objects (never
    callables), so allow-listing them does not re-open the code-execution surface that
    ``weights_only=True`` closes. Everything else a legitimate checkpoint needs is added on
    demand, per file, by :func:`safe_globals_for_checkpoint` -- we deliberately do NOT
    mass-scan/allow-list all of ``torch``/``torchmetrics`` at import time.
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


def safe_globals_for_checkpoint(path) -> list:
    """Inspect ``path`` and return the subset of its referenced globals that are safe to
    allow-list for a ``weights_only=True`` load.

    Uses ``torch.serialization.get_unsafe_globals_in_checkpoint`` (torch >= 2.6) to see which
    globals the specific file needs, then keeps only those that:

    - Are a class; never a function/callable, which PyTorch may *call* during unpickling
    - Live under a trusted package (torch/torchmetrics/darts/neuralforecast)
    - Subclass a known-safe base (``nn.Module``, ``Optimizer``, an LR scheduler, a ``torchmetrics``
      metric/collection, or a Darts likelihood), OR is one of the exact, audited ``torchmetrics`` reduction functions
      in :data:`TORCHMETRICS_SAFE_FUNCTIONS` (metrics pickle references to these).

    Anything else is left blocked so the load fails loudly instead of silently trusting an attacker-chosen global.
    Best-effort: returns ``[]`` on torch versions without the inspection API, or on any error.
    """
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if not unsafe_globals:
        return []

    bases: list = []
    # torch safe globals
    try:
        from torch.nn.modules.module import Module as _Mod
        from torch.optim import Optimizer as _Opt
        from torch.optim import lr_scheduler as _lrs

        bases += [_Mod, _Opt]
        bases.append(getattr(_lrs, "LRScheduler", getattr(_lrs, "_LRScheduler", _Mod)))
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect PyTorch safe globals: {e}")

    # torchmetrics safe globals
    try:
        import torchmetrics

        bases += [torchmetrics.Metric, torchmetrics.MetricCollection]
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect TorchMetrics safe globals: {e}")

    # Darts likelihood safe globals
    bases += likelihood_safe_globals()

    safe_bases = tuple(b for b in bases if inspect.isclass(b))
    resolved: list = []
    for name in unsafe_globals:
        if not isinstance(name, str):
            continue

        # (a) exact, audited torchmetrics reduction *functions* the metrics store in their state
        if name in TORCHMETRICS_SAFE_FUNCTIONS:
            obj = resolve_reference(name)
            if callable(obj):
                resolved.append(obj)
            continue

        # (b) classes under a trusted package that subclass a known-safe base
        if not name.startswith(TRUSTED_PREFIXES):
            continue

        obj = resolve_reference(name)
        if inspect.isclass(obj) and (not safe_bases or issubclass(obj, safe_bases)):
            resolved.append(obj)
    return resolved


def load_torch_safely(
    load_fn: Callable[..., Any],
    path,
    *,
    extra_globals: list | None = None,
    weights_only: bool | None = True,
    **kwargs,
):
    """Run ``load_fn`` (a ``weights_only=True`` ``torch.load``-backed call) inside a *scoped*
    ``torch.serialization.safe_globals`` context seeded with Darts' minimal allow-list plus the
    checkpoint-driven safe subset for ``path``.

    Scoped (a context manager around this one load) rather than a process-wide, import-time
    ``add_safe_globals`` registration: it only affects this call and auto-reverts, and nothing
    extra is imported/registered unless a checkpoint is actually loaded. On torch < 2.6 (no
    ``safe_globals``) it just calls ``load_fn`` directly.
    """
    weights_only = True if weights_only is None else weights_only
    if not weights_only:
        return load_fn(path, weights_only=weights_only, **kwargs)

    allow = dedupe_by_identity(
        (extra_globals or []) + safe_globals_for_checkpoint(path)
    )
    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=weights_only, **kwargs)
