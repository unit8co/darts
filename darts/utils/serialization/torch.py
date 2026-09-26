"""
Torch Checkpoint Serialization
----------------------------

Safe-by-default loading for PyTorch Lightning ``.ckpt`` files and Darts ``.pt`` wrapper files.
"""

import enum
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from darts.logging import get_logger
from darts.utils.serialization.base import (
    CHECKPOINT_TRUSTED_PREFIXES,
    WRAPPER_SAFE_FUNCTIONS,
    LoadScope,
    all_imported_subclasses,
    dedupe_by_identity,
    format_unpickling_error,
    is_allowed_global,
    is_blocked_global,
    resolve_reference,
)

logger = get_logger(__name__)

# Backward-compatible alias for callers/tests that import this name directly.
TRUSTED_PREFIXES = CHECKPOINT_TRUSTED_PREFIXES

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
    demand, per file, by :func:`safe_globals_for_torch_file` -- we deliberately do NOT
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


def wrapper_safe_bases() -> tuple[type, ...]:
    """Known-safe base classes for Darts ``.pt`` wrapper deserialization."""
    bases: list = list(_checkpoint_safe_bases())

    try:
        from darts import TimeSeries
        from darts.dataprocessing.encoders.encoder_base import Encoder, SingleEncoder
        from darts.dataprocessing.encoders.encoders import SequentialEncoder
        from darts.dataprocessing.transformers.base_data_transformer import (
            BaseDataTransformer,
        )
        from darts.dataprocessing.transformers.fittable_data_transformer import (
            FittableDataTransformer,
        )
        from darts.dataprocessing.transformers.invertible_data_transformer import (
            InvertibleDataTransformer,
        )
        from darts.models.forecasting.forecasting_model import (
            ForecastingModel,
            GlobalForecastingModel,
        )
        from darts.models.forecasting.torch_forecasting_model import (
            TorchForecastingModel,
        )
        from darts.utils.data.torch_datasets.utils import (
            TorchInferenceSample,
            TorchSample,
            TorchTrainingSample,
        )

        bases += [
            ForecastingModel,
            GlobalForecastingModel,
            TorchForecastingModel,
            *all_imported_subclasses(TorchForecastingModel),
            TimeSeries,
            Encoder,
            SingleEncoder,
            SequentialEncoder,
            *all_imported_subclasses(Encoder),
            BaseDataTransformer,
            FittableDataTransformer,
            InvertibleDataTransformer,
            *all_imported_subclasses(BaseDataTransformer),
            TorchSample,
            TorchTrainingSample,
            TorchInferenceSample,
        ]
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect Darts wrapper safe globals: {e}")

    try:
        from pytorch_lightning.callbacks import Callback

        bases += [Callback, *all_imported_subclasses(Callback)]
    except Exception as e:  # pragma: no cover - defensive only
        logger.debug(f"Could not collect PyTorch Lightning callback safe globals: {e}")

    return tuple(b for b in bases if inspect.isclass(b))


def _safe_bases_for_scope(scope: LoadScope) -> tuple[type, ...]:
    if scope == "wrapper":
        return wrapper_safe_bases()
    return _checkpoint_safe_bases()


def wrapper_seed_globals() -> list:
    """Return wrapper-scope globals not always reported by the torch inspection API.

    Nested numpy/pandas/sklearn objects inside a ``.pt`` file can reference dtype classes,
    ``enum.Enum``, or random-state types that ``get_unsafe_globals_in_checkpoint`` omits.
    These are seeded for every wrapper load so legitimate Darts saves succeed without
    requiring full unpickling.
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


def safe_globals_for_torch_file(path, *, scope: LoadScope = "checkpoint") -> list:
    """Inspect ``path`` and return referenced globals safe to allow-list for ``scope``.

    Uses ``torch.serialization.get_unsafe_globals_in_checkpoint`` (torch >= 2.6) to see which
    globals the specific file needs, then keeps only those that pass :func:`is_allowed_global`.

    Anything else is left blocked so the load fails loudly instead of silently trusting an
    attacker-chosen global. Best-effort: returns ``[]`` when the file has no unsafe globals.
    """
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if not unsafe_globals:
        return []

    safe_bases = _safe_bases_for_scope(scope)
    checkpoint_safe_functions = TORCHMETRICS_SAFE_FUNCTIONS
    wrapper_safe_functions = WRAPPER_SAFE_FUNCTIONS if scope == "wrapper" else None

    resolved: list = []
    blocked: list[str] = []
    for name in unsafe_globals:
        if not isinstance(name, str):
            continue

        if is_blocked_global(name):
            raise UnpicklingError(format_unpickling_error(name, path, scope=scope))

        obj = resolve_reference(name)
        if is_allowed_global(
            name,
            obj,
            scope=scope,
            safe_bases=safe_bases,
            checkpoint_safe_functions=checkpoint_safe_functions,
            wrapper_safe_functions=wrapper_safe_functions,
        ):
            resolved.append(obj)
        else:
            blocked.append(name)

    if blocked:
        raise UnpicklingError(format_unpickling_error(blocked[0], path, scope=scope))

    return resolved


def safe_globals_for_checkpoint(path) -> list:
    """Backward-compatible alias for :func:`safe_globals_for_torch_file` with checkpoint scope."""
    return safe_globals_for_torch_file(path, scope="checkpoint")


def load_torch_safely(
    load_fn: Callable[..., Any],
    path,
    *,
    extra_globals: list | None = None,
    scope: LoadScope = "checkpoint",
    weights_only: bool | None = True,
    **kwargs,
):
    """Run ``load_fn`` (a ``weights_only=True`` ``torch.load``-backed call) inside a *scoped*
    ``torch.serialization.safe_globals`` context seeded with the checkpoint-driven safe subset
    for ``path``.

    Scoped (a context manager around this one load) rather than a process-wide, import-time
    ``add_safe_globals`` registration: it only affects this call and auto-reverts, and nothing
    extra is imported/registered unless a file is actually loaded.
    """
    weights_only = True if weights_only is None else weights_only
    if not weights_only:
        return load_fn(path, weights_only=weights_only, **kwargs)

    seed_globals = wrapper_seed_globals() if scope == "wrapper" else []
    allow = dedupe_by_identity(
        (extra_globals or [])
        + seed_globals
        + safe_globals_for_torch_file(path, scope=scope)
    )
    with torch.serialization.safe_globals(allow):
        return load_fn(path, weights_only=weights_only, **kwargs)


load_torch_wrapper_safely = partial(load_torch_safely, scope="wrapper")


class UnpicklingError(RuntimeError):
    """Raised when a global referenced by a serialized file is not allow-listed."""
