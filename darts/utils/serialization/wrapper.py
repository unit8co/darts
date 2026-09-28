"""
Torch model wrapper
-------------------

Torch-only persistence for ``TorchForecastingModel`` ``.pt`` and ``_model.pth.tar``
files. The object graph (series, encoders, callbacks, user classes) is encoded by
:mod:`darts.utils.serialization.base`, which does not import PyTorch.

This module writes that state with ``torch.save`` and reads it with
``torch.load(..., weights_only=True)``. Numeric arrays are stored as tensors so
that loader needs no extra globals. Tensors, devices, dtypes, and ``nn.Module``
instances are the only types handled here.

Legacy pickles are refused unless the caller passes ``weights_only=False``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from darts.utils.serialization.base import (
    KIND_TAG,
    STATE_UNHANDLED,
    UnencodableObjectError,
    data_state,
    decode_state,
    encode_state,
    import_class,
    qualname,
)

WRAPPER_VERSION = 1

# Rebuilt by ``nn.Module.__init__`` on the torch version that loads the file.
_MODULE_INTERNALS = frozenset({
    "training",
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
})


class LegacyModelFormatError(ValueError):
    """A wrapper file is a legacy pickle and ``weights_only=False`` was not passed."""


def save_torch_wrapper(model, path) -> None:
    """Write ``model`` as a torch wrapper file.

    Encoding finishes before the file is created, so a value that cannot be stored
    does not leave a partial file. The object graph comes from :func:`encode_state`.
    """
    payload = _arrays_to_tensors(_build_payload(model))
    with open(path, "wb") as handle:
        torch.save(payload, handle)


def load_torch_wrapper(path, *, weights_only: bool = True, map_location=None):
    """Load a torch wrapper file.

    ``weights_only=True`` (default) accepts only this state-dict format and rebuilds
    the model with :func:`decode_state`. ``weights_only=False`` also unpickles legacy
    files and can execute code in them.
    """
    if not weights_only:
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if _is_payload(loaded):
            return _restore(loaded)
        return loaded

    try:
        loaded = torch.load(path, weights_only=True, map_location=map_location)
    except Exception as exc:
        raise _legacy_error(path) from exc
    if not _is_payload(loaded):
        raise _legacy_error(path)
    return _restore(loaded)


def _legacy_error(path) -> LegacyModelFormatError:
    """Error for a pickle wrapper. Names ``path`` and the ``weights_only=False`` opt-in."""
    return LegacyModelFormatError(
        f"Model file `{path}` uses the legacy pickle format and cannot be loaded safely. "
        "Re-save it with this version of Darts, or pass `weights_only=False` if you trust "
        "the file. Loading with `weights_only=False` can execute code contained in the file."
    )


def _is_payload(obj) -> bool:
    """Whether ``obj`` is a wrapper dict for :data:`WRAPPER_VERSION`."""
    return isinstance(obj, dict) and obj.get("darts_wrapper") == WRAPPER_VERSION


def _build_payload(model) -> dict:
    """In-memory wrapper dict. Numeric arrays are still NumPy at this point."""
    model_path = qualname(type(model))
    try:
        import_class(model_path)
    except ValueError as exc:
        raise UnencodableObjectError(model, model_path) from exc
    return {
        "darts_wrapper": WRAPPER_VERSION,
        "model_cls": model_path,
        "state": encode_state(model.__getstate__(), path="state", extra=_encode_torch),
    }


def _restore(payload: dict):
    """Build the model with ``__new__`` and ``__setstate__``. ``__init__`` is not called."""
    cls = _resolve_model_class(payload["model_cls"])
    model = cls.__new__(cls)
    model.__setstate__(decode_state(payload["state"], extra=_decode_torch))
    return model


def _resolve_model_class(model_cls: str):
    """Import the root class. It must subclass ``TorchForecastingModel``."""
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

    try:
        cls = import_class(model_cls, base=TorchForecastingModel)
    except ValueError as exc:
        raise ValueError(
            f"Cannot import model class `{model_cls}`. It must be a "
            "TorchForecastingModel subclass defined in an importable module."
        ) from exc
    return cls


def _arrays_to_tensors(obj):
    """Replace NumPy arrays with CPU tensors.

    ``weights_only=True`` accepts tensors, not ndarrays. Tensors already in the
    tree (parameters, buffers) are left unchanged.
    """
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj)).clone()
    if isinstance(obj, dict):
        return {key: _arrays_to_tensors(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_arrays_to_tensors(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_arrays_to_tensors(value) for value in obj)
    return obj


def _encode_torch(obj, path, stack):
    """Encode a tensor, device, dtype, or ``nn.Module``. Anything else is declined."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, torch.device):
        return {KIND_TAG: "device", "type": obj.type, "index": obj.index}
    if isinstance(obj, torch.dtype):
        return {
            KIND_TAG: "torch_dtype",
            "name": str(obj).removeprefix("torch."),
        }
    if isinstance(obj, nn.Module):
        return _encode_module(obj, path, stack)
    return STATE_UNHANDLED


def _encode_module(module: nn.Module, path: str, stack: set[int]) -> dict:
    """Public attributes plus ``state_dict``. Private module internals are skipped."""
    cls_path = qualname(type(module))
    try:
        import_class(cls_path, base=nn.Module)
    except ValueError as exc:
        raise UnencodableObjectError(module, path) from exc
    attrs = {
        key: value
        for key, value in data_state(module).items()
        if not key.startswith("_") and key not in _MODULE_INTERNALS
    }
    return {
        KIND_TAG: "nn_module",
        "cls": cls_path,
        "attrs": encode_state(attrs, path=path, stack=stack, extra=_encode_torch),
        "state": encode_state(
            dict(module.state_dict()),
            path=f"{path}.state",
            stack=stack,
            extra=_encode_torch,
        ),
    }


def _decode_torch(obj):
    """Decode a tensor or a torch record. Anything else is declined."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        decoder = _TORCH_DECODERS.get(obj.get(KIND_TAG))
        if decoder is not None:
            return decoder(obj)
    return STATE_UNHANDLED


def _decode_device(obj):
    """A ``torch.device``. No index means a device such as CPU."""
    if obj["index"] is None:
        return torch.device(obj["type"])
    return torch.device(obj["type"], obj["index"])


def _decode_torch_dtype(obj):
    """Restore a real ``torch.dtype`` attribute. The name must be an identifier."""
    name = obj["name"]
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Refusing to restore torch dtype `{name}`.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Refusing to restore torch dtype `{name}`.")
    return dtype


def _decode_module(obj):
    """``nn.Module.__init__``, then saved public attributes and ``load_state_dict``."""
    cls = import_class(obj["cls"], base=nn.Module)
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    for key, value in decode_state(obj["attrs"], extra=_decode_torch).items():
        setattr(module, key, value)
    state = decode_state(obj["state"], extra=_decode_torch)
    if state:
        module.load_state_dict(state)
    return module


_TORCH_DECODERS = {
    "device": _decode_device,
    "torch_dtype": _decode_torch_dtype,
    "nn_module": _decode_module,
}
