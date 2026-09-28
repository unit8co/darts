"""
Safe TorchForecastingModel wrapper format
------------------------------------------

The ``.pt`` / ``_model.pth.tar`` wrapper is a Darts state dict of primitives and tensors.
``torch.load(..., weights_only=True)`` reads it with no extra globals. Darts rebuilds the
model afterwards, so the file cannot choose which code runs.

Legacy pickles (full object graphs from older saves) are refused unless the caller passes
``weights_only=False``.
"""

from __future__ import annotations

import datetime
import enum
import importlib
import inspect
import pathlib
import sys
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WRAPPER_VERSION = 1

# Classes reconstructed from a name in the file. Import happens here, never via pickle.
_CALLBACK_PREFIXES = (
    "pytorch_lightning.callbacks.",
    "lightning.pytorch.callbacks.",
)
_LOGGER_PREFIXES = (
    "pytorch_lightning.loggers.",
    "lightning.pytorch.loggers.",
)
_METRIC_PREFIXES = ("torchmetrics.",)
_SKLEARN_PREFIXES = ("sklearn.preprocessing.",)
_ENUM_PREFIXES = ("darts.", "torch.")

_MISSING = object()


class UnencodableObjectError(TypeError):
    """Raised when a value cannot be stored in the safe wrapper format."""

    def __init__(self, obj: Any, path: str):
        label = _describe(obj)
        super().__init__(
            f"Cannot store {label} ({path}) in the safe model format. "
            "Custom functions and unsupported third-party objects cannot be saved this way."
        )


class LegacyModelFormatError(ValueError):
    """Raised when a wrapper file is a legacy pickle and was not explicitly trusted."""


def save_torch_wrapper(model, path) -> None:
    """Write ``model`` as a safe wrapper file. Raises before creating the file if it cannot."""
    payload = _build_payload(model)
    with open(path, "wb") as handle:
        torch.save(payload, handle)


def load_torch_wrapper(path, *, weights_only: bool = True, map_location=None):
    """Load a wrapper file.

    ``weights_only=True`` (default) accepts only the safe state-dict format.
    ``weights_only=False`` also unpickles legacy files and can execute code in them.
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
    return LegacyModelFormatError(
        f"Model file `{path}` uses the legacy pickle format and cannot be loaded safely. "
        "Re-save it with this version of Darts, or pass `weights_only=False` if you trust "
        "the file. Loading with `weights_only=False` can execute code contained in the file."
    )


def _is_payload(obj) -> bool:
    return isinstance(obj, dict) and obj.get("darts_wrapper") == WRAPPER_VERSION


def _build_payload(model) -> dict:
    return {
        "darts_wrapper": WRAPPER_VERSION,
        "model_cls": _qualname(type(model)),
        "state": _encode(model.__getstate__(), path="state"),
    }


def _restore(payload: dict):
    cls = _resolve_model_class(payload["model_cls"])
    model = cls.__new__(cls)
    model.__setstate__(_decode(payload["state"]))
    return model


def _resolve_model_class(model_cls: str):
    if not isinstance(model_cls, str) or not model_cls.startswith("darts.models."):
        raise ValueError(f"Refusing to import model class `{model_cls}`.")
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

    cls = _import_qualified(model_cls)
    if not (inspect.isclass(cls) and issubclass(cls, TorchForecastingModel)):
        raise ValueError(f"`{model_cls}` is not a TorchForecastingModel subclass.")
    return cls


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _describe(obj: Any) -> str:
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return f"function `{getattr(obj, '__qualname__', type(obj).__name__)}`"
    cls = obj if inspect.isclass(obj) else type(obj)
    return f"`{cls.__module__}.{cls.__qualname__}`"


def _import_qualified(qualname: str):
    if (
        not isinstance(qualname, str)
        or not qualname
        or ".." in qualname
        or qualname.startswith(".")
    ):
        raise ValueError(f"Refusing to import `{qualname}`.")
    parts = qualname.split(".")
    if any(not part.isidentifier() or part.startswith("__") for part in parts):
        raise ValueError(f"Refusing to import `{qualname}`.")

    for i in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        obj = module
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"Cannot import `{qualname}`.")


def _resolve_class(
    qualname: str, *, prefixes: tuple[str, ...], base: type | None = None
):
    if not isinstance(qualname, str) or not qualname.startswith(prefixes):
        raise ValueError(f"Refusing to import `{qualname}`.")
    cls = _import_qualified(qualname)
    if not inspect.isclass(cls):
        raise ValueError(f"`{qualname}` is not a class.")
    if base is not None and not issubclass(cls, base):
        raise ValueError(f"`{qualname}` is not a `{base.__name__}` subclass.")
    return cls


def _class_from_loaded_module(qualname: str):
    """Return a class already imported by this process.

    This does not import anything. A name in the file cannot load new code; it can
    only point at a class the application has already imported.
    """
    if (
        not isinstance(qualname, str)
        or ".." in qualname
        or qualname.startswith(".")
        or not qualname
    ):
        return None
    parts = qualname.split(".")
    if any(not part.isidentifier() or part.startswith("__") for part in parts):
        return None
    for i in range(len(parts) - 1, 0, -1):
        module = sys.modules.get(".".join(parts[:i]))
        if module is None:
            continue
        obj = module
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if inspect.isclass(obj):
            return obj
    return None


def _resolve_component_class(qualname: str, *, prefixes: tuple[str, ...], base: type):
    """Resolve a callback, logger, or metric.

    Library classes under ``prefixes`` are imported. A user's own subclass is
    restored only when that module is already imported.
    """
    if isinstance(qualname, str) and qualname.startswith(prefixes):
        return _resolve_class(qualname, prefixes=prefixes, base=base)
    cls = _class_from_loaded_module(qualname)
    if cls is None:
        raise ValueError(
            f"Cannot restore `{qualname}` because it is not imported. "
            "Import the class before loading this model."
        )
    if not issubclass(cls, base):
        raise ValueError(f"Refusing to restore `{qualname}`.")
    return cls


def _encode(obj: Any, *, path: str, stack: set[int] | None = None):
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, enum.Enum):
        return _encode_enum(obj)
    if obj is None or isinstance(obj, bool | int | float | str | bytes | complex):
        return obj

    if stack is None:
        stack = set()
    obj_id = id(obj)
    if obj_id in stack:
        raise UnencodableObjectError(obj, path)
    stack.add(obj_id)
    try:
        return _encode_container(obj, path=path, stack=stack)
    finally:
        stack.discard(obj_id)


def _encode_enum(obj: enum.Enum) -> dict:
    qualname = _qualname(type(obj))
    if not qualname.startswith(_ENUM_PREFIXES):
        raise UnencodableObjectError(obj, qualname)
    return {"__darts_kind__": "enum", "cls": qualname, "name": obj.name}


def _encode_container(obj: Any, *, path: str, stack: set[int]):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj, path)
    if isinstance(obj, np.random.RandomState):
        return {
            "__darts_kind__": "random_state",
            "state": _encode(obj.get_state(), path=f"{path}.state", stack=stack),
        }
    dtype_name = _numpy_dtype_name(obj)
    if dtype_name is not None:
        return {
            "__darts_kind__": "dtype",
            "name": dtype_name,
            "as_dtype": isinstance(obj, np.dtype),
        }
    if isinstance(obj, pathlib.Path):
        return {"__darts_kind__": "path", "value": str(obj)}
    if isinstance(obj, pd.Timestamp):
        return _encode_timestamp(obj)
    if isinstance(obj, pd.Timedelta):
        return {"__darts_kind__": "pd_timedelta", "value": obj.value}
    if isinstance(obj, datetime.datetime):
        return {"__darts_kind__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {"__darts_kind__": "date", "value": obj.isoformat()}
    if isinstance(obj, datetime.timedelta):
        return {
            "__darts_kind__": "timedelta",
            "days": obj.days,
            "seconds": obj.seconds,
            "microseconds": obj.microseconds,
        }
    if isinstance(obj, pd.DataFrame):
        return _encode_dataframe(obj, path)
    if isinstance(obj, pd.Series):
        return _encode_series(obj, path)
    if isinstance(obj, pd.Index):
        return _encode_index(obj, path)

    from darts import TimeSeries

    if isinstance(obj, TimeSeries):
        return _encode_timeseries(obj, path, stack)
    if isinstance(obj, dict):
        return {
            "__darts_kind__": "dict",
            "items": [
                [
                    _encode(key, path=f"{path}[{key!r}].key", stack=stack),
                    _encode(value, path=f"{path}[{key!r}]", stack=stack),
                ]
                for key, value in obj.items()
            ],
        }
    if isinstance(obj, list | tuple | set):
        kind = {list: "list", tuple: "tuple", set: "set"}[type(obj)]
        return {
            "__darts_kind__": kind,
            "items": [
                _encode(item, path=f"{path}[{i}]", stack=stack)
                for i, item in enumerate(obj)
            ],
        }
    if isinstance(obj, range):
        return {
            "__darts_kind__": "range",
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }
    if inspect.isclass(obj):
        return _encode_class(obj)
    if inspect.isroutine(obj) or inspect.ismethod(obj) or inspect.isfunction(obj):
        raise UnencodableObjectError(obj, path)

    callback = _as_callback(obj)
    if callback is not None:
        return _encode_from_init(
            obj,
            path,
            stack,
            prefixes=_CALLBACK_PREFIXES,
            base=callback,
            role="callback",
        )
    logger_base = _as_logger(obj)
    if logger_base is not None:
        return _encode_from_init(
            obj,
            path,
            stack,
            prefixes=_LOGGER_PREFIXES,
            base=logger_base,
            role="logger",
        )
    metric_base = _as_metric(obj)
    if metric_base is not None:
        return _encode_from_init(
            obj,
            path,
            stack,
            prefixes=_METRIC_PREFIXES,
            base=metric_base,
            role="metric",
        )
    if type(obj).__module__.startswith(_SKLEARN_PREFIXES):
        return _encode_sklearn(obj, path, stack)
    if isinstance(obj, nn.Module) and type(obj).__module__.startswith("torch.nn."):
        return _encode_module(obj, path, stack)
    if (
        is_dataclass(obj)
        and not isinstance(obj, type)
        and type(obj).__module__.startswith("darts.")
    ):
        return _encode_dataclass(obj, path, stack)
    if type(obj).__module__.startswith("darts.") and hasattr(obj, "__dict__"):
        return _encode_instance(obj, path, stack)
    raise UnencodableObjectError(obj, path)


def _encode_class(cls: type) -> dict:
    qualname = _qualname(cls)
    if qualname.startswith("torch.optim.lr_scheduler.") or qualname.startswith(
        "torch.optim.lr_scheduler"
    ):
        base = _lr_scheduler_base()
        prefixes = ("torch.optim.lr_scheduler.",)
    elif qualname.startswith("torch.optim."):
        base = torch.optim.Optimizer
        prefixes = ("torch.optim.",)
    elif qualname.startswith("torch.nn.") or qualname.startswith("neuralforecast."):
        base = nn.Module
        prefixes = ("torch.nn.", "neuralforecast.")
    elif qualname.startswith("darts."):
        base = None
        prefixes = ("darts.",)
    else:
        raise UnencodableObjectError(cls, qualname)
    _resolve_class(qualname, prefixes=prefixes, base=base)
    return {"__darts_kind__": "class", "cls": qualname}


def _lr_scheduler_base():
    from torch.optim import lr_scheduler

    return getattr(lr_scheduler, "LRScheduler", lr_scheduler._LRScheduler)


def _encode_ndarray(arr: np.ndarray, path: str) -> dict:
    if arr.dtype == object:
        values = [_python_scalar(item, path) for item in arr.reshape(-1).tolist()]
        return {
            "__darts_kind__": "ndarray",
            "dtype": "object",
            "shape": list(arr.shape),
            "values": values,
        }
    data = torch.from_numpy(np.ascontiguousarray(arr)).clone()
    return {"__darts_kind__": "ndarray", "dtype": str(arr.dtype), "data": data}


def _python_scalar(value, path: str):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | float | str):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value
    raise UnencodableObjectError(value, path)


def _numpy_dtype_name(obj) -> str | None:
    if isinstance(obj, np.dtype):
        return obj.name
    if inspect.isclass(obj) and issubclass(obj, np.generic):
        return np.dtype(obj).name
    return None


def _encode_timestamp(ts: pd.Timestamp) -> dict:
    return {
        "__darts_kind__": "timestamp",
        "value": ts.isoformat(),
        "tz": None if ts.tz is None else str(ts.tz),
    }


def _encode_index(index: pd.Index, path: str) -> dict:
    name = None if index.name is None else index.name
    if name is not None and not isinstance(name, str):
        raise UnencodableObjectError(name, path)
    if isinstance(index, pd.RangeIndex):
        return {
            "__darts_kind__": "index",
            "kind": "range",
            "start": int(index.start),
            "stop": int(index.stop),
            "step": int(index.step),
            "name": name,
        }
    if isinstance(index, pd.DatetimeIndex):
        freq = None if index.freq is None else index.freqstr
        return {
            "__darts_kind__": "index",
            "kind": "datetime",
            "unit": np.dtype(index.dtype).name,
            "values": torch.from_numpy(np.ascontiguousarray(index.asi8)).clone(),
            "freq": freq,
            "name": name,
        }
    if pd.api.types.is_numeric_dtype(index):
        return {
            "__darts_kind__": "index",
            "kind": "numeric",
            "dtype": str(index.dtype),
            "values": torch.from_numpy(np.ascontiguousarray(np.asarray(index))).clone(),
            "name": name,
        }
    return {
        "__darts_kind__": "index",
        "kind": "object",
        "values": [_python_scalar(value, path) for value in index.tolist()],
        "name": name,
    }


def _encode_column(values: pd.Series, path: str) -> dict:
    if pd.api.types.is_bool_dtype(values):
        return {"kind": "bool", "values": [bool(v) for v in values.tolist()]}
    if pd.api.types.is_numeric_dtype(values):
        array = np.ascontiguousarray(values.to_numpy())
        return {
            "kind": "numeric",
            "dtype": str(array.dtype),
            "values": torch.from_numpy(array).clone(),
        }
    if pd.api.types.is_datetime64_any_dtype(values):
        int_values = values.astype("int64").to_numpy()
        return {
            "kind": "datetime",
            "unit": np.dtype(values.dtype).name,
            "values": torch.from_numpy(np.ascontiguousarray(int_values)).clone(),
        }
    return {
        "kind": "object",
        "values": [_python_scalar(value, path) for value in values.tolist()],
    }


def _encode_dataframe(df: pd.DataFrame, path: str) -> dict:
    columns_name = None if df.columns.name is None else df.columns.name
    if columns_name is not None and not isinstance(columns_name, str):
        raise UnencodableObjectError(columns_name, path)
    return {
        "__darts_kind__": "dataframe",
        "columns": [str(col) for col in df.columns],
        "columns_name": columns_name,
        "index": _encode_index(df.index, f"{path}.index"),
        "data": [_encode_column(df[col], f"{path}[{col!r}]") for col in df.columns],
    }


def _encode_series(series: pd.Series, path: str) -> dict:
    name = None if series.name is None else series.name
    if name is not None and not isinstance(name, str):
        raise UnencodableObjectError(name, path)
    return {
        "__darts_kind__": "series",
        "name": name,
        "index": _encode_index(series.index, f"{path}.index"),
        "data": _encode_column(series, path),
    }


def _encode_timeseries(series, path: str, stack: set[int]) -> dict:
    values = np.ascontiguousarray(series.all_values(copy=False))
    return {
        "__darts_kind__": "timeseries",
        "dtype": str(values.dtype),
        "values": torch.from_numpy(values).clone(),
        "index": _encode_index(series.time_index, f"{path}.index"),
        "components": [str(comp) for comp in series.components],
        "static_covariates": _encode(
            series.static_covariates, path=f"{path}.static_covariates", stack=stack
        ),
        "hierarchy": _encode(series.hierarchy, path=f"{path}.hierarchy", stack=stack),
        "metadata": _encode(series.metadata, path=f"{path}.metadata", stack=stack),
    }


def _encode_from_init(
    obj,
    path: str,
    stack: set[int],
    *,
    prefixes: tuple[str, ...],
    base: type,
    role: str,
) -> dict:
    qualname = _qualname(type(obj))
    try:
        _resolve_component_class(qualname, prefixes=prefixes, base=base)
    except ValueError as exc:
        raise UnencodableObjectError(obj, path) from exc
    kwargs = {}
    signature = inspect.signature(type(obj).__init__)
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        value = _read_ctor_attr(obj, name)
        if value is _MISSING:
            if param.default is inspect.Parameter.empty:
                raise UnencodableObjectError(obj, f"{path}.{name}")
            continue
        if type(obj).__name__ == "EarlyStopping" and name == "min_delta":
            value = abs(value)
        kwargs[name] = value
    return {
        "__darts_kind__": "init",
        "cls": qualname,
        "role": role,
        "kwargs": _encode(kwargs, path=path, stack=stack),
    }


def _read_ctor_attr(obj, name: str):
    public = getattr(obj, name, _MISSING)
    private = getattr(obj, f"_{name}", _MISSING)
    if public is not _MISSING and not callable(public):
        return public
    if private is not _MISSING and not callable(private):
        return private
    return _MISSING


def _encode_sklearn(est, path: str, stack: set[int]) -> dict:
    qualname = _qualname(type(est))
    from sklearn.base import BaseEstimator

    _resolve_class(qualname, prefixes=_SKLEARN_PREFIXES, base=BaseEstimator)
    params = est.get_params(deep=False)
    fitted = {
        name: value
        for name, value in est.__dict__.items()
        if name.endswith("_") and not name.startswith("_") and not callable(value)
    }
    return {
        "__darts_kind__": "sklearn",
        "cls": qualname,
        "params": _encode(params, path=f"{path}.params", stack=stack),
        "fitted": _encode(fitted, path=f"{path}.fitted", stack=stack),
    }


def _encode_module(module: nn.Module, path: str, stack: set[int]) -> dict:
    qualname = _qualname(type(module))
    _resolve_class(qualname, prefixes=("torch.nn.",), base=nn.Module)
    signature = inspect.signature(type(module).__init__)
    kwargs = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        value = _read_ctor_attr(module, name)
        if value is _MISSING:
            if param.default is inspect.Parameter.empty:
                raise UnencodableObjectError(module, f"{path}.{name}")
            continue
        kwargs[name] = value
    return {
        "__darts_kind__": "nn_module",
        "cls": qualname,
        "kwargs": _encode(kwargs, path=path, stack=stack),
        "state": _encode(
            dict(module.state_dict()), path=f"{path}.state_dict", stack=stack
        ),
    }


def _encode_dataclass(obj, path: str, stack: set[int]) -> dict:
    payload = {field.name: getattr(obj, field.name) for field in fields(obj)}
    return {
        "__darts_kind__": "dataclass",
        "cls": _qualname(type(obj)),
        "fields": _encode(payload, path=path, stack=stack),
    }


def _encode_instance(obj, path: str, stack: set[int]) -> dict:
    qualname = _qualname(type(obj))
    _resolve_class(qualname, prefixes=("darts.",))
    return {
        "__darts_kind__": "instance",
        "cls": qualname,
        "state": _encode(dict(obj.__dict__), path=path, stack=stack),
    }


def _as_callback(obj):
    try:
        from pytorch_lightning.callbacks import Callback
    except Exception:
        return None
    return Callback if isinstance(obj, Callback) else None


def _as_logger(obj):
    try:
        from pytorch_lightning.loggers import Logger
    except Exception:
        return None
    return Logger if isinstance(obj, Logger) else None


def _as_metric(obj):
    try:
        import torchmetrics
    except Exception:
        return None
    return torchmetrics.Metric if isinstance(obj, torchmetrics.Metric) else None


def _decode(obj):
    if isinstance(obj, dict) and "__darts_kind__" in obj:
        kind = obj["__darts_kind__"]
        decoder = _DECODERS.get(kind)
        if decoder is None:
            raise ValueError(f"Unknown wrapper record `{kind}`.")
        return decoder(obj)
    if isinstance(obj, torch.Tensor):
        return obj
    if obj is None or isinstance(obj, bool | int | float | str | bytes | complex):
        return obj
    raise ValueError(
        f"Unexpected value of type `{type(obj).__name__}` in a model file."
    )


def _decode_enum(obj):
    cls = _resolve_class(obj["cls"], prefixes=_ENUM_PREFIXES, base=enum.Enum)
    return cls[obj["name"]]


def _decode_dtype(obj):
    dtype = np.dtype(obj["name"])
    return dtype if obj["as_dtype"] else dtype.type


def _decode_path(obj):
    return pathlib.Path(obj["value"])


def _decode_timestamp(obj):
    ts = pd.Timestamp(obj["value"])
    if obj["tz"] is not None:
        ts = ts.tz_localize(obj["tz"])
    return ts


def _decode_pd_timedelta(obj):
    return pd.Timedelta(obj["value"])


def _decode_datetime(obj):
    return datetime.datetime.fromisoformat(obj["value"])


def _decode_date(obj):
    return datetime.date.fromisoformat(obj["value"])


def _decode_timedelta(obj):
    return datetime.timedelta(
        days=obj["days"], seconds=obj["seconds"], microseconds=obj["microseconds"]
    )


def _decode_index(obj):
    kind = obj["kind"]
    name = obj["name"]
    if kind == "range":
        return pd.RangeIndex(
            start=obj["start"], stop=obj["stop"], step=obj["step"], name=name
        )
    if kind == "datetime":
        raw = obj["values"].numpy().astype("int64")
        values = raw.astype(obj.get("unit", "datetime64[ns]"))
        return pd.DatetimeIndex(values, freq=obj["freq"], name=name)
    if kind == "numeric":
        values = obj["values"].numpy().astype(obj["dtype"])
        return pd.Index(values, dtype=obj["dtype"], name=name)
    return pd.Index(obj["values"], name=name)


def _decode_column(spec: dict) -> np.ndarray | list:
    kind = spec["kind"]
    if kind == "numeric":
        return spec["values"].numpy().astype(spec["dtype"])
    if kind == "datetime":
        raw = spec["values"].numpy().astype("int64")
        return raw.astype(spec.get("unit", "datetime64[ns]"))
    return spec["values"]


def _decode_dataframe(obj):
    data = {
        column: _decode_column(spec)
        for column, spec in zip(obj["columns"], obj["data"])
    }
    frame = pd.DataFrame(data, index=_decode(obj["index"]), columns=obj["columns"])
    frame.columns.name = obj["columns_name"]
    return frame


def _decode_series(obj):
    values = _decode_column(obj["data"])
    return pd.Series(values, index=_decode(obj["index"]), name=obj["name"])


def _decode_timeseries(obj):
    from darts import TimeSeries

    values = obj["values"].numpy().astype(obj["dtype"])
    return TimeSeries(
        times=_decode(obj["index"]),
        values=values,
        components=obj["components"],
        static_covariates=_decode(obj["static_covariates"]),
        hierarchy=_decode(obj["hierarchy"]),
        metadata=_decode(obj["metadata"]),
        copy=False,
    )


def _decode_mapping(obj):
    return {_decode(key): _decode(value) for key, value in obj["items"]}


def _decode_list(obj):
    return [_decode(item) for item in obj["items"]]


def _decode_tuple(obj):
    return tuple(_decode(item) for item in obj["items"])


def _decode_set(obj):
    return {_decode(item) for item in obj["items"]}


def _decode_range(obj):
    return range(obj["start"], obj["stop"], obj["step"])


def _class_base(qualname: str):
    if qualname.startswith("torch.optim.lr_scheduler."):
        return _lr_scheduler_base()
    if qualname.startswith("torch.optim."):
        return torch.optim.Optimizer
    if qualname.startswith(("torch.nn.", "neuralforecast.")):
        return nn.Module
    return None


def _class_prefixes(qualname: str) -> tuple[str, ...]:
    if qualname.startswith("torch.optim.lr_scheduler."):
        return ("torch.optim.lr_scheduler.",)
    if qualname.startswith("torch.optim."):
        return ("torch.optim.",)
    if qualname.startswith(("torch.nn.", "neuralforecast.")):
        return ("torch.nn.", "neuralforecast.")
    if qualname.startswith("darts."):
        return ("darts.",)
    raise ValueError(f"Refusing to import `{qualname}`.")


def _decode_class(obj):
    qualname = obj["cls"]
    return _resolve_class(
        qualname, prefixes=_class_prefixes(qualname), base=_class_base(qualname)
    )


def _decode_init(obj):
    qualname = obj["cls"]
    role = obj.get("role")
    if role == "callback" or qualname.startswith(_CALLBACK_PREFIXES):
        from pytorch_lightning.callbacks import Callback

        base, prefixes = Callback, _CALLBACK_PREFIXES
    elif role == "logger" or qualname.startswith(_LOGGER_PREFIXES):
        from pytorch_lightning.loggers import Logger

        base, prefixes = Logger, _LOGGER_PREFIXES
    elif role == "metric" or qualname.startswith(_METRIC_PREFIXES):
        import torchmetrics

        base, prefixes = torchmetrics.Metric, _METRIC_PREFIXES
    else:
        raise ValueError(f"Refusing to import `{qualname}`.")
    cls = _resolve_component_class(qualname, prefixes=prefixes, base=base)
    return cls(**_decode(obj["kwargs"]))


def _decode_sklearn(obj):
    from sklearn.base import BaseEstimator

    cls = _resolve_class(obj["cls"], prefixes=_SKLEARN_PREFIXES, base=BaseEstimator)
    est = cls(**_decode(obj["params"]))
    for name, value in _decode(obj["fitted"]).items():
        setattr(est, name, value)
    return est


def _decode_module(obj):
    cls = _resolve_class(obj["cls"], prefixes=("torch.nn.",), base=nn.Module)
    module = cls(**_decode(obj["kwargs"]))
    state = _decode(obj["state"])
    if state:
        module.load_state_dict(state)
    return module


def _decode_dataclass(obj):
    cls = _resolve_class(obj["cls"], prefixes=("darts.",))
    if not is_dataclass(cls):
        raise ValueError(f"`{obj['cls']}` is not a dataclass.")
    return cls(**_decode(obj["fields"]))


def _decode_instance(obj):
    cls = _resolve_class(obj["cls"], prefixes=("darts.",))
    instance = cls.__new__(cls)
    instance.__dict__.update(_decode(obj["state"]))
    return instance


def _decode_random_state(obj):
    random_state = np.random.RandomState()
    random_state.set_state(_decode(obj["state"]))
    return random_state


def _decode_ndarray(obj):
    if obj["dtype"] == "object":
        return np.array(obj["values"], dtype=object).reshape(obj["shape"])
    return obj["data"].numpy().astype(obj["dtype"], copy=False)


_DECODERS = {
    "enum": _decode_enum,
    "dtype": _decode_dtype,
    "path": _decode_path,
    "timestamp": _decode_timestamp,
    "pd_timedelta": _decode_pd_timedelta,
    "datetime": _decode_datetime,
    "date": _decode_date,
    "timedelta": _decode_timedelta,
    "index": _decode_index,
    "dataframe": _decode_dataframe,
    "series": _decode_series,
    "timeseries": _decode_timeseries,
    "dict": _decode_mapping,
    "list": _decode_list,
    "tuple": _decode_tuple,
    "set": _decode_set,
    "range": _decode_range,
    "class": _decode_class,
    "init": _decode_init,
    "sklearn": _decode_sklearn,
    "nn_module": _decode_module,
    "dataclass": _decode_dataclass,
    "instance": _decode_instance,
    "ndarray": _decode_ndarray,
    "random_state": _decode_random_state,
}
