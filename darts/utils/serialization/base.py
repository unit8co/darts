"""
Model Serialization Utilities (core)
------------------------------------

Core helpers for Darts model persistence.

``encode_state`` / ``decode_state`` turn an object graph into plain data (primitives,
lists, dicts, and NumPy arrays). Classes and functions are stored by import path and
imported again on load; nothing in the payload picks a reducer.
"""

from __future__ import annotations

import datetime
import enum
import importlib
import inspect
import pathlib
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from darts.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Prefixes for trusted packages when filtering checkpoint-referenced globals.
TrustedPrefixPolicy = tuple[str, ...]

# An ``extra`` encode/decode callback returns this when it does not handle ``obj``.
STATE_UNHANDLED = object()

KIND_TAG = "__kind__"


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


class UnencodableObjectError(TypeError):
    """A value has no import path, or no supported state.

    Nested functions and classes defined inside a function cannot be saved.
    """

    def __init__(self, obj: Any, path: str):
        """Build the error from ``obj`` and where it sits in the state (``path``)."""
        label = _describe(obj)
        super().__init__(
            f"Cannot store {label} ({path}) in the safe model format. "
            "Classes must be defined in an importable module. Nested functions and "
            "classes defined inside a function cannot be saved."
        )


def qualname(cls: type) -> str:
    """Import path ``module.qualname`` for ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def import_qualified(qualname_: str):
    """Import ``qualname_`` (``module.Class.attr``) and return the object.

    Rejects relative paths, dunder parts, and non-identifiers. Importing runs that
    module's top-level code. It does not call the object.
    """
    if (
        not isinstance(qualname_, str)
        or not qualname_
        or ".." in qualname_
        or qualname_.startswith(".")
    ):
        raise ValueError(f"Refusing to import `{qualname_}`.")
    parts = qualname_.split(".")
    if any(not part.isidentifier() or part.startswith("__") for part in parts):
        raise ValueError(f"Refusing to import `{qualname_}`.")

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
    raise ValueError(f"Cannot import `{qualname_}`.")


def import_class(qualname_: str, *, base: type | None = None):
    """Import a class by `qualname_`.

    The class must already be defined in a module. ``<locals>`` and similar paths
    are rejected. When ``base`` is set, the class must subclass it.
    """
    if not isinstance(qualname_, str) or "<" in qualname_ or ">" in qualname_:
        raise ValueError(
            f"Cannot import `{qualname_}`. Define the class in an importable module."
        )
    cls = import_qualified(qualname_)
    if not inspect.isclass(cls):
        raise ValueError(f"`{qualname_}` is not a class.")
    if base is not None and not issubclass(cls, base):
        raise ValueError(f"`{qualname_}` is not a `{base.__name__}` subclass.")
    return cls


def encode_state(obj, *, path="state", stack=None, extra=None):
    """Encode ``obj`` as plain data.

    Containers are tagged with ``__darts_kind__`` so a user dict cannot look like
    a record. Each container is encoded once. Numeric arrays stay NumPy arrays.
    Classes and callables are import paths, not called constructors.

    ``extra(obj, path, stack)`` may encode a type this function does not know
    (tensors, modules). Return :data:`STATE_UNHANDLED` to decline.
    """
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
        if extra is not None:
            handled = extra(obj, path, stack)
            if handled is not STATE_UNHANDLED:
                return handled
        return _encode_value(obj, path=path, stack=stack, extra=extra)
    finally:
        stack.discard(obj_id)


def decode_state(obj, *, extra=None):
    """Inverse of :func:`encode_state`.

    ``extra(obj)`` may decode a record this function does not know. Return
    :data:`STATE_UNHANDLED` to decline. Numeric payloads may be NumPy arrays or
    any object with ``.numpy()``, such as a tensor after a weights-only load.
    """
    if extra is not None:
        handled = extra(obj)
        if handled is not STATE_UNHANDLED:
            return handled
    if isinstance(obj, dict) and KIND_TAG in obj:
        kind = obj[KIND_TAG]
        decoder = _DECODERS.get(kind)
        if decoder is None:
            raise ValueError(f"Unknown state record `{kind}`.")
        return decoder(obj, extra)
    if obj is None or isinstance(obj, bool | int | float | str | bytes | complex):
        return obj
    raise ValueError(
        f"Unexpected value of type `{type(obj).__name__}` in a model file."
    )


def _describe(obj: Any) -> str:
    """Short type label for an :class:`UnencodableObjectError` message."""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return f"function `{getattr(obj, '__qualname__', type(obj).__name__)}`"
    cls = obj if inspect.isclass(obj) else type(obj)
    return f"`{cls.__module__}.{cls.__qualname__}`"


def _encode_enum(obj: enum.Enum) -> dict:
    """Enum class path and member name. Checked before ``str`` so string enums stay enums."""
    cls_path = qualname(type(obj))
    try:
        import_class(cls_path, base=enum.Enum)
    except ValueError as exc:
        raise UnencodableObjectError(obj, cls_path) from exc
    return {KIND_TAG: "enum", "cls": cls_path, "name": obj.name}


def _encode_value(obj, *, path: str, stack: set[int], extra):
    """Pick an encoder for ``obj``. ``extra`` has already declined it."""
    if isinstance(obj, np.ndarray):
        return _encode_ndarray(obj, path)
    if isinstance(obj, np.random.RandomState):
        return {
            KIND_TAG: "random_state",
            "state": encode_state(
                obj.get_state(), path=f"{path}.state", stack=stack, extra=extra
            ),
        }
    dtype_name = _numpy_dtype_name(obj)
    if dtype_name is not None:
        return {
            KIND_TAG: "dtype",
            "name": dtype_name,
            "as_dtype": isinstance(obj, np.dtype),
        }
    if isinstance(obj, pathlib.Path):
        return {KIND_TAG: "path", "value": str(obj)}
    if isinstance(obj, pd.Timestamp):
        return _encode_timestamp(obj)
    if isinstance(obj, pd.Timedelta):
        return {KIND_TAG: "pd_timedelta", "value": obj.value}
    if isinstance(obj, datetime.datetime):
        return {KIND_TAG: "datetime", "value": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {KIND_TAG: "date", "value": obj.isoformat()}
    if isinstance(obj, datetime.timedelta):
        return {
            KIND_TAG: "timedelta",
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
        return _encode_timeseries(obj, path, stack, extra)
    if isinstance(obj, dict):
        return {
            KIND_TAG: "dict",
            "items": [
                [
                    encode_state(
                        key, path=f"{path}[{key!r}].key", stack=stack, extra=extra
                    ),
                    encode_state(
                        value, path=f"{path}[{key!r}]", stack=stack, extra=extra
                    ),
                ]
                for key, value in obj.items()
            ],
        }
    if isinstance(obj, list | tuple | set):
        kind = {list: "list", tuple: "tuple", set: "set"}[type(obj)]
        return {
            KIND_TAG: kind,
            "items": [
                encode_state(item, path=f"{path}[{i}]", stack=stack, extra=extra)
                for i, item in enumerate(obj)
            ],
        }
    if isinstance(obj, range):
        return {
            KIND_TAG: "range",
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }
    if inspect.isclass(obj):
        return _encode_class(obj)
    if inspect.isroutine(obj) or inspect.ismethod(obj) or inspect.isfunction(obj):
        return _encode_callable(obj, path)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _encode_instance(
            obj,
            path,
            stack,
            extra,
            state={field.name: getattr(obj, field.name) for field in fields(obj)},
        )
    if hasattr(obj, "__dict__"):
        return _encode_instance(obj, path, stack, extra, state=data_state(obj))
    raise UnencodableObjectError(obj, path)


def _encode_class(cls: type) -> dict:
    """Store a class by import path. A local class fails here, before any file write."""
    cls_path = qualname(cls)
    try:
        import_class(cls_path)
    except ValueError as exc:
        raise UnencodableObjectError(cls, cls_path) from exc
    return {KIND_TAG: "class", "cls": cls_path}


def _encode_callable(fn, path: str) -> dict:
    """Store a module-level function by import path. Do not call it on load."""
    module = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None)
    cls_path = f"{module}.{name}" if module and name else ""
    if "<" in cls_path or ">" in cls_path or not cls_path:
        raise UnencodableObjectError(fn, path)
    try:
        imported = import_qualified(cls_path)
    except ValueError as exc:
        raise UnencodableObjectError(fn, path) from exc
    if not callable(imported):
        raise UnencodableObjectError(fn, path)
    return {KIND_TAG: "callable", "cls": cls_path}


def _encode_ndarray(arr: np.ndarray, path: str) -> dict:
    """Object arrays become Python scalars. Numeric arrays stay NumPy."""
    if arr.dtype == object:
        values = [_python_scalar(item, path) for item in arr.reshape(-1).tolist()]
        return {
            KIND_TAG: "ndarray",
            "dtype": "object",
            "shape": list(arr.shape),
            "values": values,
        }
    return {
        KIND_TAG: "ndarray",
        "dtype": str(arr.dtype),
        "data": np.ascontiguousarray(arr),
    }


def _python_scalar(value, path: str):
    """One object-array cell as a Python scalar. NaN becomes ``None``."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | float | str):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value
    raise UnencodableObjectError(value, path)


def _numpy_dtype_name(obj) -> str | None:
    """Dtype name for a ``np.dtype`` or NumPy scalar type, otherwise ``None``."""
    if isinstance(obj, np.dtype):
        return obj.name
    if inspect.isclass(obj) and issubclass(obj, np.generic):
        return np.dtype(obj).name
    return None


def _encode_timestamp(ts: pd.Timestamp) -> dict:
    """ISO timestamp and timezone name. ``tz`` is ``None`` when the timestamp is naive."""
    return {
        KIND_TAG: "timestamp",
        "value": ts.isoformat(),
        "tz": None if ts.tz is None else str(ts.tz),
    }


def _encode_index(index: pd.Index, path: str) -> dict:
    """Range, datetime, numeric, or object index. Datetime values keep their unit."""
    name = None if index.name is None else index.name
    if name is not None and not isinstance(name, str):
        raise UnencodableObjectError(name, path)
    if isinstance(index, pd.RangeIndex):
        return {
            KIND_TAG: "index",
            "kind": "range",
            "start": int(index.start),
            "stop": int(index.stop),
            "step": int(index.step),
            "name": name,
        }
    if isinstance(index, pd.DatetimeIndex):
        freq = None if index.freq is None else index.freqstr
        # Keep the original unit. Casting to ``datetime64[ns]`` breaks daily freq.
        return {
            KIND_TAG: "index",
            "kind": "datetime",
            "unit": np.dtype(index.dtype).name,
            "values": np.ascontiguousarray(index.asi8),
            "freq": freq,
            "name": name,
        }
    if pd.api.types.is_numeric_dtype(index):
        return {
            KIND_TAG: "index",
            "kind": "numeric",
            "dtype": str(index.dtype),
            "values": np.ascontiguousarray(np.asarray(index)),
            "name": name,
        }
    return {
        KIND_TAG: "index",
        "kind": "object",
        "values": [_python_scalar(value, path) for value in index.tolist()],
        "name": name,
    }


def _encode_column(values: pd.Series, path: str) -> dict:
    """One DataFrame or Series column. Nested in a parent record, so it has no kind tag."""
    if pd.api.types.is_bool_dtype(values):
        return {"kind": "bool", "values": [bool(v) for v in values.tolist()]}
    if pd.api.types.is_numeric_dtype(values):
        array = np.ascontiguousarray(values.to_numpy())
        return {"kind": "numeric", "dtype": str(array.dtype), "values": array}
    if pd.api.types.is_datetime64_any_dtype(values):
        int_values = values.astype("int64").to_numpy()
        return {
            "kind": "datetime",
            "unit": np.dtype(values.dtype).name,
            "values": np.ascontiguousarray(int_values),
        }
    return {
        "kind": "object",
        "values": [_python_scalar(value, path) for value in values.tolist()],
    }


def _encode_dataframe(df: pd.DataFrame, path: str) -> dict:
    """Column names, column-axis name, index, and per-column values."""
    columns_name = None if df.columns.name is None else df.columns.name
    if columns_name is not None and not isinstance(columns_name, str):
        raise UnencodableObjectError(columns_name, path)
    return {
        KIND_TAG: "dataframe",
        "columns": [str(col) for col in df.columns],
        "columns_name": columns_name,
        "index": _encode_index(df.index, f"{path}.index"),
        "data": [_encode_column(df[col], f"{path}[{col!r}]") for col in df.columns],
    }


def _encode_series(series: pd.Series, path: str) -> dict:
    """Series name, index, and values."""
    name = None if series.name is None else series.name
    if name is not None and not isinstance(name, str):
        raise UnencodableObjectError(name, path)
    return {
        KIND_TAG: "series",
        "name": name,
        "index": _encode_index(series.index, f"{path}.index"),
        "data": _encode_column(series, path),
    }


def _encode_timeseries(series, path: str, stack: set[int], extra) -> dict:
    """Values, time index, components, static covariates, hierarchy, and metadata."""
    values = np.ascontiguousarray(series.all_values(copy=False))
    return {
        KIND_TAG: "timeseries",
        "dtype": str(values.dtype),
        "values": values,
        "index": _encode_index(series.time_index, f"{path}.index"),
        "components": [str(comp) for comp in series.components],
        "static_covariates": encode_state(
            series.static_covariates,
            path=f"{path}.static_covariates",
            stack=stack,
            extra=extra,
        ),
        "hierarchy": encode_state(
            series.hierarchy, path=f"{path}.hierarchy", stack=stack, extra=extra
        ),
        "metadata": encode_state(
            series.metadata, path=f"{path}.metadata", stack=stack, extra=extra
        ),
    }


def data_state(obj) -> dict:
    """``__dict__`` values that are data. Bound methods and builtins are omitted."""
    return {
        key: value
        for key, value in obj.__dict__.items()
        if not inspect.ismethod(value) and not inspect.isbuiltin(value)
    }


def _encode_instance(obj, path, stack, extra, *, state: dict) -> dict:
    """Import path plus attributes. Load uses ``__new__``, not ``__init__``."""
    cls_path = qualname(type(obj))
    try:
        import_class(cls_path)
    except ValueError as exc:
        raise UnencodableObjectError(obj, path) from exc
    return {
        KIND_TAG: "instance",
        "cls": cls_path,
        "state": encode_state(state, path=path, stack=stack, extra=extra),
    }


def _as_numpy(value) -> np.ndarray:
    """Numeric payload as a NumPy array.

    Accepts an ndarray, or a tensor-like object with ``.numpy()`` after a
    weights-only load. This module does not import PyTorch.
    """
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        array = to_numpy()
        if isinstance(array, np.ndarray):
            return array
    raise ValueError(f"Unexpected numeric payload of type `{type(value).__name__}`.")


def _decode_enum(obj, _extra=None):
    """Import the enum and return the named member."""
    cls = import_class(obj["cls"], base=enum.Enum)
    return cls[obj["name"]]


def _decode_dtype(obj, _extra=None):
    """A ``np.dtype`` when ``as_dtype`` is set, otherwise the scalar type."""
    dtype = np.dtype(obj["name"])
    return dtype if obj["as_dtype"] else dtype.type


def _decode_path(obj, _extra=None):
    """A ``pathlib.Path`` from the stored string."""
    return pathlib.Path(obj["value"])


def _decode_timestamp(obj, _extra=None):
    """A ``pd.Timestamp``, localized when a timezone was stored."""
    ts = pd.Timestamp(obj["value"])
    if obj["tz"] is not None:
        ts = ts.tz_localize(obj["tz"])
    return ts


def _decode_pd_timedelta(obj, _extra=None):
    """A ``pd.Timedelta`` from its integer nanosecond value."""
    return pd.Timedelta(obj["value"])


def _decode_datetime(obj, _extra=None):
    """A ``datetime.datetime`` from an ISO string."""
    return datetime.datetime.fromisoformat(obj["value"])


def _decode_date(obj, _extra=None):
    """A ``datetime.date`` from an ISO string."""
    return datetime.date.fromisoformat(obj["value"])


def _decode_timedelta(obj, _extra=None):
    """A ``datetime.timedelta`` from days, seconds, and microseconds."""
    return datetime.timedelta(
        days=obj["days"], seconds=obj["seconds"], microseconds=obj["microseconds"]
    )


def _decode_index(obj, _extra=None):
    """A pandas index. Datetime values are cast back to the stored unit."""
    kind = obj["kind"]
    name = obj["name"]
    if kind == "range":
        return pd.RangeIndex(
            start=obj["start"], stop=obj["stop"], step=obj["step"], name=name
        )
    if kind == "datetime":
        raw = _as_numpy(obj["values"]).astype("int64")
        values = raw.astype(obj.get("unit", "datetime64[ns]"))
        return pd.DatetimeIndex(values, freq=obj["freq"], name=name)
    if kind == "numeric":
        values = _as_numpy(obj["values"]).astype(obj["dtype"])
        return pd.Index(values, dtype=obj["dtype"], name=name)
    return pd.Index(obj["values"], name=name)


def _decode_column(spec: dict) -> np.ndarray | list:
    """One column as an array, or as Python scalars for bool and object columns."""
    kind = spec["kind"]
    if kind == "numeric":
        return _as_numpy(spec["values"]).astype(spec["dtype"])
    if kind == "datetime":
        raw = _as_numpy(spec["values"]).astype("int64")
        return raw.astype(spec.get("unit", "datetime64[ns]"))
    return spec["values"]


def _decode_dataframe(obj, extra=None):
    """A DataFrame from stored columns and its index."""
    data = {
        column: _decode_column(spec)
        for column, spec in zip(obj["columns"], obj["data"])
    }
    frame = pd.DataFrame(
        data, index=decode_state(obj["index"], extra=extra), columns=obj["columns"]
    )
    frame.columns.name = obj["columns_name"]
    return frame


def _decode_series(obj, extra=None):
    """A Series from stored values, index, and name."""
    values = _decode_column(obj["data"])
    return pd.Series(
        values, index=decode_state(obj["index"], extra=extra), name=obj["name"]
    )


def _decode_timeseries(obj, extra=None):
    """A ``TimeSeries``, including stochastic samples in ``values``."""
    from darts import TimeSeries

    values = _as_numpy(obj["values"]).astype(obj["dtype"])
    return TimeSeries(
        times=decode_state(obj["index"], extra=extra),
        values=values,
        components=obj["components"],
        static_covariates=decode_state(obj["static_covariates"], extra=extra),
        hierarchy=decode_state(obj["hierarchy"], extra=extra),
        metadata=decode_state(obj["metadata"], extra=extra),
        copy=False,
    )


def _decode_mapping(obj, extra=None):
    """A dict. Keys and values are decoded rather than left as records."""
    return {
        decode_state(key, extra=extra): decode_state(value, extra=extra)
        for key, value in obj["items"]
    }


def _decode_list(obj, extra=None):
    """A list of decoded items."""
    return [decode_state(item, extra=extra) for item in obj["items"]]


def _decode_tuple(obj, extra=None):
    """A tuple of decoded items."""
    return tuple(decode_state(item, extra=extra) for item in obj["items"])


def _decode_set(obj, extra=None):
    """A set of decoded items."""
    return {decode_state(item, extra=extra) for item in obj["items"]}


def _decode_range(obj, _extra=None):
    """A ``range`` from start, stop, and step."""
    return range(obj["start"], obj["stop"], obj["step"])


def _decode_class(obj, _extra=None):
    """Import the named class. Do not instantiate it."""
    return import_class(obj["cls"])


def _decode_callable(obj, _extra=None):
    """Import the named function. Do not call it."""
    fn = import_qualified(obj["cls"])
    if not callable(fn) or inspect.isclass(fn):
        raise ValueError(f"Refusing to restore `{obj['cls']}` as a callable.")
    return fn


def _decode_instance(obj, extra=None):
    """``__new__`` plus saved attributes. Dataclasses use ``setattr``."""
    cls = import_class(obj["cls"])
    state = decode_state(obj["state"], extra=extra)
    instance = cls.__new__(cls)
    if is_dataclass(cls):
        for key, value in state.items():
            setattr(instance, key, value)
        return instance
    instance.__dict__.update(state)
    return instance


def _decode_random_state(obj, extra=None):
    """A ``RandomState`` restored with ``set_state``, not a new seed."""
    random_state = np.random.RandomState()
    random_state.set_state(decode_state(obj["state"], extra=extra))
    return random_state


def _decode_ndarray(obj, _extra=None):
    """An ndarray. Object arrays come from Python scalars; others from the numeric payload."""
    if obj["dtype"] == "object":
        return np.array(obj["values"], dtype=object).reshape(obj["shape"])
    return _as_numpy(obj["data"]).astype(obj["dtype"], copy=False)


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
    "callable": _decode_callable,
    "instance": _decode_instance,
    "ndarray": _decode_ndarray,
    "random_state": _decode_random_state,
}
