"""
Model Serialization Utilities (core)
------------------------------------

Shared helpers for Darts model persistence.

Security model:

- Prefer **scoped** allow-lists applied only around a single load call, not process-wide
  registration at import time.
- When an inspection API exists, derive the allow-list **from the specific artifact** being
  loaded (checkpoint-driven registration) rather than pre-registering entire libraries.
- Be explicit about scope: safe loading for Lightning ``.ckpt`` files does **not** protect
  the Darts base model ``.pt`` shell or general ``.pkl`` pickles, which remain full unpickling
  surfaces (CWE-502) until a future general serialization revamp addresses them.
"""

import importlib
from collections.abc import Sequence
from typing import TypeVar

from darts.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Prefixes for trusted packages when filtering checkpoint-referenced globals.
TrustedPrefixPolicy = tuple[str, ...]


def all_subclasses(cls: type) -> set[type]:
    """Gives all currently imported subclasses for `cls` (including `cls`)."""
    found = {cls}
    for sub in cls.__subclasses__():
        found |= all_subclasses(sub)
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
