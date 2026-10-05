"""
Model Serialization Utilities
-----------------------------

Helpers for Darts model loading.
"""

from darts.utils.serialization.registry import (
    add_safe_globals,
    clear_safe_globals,
    get_safe_globals,
    safe_globals,
)

__all__ = [
    "add_safe_globals",
    "clear_safe_globals",
    "get_safe_globals",
    "safe_globals",
]
