"""
String constants for additive/multiplicative decomposition and related model modes.
"""

from typing import Literal, TypeVar

from darts.logging import raise_log

ADDITIVE: Literal["additive"] = "additive"
MULTIPLICATIVE: Literal["multiplicative"] = "multiplicative"
LINEAR: Literal["linear"] = "linear"
EXPONENTIAL: Literal["exponential"] = "exponential"

T = TypeVar("T")


def validate_mode(
    value: T,
    *,
    param: str,
    allowed: set,
) -> T:
    """Return ``value`` if it is in ``allowed`` (identity preserved for type checkers).

    Pass ``None`` in ``allowed`` to accept a ``None`` value (e.g. to disable a component).
    """
    if (value is None or isinstance(value, str)) and value in allowed:
        return value
    raise_log(
        ValueError(f"Unknown value for `{param}`: None. Expected one of {allowed}."),
    )
