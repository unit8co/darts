"""
Lag configuration for tabularization.
-------------------------------------

Internal semantic layer for the lags of a tabularized model: parsing and validating user-provided
lags, handling ``output_chunk_shift`` and resolving component-specific lags against the components
of the actual series. This module is not part of the public Darts API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn

from darts.logging import raise_log

LagsKind = Literal["target", "past", "future"]

_LAGS_ARG_NAMES: dict[str, str] = {
    "target": "lags",
    "past": "lags_past_covariates",
    "future": "lags_future_covariates",
}
_DEFAULT_LAGS_KEY = "default_lags"


def _validate_kind(kind: str) -> LagsKind:
    if kind not in _LAGS_ARG_NAMES:
        raise_log(ValueError(f"Unknown lags kind: {kind!r}."))
    return kind


def _raise_invalid_component_lags(
    lags_name: str,
    comp_name: str,
    supported_types: str,
    comp_lags: object,
) -> NoReturn:
    raise_log(
        ValueError(
            f"`{lags_name}` - `{comp_name}`: must be either a {supported_types}. "
            f"Given : {type(comp_lags)}."
        )
    )


def _normalize_component_lags(
    comp_lags: object, comp_name: str, kind: LagsKind
) -> tuple[int, ...]:
    """Validate the lags of one component and convert them to a sorted tuple."""
    lags_name = _LAGS_ARG_NAMES[kind]

    if kind == "future":
        if isinstance(comp_lags, tuple):
            if not (
                len(comp_lags) == 2
                and isinstance(comp_lags[0], int)
                and isinstance(comp_lags[1], int)
            ):
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: tuple must be of length 2, and must "
                        "contain two integers."
                    )
                )
            if isinstance(comp_lags[0], bool) or isinstance(comp_lags[1], bool):
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: tuple must contain integers, not bool."
                    )
                )
            if not (comp_lags[0] >= 0 and comp_lags[1] >= 0):
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: tuple must contain positive integers. "
                        f"Given: {comp_lags}."
                    )
                )
            if comp_lags[0] == 0 and comp_lags[1] == 0:
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: tuple cannot be (0, 0) as it corresponds "
                        "to an empty list of lags."
                    )
                )
            return tuple(range(-comp_lags[0], comp_lags[1]))

        if isinstance(comp_lags, list):
            if len(comp_lags) == 0:
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: list must contain at least one lag."
                    )
                )
            for lag in comp_lags:
                if not isinstance(lag, int) or isinstance(lag, bool):
                    raise_log(
                        ValueError(
                            f"`{lags_name}` - `{comp_name}`: list must contain "
                            f"only integers. Given: {comp_lags}."
                        )
                    )
            return tuple(sorted(comp_lags))

        _raise_invalid_component_lags(
            lags_name, comp_name, "tuple or a list", comp_lags
        )

    if isinstance(comp_lags, int):
        if comp_lags <= 0:
            raise_log(
                ValueError(
                    f"`{lags_name}` - `{comp_name}`: integer must be strictly "
                    f"positive . Given: {comp_lags}."
                )
            )
        return tuple(range(-comp_lags, 0))

    if isinstance(comp_lags, list):
        if len(comp_lags) == 0:
            raise_log(
                ValueError(
                    f"`{lags_name}` - `{comp_name}`: list must contain at least one lag."
                )
            )
        for lag in comp_lags:
            if not isinstance(lag, int) or (lag >= 0):
                raise_log(
                    ValueError(
                        f"`{lags_name}` - `{comp_name}`: list must contain "
                        f"only strictly negative integers. Given: {comp_lags}."
                    )
                )
        return tuple(sorted(comp_lags))

    _raise_invalid_component_lags(
        lags_name, comp_name, "strictly positive integer or a list", comp_lags
    )


@dataclass
class Lags:
    """Semantic representation of the lags of one variate.

    A ``Lags`` instance is either shared by every component of the series (``shared``) or
    component-specific (``component_lags``, with an optional ``default``). It is parsed with
    :meth:`parse`, resolved in place against the actual components with :meth:`resolve`, and can be
    shifted with :meth:`shifted`.
    """

    kind: LagsKind
    shared: tuple[int, ...] | None = None
    component_lags: dict[str, tuple[int, ...]] = field(default_factory=dict)
    default: tuple[int, ...] | None = None
    _resolved: dict[str, tuple[int, ...]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        self.kind = _validate_kind(self.kind)
        if self.shared is not None:
            self.shared = tuple(self.shared)
        self.component_lags = {
            comp: tuple(lags) for comp, lags in self.component_lags.items()
        }
        if self.default is not None:
            self.default = tuple(self.default)
        if self.shared is not None and self.component_lags:
            raise_log(
                ValueError(
                    "`Lags` cannot define both shared and component-specific lags."
                )
            )
        if self.shared is None and not self.component_lags:
            raise_log(
                ValueError("`Lags` must define shared or component-specific lags.")
            )
        self._resolved = None

    @classmethod
    def parse(cls, value: object, *, kind: LagsKind) -> Lags:
        """Parse the user representation of the lags of one variate.

        Accepted values are ``int``, ``list[int]``, and ``dict[str, int | list[int]]``
        for target and past covariates. Future covariates additionally accept
        ``tuple[int, int]`` for both shared and component-specific lags.
        """
        kind = _validate_kind(kind)

        if isinstance(value, dict):
            if len(value) == 0:
                raise_log(
                    ValueError(
                        f"When passed as a dictionary, `{_LAGS_ARG_NAMES[kind]}` must contain "
                        "at least one key."
                    )
                )
            if list(value.keys()) == [_DEFAULT_LAGS_KEY]:
                return cls(
                    kind=kind,
                    shared=_normalize_component_lags(
                        value[_DEFAULT_LAGS_KEY], _DEFAULT_LAGS_KEY, kind
                    ),
                )

            component_lags: dict[str, tuple[int, ...]] = {}
            default: tuple[int, ...] | None = None
            for comp_name, comp_lags in value.items():
                normalized = _normalize_component_lags(comp_lags, comp_name, kind)
                if comp_name == _DEFAULT_LAGS_KEY:
                    default = normalized
                else:
                    component_lags[comp_name] = normalized
            return cls(kind=kind, component_lags=component_lags, default=default)

        return cls(
            kind=kind,
            shared=_normalize_component_lags(value, _DEFAULT_LAGS_KEY, kind),
        )

    def resolve(self, components: Sequence[str]) -> None:
        """Resolve the lags in place against the components of the series.

        The lags are validated before anything is assigned: a failed resolution keeps the previous
        resolution.
        """
        self._resolved = self._resolve_mapping(components)

    def _resolve_mapping(self, components: Sequence[str]) -> dict[str, tuple[int, ...]]:
        """Validate the lags against ``components`` and return the per-component mapping."""
        components = tuple(components)

        if self.shared is not None:
            return {comp: self.shared for comp in components}

        required = set(components)
        provided = set(self.component_lags)
        wrong_components = [
            comp for comp in self.component_lags if comp not in required
        ]
        missing_components = [comp for comp in components if comp not in provided]
        lags_name = _LAGS_ARG_NAMES[self.kind]

        if len(wrong_components) > 0:
            raise_log(
                ValueError(
                    f"The `{lags_name}` dictionary specifies lags for components that are not "
                    f"present in the series : {wrong_components}. They must be removed to avoid any ambiguity."
                )
            )
        if len(missing_components) > 0 and self.default is None:
            raise_log(
                ValueError(
                    f"The {lags_name} dictionary is missing the lags for the following components "
                    f"present in the series: {missing_components}. The key 'default_lags' can "
                    "be used to provide lags for all the non-explicitely defined components."
                )
            )

        return {
            comp: self.component_lags.get(comp, self.default) for comp in components
        }

    def shifted(self, n: int) -> Lags:
        """Return an unresolved copy with every lag shifted by ``n`` steps."""
        return Lags(
            kind=self.kind,
            shared=(
                None if self.shared is None else tuple(lag + n for lag in self.shared)
            ),
            component_lags={
                comp: tuple(lag + n for lag in lags)
                for comp, lags in self.component_lags.items()
            },
            default=(
                None if self.default is None else tuple(lag + n for lag in self.default)
            ),
        )

    @property
    def is_component_specific(self) -> bool:
        return self.shared is None

    @property
    def resolved_per_component(self) -> dict[str, tuple[int, ...]]:
        if self._resolved is None:
            raise_log(
                ValueError("`Lags` have not been resolved yet; call `resolve()` first.")
            )
        return self._resolved

    @property
    def extreme_lags(self) -> tuple[int, int]:
        """The minimum and maximum of the declared lags.

        They include the `'default_lags'` even if every component has explicit lags, and do not depend on the
        resolution. The model uses them for its extreme lags and for the input windows of `predict()`.
        """
        if self.shared is not None:
            values = list(self.shared)
        else:
            values = [lag for lags in self.component_lags.values() for lag in lags]
            if self.default is not None:
                values.extend(self.default)
        return _min_max(values)

    @property
    def resolved_extreme_lags(self) -> tuple[int, int]:
        """The minimum and maximum of the lags used by the resolved components.

        The tabularization uses them to compute the feature times. Shared lags do not need a resolution.
        """
        if self.shared is not None:
            values = list(self.shared)
        else:
            values = [
                lag for lags in self.resolved_per_component.values() for lag in lags
            ]
        return _min_max(values)


def _min_max(values: list[int]) -> tuple[int, int]:
    if len(values) == 0:
        raise_log(ValueError("Lags must contain at least one lag."))
    return min(values), max(values)


__all__ = [
    "Lags",
    "LagsKind",
]
