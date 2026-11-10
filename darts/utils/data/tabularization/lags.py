"""
Lag configuration for tabularization.
-------------------------------------

Internal semantic layer for the lags of a tabularized model: parsing and validating user-provided
lags, handling ``output_chunk_shift`` and resolving component-specific lags against the components
of the actual series. This module is not part of the public Darts API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        """Resolve the lags in place against the components of the series."""
        components = tuple(components)
        self._resolved = None

        if self.shared is not None:
            self._resolved = {comp: self.shared for comp in components}
            return

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

        self._resolved = {
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
        """The minimum and maximum lag values."""
        if self._resolved is not None:
            values = [lag for lags in self._resolved.values() for lag in lags]
        elif self.shared is not None:
            values = list(self.shared)
        else:
            values = [lag for lags in self.component_lags.values() for lag in lags]
            if self.default is not None:
                values.extend(self.default)

        if len(values) == 0:
            raise_log(ValueError("Lags must contain at least one lag."))
        return min(values), max(values)


@dataclass(frozen=True)
class ResolvedLagConfig:
    """Internal lag configuration resolved against the components of the training series."""

    target: Mapping[str, tuple[int, ...]] | None
    past: Mapping[str, tuple[int, ...]] | None
    future: Mapping[str, tuple[int, ...]] | None
    is_component_specific: Mapping[str, bool]
    legacy_lags: Mapping[str, tuple[int, ...]]

    def __post_init__(self):
        for name in ("target", "past", "future", "legacy_lags"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    {key: tuple(lags) for key, lags in value.items()},
                )
        object.__setattr__(
            self, "is_component_specific", dict(self.is_component_specific)
        )

    def legacy_lags_dict(self) -> dict[str, list[int]]:
        """The ``model.lags`` representation of the resolved lags."""
        return {
            kind: list(self.legacy_lags[kind])
            for kind in _LAGS_ARG_NAMES
            if kind in self.legacy_lags
        }

    def legacy_component_lags_dict(self) -> dict[str, dict[str, list[int]]]:
        """The ``model.component_lags`` representation of the resolved lags."""
        resolved: dict[str, dict[str, list[int]]] = {}
        for kind in _LAGS_ARG_NAMES:
            if not self.is_component_specific.get(kind, False):
                continue
            component_lags = getattr(self, kind)
            if component_lags is not None:
                resolved[kind] = {
                    comp: list(lags) for comp, lags in component_lags.items()
                }
        return resolved


@dataclass(frozen=True)
class LagConfig:
    """Internal, unresolved lag configuration of a forecast model."""

    target: Lags | None = None
    past: Lags | None = None
    future: Lags | None = None
    output_chunk_length: int = 1
    output_chunk_shift: int = 0

    def __post_init__(self):
        if self.target is None and self.past is None and self.future is None:
            raise_log(
                ValueError(
                    "At least one of `lags`, `lags_future_covariates` or "
                    "`lags_past_covariates` must be not None."
                )
            )

    @classmethod
    def from_user(
        cls,
        lags: object | None = None,
        lags_past_covariates: object | None = None,
        lags_future_covariates: object | None = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
    ) -> LagConfig:
        """Build a config from the public ``SKLearnModel`` constructor arguments."""
        if (
            lags is None
            and lags_past_covariates is None
            and lags_future_covariates is None
        ):
            raise_log(
                ValueError(
                    "At least one of `lags`, `lags_future_covariates` or "
                    "`lags_past_covariates` must be not None."
                )
            )

        return cls(
            target=(None if lags is None else Lags.parse(lags, kind="target")),
            past=(
                None
                if lags_past_covariates is None
                else Lags.parse(lags_past_covariates, kind="past")
            ),
            future=(
                None
                if lags_future_covariates is None
                else Lags.parse(lags_future_covariates, kind="future")
            ),
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
        )

    def resolve(
        self,
        target_components: Sequence[str],
        past_components: Sequence[str] | None = None,
        future_components: Sequence[str] | None = None,
    ) -> ResolvedLagConfig:
        """Resolve component-specific lags and expand ``default_lags``."""
        components = {
            "target": tuple(target_components),
            "past": None if past_components is None else tuple(past_components),
            "future": None if future_components is None else tuple(future_components),
        }

        errors: list[str] = []
        resolved: dict[str, Lags] = {}
        legacy_lags: dict[str, tuple[int, ...]] = {}
        for kind, lags in (
            ("target", self.target),
            ("past", self.past),
            ("future", self.future),
        ):
            if lags is None:
                continue
            kind_components = components[kind]
            if kind_components is None:
                raise_log(
                    ValueError(
                        f"`{_LAGS_ARG_NAMES[kind]}` lags are defined, but no "
                        f"{kind} components were provided."
                    )
                )

            # `output_chunk_shift` is only applied to the future covariates, exactly as in
            # the legacy lag generation. `shifted()` also returns an unresolved copy, so
            # the config itself is never mutated by `resolve()`.
            shift = self.output_chunk_shift if kind == "future" else 0
            lags = lags.shifted(shift)

            # The legacy `model.lags` representation keeps the full list for shared lags,
            # and the extreme values over the user mapping for component-specific lags
            # (including `default_lags`, even when every component defines its own lags).
            # Capture it before `resolve()` narrows the mapping to the actual components.
            legacy_lags[kind] = (
                lags.extreme_lags if lags.is_component_specific else lags.shared
            )

            try:
                lags.resolve(kind_components)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            resolved[kind] = lags

        if errors:
            raise_log(ValueError("\n".join(errors)))

        per_component = {
            kind: lags.resolved_per_component for kind, lags in resolved.items()
        }
        is_component_specific = {
            kind: lags.is_component_specific for kind, lags in resolved.items()
        }

        return ResolvedLagConfig(
            target=per_component.get("target"),
            past=per_component.get("past"),
            future=per_component.get("future"),
            is_component_specific=is_component_specific,
            legacy_lags=legacy_lags,
        )

    def _lags_for_kind(self, kind: LagsKind) -> Lags | None:
        if kind == "target":
            return self.target
        if kind == "past":
            return self.past
        return self.future

    def legacy_lags_dict(self) -> dict[str, list[int]]:
        """The pre-fit ``model.lags`` representation of this config."""
        processed: dict[str, list[int]] = {}
        for kind in _LAGS_ARG_NAMES:
            lags = self._lags_for_kind(kind)
            if lags is None:
                continue
            if kind == "future" and self.output_chunk_shift:
                lags = lags.shifted(self.output_chunk_shift)
            processed[kind] = list(
                lags.extreme_lags if lags.is_component_specific else lags.shared
            )
        return processed

    def legacy_component_lags_dict(self) -> dict[str, dict[str, list[int]]]:
        """The pre-fit ``model.component_lags`` representation of this config."""
        processed: dict[str, dict[str, list[int]]] = {}
        for kind in _LAGS_ARG_NAMES:
            lags = self._lags_for_kind(kind)
            if lags is None or not lags.is_component_specific:
                continue
            if kind == "future" and self.output_chunk_shift:
                lags = lags.shifted(self.output_chunk_shift)
            component_lags = {
                comp: list(comp_lags) for comp, comp_lags in lags.component_lags.items()
            }
            if lags.default is not None:
                component_lags[_DEFAULT_LAGS_KEY] = list(lags.default)
            processed[kind] = component_lags
        return processed


__all__ = [
    "LagConfig",
    "Lags",
    "LagsKind",
    "ResolvedLagConfig",
]
