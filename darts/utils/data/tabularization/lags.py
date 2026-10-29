"""
Lag configuration for tabularization.
-------------------------------------

Internal semantic layer for the lags of a tabularized model: parsing and validating user-provided
lags, handling ``output_chunk_shift`` and resolving component-specific lags against the components
of the actual series. This module is not part of the public Darts API.

The classes are intentionally independent from the tabularization and model layers. This keeps
the validation in a single place while the call sites are migrated incrementally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
_STEPWISE_ARG_NAME = "lags_future_covariates_stepwise"
_DEFAULT_LAGS_KEY = "default_lags"


def _validate_kind(kind: str) -> LagsKind:
    if kind not in _LAGS_ARG_NAMES:
        raise_log(ValueError(f"Unknown lags kind: {kind!r}."))
    return kind


def _has_stepwise_flags(stepwise: bool | Mapping[str, bool]) -> bool:
    """Whether at least one component is flagged as step-wise."""
    if isinstance(stepwise, Mapping):
        return any(stepwise.values())
    return bool(stepwise)


def _normalize_stepwise(
    stepwise: bool | Mapping[str, bool],
) -> bool | dict[str, bool]:
    """Validate and copy the step-wise flags."""
    if isinstance(stepwise, bool):
        return stepwise
    if not isinstance(stepwise, Mapping):
        raise_log(
            ValueError(
                f"`{_STEPWISE_ARG_NAME}` must be a boolean or a dictionary mapping "
                "`future_covariates` component names to booleans. Given: "
                f"{type(stepwise)}."
            )
        )
    if len(stepwise) == 0:
        raise_log(
            ValueError(
                f"When passed as a dictionary, `{_STEPWISE_ARG_NAME}` must contain at least "
                "one key."
            )
        )
    for comp_name, is_stepwise in stepwise.items():
        if not isinstance(is_stepwise, bool):
            raise_log(
                ValueError(
                    f"`{_STEPWISE_ARG_NAME}` - `{comp_name}`: must be a boolean. "
                    f"Given: {type(is_stepwise)}."
                )
            )
    return dict(stepwise)


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


@dataclass(frozen=True)
class ResolvedLags:
    """Lags resolved against the components of one series."""

    per_component: Mapping[str, tuple[int, ...]]
    is_component_specific: bool
    legacy_lags: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(
            self,
            "per_component",
            {comp: tuple(lags) for comp, lags in self.per_component.items()},
        )
        object.__setattr__(self, "legacy_lags", tuple(self.legacy_lags))


class Lags(ABC):
    """Semantic representation of the lags of one variate."""

    _kind: LagsKind

    @classmethod
    def from_user(cls, value: object, *, kind: LagsKind) -> Lags:
        """
        Parse the user representation of the lags of one variate.

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
                return SharedLags(
                    _normalize_component_lags(
                        value[_DEFAULT_LAGS_KEY], _DEFAULT_LAGS_KEY, kind
                    ),
                    _kind=kind,
                )

            per_component: dict[str, tuple[int, ...]] = {}
            default: tuple[int, ...] | None = None
            for comp_name, comp_lags in value.items():
                component_lags = _normalize_component_lags(comp_lags, comp_name, kind)
                if comp_name == _DEFAULT_LAGS_KEY:
                    default = component_lags
                else:
                    per_component[comp_name] = component_lags
            return ComponentLags(per_component, default=default, _kind=kind)

        return SharedLags(
            _normalize_component_lags(value, _DEFAULT_LAGS_KEY, kind), _kind=kind
        )

    def resolve(self, components: Sequence[str]) -> ResolvedLags:
        """Resolve the lags against the components of the series."""
        try:
            return _resolve_lags(self, components, self._kind)
        except ValueError as exc:
            raise_log(exc)

    @abstractmethod
    def shifted(self, n: int) -> Lags:
        """Return a copy with every lag shifted by ``n`` steps."""

    @property
    @abstractmethod
    def legacy_lags(self) -> tuple[int, ...]:
        """The lag values used by the legacy ``model.lags`` representation."""


@dataclass(frozen=True)
class SharedLags(Lags):
    """The same lags for every component."""

    lags: tuple[int, ...]
    _kind: LagsKind = field(default="target", repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "lags", tuple(self.lags))

    def shifted(self, n: int) -> SharedLags:
        return SharedLags(tuple(lag + n for lag in self.lags), _kind=self._kind)

    @property
    def legacy_lags(self) -> tuple[int, ...]:
        return self.lags


@dataclass(frozen=True)
class ComponentLags(Lags):
    """Component-specific lags, with an optional default for unspecified components."""

    per_component: Mapping[str, tuple[int, ...]]
    default: tuple[int, ...] | None = None
    _kind: LagsKind = field(default="target", repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "per_component",
            {comp: tuple(lags) for comp, lags in self.per_component.items()},
        )
        if self.default is not None:
            object.__setattr__(self, "default", tuple(self.default))

    def shifted(self, n: int) -> ComponentLags:
        return ComponentLags(
            {
                comp: tuple(lag + n for lag in lags)
                for comp, lags in self.per_component.items()
            },
            default=(
                None if self.default is None else tuple(lag + n for lag in self.default)
            ),
            _kind=self._kind,
        )

    @property
    def legacy_lags(self) -> tuple[int, ...]:
        all_lags = [lag for lags in self.per_component.values() for lag in lags]
        if self.default is not None:
            all_lags.extend(self.default)
        if len(all_lags) == 0:
            raise ValueError("Component-specific lags must contain at least one lag.")
        return min(all_lags), max(all_lags)


@dataclass(frozen=True)
class FutureLags:
    """Lags and step-wise flags for the future covariates."""

    lags: Lags
    stepwise: bool | Mapping[str, bool] = False

    def __post_init__(self):
        stepwise = _normalize_stepwise(self.stepwise)
        object.__setattr__(self, "stepwise", stepwise)

        if isinstance(stepwise, dict) and isinstance(self.lags, ComponentLags):
            missing_lags = [
                comp_name
                for comp_name, is_stepwise in stepwise.items()
                if is_stepwise
                and comp_name != _DEFAULT_LAGS_KEY
                and comp_name not in self.lags.per_component
            ]
            if missing_lags and self.lags.default is None:
                raise_log(
                    ValueError(
                        f"The `{_STEPWISE_ARG_NAME}` dictionary flags components for which "
                        f"`{_LAGS_ARG_NAMES['future']}` does not define any lags: "
                        f"{sorted(missing_lags)}. Either add them to "
                        f"`{_LAGS_ARG_NAMES['future']}` or provide its 'default_lags' key."
                    )
                )

    @classmethod
    def from_user(
        cls,
        value: object | None,
        *,
        stepwise: bool | Mapping[str, bool] = False,
    ) -> FutureLags | None:
        normalized = _normalize_stepwise(stepwise)
        if value is None:
            if _has_stepwise_flags(normalized):
                raise_log(
                    ValueError(
                        f"`{_STEPWISE_ARG_NAME}` requires "
                        f"`{_LAGS_ARG_NAMES['future']}` to be not None."
                    )
                )
            return None
        return cls(Lags.from_user(value, kind="future"), stepwise=normalized)

    def shifted(self, n: int) -> FutureLags:
        return FutureLags(self.lags.shifted(n), stepwise=self.stepwise)

    @property
    def has_stepwise(self) -> bool:
        return _has_stepwise_flags(self.stepwise)


@dataclass(frozen=True)
class ResolvedLagConfig:
    """Internal lag configuration resolved against the components of the training series."""

    target: Mapping[str, tuple[int, ...]] | None
    past: Mapping[str, tuple[int, ...]] | None
    future: Mapping[str, tuple[int, ...]] | None
    future_stepwise: Mapping[str, bool] | None
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
            self,
            "future_stepwise",
            None if self.future_stepwise is None else dict(self.future_stepwise),
        )
        object.__setattr__(
            self, "is_component_specific", dict(self.is_component_specific)
        )

    @property
    def has_stepwise(self) -> bool:
        return bool(self.future_stepwise) and any(self.future_stepwise.values())

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
    future: FutureLags | None = None
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
        lags_future_covariates_stepwise: bool | Mapping[str, bool] = False,
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

        stepwise = _normalize_stepwise(lags_future_covariates_stepwise)
        if lags_future_covariates is None and _has_stepwise_flags(stepwise):
            raise_log(
                ValueError(
                    f"`{_STEPWISE_ARG_NAME}` requires "
                    f"`{_LAGS_ARG_NAMES['future']}` to be not None."
                )
            )

        return cls(
            target=(None if lags is None else Lags.from_user(lags, kind="target")),
            past=(
                None
                if lags_past_covariates is None
                else Lags.from_user(lags_past_covariates, kind="past")
            ),
            future=FutureLags.from_user(lags_future_covariates, stepwise=stepwise),
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
        resolved: dict[str, ResolvedLags] = {}
        for kind, lags in (
            ("target", self.target),
            ("past", self.past),
            ("future", None if self.future is None else self.future.lags),
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
            # the legacy lag generation.
            if kind == "future" and self.output_chunk_shift:
                lags = lags.shifted(self.output_chunk_shift)

            try:
                resolved[kind] = _resolve_lags(lags, kind_components, kind)
            except ValueError as exc:
                errors.append(str(exc))

        future_stepwise: dict[str, bool] | None = None
        if self.future is not None and components["future"] is not None:
            if self.output_chunk_length == 1:
                future_stepwise = {}
            else:
                try:
                    future_stepwise = _resolve_stepwise(
                        self.future, components["future"]
                    )
                except ValueError as exc:
                    errors.append(str(exc))

        if errors:
            raise_log(ValueError("\n".join(errors)))

        per_component = {kind: resolved[kind].per_component for kind in resolved}
        is_component_specific = {
            kind: resolved[kind].is_component_specific for kind in resolved
        }
        legacy_lags = {kind: resolved[kind].legacy_lags for kind in resolved}

        return ResolvedLagConfig(
            target=per_component.get("target"),
            past=per_component.get("past"),
            future=per_component.get("future"),
            future_stepwise=future_stepwise,
            is_component_specific=is_component_specific,
            legacy_lags=legacy_lags,
        )

    def _lags_for_kind(self, kind: LagsKind) -> Lags | None:
        if kind == "target":
            return self.target
        if kind == "past":
            return self.past
        return None if self.future is None else self.future.lags

    def legacy_lags_dict(self) -> dict[str, list[int]]:
        """The pre-fit ``model.lags`` representation of this config."""
        processed: dict[str, list[int]] = {}
        for kind in _LAGS_ARG_NAMES:
            lags = self._lags_for_kind(kind)
            if lags is None:
                continue
            if kind == "future" and self.output_chunk_shift:
                lags = lags.shifted(self.output_chunk_shift)
            processed[kind] = list(lags.legacy_lags)
        return processed

    def legacy_component_lags_dict(self) -> dict[str, dict[str, list[int]]]:
        """The pre-fit ``model.component_lags`` representation of this config."""
        processed: dict[str, dict[str, list[int]]] = {}
        for kind in _LAGS_ARG_NAMES:
            lags = self._lags_for_kind(kind)
            if not isinstance(lags, ComponentLags):
                continue
            if kind == "future" and self.output_chunk_shift:
                lags = lags.shifted(self.output_chunk_shift)
            component_lags = {
                comp: list(comp_lags) for comp, comp_lags in lags.per_component.items()
            }
            if lags.default is not None:
                component_lags[_DEFAULT_LAGS_KEY] = list(lags.default)
            processed[kind] = component_lags
        return processed


def _resolve_lags(
    lags: Lags, components: Sequence[str], kind: LagsKind
) -> ResolvedLags:
    """Resolve the lags against ``components``; raise ``ValueError`` on invalid components."""
    components = tuple(components)

    if isinstance(lags, SharedLags):
        return ResolvedLags(
            per_component={comp: lags.lags for comp in components},
            is_component_specific=False,
            legacy_lags=lags.legacy_lags,
        )

    if not isinstance(lags, ComponentLags):
        raise TypeError(f"Unsupported lags type: {type(lags)}.")

    required = set(components)
    provided = set(lags.per_component)
    wrong_components = [comp for comp in lags.per_component if comp not in required]
    missing_components = [comp for comp in components if comp not in provided]
    lags_name = _LAGS_ARG_NAMES[kind]

    if len(wrong_components) > 0:
        raise ValueError(
            f"The `{lags_name}` dictionary specifies lags for components that are not "
            f"present in the series : {wrong_components}. They must be removed to avoid any ambiguity."
        )
    if len(missing_components) > 0 and lags.default is None:
        raise ValueError(
            f"The {lags_name} dictionary is missing the lags for the following components "
            f"present in the series: {missing_components}. The key 'default_lags' can be used to provide lags for "
            "all the non-explicitely defined components."
        )

    return ResolvedLags(
        per_component={
            comp: lags.per_component.get(comp, lags.default) for comp in components
        },
        is_component_specific=True,
        legacy_lags=lags.legacy_lags,
    )


def _resolve_stepwise(future: FutureLags, components: Sequence[str]) -> dict[str, bool]:
    """Resolve and validate the step-wise flags against the future components."""
    stepwise = future.stepwise
    if not _has_stepwise_flags(stepwise):
        return {}
    if isinstance(stepwise, bool):
        stepwise = {_DEFAULT_LAGS_KEY: stepwise}

    components = tuple(components)
    required = set(components)
    provided = set(stepwise)
    wrong_components = [
        comp for comp in stepwise if comp != _DEFAULT_LAGS_KEY and comp not in required
    ]
    missing_components = [comp for comp in components if comp not in provided]

    if len(wrong_components) > 0:
        raise ValueError(
            f"The `{_STEPWISE_ARG_NAME}` dictionary specifies flags for components that are not "
            f"present in the series : {wrong_components}. They must be removed to avoid any ambiguity."
        )
    if len(missing_components) > 0 and _DEFAULT_LAGS_KEY not in provided:
        raise ValueError(
            f"The {_STEPWISE_ARG_NAME} dictionary is missing the flags for the following "
            f"components present in the series: {missing_components}. The key 'default_lags' can be used to "
            "provide flags for all the non-explicitely defined components."
        )

    default_stepwise = stepwise.get(_DEFAULT_LAGS_KEY, False)
    return {comp: bool(stepwise.get(comp, default_stepwise)) for comp in components}


__all__ = [
    "ComponentLags",
    "FutureLags",
    "LagConfig",
    "Lags",
    "ResolvedLagConfig",
    "ResolvedLags",
    "SharedLags",
]
