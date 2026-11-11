"""
Forecast configuration.
-----------------------

Configuration shared by the tabularization, model and historical-forecast layers. It owns the
parsed lags and the output-chunk strategy, and resolves the lags in place against the components
of the actual series.

Build it directly with the same arguments as ``SKLearnModel``. ``ForecastConfig`` is part of the
public tabularization API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from darts.logging import raise_log
from darts.utils.data.tabularization.lags import (
    _LAGS_ARG_NAMES,
    Lags,
    LagsKind,
)
from darts.utils.data.tabularization.output_chunk import (
    DirectOutputStrategy,
    OutputChunkStrategy,
    RecursiveOutputStrategy,
)


@dataclass(init=False)
class ForecastConfig:
    """Configuration of a tabularized forecasting model.

    Build it directly with the same arguments as ``SKLearnModel``. Call :meth:`resolve` once the
    components of the actual series are known; the lags are resolved in place.
    """

    target: Lags | None
    past: Lags | None
    future: Lags | None
    strategy: OutputChunkStrategy
    output_chunk_shift: int

    def __init__(
        self,
        lags: object | None = None,
        lags_past_covariates: object | None = None,
        lags_future_covariates: object | None = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        multi_models: bool | None = True,
    ):
        self.target = None if lags is None else Lags.parse(lags, kind="target")
        self.past = (
            None
            if lags_past_covariates is None
            else Lags.parse(lags_past_covariates, kind="past")
        )
        self.future = (
            None
            if lags_future_covariates is None
            else Lags.parse(lags_future_covariates, kind="future")
        )

        # `output_chunk_shift` is applied exactly once, at construction time.
        if self.future is not None and output_chunk_shift:
            self.future = self.future.shifted(output_chunk_shift)

        strategy_cls = (
            DirectOutputStrategy
            if multi_models or output_chunk_length == 1
            else RecursiveOutputStrategy
        )
        self.strategy = strategy_cls(output_chunk_length)
        self.output_chunk_shift = output_chunk_shift
        self._validate()

    @classmethod
    def _from_lags(
        cls,
        *,
        target: Lags | None = None,
        past: Lags | None = None,
        future: Lags | None = None,
        strategy: OutputChunkStrategy,
        output_chunk_shift: int = 0,
    ) -> ForecastConfig:
        """Build a config from already-parsed lags, without applying the shift again."""
        config = cls.__new__(cls)
        config.target = target
        config.past = past
        config.future = future
        config.strategy = strategy
        config.output_chunk_shift = output_chunk_shift
        config._validate()
        return config

    def _validate(self) -> None:
        if self.target is None and self.past is None and self.future is None:
            raise_log(
                ValueError(
                    "At least one of `lags`, `lags_future_covariates` or "
                    "`lags_past_covariates` must be not None."
                )
            )

    def resolve(
        self,
        target_components: Sequence[str],
        past_components: Sequence[str] | None = None,
        future_components: Sequence[str] | None = None,
    ) -> ForecastConfig:
        """Resolve the lags in place against the components of the series.

        Every variate is validated before anything is assigned: a failed resolution leaves the
        config unchanged.
        """
        components = {
            "target": tuple(target_components),
            "past": None if past_components is None else tuple(past_components),
            "future": None if future_components is None else tuple(future_components),
        }

        mappings: list[tuple[Lags, dict[str, tuple[int, ...]]]] = []
        errors: list[str] = []
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
            try:
                mappings.append((lags, lags._resolve_mapping(kind_components)))
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            raise_log(ValueError("\n".join(errors)))
        for lags, mapping in mappings:
            lags._resolved = mapping
        return self

    @property
    def output_chunk_length(self) -> int:
        return self.strategy.output_chunk_length

    @property
    def pred_dim(self) -> int:
        return self.strategy.pred_dim

    @property
    def feature_lag_offset(self) -> int:
        return self.strategy.feature_lag_offset

    @property
    def multi_models(self) -> bool:
        return self.strategy.name == "direct"

    def lags_for(self, kind: LagsKind) -> Lags | None:
        """The parsed lags of ``kind``, or ``None`` when they were not defined."""
        if kind == "target":
            return self.target
        if kind == "past":
            return self.past
        if kind == "future":
            return self.future
        raise_log(ValueError(f"Unknown lags kind: {kind!r}."))

    def extreme_lags(self, kind: LagsKind) -> tuple[int, int] | None:
        lags = self.lags_for(kind)
        return None if lags is None else lags.extreme_lags

    def resolved_per_component(
        self, kind: LagsKind
    ) -> dict[str, tuple[int, ...]] | None:
        lags = self.lags_for(kind)
        return None if lags is None else lags.resolved_per_component

    def is_component_specific(self, kind: LagsKind) -> bool:
        lags = self.lags_for(kind)
        return False if lags is None else lags.is_component_specific

    def has_lags(self, kind: LagsKind) -> bool:
        return self.lags_for(kind) is not None


__all__ = ["ForecastConfig"]
