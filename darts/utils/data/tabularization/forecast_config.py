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

    Holds the lags of the target series, past covariates and future covariates, and the output chunk
    settings (`output_chunk_length`, `output_chunk_shift` and `multi_models`). It takes the same arguments as
    :class:`~darts.models.forecasting.sklearn_model.SKLearnModel`, and is the first argument of the public
    tabularization helpers (:func:`~darts.utils.data.tabularization.create_lagged_data`,
    :func:`~darts.utils.data.tabularization.create_lagged_training_data`,
    :func:`~darts.utils.data.tabularization.create_lagged_prediction_data` and
    :func:`~darts.utils.data.tabularization.create_lagged_component_names`).

    The lags are validated when the config is created. Component-specific lags are resolved against the
    components of the actual series with :meth:`resolve`; the public helpers resolve a copy and leave the config
    unchanged.

    Parameters
    ----------
    lags
        Lagged target `series` values used as features.
        If an integer, must be > 0. Uses the last `n=lags` past lags; e.g. `(-1, -2, ..., -lags)`, where `0`
        corresponds the first predicted time step of each sample. If `output_chunk_shift > 0`, then
        lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
        If a list of integers, each value must be < 0. Uses only the specified values as lags.
        If a dictionary, the keys correspond to the `series` component names (of the first series when
        using multiple series) and the values correspond to the component lags (integer or list of integers). The
        key 'default_lags' can be used to provide default lags for un-specified components.
    lags_past_covariates
        Lagged `past_covariates` values used as features.
        If an integer, must be > 0. Uses the last `n=lags_past_covariates` past lags; e.g. `(-1, -2, ..., -lags)`,
        where `0` corresponds to the first predicted time step of each sample. If `output_chunk_shift > 0`, then
        lag `-1` translates to `-1 - output_chunk_shift` steps before the first prediction step.
        If a list of integers, each value must be < 0. Uses only the specified values as lags.
        If a dictionary, the keys correspond to the `past_covariates` component names (of the first series when
        using multiple series) and the values correspond to the component lags (integer or list of integers). The
        key 'default_lags' can be used to provide default lags for un-specified components.
    lags_future_covariates
        Lagged `future_covariates` values used as features. The lags are always relative to the first step in the
        output chunk, even when `output_chunk_shift > 0`.
        If a tuple of `(past, future)`, both values must be >= 0 and not both 0. Uses the last `n=past` past lags
        and `n=future` future lags; e.g. `(-past, -(past - 1), ..., -1, 0, 1, .... future - 1)`, where `0`
        corresponds the first predicted time step of each sample.
        If a list of integers, uses only the specified values as lags.
        If a dictionary, the keys correspond to the `future_covariates` component names (of the first series when
        using multiple series) and the values correspond to the component lags (tuple or list of integers). The key
        'default_lags' can be used to provide default lags for un-specified components.
    output_chunk_length
        Number of time steps predicted at once (per chunk) by the model. Default: ``1``.
    output_chunk_shift
        Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
        input chunk end). The `lags_future_covariates` are relative to the first step in the shifted output chunk.
        Default: ``0``.
    multi_models
        If ``True``, a separate model is used for each step in the output chunk, and the labels contain all
        `output_chunk_length` steps. If ``False``, a single model predicts the last step of the output chunk, and
        the labels contain only that step. Ignored when `output_chunk_length = 1`. Default: ``True``.

    Raises
    ------
    ValueError
        If none of `lags`, `lags_past_covariates` or `lags_future_covariates` is given, or if any of them, or
        `output_chunk_length`, is invalid.

    Examples
    --------
    >>> from darts.utils.data.tabularization import ForecastConfig, create_lagged_training_data
    >>> from darts.utils.timeseries_generation import linear_timeseries
    >>> series = linear_timeseries(length=10)
    >>> config = ForecastConfig(lags=[-2, -1], output_chunk_length=2)
    >>> X, y, times, _, _ = create_lagged_training_data(
    ...     config, series, uses_static_covariates=False
    ... )
    >>> X.to_numpy().shape, y.shape
    ((7, 2, 1), (7, 2, 1))
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

        Shared lags are applied to every component; component-specific lags are matched by component name, and
        `'default_lags'` fill the components without explicit lags. Every variate is validated before anything is
        assigned: a failed resolution leaves the config unchanged.

        Parameters
        ----------
        target_components
            The component names of the target series. Can be empty if the config has no target lags.
        past_components
            The component names of the past covariates. Required if the config has past covariates lags.
        future_components
            The component names of the future covariates. Required if the config has future covariates lags.

        Returns
        -------
        ForecastConfig
            The same config, resolved.

        Raises
        ------
        ValueError
            If lags are defined for a variate without components, or if the component-specific lags name
            components that are not in the series or miss components without `'default_lags'`. The errors of all
            variates are reported together.
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
