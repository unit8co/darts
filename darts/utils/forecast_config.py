"""
Forecast configuration.
-----------------------

High-level configuration shared by the tabularization, model and historical-forecast layers. It
combines the unresolved lag configuration with the output-chunk strategy. The resolved variant
adds the component mapping produced by :meth:`LagConfig.resolve`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from darts.utils.data.tabularization.lags import LagConfig, ResolvedLagConfig
from darts.utils.output_chunk import (
    DirectOutputStrategy,
    OutputChunkStrategy,
    RecursiveOutputStrategy,
)


@dataclass(frozen=True)
class ForecastConfig:
    """Unresolved configuration of a tabularized forecasting model."""

    lags: LagConfig
    strategy: OutputChunkStrategy

    @classmethod
    def from_user(
        cls,
        lags: object | None = None,
        lags_past_covariates: object | None = None,
        lags_future_covariates: object | None = None,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        multi_models: bool | None = True,
        lags_future_covariates_stepwise: bool | dict[str, bool] = False,
    ) -> ForecastConfig:
        """Build a config from the public model constructor arguments."""
        lag_config = LagConfig.from_user(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            lags_future_covariates_stepwise=lags_future_covariates_stepwise,
        )
        strategy_cls = (
            DirectOutputStrategy
            if multi_models or output_chunk_length == 1
            else RecursiveOutputStrategy
        )
        return cls(lags=lag_config, strategy=strategy_cls(output_chunk_length))

    def resolve(
        self,
        target_components: Sequence[str],
        past_components: Sequence[str] | None = None,
        future_components: Sequence[str] | None = None,
    ) -> ResolvedForecastConfig:
        """Resolve the lag configuration against the training components."""
        return ResolvedForecastConfig(
            lags=self.lags.resolve(
                target_components,
                past_components,
                future_components,
            ),
            strategy=self.strategy,
        )

    @property
    def output_chunk_length(self) -> int:
        return self.lags.output_chunk_length

    @property
    def output_chunk_shift(self) -> int:
        return self.lags.output_chunk_shift


@dataclass(frozen=True)
class ResolvedForecastConfig:
    """Forecast configuration resolved against the training components."""

    lags: ResolvedLagConfig
    strategy: OutputChunkStrategy


__all__ = ["ForecastConfig", "ResolvedForecastConfig"]
