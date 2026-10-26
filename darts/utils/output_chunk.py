"""
Output-chunk strategies.
------------------------

Strategies that define how a forecast horizon and target components map to the estimators of a
multi-output model. They are independent from the feature containers and from the model so they
can be reused by the wrapper, ``get_estimator()`` and the optimized historical forecasts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from darts.logging import raise_log


class OutputChunkStrategy(ABC):
    """How output positions are assigned to estimators."""

    name: str
    n_horizons: int
    pred_dim: int
    feature_lag_offset: int

    def __init__(self, output_chunk_length: int):
        if not isinstance(output_chunk_length, int) or isinstance(
            output_chunk_length, bool
        ):
            raise_log(
                ValueError(
                    "`output_chunk_length` must be a positive integer greater than 0. "
                    f"Given: {output_chunk_length}."
                ),
            )
        if output_chunk_length < 1:
            raise_log(
                ValueError(
                    "`output_chunk_length` must be a positive integer greater than 0. "
                    f"Given: {output_chunk_length}."
                ),
            )
        self.output_chunk_length = output_chunk_length

    def n_outputs(self, n_components: int) -> int:
        """Number of estimator outputs for ``n_components`` target components."""
        self._validate_n_components(n_components)
        return self.pred_dim * n_components

    @abstractmethod
    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        """Index of the estimator for ``horizon`` and ``component``."""

    @abstractmethod
    def horizon_of(self, output_idx: int, n_components: int) -> int:
        """Horizon that the output at ``output_idx`` belongs to."""

    @abstractmethod
    def component_of(self, output_idx: int, n_components: int) -> int:
        """Target component that the output at ``output_idx`` belongs to."""

    def chain_columns(self, output_idx: int, n_components: int) -> tuple[int, ...]:
        """Output columns chained into the features of ``output_idx``."""
        return ()

    def fit_order(self, n_outputs: int, n_components: int) -> list[list[int]]:
        """Groups of outputs that can be fitted in parallel."""
        self._validate_outputs(n_outputs, n_components)
        return [list(range(n_outputs))]

    def predict_order(self, n_outputs: int, n_components: int) -> list[list[int]]:
        """Groups of outputs that can be predicted in parallel."""
        self._validate_outputs(n_outputs, n_components)
        return [list(range(n_outputs))]

    @abstractmethod
    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
        per_horizon_features: bool = False,
    ) -> bool:
        """Whether the model needs the multi-output wrapper."""

    def _validate_n_components(self, n_components: int) -> None:
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise_log(
                ValueError(
                    f"`n_components` must be a positive integer. Given: {n_components}."
                ),
            )
        if n_components < 1:
            raise_log(
                ValueError(
                    f"`n_components` must be a positive integer. Given: {n_components}."
                ),
            )

    def _validate_outputs(self, n_outputs: int, n_components: int) -> None:
        expected = self.n_outputs(n_components)
        if n_outputs != expected:
            raise_log(
                ValueError(
                    f"`n_outputs` must be {expected} for `n_components={n_components}`; "
                    f"received {n_outputs}."
                ),
            )

    def _validate_horizon(self, horizon: int) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise_log(
                ValueError(
                    f"`horizon` must be an integer in `[0, {self.output_chunk_length - 1}]`; "
                    f"received {horizon}."
                ),
            )
        if not 0 <= horizon < self.output_chunk_length:
            raise_log(
                ValueError(
                    f"`horizon` must be in `[0, {self.output_chunk_length - 1}]`; "
                    f"received {horizon}."
                ),
            )

    def _validate_component(self, component: int, n_components: int) -> None:
        self._validate_n_components(n_components)
        if not isinstance(component, int) or isinstance(component, bool):
            raise_log(
                ValueError(
                    f"`component` must be an integer in `[0, {n_components - 1}]`; "
                    f"received {component}."
                ),
            )
        if not 0 <= component < n_components:
            raise_log(
                ValueError(
                    f"`component` must be in `[0, {n_components - 1}]`; "
                    f"received {component}."
                ),
            )

    def _validate_output_idx(self, output_idx: int, n_components: int) -> None:
        n_outputs = self.n_outputs(n_components)
        if not isinstance(output_idx, int) or isinstance(output_idx, bool):
            raise_log(
                ValueError(
                    f"`output_idx` must be an integer in `[0, {n_outputs - 1}]`; "
                    f"received {output_idx}."
                ),
            )
        if not 0 <= output_idx < n_outputs:
            raise_log(
                ValueError(
                    f"`output_idx` must be in `[0, {n_outputs - 1}]`; "
                    f"received {output_idx}."
                ),
            )


class DirectOutputStrategy(OutputChunkStrategy):
    """One estimator per horizon and target component (``multi_models=True``)."""

    name = "direct"

    def __init__(self, output_chunk_length: int):
        super().__init__(output_chunk_length)
        self.n_horizons = output_chunk_length
        self.pred_dim = output_chunk_length
        self.feature_lag_offset = 0

    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        self._validate_horizon(horizon)
        self._validate_component(component, n_components)
        return horizon * n_components + component

    def horizon_of(self, output_idx: int, n_components: int) -> int:
        self._validate_output_idx(output_idx, n_components)
        return output_idx // n_components

    def component_of(self, output_idx: int, n_components: int) -> int:
        self._validate_output_idx(output_idx, n_components)
        return output_idx % n_components

    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
        per_horizon_features: bool = False,
    ) -> bool:
        return not native_multioutput or sample_weight or per_horizon_features


class RecursiveOutputStrategy(OutputChunkStrategy):
    """One estimator per target component predicting the last chunk step (``multi_models=False``)."""

    name = "recursive"

    def __init__(self, output_chunk_length: int):
        super().__init__(output_chunk_length)
        self.n_horizons = 1
        self.pred_dim = 1
        self.feature_lag_offset = output_chunk_length - 1

    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        self._validate_horizon(horizon)
        self._validate_component(component, n_components)
        return component

    def horizon_of(self, output_idx: int, n_components: int) -> int:
        self._validate_output_idx(output_idx, n_components)
        return 0

    def component_of(self, output_idx: int, n_components: int) -> int:
        self._validate_output_idx(output_idx, n_components)
        return output_idx

    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
        per_horizon_features: bool = False,
    ) -> bool:
        return not native_multioutput or sample_weight


class DirRecOutputStrategy(OutputChunkStrategy):
    """Reserved interface for the future DirRec strategy.

    The actual chained implementation is intentionally left for the DirRec feature track.
    """

    name = "dirrec"

    def __init__(self, output_chunk_length: int):
        super().__init__(output_chunk_length)
        self.n_horizons = output_chunk_length
        self.pred_dim = output_chunk_length
        self.feature_lag_offset = 0

    @staticmethod
    def _not_implemented() -> None:
        raise_log(
            NotImplementedError(
                "`DirRecOutputStrategy` is reserved for the DirRec feature and is not "
                "implemented yet."
            )
        )

    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        self._not_implemented()

    def horizon_of(self, output_idx: int, n_components: int) -> int:
        self._not_implemented()

    def component_of(self, output_idx: int, n_components: int) -> int:
        self._not_implemented()

    def chain_columns(self, output_idx: int, n_components: int) -> tuple[int, ...]:
        self._not_implemented()

    def fit_order(self, n_outputs: int, n_components: int) -> list[list[int]]:
        self._not_implemented()

    def predict_order(self, n_outputs: int, n_components: int) -> list[list[int]]:
        self._not_implemented()

    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
        per_horizon_features: bool = False,
    ) -> bool:
        return True


__all__ = [
    "DirRecOutputStrategy",
    "DirectOutputStrategy",
    "OutputChunkStrategy",
    "RecursiveOutputStrategy",
]
