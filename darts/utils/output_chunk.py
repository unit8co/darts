"""
Output-chunk strategies.
------------------------

Internal strategies that define how a forecast horizon and target components map to the estimators
of a multi-output model. They are independent from the model so they can be reused by
``get_estimator()`` and the optimized historical forecasts. This module is not part of the public
Darts API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from darts.logging import raise_log


class OutputChunkStrategy(ABC):
    """How output positions are assigned to estimators.

    Internal API shared by ``SKLearnModel`` and the optimized historical forecasts. It is
    not part of the public Darts API.
    """

    name: str
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

    def output_horizons(self, n_components: int) -> np.ndarray:
        """The forecast horizon of every estimator output."""
        self._validate_n_components(n_components)
        n_outputs = self.n_outputs(n_components)
        if self.name == "recursive":
            return np.zeros(n_outputs, dtype=int)
        return np.repeat(np.arange(self.pred_dim), n_components)

    @abstractmethod
    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
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


class DirectOutputStrategy(OutputChunkStrategy):
    """One estimator per horizon and target component (``multi_models=True``)."""

    name = "direct"

    def __init__(self, output_chunk_length: int):
        super().__init__(output_chunk_length)
        self.pred_dim = output_chunk_length
        self.feature_lag_offset = 0

    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        self._validate_horizon(horizon)
        self._validate_component(component, n_components)
        return horizon * n_components + component

    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
    ) -> bool:
        return not native_multioutput or sample_weight


class RecursiveOutputStrategy(OutputChunkStrategy):
    """One estimator per target component predicting the last chunk step (``multi_models=False``)."""

    name = "recursive"

    def __init__(self, output_chunk_length: int):
        super().__init__(output_chunk_length)
        self.pred_dim = 1
        self.feature_lag_offset = output_chunk_length - 1

    def estimator_index(self, horizon: int, component: int, n_components: int) -> int:
        self._validate_horizon(horizon)
        self._validate_component(component, n_components)
        return component

    def requires_wrapper(
        self,
        *,
        native_multioutput: bool,
        sample_weight: bool,
    ) -> bool:
        return not native_multioutput or sample_weight


__all__ = [
    "DirectOutputStrategy",
    "OutputChunkStrategy",
    "RecursiveOutputStrategy",
]
