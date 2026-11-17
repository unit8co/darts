"""
Lagged feature container.
-------------------------

Container returned by the public tabularization helpers for the lagged features matrix. The model and the
tabularization core work with plain arrays; only the public helpers wrap them.

``LaggedFeatures`` is part of the public tabularization API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darts.logging import raise_log


@dataclass
class LaggedFeatures:
    """Lagged features created by the tabularization helpers.

    Holds the feature matrix returned by :func:`~darts.utils.data.tabularization.create_lagged_data`,
    :func:`~darts.utils.data.tabularization.create_lagged_training_data` and
    :func:`~darts.utils.data.tabularization.create_lagged_prediction_data`. Use :meth:`to_numpy` to get the
    matrix. See :func:`~darts.utils.data.tabularization.create_lagged_data` for the order of the features, and
    :func:`~darts.utils.data.tabularization.create_lagged_component_names` for their names.

    Parameters
    ----------
    array
        The feature matrix, with shape `(n_observations, n_lagged_features)` or
        `(n_observations, n_lagged_features, n_samples)`.

    Raises
    ------
    ValueError
        If `array` has less than 2 dimensions.
    """

    array: np.ndarray

    def __post_init__(self):
        self.array = np.asarray(self.array)
        if self.array.ndim < 2:
            raise_log(
                ValueError(
                    "`LaggedFeatures.array` must have at least 2 dimensions "
                    f"(`n_observations`, `n_features`[, `n_samples`]); received "
                    f"{self.array.ndim}."
                )
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def __len__(self) -> int:
        return len(self.array)

    def to_numpy(self) -> np.ndarray:
        """Return the underlying feature matrix."""
        return self.array


__all__ = ["LaggedFeatures"]
