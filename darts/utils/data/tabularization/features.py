"""
Lagged feature containers.
--------------------------

Container that represents the lagged features matrix together with its forecast-horizon
metadata. It is independent from the lag semantics and from the tabularization code, so the model
and the multi-output wrapper can consume it without branching on the container type.

``LaggedFeatures`` is part of the public tabularization API.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    n_horizons
        The number of forecast horizons the features are used for. Default: ``1``.
    output_horizons
        Optionally, the horizon in `[0, n_horizons)` of each estimator output. If ``None``, every output uses
        horizon ``0``.

    Raises
    ------
    ValueError
        If `array` has less than 2 dimensions, if `n_horizons` is not a positive integer, or if `output_horizons`
        is not a one-dimensional integer array with values in `[0, n_horizons)`.
    """

    array: np.ndarray
    n_horizons: int = 1
    output_horizons: np.ndarray | None = None

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
        if not isinstance(self.n_horizons, int) or isinstance(self.n_horizons, bool):
            raise_log(
                ValueError(
                    f"`n_horizons` must be an integer greater than 0. Given: {self.n_horizons}."
                )
            )
        if self.n_horizons < 1:
            raise_log(
                ValueError(
                    f"`n_horizons` must be an integer greater than 0. Given: {self.n_horizons}."
                )
            )
        if self.output_horizons is not None:
            output_horizons = np.asarray(self.output_horizons)
            if output_horizons.ndim != 1:
                raise_log(
                    ValueError(
                        "`output_horizons` must be a one-dimensional array; received "
                        f"{output_horizons.ndim} dimensions."
                    )
                )
            if not np.issubdtype(output_horizons.dtype, np.integer):
                raise_log(
                    ValueError(
                        "`output_horizons` must contain integers; received dtype "
                        f"{output_horizons.dtype}."
                    )
                )
            if np.any((output_horizons < 0) | (output_horizons >= self.n_horizons)):
                raise_log(
                    ValueError(
                        "`output_horizons` must be in "
                        f"`[0, {self.n_horizons - 1}]`; received {output_horizons}."
                    )
                )
            self.output_horizons = output_horizons

    @classmethod
    def _from_array(
        cls,
        array: np.ndarray,
        n_horizons: int,
        output_horizons: np.ndarray | None = None,
    ) -> LaggedFeatures:
        """Internal fast path for already validated arrays and metadata."""
        obj = object.__new__(cls)
        obj.array = array
        obj.n_horizons = n_horizons
        obj.output_horizons = output_horizons
        return obj

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

    def for_horizon(self, h: int) -> np.ndarray:
        """Return the feature matrix of horizon ``h``.

        In this version the same base matrix is shared by every horizon.
        """
        if not isinstance(h, int) or isinstance(h, bool):
            raise_log(
                ValueError(
                    f"Horizon must be an integer in `[0, {self.n_horizons - 1}]`; received `{h}`."
                )
            )
        if not 0 <= h < self.n_horizons:
            raise_log(
                IndexError(
                    f"Horizon must be in `[0, {self.n_horizons - 1}]`; received `{h}`."
                )
            )
        return self.array

    def for_output(self, output_idx: int) -> np.ndarray:
        """Return the feature matrix for the estimator output ``output_idx``."""
        if self.output_horizons is None:
            h = 0
        else:
            if not isinstance(output_idx, int) or isinstance(output_idx, bool):
                raise_log(
                    ValueError(
                        "`output_idx` must be an integer in "
                        f"`[0, {len(self.output_horizons) - 1}]`; received `{output_idx}`."
                    )
                )
            if not 0 <= output_idx < len(self.output_horizons):
                raise_log(
                    IndexError(
                        "`output_idx` must be in "
                        f"`[0, {len(self.output_horizons) - 1}]`; received `{output_idx}`."
                    )
                )
            h = int(self.output_horizons[output_idx])
        return self.for_horizon(h)

    def with_base(self, array: np.ndarray) -> LaggedFeatures:
        """Return a new container with the base matrix replaced."""
        return LaggedFeatures(
            array=array,
            n_horizons=self.n_horizons,
            output_horizons=self.output_horizons,
        )

    def repeat(self, repeats: int, axis: int = 0) -> LaggedFeatures:
        """Repeat the observations ``repeats`` times along ``axis=0``."""
        if axis != 0:
            raise_log(
                ValueError("`LaggedFeatures.repeat()` only supports `axis=0`."),
            )
        return self.with_base(np.repeat(self.array, repeats, axis=0))

    def select_samples(self, index: int) -> LaggedFeatures:
        """Select the sample at position ``index`` from a 3D feature matrix."""
        if self.array.ndim != 3:
            raise_log(
                ValueError(
                    "`select_samples()` requires a 3D feature matrix "
                    f"(`n_observations`, `n_features`, `n_samples`); received `ndim={self.ndim}`."
                )
            )
        return self.with_base(self.array[..., index])

    @classmethod
    def concatenate(cls, items: Sequence[LaggedFeatures]) -> LaggedFeatures:
        """Concatenate the observations of several containers."""
        if not items:
            raise_log(ValueError("Cannot concatenate an empty sequence."))

        first = items[0]
        for item in items[1:]:
            if item.n_horizons != first.n_horizons:
                raise_log(
                    ValueError(
                        "All `LaggedFeatures` must share the same number of horizons to be "
                        "concatenated."
                    ),
                )
            if not np.array_equal(item.output_horizons, first.output_horizons):
                raise_log(
                    ValueError(
                        "All `LaggedFeatures` must share the same `output_horizons` to be "
                        "concatenated."
                    ),
                )

        return cls(
            array=np.concatenate([item.array for item in items], axis=0),
            n_horizons=first.n_horizons,
            output_horizons=first.output_horizons,
        )


__all__ = ["LaggedFeatures"]
