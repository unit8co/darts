"""
Lagged feature containers.
--------------------------

Internal containers that represent the lagged features matrix together with its forecast-horizon
metadata. They are independent from the lag semantics and from the tabularization code, so the
model and the multi-output wrapper can consume them without branching on the container type.
This module is not part of the public Darts API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np

from darts.logging import raise_log


class LaggedFeatures(ABC):
    """Internal feature matrix that can be materialized for each forecast horizon."""

    n_horizons: int

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        """The underlying feature matrix."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the features array of a single horizon."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """The number of dimensions of the features array of a single horizon."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The dtype of the features array."""

    @abstractmethod
    def horizon(self, h: int) -> Any:
        """Return the features array of horizon ``h``."""

    @abstractmethod
    def with_base(self, array: np.ndarray) -> Self:
        """Return a new container with the base feature matrix replaced."""

    @abstractmethod
    def with_format_fn(self, format_fn: Callable[[np.ndarray], Any] | None) -> Self:
        """Return a new container that applies ``format_fn`` on materialization."""

    @abstractmethod
    def repeat(self, repeats: int, axis: int = 0) -> Self:
        """Repeat the observations ``repeats`` times."""

    @classmethod
    @abstractmethod
    def concatenate(cls, items: Sequence[Self]) -> Self:
        """Concatenate the observations of several containers."""

    def __len__(self) -> int:
        return len(self.array)


class ArrayLaggedFeatures(np.ndarray, LaggedFeatures):
    """A regular feature matrix that is identical for every forecast horizon.

    The container is a zero-copy view of the original array: :meth:`array` returns a base
    ``np.ndarray`` view and :meth:`horizon` returns the container itself for every valid horizon.
    """

    n_horizons: int
    format_fn: Callable[[np.ndarray], Any] | None

    def __new__(
        cls,
        array: np.ndarray,
        *,
        n_horizons: int = 1,
        format_fn: Callable[[np.ndarray], Any] | None = None,
    ) -> ArrayLaggedFeatures:
        if not isinstance(n_horizons, int) or isinstance(n_horizons, bool):
            raise_log(
                ValueError(
                    f"`n_horizons` must be an integer greater than 0. Given: {n_horizons}."
                ),
            )
        if n_horizons < 1:
            raise_log(
                ValueError(
                    f"`n_horizons` must be an integer greater than 0. Given: {n_horizons}."
                ),
            )
        features = np.asarray(array).view(cls)
        features.n_horizons = n_horizons
        features.format_fn = format_fn
        return features

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_horizons = getattr(obj, "n_horizons", 1)
        self.format_fn = getattr(obj, "format_fn", None)

    @classmethod
    def wrap(
        cls,
        array: np.ndarray,
        *,
        n_horizons: int = 1,
        format_fn: Callable[[np.ndarray], Any] | None = None,
    ) -> ArrayLaggedFeatures:
        """Wrap ``array`` in a zero-copy container."""
        if (
            isinstance(array, cls)
            and array.n_horizons == n_horizons
            and array.format_fn is format_fn
        ):
            return array
        return cls(array, n_horizons=n_horizons, format_fn=format_fn)

    @property
    def array(self) -> np.ndarray:
        return self.view(np.ndarray)

    @property
    def shape(self) -> tuple[int, ...]:
        return super().shape

    @property
    def ndim(self) -> int:
        return super().ndim

    @property
    def dtype(self) -> np.dtype:
        return super().dtype

    def horizon(self, h: int) -> Any:
        if not 0 <= h < self.n_horizons:
            raise_log(
                IndexError(
                    f"Horizon must be in `[0, {self.n_horizons - 1}]`; received `{h}`."
                ),
            )
        if self.format_fn is not None:
            return self.format_fn(self.array)
        return self

    def with_base(self, array: np.ndarray) -> ArrayLaggedFeatures:
        return type(self).wrap(
            array, n_horizons=self.n_horizons, format_fn=self.format_fn
        )

    def with_format_fn(
        self, format_fn: Callable[[np.ndarray], Any] | None
    ) -> ArrayLaggedFeatures:
        return type(self).wrap(
            self.array, n_horizons=self.n_horizons, format_fn=format_fn
        )

    def repeat(self, repeats: int, axis: int = 0) -> ArrayLaggedFeatures:
        repeated = np.asarray(self).repeat(repeats, axis=axis)
        return type(self).wrap(
            repeated, n_horizons=self.n_horizons, format_fn=self.format_fn
        )

    @classmethod
    def concatenate(cls, items: Sequence[ArrayLaggedFeatures]) -> ArrayLaggedFeatures:
        if not items:
            raise_log(ValueError("Cannot concatenate an empty sequence."))

        first = items[0]
        n_horizons = getattr(first, "n_horizons", 1)
        format_fn = getattr(first, "format_fn", None)
        for item in items[1:]:
            if getattr(item, "n_horizons", 1) != n_horizons:
                raise_log(
                    ValueError(
                        "All `ArrayLaggedFeatures` must share the same number of horizons "
                        "to be concatenated."
                    ),
                )
            if getattr(item, "format_fn", None) is not format_fn:
                raise_log(
                    ValueError(
                        "All `ArrayLaggedFeatures` must share the same `format_fn` to be "
                        "concatenated."
                    ),
                )

        return cls.wrap(
            np.concatenate([np.asarray(item) for item in items], axis=0),
            n_horizons=n_horizons,
            format_fn=format_fn,
        )


__all__ = [
    "ArrayLaggedFeatures",
    "LaggedFeatures",
]
