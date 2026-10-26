"""
Lagged feature containers.
--------------------------

Containers that represent the lagged features matrix as a function of the forecast horizon. They
are independent from the lag semantics and from the tabularization code, so the model and the
multi-output wrapper can consume them without branching on the container type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Self

import numpy as np

from darts.logging import raise_log


class LaggedFeatures(ABC):
    """A feature matrix that can be materialized for each forecast horizon."""

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


class StepwiseLaggedFeatures(LaggedFeatures):
    """Compact per-horizon features for step-wise future covariates lags.

    The container stores the features array for horizon ``0`` (``array``), the values of the
    step-wise columns for every horizon (``step_values``), and the positions of those columns
    (``step_cols``). Rather than storing ``n_horizons`` copies of the complete features matrix,
    only the step-wise columns are materialized per horizon.

    Invariant: ``step_values[0] == array[:, step_cols]``.

    Non-step-wise columns (target lags, past covariates lags, absolute future covariates lags,
    static covariates) are only modified through ``array``, so in-place updates such as the
    auto-regression writing target columns are reflected in every horizon.
    """

    def __init__(
        self,
        array: np.ndarray,
        step_values: np.ndarray,
        step_cols: np.ndarray | Sequence[int],
        format_fn: Callable[[np.ndarray], Any] | None = None,
    ):
        step_cols = np.asarray(step_cols, dtype=int)
        if step_values.ndim != array.ndim + 1:
            raise_log(
                ValueError(
                    "`step_values` must have exactly one more dimension than `array` (the "
                    f"leading horizon axis); received `array.ndim={array.ndim}` and "
                    f"`step_values.ndim={step_values.ndim}`."
                ),
            )
        if step_values.shape[1] != array.shape[0]:
            raise_log(
                ValueError(
                    "`step_values` and `array` must have the same number of observations; "
                    f"received {step_values.shape[1]} and {array.shape[0]}."
                ),
            )
        if step_values.shape[2] != len(step_cols):
            raise_log(
                ValueError(
                    "The number of step-wise columns in `step_values` must match the length "
                    f"of `step_cols`; received {step_values.shape[2]} and {len(step_cols)}."
                ),
            )
        if array.ndim > 2 and step_values.shape[3:] != array.shape[2:]:
            raise_log(
                ValueError(
                    "`step_values` and `array` must have the same samples dimension; "
                    f"received {step_values.shape[3:]} and {array.shape[2:]}."
                ),
            )
        self._array = array
        self.step_values = step_values
        self.step_cols = step_cols
        self.format_fn = format_fn

    @property
    def n_horizons(self) -> int:
        return self.step_values.shape[0]

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    def __getitem__(self, key) -> Self:
        """Index the observations axis and, when present, the samples axis.

        The features axis must be left untouched (either omitted or indexed with ``:``).
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise_log(
                IndexError(
                    f"Too many indices for `StepwiseLaggedFeatures` with `ndim={self.ndim}`: "
                    f"received {len(key)}."
                ),
            )
        if len(key) >= 2 and not _is_full_slice(key[1]):
            raise_log(
                IndexError(
                    "Indexing the features axis of a `StepwiseLaggedFeatures` is not "
                    "supported; use `horizon()` to materialize the features array of a given "
                    "horizon."
                ),
            )
        rows_key = key[0]
        if isinstance(rows_key, int | np.integer):
            raise_log(
                IndexError(
                    "Integer indexing of the observations axis of a `StepwiseLaggedFeatures` "
                    "is not supported (it would drop the axis); use a slice instead."
                ),
            )
        return self.__class__(
            array=self._array[key],
            step_values=self.step_values[(slice(None),) + key],
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    def horizon(self, h: int) -> Any:
        if not 0 <= h < self.n_horizons:
            raise_log(
                IndexError(
                    f"Horizon must be in `[0, {self.n_horizons - 1}]`; received `{h}`."
                ),
            )
        if h == 0:
            features = self._array
        else:
            features = self._array.copy()
            features[:, self.step_cols] = self.step_values[h]
        return self.format_fn(features) if self.format_fn is not None else features

    def horizons(self):
        """Yield the features array of every horizon, in order."""
        for h in range(self.n_horizons):
            yield self.horizon(h)

    def with_base(self, array: np.ndarray) -> StepwiseLaggedFeatures:
        return self.__class__(
            array=array,
            step_values=self.step_values,
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    def with_format_fn(
        self, format_fn: Callable[[np.ndarray], Any] | None
    ) -> StepwiseLaggedFeatures:
        return self.__class__(
            array=self._array,
            step_values=self.step_values,
            step_cols=self.step_cols,
            format_fn=format_fn,
        )

    def repeat(self, repeats: int, axis: int = 0) -> StepwiseLaggedFeatures:
        if axis != 0:
            raise_log(
                ValueError("`StepwiseLaggedFeatures.repeat()` only supports `axis=0`."),
            )
        return self.__class__(
            array=np.repeat(self._array, repeats, axis=0),
            step_values=np.repeat(self.step_values, repeats, axis=1),
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    @classmethod
    def concatenate(
        cls, items: Sequence[StepwiseLaggedFeatures]
    ) -> StepwiseLaggedFeatures:
        if not items:
            raise_log(ValueError("Cannot concatenate an empty sequence."))

        first = items[0]
        for item in items[1:]:
            if item.n_horizons != first.n_horizons or not np.array_equal(
                item.step_cols, first.step_cols
            ):
                raise_log(
                    ValueError(
                        "All `StepwiseLaggedFeatures` must share the same number of horizons "
                        "and step-wise columns to be concatenated."
                    ),
                )

        return cls(
            array=np.concatenate([item.array for item in items], axis=0),
            step_values=np.concatenate([item.step_values for item in items], axis=1),
            step_cols=first.step_cols,
            format_fn=first.format_fn,
        )

    def __repr__(self) -> str:
        return (
            f"StepwiseLaggedFeatures(shape={self.shape}, "
            f"n_horizons={self.n_horizons}, step_cols={self.step_cols.tolist()})"
        )


def _is_full_slice(key) -> bool:
    return isinstance(key, slice) and key == slice(None)


__all__ = [
    "ArrayLaggedFeatures",
    "LaggedFeatures",
    "StepwiseLaggedFeatures",
]
