"""
Step-wise Lagged Features
-------------------------

Compact container for lagged features whose future covariates lags are relative to the forecasted step of the
output chunk (see `lags_future_covariates_stepwise` in `SKLearnModel`).
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from darts.logging import raise_log


class StepwiseLaggedFeatures:
    """
    Compact representation of the per-horizon features arrays `X_0, ..., X_{output_chunk_length - 1}` that arise
    when some future covariates lags are relative to the forecasted step ("step-wise" lags).

    Rather than storing `output_chunk_length` copies of the full features array, the container stores:

    - `base`: the features array `X_0` for horizon `0`, with shape `(n_observations, n_features[, n_samples])`.
      The columns of the step-wise features contain the values for horizon `0`.
    - `step_values`: the values of the step-wise columns for every horizon, with shape
      `(n_horizons, n_observations, n_step_cols[, n_samples])`.
    - `step_cols`: the (sorted) indices of the step-wise columns along the features axis of `base`.

    Invariant: `step_values[0] == base[:, step_cols]`.

    The features array for horizon `h` is obtained with `horizon(h)`: a copy of `base` with the step-wise columns
    replaced by `step_values[h]` (horizon `0` returns `base` itself, without copying). Non-step-wise columns (target
    lags, past covariates lags, absolute future covariates lags, static covariates) are only ever modified through
    `base`, so that in-place updates (e.g. auto-regression writing target columns) are reflected in every horizon.

    `format_fn` is an optional hook applied to the array returned by `horizon()` (e.g. to convert it into a
    `pd.DataFrame` for models with categorical features).

    Only the row axis (and, when present, the samples axis) can be indexed with `__getitem__`; indexing the
    features axis is not supported since it would invalidate `step_cols`.
    """

    def __init__(
        self,
        base: np.ndarray,
        step_values: np.ndarray,
        step_cols: np.ndarray | Sequence[int],
        format_fn: Callable[[np.ndarray], Any] | None = None,
    ):
        step_cols = np.asarray(step_cols, dtype=int)
        if step_values.ndim != base.ndim + 1:
            raise_log(
                ValueError(
                    "`step_values` must have exactly one more dimension than `base` (the leading horizon axis); "
                    f"received `base.ndim={base.ndim}` and `step_values.ndim={step_values.ndim}`."
                ),
            )
        if step_values.shape[1] != base.shape[0]:
            raise_log(
                ValueError(
                    "`step_values` and `base` must have the same number of observations; received "
                    f"{step_values.shape[1]} and {base.shape[0]}."
                ),
            )
        if step_values.shape[2] != len(step_cols):
            raise_log(
                ValueError(
                    "The number of step-wise columns in `step_values` must match the length of `step_cols`; "
                    f"received {step_values.shape[2]} and {len(step_cols)}."
                ),
            )
        if base.ndim > 2 and step_values.shape[3:] != base.shape[2:]:
            raise_log(
                ValueError(
                    "`step_values` and `base` must have the same samples dimension; received "
                    f"{step_values.shape[3:]} and {base.shape[2:]}."
                ),
            )
        self.base = base
        self.step_values = step_values
        self.step_cols = step_cols
        self.format_fn = format_fn

    # -- array-like interface --------------------------------------------------------------------------------------

    @property
    def n_horizons(self) -> int:
        """The number of horizons (i.e. `output_chunk_length`) stored in the container."""
        return self.step_values.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the features array of any single horizon (identical to `base.shape`)."""
        return self.base.shape

    @property
    def ndim(self) -> int:
        return self.base.ndim

    @property
    def dtype(self):
        return self.base.dtype

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key) -> "StepwiseLaggedFeatures":
        """
        Indexes the observations axis and, when present, the samples axis. The features axis must be left untouched
        (either omitted or indexed with `:`).
        """
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise_log(
                IndexError(
                    f"Too many indices for `StepwiseLaggedFeatures` with `ndim={self.ndim}`: received {len(key)}."
                ),
            )
        if len(key) >= 2 and not _is_full_slice(key[1]):
            raise_log(
                IndexError(
                    "Indexing the features axis of a `StepwiseLaggedFeatures` is not supported; use `horizon()` "
                    "to materialize the features array of a given horizon."
                ),
            )
        rows_key = key[0]
        if isinstance(rows_key, int | np.integer):
            raise_log(
                IndexError(
                    "Integer indexing of the observations axis of a `StepwiseLaggedFeatures` is not supported "
                    "(it would drop the axis); use a slice instead."
                ),
            )
        base_key = (rows_key,) + key[1:]
        step_key = (slice(None), rows_key, slice(None)) + key[2:]
        return StepwiseLaggedFeatures(
            base=self.base[base_key],
            step_values=self.step_values[step_key],
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    def horizon(self, h: int) -> Any:
        """
        Returns the features array `X_h` for horizon `h` (after applying `format_fn`, if any). Horizon `0` returns
        `base` without copying.
        """
        if not 0 <= h < self.n_horizons:
            raise_log(
                IndexError(
                    f"Horizon must be in `[0, {self.n_horizons - 1}]`; received `{h}`."
                ),
            )
        if h == 0:
            X_h = self.base
        else:
            X_h = self.base.copy()
            X_h[:, self.step_cols] = self.step_values[h]
        return self.format_fn(X_h) if self.format_fn is not None else X_h

    def horizons(self):
        """Yields the features array of every horizon, in order."""
        for h in range(self.n_horizons):
            yield self.horizon(h)

    def with_base(self, base: np.ndarray) -> "StepwiseLaggedFeatures":
        """
        Returns a new container with `base` replaced. The new `base` must keep the step-wise columns at the same
        positions (e.g. columns were appended to the right).
        """
        return StepwiseLaggedFeatures(
            base=base,
            step_values=self.step_values,
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    def repeat(self, repeats: int, axis: int = 0) -> "StepwiseLaggedFeatures":
        """Repeats the observations `repeats` times (only `axis=0` is supported)."""
        if axis != 0:
            raise_log(
                ValueError("`StepwiseLaggedFeatures.repeat()` only supports `axis=0`."),
            )
        return StepwiseLaggedFeatures(
            base=np.repeat(self.base, repeats, axis=0),
            step_values=np.repeat(self.step_values, repeats, axis=1),
            step_cols=self.step_cols,
            format_fn=self.format_fn,
        )

    @classmethod
    def concatenate(
        cls, items: Sequence["StepwiseLaggedFeatures"]
    ) -> "StepwiseLaggedFeatures":
        """Concatenates the observations of several containers sharing the same layout."""
        if not items:
            raise_log(ValueError("Cannot concatenate an empty sequence."))
        first = items[0]
        for item in items[1:]:
            if item.n_horizons != first.n_horizons or not np.array_equal(
                item.step_cols, first.step_cols
            ):
                raise_log(
                    ValueError(
                        "All `StepwiseLaggedFeatures` must share the same number of horizons and step-wise "
                        "columns to be concatenated."
                    ),
                )
        return cls(
            base=np.concatenate([item.base for item in items], axis=0),
            step_values=np.concatenate([item.step_values for item in items], axis=1),
            step_cols=first.step_cols,
            format_fn=first.format_fn,
        )

    def __repr__(self) -> str:
        return (
            f"StepwiseLaggedFeatures(shape={self.shape}, n_horizons={self.n_horizons}, "
            f"step_cols={self.step_cols.tolist()})"
        )


def _is_full_slice(key) -> bool:
    return isinstance(key, slice) and key == slice(None)


def concatenate_lagged_features(
    items: Sequence[np.ndarray | StepwiseLaggedFeatures],
) -> np.ndarray | StepwiseLaggedFeatures:
    """Concatenates lagged features along the observations axis, dispatching on the type of the items."""
    if isinstance(items[0], StepwiseLaggedFeatures):
        return StepwiseLaggedFeatures.concatenate(items)
    return np.concatenate(items, axis=0)
