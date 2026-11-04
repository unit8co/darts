"""
Lagged data containers.
-----------------------

Internal container returned by the config-driven tabularization core. It keeps the
feature matrix together with its labels, time index and ancillary data so call
sites do not have to unpack positional tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from darts.utils.data.tabularization.features import LaggedFeatures

__all__ = ["LaggedData"]


@dataclass
class LaggedData:
    """Tabularized features, labels and their metadata.

    ``X`` is a lagged-features container (or a sequence of them when
    ``concatenate=False``). ``y`` is ``None`` for prediction data. The remaining
    fields mirror the values returned by the historical positional API.
    """

    X: LaggedFeatures | Sequence[LaggedFeatures]
    y: np.ndarray | Sequence[np.ndarray] | None
    times: Sequence[pd.Index]
    sample_weights: np.ndarray | Sequence[np.ndarray] | None = None
    series_lengths: list[int] | None = None
    last_static_covariates_shape: tuple[int, int] | None = None

    def to_tuple(self):
        """Return the historical positional tuple used by the private wrappers."""
        return (
            self.X,
            self.y,
            self.times,
            self.last_static_covariates_shape,
            self.sample_weights,
        )
