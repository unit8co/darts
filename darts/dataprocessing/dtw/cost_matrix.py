import array
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Tuple

import numpy as np

from .window import CRWindow, Window

Elem = Tuple[int, int]


class CostMatrix(ABC):
    """
    (n+1) x (m+1) Matrix
    Cell (i,j) corresponds to minimum total cost/distance of matching elements (i-1, j-1) in series1 and series2.

    Row 0 and column 0, are typically set to infinity, to prevent matching before the first element
    """

    n: int
    m: int

    @abstractmethod
    def fill(self, value: float):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        """
        Returns
        -------
        Dense n x m numpy array, where empty cells are set to np.inf
        """
        pass

    @staticmethod
    def _from_window(window: Window):
        """
        Creates a cost matrix from a window.
        Depending on the density of the active cells in the window,
        will select either a dense or sparse storage representation.

        Parameters
        ----------
        window
            Takes a `Window` defining which cells are active and which are empty

        Returns
        -------
        CostMatrix
        """
        density = len(window) / ((window.n + 1) * (window.m + 1))

        # In the future it might be worth implementing
        # a sparse cost matrix based on column_index/column_length
        # that will work even for non-contiguous windows
        if isinstance(window, CRWindow) and density < 0.5:
            return SparseCostMatrix(window)
        else:
            return DenseCostMatrix(window.n, window.m)


class DenseCostMatrix(np.ndarray, CostMatrix):
    def __new__(self, n, m):
        self.n = n
        self.m = m
        return super().__new__(self, (n + 1, m + 1), float)

    def to_dense(self) -> np.ndarray:
        return self[1:, 1:]

    def __iter__(self):
        for n in range(1, self.n):
            for m in range(1, self.m):
                yield n, m


class SparseCostMatrix(CostMatrix):
    def __init__(self, window: CRWindow):
        self.n = window.n
        self.m = window.m
        self.window = window
        self.offsets = np.empty(self.n + 2, dtype=int)
        self.column_ranges = window.column_ranges
        self.offsets[0] = 0
        np.cumsum(window.column_lengths(), out=self.offsets[1:])

        len = self.offsets[-1]

        self.offsets = array.array("i", self.offsets)
        self.dense = array.array("f", repeat(np.inf, len))

    def fill(self, value):
        if value != np.inf:  # should already be cleared to np.inf
            for i in range(len(self.dense)):
                self.dense[i] = value

    def to_dense(self) -> np.ndarray:
        matrix = np.empty((self.n, self.m))
        matrix.fill(np.inf)

        if isinstance(self.window, CRWindow):
            ranges = self.window.column_ranges
            lengths = self.window.column_lengths()

            # TODO express only in terms of numpy operations
            for i in range(1, self.n + 1):
                start = ranges[i * 2 + 0] - 1
                end = ranges[i * 2 + 1] - 1
                len = lengths[i]
                offset = self.offsets[i]

                matrix[i - 1][start:end] = self.dense[offset : offset + len]
        else:
            for i in range(1, self.n + 1):
                column_start = self.offsets[i]

                for j in range(1, self.m + 1):
                    column_idx = self.window.column_index(i)
                    if column_idx == -1:
                        continue

                    matrix[i - 1, j - 1] = self.dense[column_start + column_idx]

        return matrix

    # PERFORMANCE: hot functions, avoid calling functions/excessive checking
    def __getitem__(self, elem: Elem):
        i, j = elem

        start = self.column_ranges[i * 2 + 0]
        end = self.column_ranges[i * 2 + 1]
        if start <= j < end:
            return self.dense[self.offsets[i] + j - start]
        return np.inf

    def __setitem__(self, elem, value):
        i, j = elem

        start = self.column_ranges[i * 2 + 0]

        self.dense[self.offsets[i] + j - start] = value

    def __iter__(self):
        return self.window.__iter__()
