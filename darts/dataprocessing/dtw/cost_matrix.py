from typing import Tuple, Dict
import numpy as np

from .window import Window, CRWindow
from abc import abstractmethod
import array
from itertools import repeat

Elem = Tuple[int, int]

class CostMatrix:
    n: int
    m: int

    @abstractmethod
    def fill(self, value: np.float): pass

    @abstractmethod
    def __getitem__(self, item): pass

    @abstractmethod
    def __setitem__(self, key, value): pass

    @abstractmethod
    def __iter__(self): pass

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        pass

    @staticmethod
    def from_window(window: Window):
        density = len(window) / ((window.n+1)*(window.m+1))

        if isinstance(window, CRWindow) and density < 0.5:
            return SparseCostMatrix(window)
        else:
            return DenseCostMatrix(window.n, window.m)


class DenseCostMatrix(np.ndarray, CostMatrix):
    def __new__(self, n, m):
        self.n = n
        self.m = m
        return super().__new__(self, (n+1, m+1), np.float)

    def to_dense(self) -> np.ndarray:
        return self


class SparseCostMatrix(CostMatrix):
    def __init__(self, window: CRWindow):
        self.n = window.n
        self.m = window.m
        self.shape = (self.n+1, self.m+1)
        self.window = window
        self.offsets = np.empty(self.n+2, dtype=int)
        self.column_ranges = window.column_ranges
        self.offsets[0] = 0
        np.cumsum(window.column_lengths(), out=self.offsets[1:])

        len = self.offsets[-1]

        self.offsets = array.array('i', self.offsets)
        self.dense = array.array('f', repeat(np.inf, len))

    def fill(self, value):
        if value != np.inf: # should already be cleared to np.inf
            for i in range(len(self.dense)):
                self.dense[i] = value

    def to_dense(self) -> np.ndarray:
        matrix = np.empty((self.n+1, self.m+1))
        matrix.fill(np.inf)

        if isinstance(self.window, CRWindow):
            ranges = self.window.column_ranges
            lengths = self.window.column_lengths()

            # TODO express only in terms of numpy operations
            for i in range(0, self.n+1):
                start = self.window.column_ranges[i*2 + 0]
                end = self.window.column_ranges[i*2 + 1]
                len = lengths[i]
                offset = self.offsets[i]

                matrix[i][start:end] = self.dense[offset:offset + len]
        else:
            for i in range(0, self.n+1):
                column_start = self.offsets[i]

                for j in range(0, self.m+1):
                    column_idx = self.window.column_index(i)
                    if column_idx == -1: continue

                    matrix[i,j] = self.dense[column_start + column_idx]

        return matrix

    # PERFORMANCE: hot functions, avoid calling functions
    def __getitem__(self, elem: Elem):
        i, j = elem

        start = self.column_ranges[i * 2 + 0]
        end = self.column_ranges[i * 2 + 1]
        if start <= j < end: return self.dense[self.offsets[i] + j - start]
        return np.inf

    def __setitem__(self, elem, value):
        i, j = elem

        start = self.column_ranges[i * 2 + 0]
        end = self.column_ranges[i * 2 + 1]
        #assert(start <= j < end)

        self.dense[self.offsets[i] + j - start] = value

    def __iter__(self):
        return self.window.__iter__()