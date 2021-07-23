from typing import Tuple, Dict
import numpy as np

from .window import Window, CRWindow
from abc import abstractmethod

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

        if density < 0.5:
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
    def __init__(self, window: Window):
        self.n = window.n
        self.m = window.m
        self.shape = (self.n+1, self.m+1)
        self.window = window
        self.offsets = np.empty(self.n+2, dtype=int)
        self.offsets[0] = 0
        self.offsets[1:] = np.cumsum(window.column_lengths())
        self.dense = np.empty(self.offsets[-1])

    def fill(self, value):
        self.dense.fill(-1)

    def to_dense(self) -> np.ndarray:
        matrix = np.empty((self.n+1, self.m+1))
        matrix.fill(np.inf)

        if isinstance(self.window, CRWindow):
            ranges = self.window.column_ranges
            lengths = self.window.column_lengths()

            # TODO express only in terms of numpy operations
            for i in range(0, self.n+1):
                start, end = self.window.column_ranges[i]
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

    def index(self, key: Elem):
        column_index = self.window.column_index(key)
        if column_index == -1: return -1
        return self.offsets[key[0]] + column_index

    def __getitem__(self, key: Elem):
        idx = self.index(key)
        return self.dense[idx] if idx != -1 else np.inf

    def __setitem__(self, key, value):
        idx = self.index(key)
        assert(idx != -1)
        self.dense[idx] = value

    def __iter__(self):
        return self.window.__iter__()