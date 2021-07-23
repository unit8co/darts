from typing import Iterable, Tuple
from dataclasses import dataclass
import numpy as np
from ...logging import raise_if_not
from abc import abstractmethod


class Window:
    n: int
    m: int

    def init_size(self, n: int, m: int):
        self.n = n
        self.m = m

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def column_index(self, elem: (int, int)) -> int:
        """
        Parameters
        ----------
        self
        elem: (i,j) index

        Returns
        -------
        The number of active grid cells before row element j, in column i,
        If (i,j) is not an active grid cell returns -1
        """

    def __contains__(self, item):
        return self.column_index(item) != -1

    def raise_if_index_error(self, elem: (int,int), names: Tuple[str,str] = ("i","j")):
        i, j = elem

        if i == 0 and j == 0: return
        if i < 1 or i > self.n: raise IndexError(f"{names[0]} >= 1 and {names[0]} <= n, got {i}")
        if j < 1 or j > self.m: raise IndexError(f"{names[1]} >= 1 and {names[1]} <= n, got {j}")

    @abstractmethod
    def column_length(self, column: int) -> int:
        """
        Parameters
        ----------
        self
        column: int

        Returns
        -------
        The number of active grid cells in a column.
        """

    def column_lengths(self) -> np.ndarray:
        """
        Parameters
        ----------
        self

        Returns
        -------
        The number of active grid cells in each column.
        """

        return [self.column_length(i) for i in range(0, self.n + 1)]

    @abstractmethod
    def __iter__(self):
        pass


class NoWindow(Window):
    def __len__(self):
        return self.n*self.m + 1 # include (0,0) element

    def column_index(self, elem: (int, int)):
        return elem[1]-1

    def column_length(self, column: int) -> int:
        return self.m

    def column_lengths(self) -> np.ndarray:
        result = np.empty((self.n+1))
        result.fill(self.m)
        result[0] = 1

    def __iter__(self):
        for i in range(1, self.n+1):
            for j in range(1, self.m+1):
                yield i, j

def gtz(value): # greater than zero
    return value if value > 0 else 0

class CRWindow:
    """
    Arbitrary contiguous column windows
    """

    length: int
    column_ranges: np.ndarray #2d array [start,end] for each column

    def __init__(self, n: int, m: int, ranges: np.ndarray = None):
        self.n = n
        self.m = n

        if not ranges is None:
            raise_if_not(ranges.shape == (n, 2), f"Expects a 2d array with [start, end] for each column and shape = ({n}, 2)")

            ranges = np.insert(ranges, 0, [0,1], axis=0)
            ranges[1:] += 1

            flatten = ranges.reshape((-1))
            start = flatten[0::2]
            end = flatten[1::2]

            diff = np.maximum(end - start, 0)
            self.length = np.sum(diff)
            self.column_ranges = ranges
        else:
            self.column_ranges = np.zeros((n + 1) * 2, dtype=int)
            self.column_ranges[0::2] = self.m #start
            self.column_ranges[1::2] = 0 #end
            self.column_ranges = np.reshape(self.column_ranges, (-1,2))
            self.column_ranges[0] = (0, 1)
            self.length = 1

    def add_range(self, column: int, start: int, end: int):
        if start < 1 or start > self.n: raise IndexError(f"Start must be >=1 and <=n, got {start}")
        if end < 1 or end > self.n+1: raise IndexError(f"End must be >=1 and <=n+1, got {end}")

        orig_start, orig_end = self.column_ranges[column]
        start, end = min(orig_start, start), max(orig_end, end)

        orig_row_length = gtz(orig_end - orig_start)
        row_length = gtz(end - start)

        self.length += row_length - orig_row_length
        self.column_ranges[column] = (start, end)

    def add(self, elem: (int, int)):
        self.add_range(elem[0], elem[1], elem[1]+1)


    def column_length(self, column: int) -> int:
        start, end = self.column_ranges[column]
        return gtz(end - start)

    def column_index(self, elem: (int, int)) -> int:
        i, j = elem

        start,end = self.column_ranges[i]
        if j < start or j >= end: return -1
        else: return j - start

    def __iter__(self):
        for i in range(1, self.n+1):
            start, end = self.column_ranges[i]
            for j in range(start, end):
                yield i, j

    def column_lengths(self) -> np.ndarray:
        ranges = self.column_ranges.reshape((-1,))
        start = ranges[::2]
        end = ranges[1::2]

        return np.maximum(end-start, 0)

    def __len__(self):
        return self.length


class Itakura(CRWindow):
    """
    Forms the itakura parallelogram, where max_slope determines the slope of the steeper side.
    """

    def __init__(self, max_slope: int):
        self.max_slope = max_slope

    def init_size(self, n: int, m: int):
        self.n = n
        self.m = m

        max_slope = self.max_slope
        diagonal_slope = m/n #rise over run
        raise_if_not(max_slope > diagonal_slope, f"Itakura slope {max_slope} must be greater than {diagonal_slope} to form valid parallelogram.")

        diff_slope = max_slope - diagonal_slope
        min_slope = diagonal_slope - diff_slope

        first_half = self.n//2
        second_half = self.n - first_half

        ranges = np.zeros(self.n*2, dtype=float)
        ranges[:first_half*2] = np.repeat(np.arange(first_half), 2)
        ranges[first_half*2:] = np.repeat(np.arange(second_half), 2)
        ranges = ranges.reshape((-1, 2))

        ranges[:first_half] *= (min_slope, max_slope)
        ranges[first_half:] *= (max_slope, min_slope)
        ranges[first_half:] += ranges[first_half-1]
        ranges += [0,1]

        ranges = ranges.reshape((-1,))
        ranges[0::2] = np.maximum(0, ranges[::2])
        ranges[1::2] = np.minimum(self.m, ranges[1::2])
        ranges = ranges.reshape((-1, 2))
        ranges = ranges.astype(int)

        super().__init__(n, m, ranges)

class SakoeChiba(CRWindow):
    """
    Forms a diagonal window where window_size controls the maximum allowed shift between the two series.
    If both time-series have the same time axis, window_size corresponds to the maximum number of time periods
    """
    def __init__(self, window_size: int):
        self.window_size = window_size

    def init_size(self, n: int, m: int):
        self.n = n
        self.m = m

        diff = abs(n-m)
        raise_if_not(diff < self.window_size, f"Window size must at least cover size difference (diff)")

        ranges = np.repeat(np.arange(n), 2)
        ranges[0::2] -= self.window_size,
        ranges[1::2] += self.window_size

        ranges[0::2] = np.maximum(0, ranges[0::2])
        ranges[1::2] = np.minimum(self.m, ranges[1::2]+1)
        ranges = np.reshape(ranges, (-1, 2))

        super().__init__(n, m, ranges)



"""
    def column_range(self, i: int) -> (int,int):
        column = i-1

        min_slope = self.min_slope
        max_slope = self.max_slope

        half_way = column*2>self.n

        if half_way:
            half_column = self.n//2 if half_way else 0
            rem_column = column - half_column
        else:
            half_column = column
            rem_column = 0

        start = half_column * min_slope + rem_column * max_slope
        end = half_column * max_slope + rem_column * min_slope

        start = int(max(1, np.floor(start)+1))
        end = int(min(self.m+1, np.floor(end)+2))
        return start, end
"""