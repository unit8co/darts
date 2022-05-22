import array
from abc import ABC, abstractmethod
from math import atan, tan

import numpy as np

from darts.logging import raise_if, raise_if_not


class Window(ABC):
    n: int
    m: int

    def init_size(self, n: int, m: int):
        """
        Called by dtw to initialize the window to a certain size.

        Parameters
        ----------
        n
            The width of the window, must be equal to the length of series1
        m
            The height of the window, must be equal to the length of series2
        """
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
        elem
            (i,j) index, where i indexes columns and j rows

        Returns
        -------
        int
            The number of active grid cells before row element j, in column i,
            If (i,j) is not an active grid cell returns -1
        """

    def __contains__(self, item):
        return self.column_index(item) != -1

    @abstractmethod
    def column_length(self, column: int) -> int:
        """
        Parameters
        ----------
        column
            A column in the window, must be within 0 < column < n+1

        Returns
        -------
        int
            The number of active grid cells in a column.
        """

    def column_lengths(self) -> np.ndarray:
        """
        Parameters
        ----------
        self

        Returns
        -------
        np.ndarray of shape (n+1,)
            Containing The number of active grid cells in each column.
        """

        return np.array(self.column_length(i) for i in range(0, self.n + 1))

    @abstractmethod
    def __iter__(self):
        """
        Returns
        -------
        Iterator
            Iterate over all active cells in the window, yielding (i,j) tuple.
            Expected to start with index (1,1) and end with index (n+1, m+1).
        """
        pass


class NoWindow(Window):
    """
    Window covers the entire grid,
    meaning every possible alignment between series1 and series2 is considered.
    """

    def __len__(self):
        return self.n * self.m + 1  # include (0,0) element

    def column_index(self, elem: (int, int)):
        return elem[1] - 1

    def column_length(self, column: int) -> int:
        return self.m

    def column_lengths(self) -> np.ndarray:
        result = np.empty(self.n + 1)
        result.fill(self.m)
        result[0] = 1

    def __iter__(self):
        for i in range(1, self.n + 1):
            for j in range(1, self.m + 1):
                yield i, j


def gtz(value):  # greater than zero
    return value if value > 0 else 0


class CRWindow:
    """
    Compressed row representation window.
    Stores the range of active grid cells in each column.
    Any window with contiguous columns can be expressed as an CRWindow.
    Supports efficient iterative construction and updates.
    """

    length: int
    column_ranges: array.array

    def __init__(self, n: int, m: int, ranges: np.ndarray = None):
        """
        Parameters
        ----------
        n
            The width of the window, must be equal to the length of series1
        m
            The height of the window, must be equal to the length of series2
        ranges
            Ranges of active cells within a column [[start_column0, end_column0], ...]
            with shape (n, 2) and where start >= 0 and end <= m.
        """

        self.n = n
        self.m = m

        if ranges is not None:
            raise_if_not(
                ranges.shape == (n, 2),
                f"Expects a 2d array with [start, end] for each column and shape = ({n}, 2)",
            )

            ranges = np.insert(ranges, 0, [0, 1], axis=0)
            start = ranges[:, 0]
            end = ranges[:, 1]

            raise_if(np.any(start < 0), "Start must be >=0")
            raise_if(np.any(end > m), "End must be <m")

            diff = np.maximum(end - start, 0)
            self.length = np.sum(diff)

            ranges[1:] += 1
            ranges = ranges.flatten()
        else:
            ranges = np.zeros((n + 1) * 2, dtype=int)
            ranges[0::2] = self.m  # start
            ranges[1::2] = 0  # end
            ranges = array.array("i", ranges)

            ranges[0] = 0
            ranges[1] = 1
            self.length = 1

        self.column_ranges = array.array("i", ranges)

    def add_range(self, column: int, start: int, end: int):
        """
        Extends the active cells in the column by the range (start,end).
        Ranges smaller than the current one are ignored.
        Note (1, m+1), not (0,m) corresponds to an entire column.

        Parameters
        ----------
        column
            Column int index
        start
            Row element int index where start >= 1 and start <= end
        end:
            Row element int index where end >= 1 and end <= m+1
        """

        if start < 1 or start > self.m:
            raise IndexError(f"Start must be >=1 and <=m, got {start}")
        if end < 1 or end > self.m + 1:
            raise IndexError(f"End must be >=1 and <=m+1, got {end}")

        start_idx = column * 2 + 0
        end_idx = column * 2 + 1

        orig_start = self.column_ranges[start_idx]
        orig_end = self.column_ranges[end_idx]

        start, end = min(orig_start, start), max(orig_end, end)

        orig_row_length = gtz(orig_end - orig_start)
        row_length = gtz(end - start)

        self.length += row_length - orig_row_length
        self.column_ranges[start_idx] = start
        self.column_ranges[end_idx] = end

    def add(self, elem: (int, int)):
        """
        Mark grid cell as active.

        Parameters
        ----------
        elem
            Tuple of grid cell index (column, row)
        """

        self.add_range(elem[0], elem[1], elem[1] + 1)

    def column_length(self, column: int) -> int:
        start, end = self.column_ranges[column]
        return gtz(end - start)

    def column_index(self, elem: (int, int)) -> int:
        i, j = elem

        start, end = self.column_ranges[i]
        if j < start or j >= end:
            return -1
        else:
            return j - start

    def __contains__(self, elem: (int, int)) -> bool:
        i, j = elem
        start, end = self.column_ranges[i]
        return start <= j < end

    def __iter__(self):
        for i in range(1, self.n + 1):
            start = self.column_ranges[i * 2 + 0]
            end = self.column_ranges[i * 2 + 1]

            for j in range(start, end):
                yield i, j

    def column_lengths(self) -> np.ndarray:
        ranges = self.column_ranges
        start = ranges[::2]
        end = ranges[1::2]

        return np.maximum(np.subtract(end, start), 0)

    def __len__(self):
        return self.length


class Itakura(CRWindow):
    """
    Forms the Itakura parallelogram, where max_slope determines the slope of the steeper side.

                                         x
                                     xxxx
                      B           xxxxxx
                            xxxxxxxxxxx
                        xxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxxx
                 xxxxxxxxxxxxxxxxxxxx     C
               xxxxxxxxxxxxxxxxxxxxx
              xxxxxxxxxxxxxxxxxxxxx
             xxxxxxxxxxxxxxxxxxxxx
        A   xxxxxxxxxxxxxxxxxxxxx
           xxxxxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxxxxx
         xxxxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxx       D
        xxxxxxxxxxxx
       xxxxxxxxx
      xxxxxx
     xxx
    x
    """

    def __init__(self, max_slope: float):
        self.max_slope = max_slope

    def init_size(self, n: int, m: int):
        self.n = n
        self.m = m

        max_slope = self.max_slope
        diagonal_slope = m / n  # rise over run
        raise_if_not(
            max_slope > diagonal_slope,
            f"Itakura slope {max_slope} must be greater than {diagonal_slope} to form valid parallelogram.",
        )

        max_slope_angle = atan(max_slope)
        diagonal_slope_angle = atan(diagonal_slope)

        diff_slope_angle = max_slope_angle - diagonal_slope_angle
        min_slope = tan(diagonal_slope_angle - diff_slope_angle)

        # Derivation for determining how wide the steep top sides (A) and shallow bottom (D) are

        # max_slope*x + (n-x)*min_slope = m
        # max_slope*x + n*min_slope - min_slope*x = m
        # (max_slope - min_slope)*x = m - n*min_slope
        # x = (m - n*min_slope) / (max_slope - min_slope)

        ranges = np.zeros((self.n, 2), dtype=float)

        shallow_bottom = int(
            np.round((m - n * max_slope) / (min_slope - max_slope)) + 1
        )
        ranges[:shallow_bottom, 0] = np.arange(shallow_bottom)
        ranges[shallow_bottom:, 0] = np.arange(n - shallow_bottom) + 1

        ranges[:shallow_bottom, 0] *= min_slope
        ranges[shallow_bottom:, 0] *= max_slope
        ranges[shallow_bottom:, 0] += ranges[shallow_bottom - 1, 0]

        steep_top = int(np.round((m - n * min_slope) / (max_slope - min_slope)))
        ranges[:steep_top, 1] = np.arange(steep_top) + 1
        ranges[steep_top:, 1] = np.arange(n - steep_top) + 1

        ranges[:steep_top:, 1] *= max_slope
        ranges[steep_top:, 1] *= min_slope
        ranges[steep_top:, 1] += ranges[steep_top - 1, 1]

        np.floor(ranges[:, 0], out=ranges[:, 0])
        np.ceil(ranges[:, 1], out=ranges[:, 1])

        ranges = np.maximum([0, 1], ranges)
        ranges = np.minimum([self.m - 1, self.m], ranges)
        ranges = ranges.astype(int)
        ranges[0][0] = 0

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

        diff = abs(n - m)
        raise_if_not(
            diff < self.window_size,
            f"Window size must at least cover size difference ({diff})",
        )

        ranges = np.repeat(np.arange(n), 2)
        ranges[0::2] -= (self.window_size,)
        ranges[1::2] += self.window_size

        ranges[0::2] = np.maximum(0, ranges[0::2])
        ranges[1::2] = np.minimum(self.m, ranges[1::2] + 1)
        ranges = np.reshape(ranges, (-1, 2))

        super().__init__(n, m, ranges)
