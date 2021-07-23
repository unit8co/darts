import numpy as np
import xarray as xr
from typing import Callable, Union, Tuple
import copy

from .window import Window, CRWindow, NoWindow
from .cost_matrix import CostMatrix
from ...timeseries import TimeSeries
from ...logging import raise_if_not

SerieValues = Union[np.ndarray, np.float]
SeriesValue = Union[np.ndarray, np.float]
DistanceFunc = Callable[[SeriesValue, SeriesValue], float]


# CORE ALGORITHM
def _dtw_cost_matrix(x: SerieValues, y: SerieValues, dist: DistanceFunc, window: Window) -> np.ndarray:
    n = len(x)
    m = len(y)

    dtw = CostMatrix.from_window(window)

    dtw.fill(np.inf)
    dtw[0, 0] = 0

    for i, j in window:
        cost = dist(x[i-1], y[j-1])
        min_cost_prev = min(dtw[i - 1, j],
                               dtw[i, j - 1],
                               dtw[i - 1, j - 1])
        dtw[i, j] = cost + min_cost_prev

    return dtw


def _dtw_path(dtw: CostMatrix) -> np.ndarray:
    i = dtw.n
    j = dtw.m

    path = []

    while i > 0 or j > 0:
        path.append((i-1, j-1))

        stencil = [
            (i - 1, j - 1),  # diagonal
            (i - 1, j),  # left
            (i, j - 1),  # down
        ]

        costs = [dtw[i, j] for i, j in stencil]
        index_min = costs.index(min(costs))

        i, j = stencil[index_min]

    path.reverse()
    return np.array(path)


# MULTI-SCALE
def _downsample(high_res: np.ndarray):
    needs_padding = len(high_res) & 1
    if needs_padding:
        high_res = np.append(high_res, high_res[-1])

    low_res = np.reshape(high_res, (-1, 2))
    low_res = np.mean(low_res, axis=1)

    return low_res


def _expand_window(low_res_path: np.ndarray,
                   n: int,
                   m: int,
                   radius: int) -> Window:
    high_res_grid = CRWindow(n, m)

    def is_valid(cell):
        valid_x = 1 <= cell[0] <= n
        valid_y = 1 <= cell[1] <= m

        return valid_x and valid_y

    # up-sample path, 1 grid point computed at half-resolution becomes at least 4
    # (1,2) and (2,1) allows for better DWT solution, as it isn't diagonally constrained
    #
    # 4-sample         6-sample
    #     X X             X X
    #     X X    vs.    X X X
    # X X             X X X
    # X X             X X

    # TODO: optimize by using add_range instead of individually adding each element
    pattern = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]

    for i,j in low_res_path:
        for a, b in pattern:
            # +1 offset since x[0], y[0] maps to DTW[1][1]
            cell = (i*2 + 1 + a, j*2 + 1 + b)
            if is_valid(cell):
                high_res_grid.add(cell)

    # keeps track of path border
    outer = list(high_res_grid)
    next_outer = []
    for it in range(radius):
        for i, j in outer:
            # expand 1x1 to 3x3 block
            for b in range(-1, 2):
                cell = (i + a, j + b)
                if is_valid(cell) and not cell in high_res_grid:
                    high_res_grid[cell] = it
                    next_outer.append(cell)

        outer, next_outer = next_outer, outer
        next_outer.clear()

    return high_res_grid


def _fast_dtw(x: SerieValues,
              y: SerieValues,
              dist: DistanceFunc,
              radius: int,
              depth: int = 0) -> CostMatrix:
    n = len(x)
    m = len(y)
    min_size = radius+2

    if n < min_size or m < min_size or radius == -1:
        window = NoWindow()
        window.init_size(n, m)
        cost = _dtw_cost_matrix(x, y, dist, window)

        return cost

    half_x = _downsample(x)
    half_y = _downsample(y)

    low_res_cost = _fast_dtw(half_x, half_y, dist, radius, depth + 1)
    low_res_path = _dtw_path(low_res_cost)
    window = _expand_window(low_res_path, len(x), len(y), radius)
    cost = _dtw_cost_matrix(x, y, dist, window)

    return cost


# Public API Functions
class DTWAlignment:
    n: int
    m: int
    actual_series: TimeSeries
    pred_series: TimeSeries
    cost: CostMatrix

    def __init__(self,
                 actual_series: TimeSeries,
                 pred_series: TimeSeries,
                 cost: CostMatrix):

        self.n = len(actual_series)
        self.m = len(pred_series)
        self.actual_series = actual_series
        self.pred_series = pred_series
        self.cost = cost

    from ._plot import plot, plot_alignment

    def path(self):
        if hasattr(self, "_path"): return self._path
        self._path = _dtw_path(self.cost)
        return self._path

    def distance(self):
        return self.cost[(self.n, self.m)]

    def normalized_distance(self):
        if hasattr(self, "_normalized_distance"): return self._normalized_distance

        path = self.path()
        self._normalized_distance = self.distance() / len(path)
        return self._normalized_distance

    def warped(self, take_dates = True, unique_dates = False) -> (TimeSeries, TimeSeries):
        """

        that are warped such that the pairwise distance between them is minimized.
        This allows any other metric to be used to compare them.

        Dynamic Time Warping can be applied to determine how closely the shape of the signal matches,
        irrespective of phase or speed differences.

        Parameters
        ----------
        actual_series
        pred_series
        radius

        Returns
        -------
        Two new time series of the same length,
        """

        actual_series = self.actual_series
        pred_series = self.pred_series

        path = self.path().reshape((-1,))

        actual_series_xa = actual_series.data_array(copy=False)
        pred_series_xa = pred_series.data_array(copy=False)



        warped_actual_series_xa = actual_series_xa[path[0::2]]
        warped_pred_series_xa = pred_series_xa[path[1::2]]

        actual_time_dim = actual_series_xa.dims[0]
        pred_time_dim = pred_series_xa.dims[0]

        if unique_dates:
            warped_actual_series_xa[actual_time_dim] = pd.date_range(pd.Timestamp(2017, 1, 1, 12),
                                                                     periods=len(warped_actual_series_xa))

        if take_dates:
            warped_pred_series_xa[pred_time_dim] = warped_actual_series_xa[actual_time_dim]

        return TimeSeries.from_xarray(warped_actual_series_xa), TimeSeries.from_xarray(warped_pred_series_xa)


def default_distance_multi(x_values: np.ndarray, y_values: np.ndarray):
    return np.sum(np.abs(x_values - y_values))


def default_distance_uni(x_value: float, y_value: float):
    return abs(x_value - y_value)


def dtw(actual_series: TimeSeries,
        pred_series: TimeSeries,
        window: Window = NoWindow(),
        distance : Union[DistanceFunc, None] = None,
        multigrid_radius = -1
        ) -> [(int,int)]:
    """
    Generates a list of indices i,j that correspond to the smallest pairwise distance between the two time series,
    where i, j are monotonically increasing.

    Parameters
    ----------
    actual_series
    pred_series
    radius:
        Default radius of -1 results in an exact evaluation of the dynamic time warping algorithm,
        which runs in O(n^2) time. Exact evaluation will result in a performance warning on large datasets.

        Increasing radius trades solution accuracy for performance.

    Returns
    -------

    """

    both_univariate = actual_series.is_univariate and pred_series.is_univariate

    if distance is None:
        raise_if_not(actual_series.n_components == pred_series.n_components,
                     "Expected series to have same number of components, or to supply custom distance function")

        distance = default_distance_uni if both_univariate else default_distance_multi

    if both_univariate:
        values_x = actual_series.univariate_values(copy=False)
        values_y = pred_series.univariate_values(copy=False)
    else:
        values_x = actual_series.values(copy=False)
        values_y = pred_series.values(copy=False)

    window = copy.deepcopy(window)
    window.init_size(len(values_x), len(values_y))

    if multigrid_radius >= 0:
        raise_if_not(isinstance(window, NoWindow), "Multi-grid solver does not currently support windows")
        cost_matrix = _fast_dtw(values_x, values_y, distance, multigrid_radius)
    else:
        print(len(window))
        cost_matrix = _dtw_cost_matrix(values_x, values_y, distance, window)

    return DTWAlignment(actual_series, pred_series, cost_matrix)
