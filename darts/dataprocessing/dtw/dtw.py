import numpy as np
import xarray as xr
import pandas as pd
from typing import Callable, Union, Tuple
import copy

from .window import Window, CRWindow, NoWindow
from .cost_matrix import CostMatrix
from ...timeseries import TimeSeries
from ...logging import get_logger, raise_if_not, raise_if

logger = get_logger(__name__)

SeriesValue = Union[np.ndarray, np.float]
DistanceFunc = Callable[[SeriesValue, SeriesValue], float]


# CORE ALGORITHM
def _dtw_cost_matrix(x: np.ndarray, y: np.ndarray, dist: DistanceFunc, window: Window) -> np.ndarray:
    n = len(x)
    m = len(y)

    dtw = CostMatrix.from_window(window)

    dtw.fill(np.inf)
    dtw[0, 0] = 0

    for i, j in window:
        cost = dist(x[i - 1], y[j - 1])
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
        path.append((i - 1, j - 1))

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
def _down_sample(high_res: np.ndarray):
    needs_padding = len(high_res) & 1
    if needs_padding:
        high_res = np.append(high_res, high_res[-1])

    low_res = np.reshape(high_res, (-1, 2))
    low_res = np.mean(low_res, axis=1)

    return low_res


def _expand_window(low_res_path: np.ndarray,
                   n: int,
                   m: int,
                   radius: int) -> CRWindow:
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

    pattern = [(0, 0, 2), (1, 0, 3), (2, 1, 2)]

    for i, j in low_res_path:
        for column, start, end in pattern:
            # +1 offset since x[0], y[0] maps to DTW[1][1]

            column += i * 2 + 1

            # ensure no out of bounds
            # enlarge vertically by radius is O(1), due to compressed row representation
            start = max(1, min(m + 1, start + j * 2 - radius))
            end = max(1, min(m + 1, end + j * 2 + 1 + radius))

            # to create an nxn block, add (0-n) range n times
            # ensure no out of bounds
            #   if column is too large m-column -> 0
            #   if column-k is too small, column < radius+1
            for k in range(0, min(radius + 1, column, n - column + 1)):
                high_res_grid.add_range(column - k, start, end)

    return high_res_grid


def _fast_dtw(x: np.ndarray,
              y: np.ndarray,
              dist: DistanceFunc,
              radius: int,
              depth: int = 0) -> CostMatrix:
    n = len(x)
    m = len(y)
    min_size = radius + 2

    if n < min_size or m < min_size or radius == -1:
        window = NoWindow()
        window.init_size(n, m)
        cost = _dtw_cost_matrix(x, y, dist, window)

        return cost

    half_x = _down_sample(x)
    half_y = _down_sample(y)

    low_res_cost = _fast_dtw(half_x, half_y, dist, radius, depth + 1)
    low_res_path = _dtw_path(low_res_cost)
    window = _expand_window(low_res_path, len(x), len(y), radius)
    cost = _dtw_cost_matrix(x, y, dist, window)

    return cost


# Public API Functions
class DTWAlignment:
    n: int
    m: int
    series1: TimeSeries
    series2: TimeSeries
    cost: CostMatrix

    def __init__(self,
                 series1: TimeSeries,
                 series2: TimeSeries,
                 cost: CostMatrix):

        self.n = len(series1)
        self.m = len(series2)
        self.series1 = series1
        self.series2 = series2
        self.cost = cost

    from ._plot import plot, plot_alignment

    def path(self) -> np.ndarray:
        """
        Returns
        -------
        An array of indices [(i0,j0), ...], corresponding to the alignment between series1 and series2 respectively.
        Indices are in monotonic order, path[n] >= path[n-1]
        """

        if hasattr(self, "_path"): return self._path
        self._path = _dtw_path(self.cost)
        return self._path

    def distance(self):
        """
        Returns
        -------
        The total distance between pair-wise elements in the two series after warping.
        """
        return self.cost[(self.n, self.m)]

    def mean_distance(self):
        """
        Returns
        -------
        The mean distance between pair-wise elements in the two series after warping.
        """
        if hasattr(self, "_normalized_distance"): return self._normalized_distance

        path = self.path()
        self._normalized_distance = self.distance() / len(path)
        return self._normalized_distance

    def warped(self, take_dates: Union[None,bool] = None, range_index: Union[None,bool] =None) -> (TimeSeries, TimeSeries):
        """
        Warps the two time series according to the alignment that minimizes the pair-wise distance.

        Defaults to indexing the two warped time series with a range index.
        This will bring two time series that are out-of-phase back into phase.

        If neither take_dates or range_index are enabled,
        the warped series will have the same time-index as the input series.
        In which case only comparing the values of the two warped series directly makes sense,
        as they have different time axes.

        Parameters
        ----------
        take_dates
            Boolean indicating whether to apply time index from series1 to the warped series.
            Requires DatetimeIndex time index for series1.

        range_index
            Boolean indicating whether to assign a pd.RangeIndex to the warped series.

        Returns
        -------
        Two new time series of the same length, which may contained repeated values.
        """

        series1 = self.series1
        series2 = self.series2

        raise_if(take_dates and range_index, "take_dates and range_index are mutually exclusive, set only one to true")
        raise_if(take_dates and isinstance(self.series1.time_index, pd.RangeIndex), "take_dates requires datetime index for series1")

        if take_dates is None and range_index is None:
            range_index = True

        path = self.path().reshape((-1,))

        xa1 = series1.data_array(copy=False)
        xa2 = series2.data_array(copy=False)

        warped_actual_series_xa = xa1[path[0::2]]
        warped_pred_series_xa = xa2[path[1::2]]

        actual_time_dim = warped_actual_series_xa.dims[0]
        pred_time_dim = warped_pred_series_xa.dims[0]

        if range_index:
            warped_actual_series_xa = warped_actual_series_xa.reset_index(dims_or_levels=actual_time_dim)
            warped_pred_series_xa = warped_pred_series_xa.reset_index(dims_or_levels=pred_time_dim)

        elif take_dates:
            time_index = warped_actual_series_xa[actual_time_dim]
            time_index = time_index.rename({actual_time_dim: pred_time_dim})
            warped_pred_series_xa[pred_time_dim] = time_index

        return TimeSeries.from_xarray(warped_actual_series_xa), TimeSeries.from_xarray(warped_pred_series_xa)


def default_distance_multi(x_values: np.ndarray, y_values: np.ndarray):
    return np.sum(np.abs(x_values - y_values))


def default_distance_uni(x_value: float, y_value: float):
    return abs(x_value - y_value)


def dtw(series1: TimeSeries,
        series2: TimeSeries,
        window: Window = NoWindow(),
        distance: Union[DistanceFunc, None] = None,
        multi_grid_radius: int = -1
        ) -> [(int, int)]:
    """
    Determines the optimal alignment between two time series series1 and series2, according to the Dynamic Time Warping algorithm.
    The alignment minimizes the distance between pair-wise elements after warping. All elements in the two series are matched
    and are in strictly monotonically increasing order. Considers only the values in the series, ignoring the time axis.

    Dynamic Time Warping can be applied to determine how closely two time series correspond,
    irrespective of phase, length or speed differences.

    Parameters
    ----------
    series1:
        `TimeSeries`

    series2:
        `TimeSeries`

    window:
        Used to constrain the search for the optimal alignment: see SakoeChiba and Itakura.
        Default considers all possible alignments.

    distance:
        Function taking as input either two `floats` for univariate series or two `np.ndarray`,
        and returning the distance between them.

        Defaults to the abs difference for univariate-data and the
        sum of the abs difference for multi-variate series.

    multi_grid_radius: int
        Default radius of -1 results in an exact evaluation of the dynamic time warping algorithm.
        Without constraints DTW runs in O(nxm) time where n,m are the size of the series. Exact evaluation with no constraints,
        will result in a performance warning on large datasets.

        Setting multigrid_radius to a value other than -1, will enable the approximate multi-grid solver,
        which executes in linear time, vs quadratic time for exact evaluation.
        Increasing radius trades solution accuracy for performance.

    Returns
    -------
    `DTWAlignment`
    """

    if multi_grid_radius == -1 and type(window) is NoWindow and len(series1) * len(series2) > 10 ** 6:
        logger.warn("Exact evaluation will result in poor performance on large datasets."
                    " Consider enabling multi-grid or using a window.")

    both_univariate = series1.is_univariate and series2.is_univariate

    if distance is None:
        raise_if_not(series1.n_components == series2.n_components,
                     "Expected series to have same number of components, or to supply custom distance function", logger)

        distance = default_distance_uni if both_univariate else default_distance_multi

    if both_univariate:
        values_x = series1.univariate_values(copy=False)
        values_y = series2.univariate_values(copy=False)
    else:
        values_x = series1.values(copy=False)
        values_y = series2.values(copy=False)

    raise_if(np.any(np.isnan(values_x)), "Dynamic Time Warping does not support nan values. "
                                         "You can use the module darts.utils.missing_values to fill them, "
                                         "before passing them to dtw.", logger)
    raise_if(np.any(np.isnan(values_y)), "Dynamic Time Warping does not support nan values. "
                                         "You can use the module darts.utils.missing_values to fill them,"
                                         "before passing it into dtw", logger)

    window = copy.deepcopy(window)
    window.init_size(len(values_x), len(values_y))

    raise_if(multi_grid_radius < -1, "Expected multi-grid radius to be positive or -1")

    if multi_grid_radius >= 0:
        raise_if_not(isinstance(window, NoWindow), "Multi-grid solver does not currently support windows", logger)
        cost_matrix = _fast_dtw(values_x, values_y, distance, multi_grid_radius)
    else:
        cost_matrix = _dtw_cost_matrix(values_x, values_y, distance, window)

    return DTWAlignment(series1, series2, cost_matrix)
