import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import transforms

from ...logging import raise_if_not


def plot(self, show_series : bool = False, show_path : bool = True, show_cost : bool = True):
    """
    Parameters
    ----------
    show_series
    show_path
    show_cost

    Returns
    -------

    """

    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)

    #ax1 = fig.add_subplot(gs[0, 0])
    #ax2 = fig.add_subplot(gs[0, 1])
    #ax3 = fig.add_subplot(gs[1, :])

    warp = fig.add_subplot(gs[0:2,1:3])
    left = fig.add_subplot(gs[0:2,0])
    bottom = fig.add_subplot(gs[2, 1:3])

    warp.title.set_text("Cost Matrix")
    warp.set_aspect("auto")
    warp.set_xlim([0, self.n])
    warp.set_ylim([0, self.m])

    if show_cost:
        cost_matrix = self.cost.to_dense()
        cost_matrix = np.transpose(cost_matrix[1:, 1:])
        warp.imshow(cost_matrix, cmap="summer", interpolation='nearest', origin="lower")

    if show_path:
        path = self.path()
        path = path.reshape((-1,))

        i_coords = path[::2]
        j_coords = path[1::2]
        warp.plot(i_coords, j_coords)

    self.pred_series.plot(ax= left, y=self.pred_series._time_dim)
    self.actual_series.plot(ax = bottom)

    bottom.set_aspect('auto')
    left.set_aspect('auto')


def plot_alignment(self, actual_series_y_offset = 0, pred_series_y_offset = 0):
    """
    Parameters
    ----------
    self
    diff

    Returns
    -------

    """

    time_series1 = self.actual_series
    time_series2 = self.pred_series

    raise_if_not(time_series1.is_univariate and time_series2.is_univariate, "plot_alignment only supports univariate TimeSeries")

    time_series1 += actual_series_y_offset
    time_series2 += pred_series_y_offset

    series1 = time_series1.pd_series()
    series2 = time_series2.pd_series()

    path = self.path()
    n = len(path)

    path = path.reshape((2 * len(path),))

    x_coords1 = np.array(series1.index, dtype="datetime64[s]")[path[::2]]
    x_coords2 = np.array(series2.index, dtype="datetime64[s]")[path[1::2]]
    y_coords1 = np.array(series1, dtype=np.float)[path[::2]]
    y_coords2 = np.array(series2, dtype=np.float)[path[1::2]]

    x_coords = np.empty(n * 3, dtype="datetime64[s]")
    y_coords = np.empty(n * 3, dtype=np.float)

    x_coords[0::3] = x_coords1
    x_coords[1::3] = x_coords2
    x_coords[2::3] = np.nan

    y_coords[0::3] = y_coords1
    y_coords[1::3] = y_coords2
    y_coords[2::3] = np.nan

    arr = xr.DataArray(y_coords, dims=["value"], coords={"value": x_coords})
    xr.plot.line(arr, x="value")

    time_series1.plot()
    time_series2.plot()