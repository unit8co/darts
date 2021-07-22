from typing import Union, Tuple

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import transforms

from ...logging import raise_if_not


def plot(self, show_series: bool = False, show_cost: bool = True):
    """
    Plot the warp path.

    Parameters
    ----------
    show_series:
        Boolean indicating whether to plot the two time series.
        Series1 will be plotted below the cost matrix and series2 to the left of the cost matrix.

    show_cost:
        Boolean indicating whether to plot the cost matrix.
        Useful for visualizing the effect of the window function.

    Returns
    -------
    """

    if show_series:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[0.6, 0.4])

        warp = fig.add_subplot(gs[0, 1])
        left = fig.add_subplot(gs[0, 0])
        bottom = fig.add_subplot(gs[1, 1])

    else:
        fig, warp = plt.subplots()

    warp.title.set_text("Cost Matrix")
    warp.set_xlim([0, self.n])
    warp.set_ylim([0, self.m])
    warp.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if show_cost:
        cost_matrix = self.cost.to_dense()
        cost_matrix = np.transpose(cost_matrix[1:, 1:])
        warp.imshow(cost_matrix, cmap="summer", interpolation='none', origin="lower", extent=[0, self.n, 0, self.m])

    show_path = True
    if show_path:
        path = self.path()
        path = path.reshape((-1,))

        i_coords = path[::2] + 0.5
        j_coords = path[1::2] + 0.5
        warp.plot(i_coords, j_coords)

    if show_series:
        self.series1.plot(ax=bottom)
        self.series2.plot(ax=left, y=self.series2._time_dim)

        bottom.set_xlim([self.series1.start_time(), self.series1.end_time()])
        left.set_ylim([self.series2.start_time(), self.series2.end_time()])

        bottom.legend()
        bottom.set_title('')
        left.legend()
        left.set_title('')

    warp.set_aspect('auto')


def plot_alignment(self, series1_y_offset=0, series2_y_offset=0, components: Union[Tuple[Union[str, int], Union[str, int]]] = (0,0)):
    """
    Plots the uni-variate component of each series,
    with lines between them indicating the alignment selected by the DTW algorithm.

    Parameters
    ----------
    series2_y_offset
        Offset series1 vertically for ease of viewing.

    series1_y_offset
        Offset series2 vertically for ease of viewing.

    components
        Tuple of component index for series1 and component index for series2.

    Returns
    -------
    """

    series1 = self.series1
    series2 = self.series2

    (component1, component2) = components

    if not series1.is_univariate: series1 = series1.univariate_component(component1)
    if not series2.is_univariate: series2 = series2.univariate_component(component2)

    series1 += series1_y_offset
    series2 += series2_y_offset

    xa1 = series1.data_array(copy=False)
    xa2 = series2.data_array(copy=False)

    path = self.path()
    n = len(path)

    path = path.reshape((-1,))

    time_dim1 = series1._time_dim
    time_dim2 = series2._time_dim

    x_coords1 = np.array(xa1[time_dim1], dtype="datetime64[s]")[path[::2]]
    x_coords2 = np.array(xa2[time_dim2], dtype="datetime64[s]")[path[1::2]]

    y_coords1 = series1.univariate_values()[path[0::2]]
    y_coords2 = series2.univariate_values()[path[1::2]]

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

    series1.plot()
    series2.plot()
