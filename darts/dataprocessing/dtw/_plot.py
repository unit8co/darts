from typing import Union

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def plot(
    self,
    new_plot: bool = False,
    show_series: bool = False,
    show_cost: bool = True,
    cost_cmap: str = "summer",
    args_path: dict = {},
    args_cost: dict = {},
    args_series1: dict = {},
    args_series2: dict = {},
):
    """
    Plot the warp path.

    Parameters
    ----------
    new_plot
        Boolean value indicating whether to spawn a new figure.
    show_series
        Boolean value indicating whether to additionally plot the two time series.
        Series1 will be plotted below the cost matrix and series2 to the left of the cost matrix.
    show_cost
        Boolean value indicating whether to additionally plot the cost matrix.
        Cost Matrix will be displayed as a heatmap.
        Useful for visualizing the effect of the window function.
    cost_cmap
        Colormap style for cost matrix plot
    args_path
        Some keyword arguments to pass to `plot()` function for warp path
    args_cost
        Some keyword arguments to pass to `imshow` function for cost matrix
    args_series1
        Some keyword arguments to pass to `plot()` method for series1
    args_series2
        Some keyword arguments to pass to `plot()` method for series2
    """

    if new_plot:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    if show_series:
        gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[0.6, 0.4])

        warp = fig.add_subplot(gs[0, 1])
        left = fig.add_subplot(gs[0, 0])
        bottom = fig.add_subplot(gs[1, 1])

    else:
        warp = plt.gca()

    warp.title.set_text("Cost Matrix")
    warp.set_xlim([0, self.n])
    warp.set_ylim([0, self.m])
    warp.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if show_cost:
        cost_matrix = self.cost.to_dense()
        cost_matrix = np.transpose(cost_matrix[1:, 1:])
        warp.imshow(
            cost_matrix,
            cmap=cost_cmap,
            interpolation="none",
            origin="lower",
            extent=[0, self.n, 0, self.m],
            **args_cost,
        )

    show_path = True
    if show_path:
        path = self.path()

        i_coords = path[:, 0] + 0.5
        j_coords = path[:, 1] + 0.5
        warp.plot(i_coords, j_coords, **args_path)

    if show_series:
        self.series1.plot(ax=bottom, **args_series1)
        self.series2.plot(ax=left, y=self.series2._time_dim, **args_series2)

        bottom.set_xlim([self.series1.start_time(), self.series1.end_time()])
        left.set_ylim([self.series2.start_time(), self.series2.end_time()])

        bottom.legend()
        bottom.set_title("")
        left.legend()
        left.set_title("")

    warp.set_aspect("auto")


def plot_alignment(
    self,
    new_plot: bool = False,
    series1_y_offset: float = 0,
    series2_y_offset: float = 0,
    components: Union[tuple[Union[str, int], Union[str, int]]] = (0, 0),
    args_line: dict = {},
    args_series1: dict = {},
    args_series2: dict = {},
):
    """
    Plots the uni-variate component of each series,
    with lines between them indicating the alignment selected by the DTW algorithm.

    Parameters
    ----------
    new_plot
        whether to spawn a new Figure
    series2_y_offset
        offset series1 vertically for ease of viewing.
    series1_y_offset
        offset series2 vertically for ease of viewing.
    components
        Tuple of component index for series1 and component index for series2.
    args_line
        Some keywords arguments to pass to `plot()` method for line
    args_series1
        Some keyword arguments to pass to `plot()` method for series1
    args_series2
        Some keyword arguments to pass to `plot()` method for series2
    """

    series1 = self.series1
    series2 = self.series2

    (component1, component2) = components

    if not series1.is_univariate:
        series1 = series1.univariate_component(component1)
    if not series2.is_univariate:
        series2 = series2.univariate_component(component2)

    series1 += series1_y_offset
    series2 += series2_y_offset

    xa1 = series1.data_array(copy=False)
    xa2 = series2.data_array(copy=False)

    path = self.path()
    n = len(path)

    time_dim1 = series1.time_dim
    time_dim2 = series2.time_dim

    x_coords1 = np.array(xa1[time_dim1], dtype=xa1[time_dim1].dtype)[path[:, 0]]
    x_coords2 = np.array(xa2[time_dim2], dtype=xa2[time_dim2].dtype)[path[:, 1]]

    y_coords1 = series1.univariate_values()[path[:, 0]]
    y_coords2 = series2.univariate_values()[path[:, 1]]

    if series1.has_datetime_index:
        x_dtype = xa1[time_dim1].dtype
        x_nan = np.datetime64("NaT")
    else:
        x_dtype = np.float64
        x_nan = np.nan

    x_coords = np.empty(n * 3, dtype=x_dtype)
    y_coords = np.empty(n * 3, dtype=np.float64)

    x_coords[0::3] = x_coords1
    x_coords[1::3] = x_coords2
    x_coords[2::3] = x_nan

    y_coords[0::3] = y_coords1
    y_coords[1::3] = y_coords2
    y_coords[2::3] = np.nan

    arr = xr.DataArray(y_coords, dims=["value"], coords={"value": x_coords})
    xr.plot.line(arr, x="value", **args_line)

    series1.plot(**args_series1)
    series2.plot(**args_series2)
