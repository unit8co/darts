"""
darts
-----
"""

import os

import matplotlib as mpl
from matplotlib import cycler

from darts.timeseries import TimeSeries, concatenate, slice_intersect

__version__ = "0.34.0"

colors = cycler(
    color=["black", "003DFD", "b512b8", "11a9ba", "0d780f", "f77f07", "ba0f0f"]
)

u8plots_mplstyle = {
    "font.family": "sans serif",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "axes.labelcolor": "#333333",
    "axes.labelweight": 600,
    "axes.linewidth": 1,
    "axes.prop_cycle": colors,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "grid.color": "#dedede",
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "xtick.color": "#333333",
    "xtick.labelsize": "small",
    "ytick.color": "#333333",
    "ytick.labelsize": "small",
    "xtick.bottom": False,
}


if os.getenv("DARTS_CONFIGURE_MATPLOTLIB", "1") != "0":
    mpl.rcParams.update(u8plots_mplstyle)

__all__ = ["TimeSeries", "concatenate", "slice_intersect"]
