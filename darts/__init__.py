"""
darts
-----
"""

from .timeseries import TimeSeries
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler

# Enums
from enum import Enum


class SeasonalityMode(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'
    NONE = None


class TrendMode(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'


class ModelMode(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'


__version__ = 'dev'

colors = cycler(color=['black', '003DFD', 'b512b8', '11a9ba', '0d780f', 'f77f07', 'ba0f0f'])

u8plots_mplstyle = {
    'font.family' : 'sans serif',
    'axes.edgecolor' : 'black',
    'axes.grid' : True,
    'axes.labelcolor': '#333333',
    'axes.labelweight' : 600,
    'axes.linewidth' : 1,
    'axes.prop_cycle' : colors,
    'axes.spines.top' : False,
    'axes.spines.right' : False,
    'axes.spines.bottom' : False,
    'axes.spines.left' : False,
    'grid.color' : '#d0d0d0',
    'legend.frameon' : False,
    'lines.linewidth' : 1.3,
    'xtick.bottom' : False,
    'xtick.color': '#333333',
    'xtick.labelsize':'small',
    'ytick.color': '#333333',
    'ytick.labelsize':'small',
    'xtick.bottom' : False,
}


mpl.rcParams.update(u8plots_mplstyle)
