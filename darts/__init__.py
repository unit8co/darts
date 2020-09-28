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
    'font.family' : 'Arial',
    'figure.facecolor' : '#f0f0f0',
    'axes.facecolor' : '#f0f0f0',
    'axes.prop_cycle' : colors,
    'lines.linewidth' : 1.3,
    'axes.spines.top' : False,
    'axes.spines.right' : False,
    'axes.spines.bottom' : False,
    'axes.spines.left' : False,
    'axes.linewidth' : 1,
    'axes.edgecolor' : 'black',
    'xtick.bottom' : False,
    'ytick.left' : False,
    'axes.grid' : True,
    'grid.color' : '#d0d0d0',
    'grid.alpha' : 0.5,
    'legend.frameon' : False}

mpl.rcParams.update(u8plots_mplstyle)
