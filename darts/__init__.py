"""
darts
-----
"""

from .timeseries import TimeSeries
import matplotlib.pyplot as plt

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

plt.style.use('../darts/u8plots.mplstyle')
