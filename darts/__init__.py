"""
darts
-----
"""

from .timeseries import TimeSeries

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
