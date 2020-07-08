"""
darts
-----
"""

from .timeseries import TimeSeries

# Enums
from enum import Enum


class Season(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'
    NONE = None


class Trend(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'


class Model(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'


__version__ = 'dev'
