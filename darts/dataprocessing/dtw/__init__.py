"""
Dynamic Time Warping (DTW)
--------------------------

Tools for computing Dynamic Time Warping alignment between time series, including cost matrices,
alignment algorithms, and windowing constraints (Sakoe-Chiba, Itakura).
"""

from darts.dataprocessing.dtw.cost_matrix import CostMatrix
from darts.dataprocessing.dtw.dtw import DTWAlignment, dtw
from darts.dataprocessing.dtw.window import (
    CRWindow,
    Itakura,
    NoWindow,
    SakoeChiba,
    Window,
)

__all__ = [
    "CostMatrix",
    "DTWAlignment",
    "dtw",
    "CRWindow",
    "Itakura",
    "NoWindow",
    "SakoeChiba",
    "Window",
]
