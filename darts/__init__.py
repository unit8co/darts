"""
Darts
-----

A Python library for user-friendly forecasting and anomaly detection on time series.
"""

from darts.config import (
    describe_option,
    get_option,
    option_context,
    reset_option,
    set_option,
)
from darts.timeseries import TimeSeries, concatenate, slice_intersect

__version__ = "0.40.0"

__all__ = [
    "TimeSeries",
    "concatenate",
    "slice_intersect",
    "get_option",
    "set_option",
    "reset_option",
    "describe_option",
    "option_context",
]
