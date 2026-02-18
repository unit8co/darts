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
from darts.timeseries import (
    TimeSeries,
    concatenate,
    slice_intersect,
    to_group_dataframe,
)

__version__ = "0.41.0"

__all__ = [
    "TimeSeries",
    "concatenate",
    "slice_intersect",
    "to_group_dataframe",
    "get_option",
    "set_option",
    "reset_option",
    "describe_option",
    "option_context",
]
