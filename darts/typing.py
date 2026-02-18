"""DARTS TYPES

tldr;   This file defines reusable types for the Darts Eco-System.
"""

from typing import TypeAlias

import pandas as pd

TimeIndex: TypeAlias = pd.DatetimeIndex | pd.RangeIndex  # | pd.Index

# NOTE: this type is only used in darts/utils/historical_forecasts/utils.py
# TODO: check if it is actually required. if no: delete!
ExtendedTimeIndex: TypeAlias = (
    TimeIndex | tuple[int, int] | tuple[pd.Timestamp, pd.Timestamp]
)
