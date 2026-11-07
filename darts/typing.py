"""
Typing
------

This module contains custom Darts types and type aliases.
"""

from collections.abc import Sequence
from datetime import tzinfo
from typing import TYPE_CHECKING, TypeAlias, Union

import pandas as pd

# prevent circular import at runtime (in timeseries.py)
if TYPE_CHECKING:
    from darts.timeseries import TimeSeries

TimeIndex: TypeAlias = pd.DatetimeIndex | pd.RangeIndex

# req. Union with string literals (again, to avoid circular dependency issues)
TimeSeriesLike: TypeAlias = Union["TimeSeries", Sequence["TimeSeries"]]

# timestamps can be either a `Timestamp` for series with `DatetimeIndex` or `int` for series with `RangeIndex`
TimeStamp: TypeAlias = pd.Timestamp | int
IntoTimeStamp: TypeAlias = TimeStamp | str

# frequency can be either a `BaseOffset` for series with `DatetimeIndex` or `int` for series with `RangeIndex`
Frequency: TypeAlias = pd.offsets.BaseOffset | int
IntoFrequency: TypeAlias = Frequency | str

TimeZone: TypeAlias = str | tzinfo | None
