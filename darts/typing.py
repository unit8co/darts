"""DARTS TYPES

tldr;   This file defines reusable types for the Darts Eco-System.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, Union

import pandas as pd

# prevent circular import at runtime (in timeseries.py)
if TYPE_CHECKING:
    from darts.timeseries import TimeSeries

TimeIndex: TypeAlias = pd.DatetimeIndex | pd.RangeIndex

# req. Union with string literals (again, to avoid circular dependency issues)
TimeSeriesLike: TypeAlias = Union["TimeSeries", Sequence["TimeSeries"]]
