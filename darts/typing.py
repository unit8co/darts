"""DARTS TYPES

tldr;   This file defines reusable types for the Darts Eco-System.
"""

from typing import TypeAlias

import pandas as pd

TimeIndex: TypeAlias = pd.DatetimeIndex | pd.RangeIndex  # | pd.Index
