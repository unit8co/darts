from enum import Enum
from typing import Union

import pandas as pd

from darts import TimeSeries
from darts.logging import raise_if_not

# Those freqs can be used to divide Time deltas (the others can't):
DIVISIBLE_FREQS = {"D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"}


class CovariateType(Enum):
    PAST = "past"
    FUTURE_PAST = "future_past"
    HISTORIC_FUTURE = "historic_future"
    FUTURE = "future"
    NONE = None


def _get_matching_index(ts_target: TimeSeries, ts_covariate: TimeSeries, idx: int):
    """
    Given two overlapping series `ts_target` and `ts_covariate` and an index point `idx` of `ts_target`, returns the
    matching index point in `ts_covariate`, based on the ending times of the two series.
    The indices are starting from the end of the series.

    This function is used to jointly slice target and covariate series in datasets. It supports both datetime and
    integer indexed series.

    Note: this function does not check if the matching index value is in `ts_covariate` or not.
    """
    raise_if_not(
        ts_target.freq == ts_covariate.freq,
        "The dataset contains some target/covariates series pair that have incompatible "
        'time axes (not the same "freq") and thus cannot be matched',
    )

    freq = ts_target.freq

    return idx + _index_diff(
        self=ts_target.end_time(), other=ts_covariate.end_time(), freq=freq
    )


def _index_diff(
    self: Union[pd.Timestamp, int], other: Union[pd.Timestamp, int], freq: pd.offsets
):
    """Returns the difference between two indexes `other` and `self` (`other` - `self`) of frequency `freq`."""
    if isinstance(freq, int):
        return int(other - self)

    elif freq.freqstr in DIVISIBLE_FREQS:
        return int((other - self) / freq)

    # /!\ THIS IS TAKING LINEAR TIME IN THE LENGTH OF THE SERIES
    # it won't scale if the end of target and covariates are far apart and the freq is not in DIVISIBLE_FREQS
    # (Not sure there's a way around it for exotic freqs)
    if other >= self:
        return -1 + len(pd.date_range(start=self, end=other, freq=freq))
    else:
        return 1 - len(pd.date_range(start=other, end=self, freq=freq))
