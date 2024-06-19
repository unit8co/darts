from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)

# Those freqs can be used to divide Time deltas (the others can't):
DIVISIBLE_FREQS = {"D", "h", "H", "T", "min", "s", "S", "L", "ms", "U", "us", "N", "ns"}
# supported built-in sample weight generators for regression and torch models
SUPPORTED_SAMPLE_WEIGHT = {"linear", "exponential"}


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
    if ts_target.freq != ts_covariate.freq:
        raise_log(
            ValueError(
                "The dataset contains some target/covariates series pair that have incompatible "
                'time axes (not the same "freq") and thus cannot be matched'
            ),
            logger=logger,
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


def _process_sample_weight(sample_weight, target_series):
    if sample_weight is None:
        return None

    if target_series is None:
        raise_log(
            ValueError("Must supply target `series` when using `sample_weight`."),
            logger=logger,
        )

    # get sample weights
    if isinstance(sample_weight, str):
        if sample_weight not in SUPPORTED_SAMPLE_WEIGHT:
            raise_log(
                ValueError(
                    f"Invalid `sample_weight` value: `'{sample_weight}'`. "
                    f"If a string, must be one of: {SUPPORTED_SAMPLE_WEIGHT}."
                ),
                logger=logger,
            )
        # create global time weights based on the longest target series
        max_len = max(len(target_i) for target_i in target_series)
        if sample_weight == "linear":
            weights = np.linspace(0, 1, max_len)
        else:  # "exponential"
            time_steps = np.linspace(0, 1, max_len)
            weights = np.exp(-10 * (1 - time_steps))
        weights = np.expand_dims(weights, -1).astype(target_series[0].dtype)

        # create sequence of series for tabularization
        sample_weight = [
            TimeSeries.from_times_and_values(
                times=target_i.time_index,
                values=weights[-len(target_i) :],
            )
            for target_i in target_series
        ]

    sample_weight = series2seq(sample_weight)
    if len(target_series) != len(sample_weight):
        raise_log(
            ValueError(
                "The provided sequence of target `series` must have the same length as "
                "the provided sequence of `sample_weight`."
            ),
            logger=logger,
        )
    return sample_weight
