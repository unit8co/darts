import pandas as pd

from ...timeseries import TimeSeries
from ...logging import raise_if_not

# Those freqs can be used to divide Time deltas (the others can't):
DIVISIBLE_FREQS = {'D', 'H', 'T', 'min', 'S', 'L', 'ms', 'U', 'us', 'N'}


def _get_matching_index(ts_target: TimeSeries,
                        ts_covariate: TimeSeries,
                        idx: int):
    """
    Given two overlapping series `ts_target` and `ts_covariate` and an index point `idx` of `ts_target`, returns the matching
    index point in `ts_covariate`, based on the ending times of the two series.
    The indices are starting from the end of the series.

    This function is used to jointly slice target and covariate series in datasets. It supports both datetime and
    integer indexed series.

    Note: this function does not check if the matching index value is in `ts_covariate` or not.
    """
    raise_if_not(ts_target.freq == ts_covariate.freq,
                 'The dataset contains some target/covariates series pair that have incompatible '
                 'time axes (not the same "freq") and thus cannot be matched')

    freq = ts_target.freq

    if isinstance(freq, int):
        return idx + int(ts_covariate.end_time() - ts_target.end_time())

    elif ts_target.freq.freqstr in DIVISIBLE_FREQS:
        return idx + int((ts_covariate.end_time() - ts_target.end_time()) / freq)

    # /!\ THIS IS TAKING LINEAR TIME IN THE LENGTH OF THE SERIES
    # it won't scale if the end of target and covariates are far apart and the freq is not in DIVISIBLE_FREQS
    # (Not sure there's a way around it for exotic freqs)
    if ts_covariate.end_time() >= ts_target.end_time():
        return idx - 1 + len(
            pd.date_range(start=ts_target.end_time(), end=ts_covariate.end_time(), freq=ts_target.freq))
    else:
        return idx + 1 - len(
            pd.date_range(start=ts_covariate.end_time(), end=ts_target.end_time(), freq=ts_target.freq))
