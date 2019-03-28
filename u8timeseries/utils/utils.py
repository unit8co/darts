from ..timeseries import TimeSeries
from typing import List


def retain_period_common_to_all(series: List[TimeSeries]) -> List[TimeSeries]:
    """
    Trims all series in the provided list, if necessary, so that the return time series have
    the same time index (corresponding to largest duration common to all series).

    Raises an error if no such time index exists.
    :param series:
    :return:
    """

    last_first = max(map(lambda s: s.start_time(), series))
    first_last = min(map(lambda s: s.end_time(), series))

    if last_first >= first_last:
        raise ValueError('The provided time series must have nonzero overlap')

    return list(map(lambda s: s.slice(last_first, first_last), series))
