from ..timeseries import TimeSeries
from ..custom_logging import raise_log, get_logger
from typing import List

logger = get_logger(__name__)

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
        raise_log(ValueError('The provided time series must have nonzero overlap'), logger)

    return list(map(lambda s: s.slice(last_first, first_last), series))
