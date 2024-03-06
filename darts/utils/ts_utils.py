"""
Additional util functions
-------------------------
"""

from typing import List, Optional, Sequence, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_log

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    get_ipython = None

logger = get_logger(__name__)


# TODO: we do not check the time index here
def retain_period_common_to_all(series: List[TimeSeries]) -> List[TimeSeries]:
    """
    Trims all series in the provided list, if necessary, so that the returned time series have
    a common span (corresponding to largest time sub-interval common to all series).

    Parameters
    ----------
    series
        The list of series to consider.

    Raises
    ------
    ValueError
        If no common time sub-interval exists

    Returns
    -------
    List[TimeSeries]
        A list of series, where each series have the same span
    """

    last_first = max(map(lambda s: s.start_time(), series))
    first_last = min(map(lambda s: s.end_time(), series))

    if last_first >= first_last:
        raise_log(
            ValueError("The provided time series must have nonzero overlap"), logger
        )

    return list(map(lambda s: s.slice(last_first, first_last), series))


def series2seq(
    ts: Optional[Union[TimeSeries, Sequence[TimeSeries]]]
) -> Optional[Sequence[TimeSeries]]:
    """If `ts` is a single TimeSeries, return it as a list of a single TimeSeries.

    Parameters
    ----------
    ts
        None, a single TimeSeries, or a sequence of TimeSeries

    Returns
    -------
        `ts` if `ts` is not a TimeSeries, else `[ts]`

    """
    return [ts] if isinstance(ts, TimeSeries) else ts


def seq2series(
    ts: Optional[Union[TimeSeries, Sequence[TimeSeries]]]
) -> Optional[TimeSeries]:
    """If `ts` is a Sequence with only a single series, return the single series as TimeSeries.

    Parameters
    ----------
    ts
        None, a single TimeSeries, or a sequence of TimeSeries

    Returns
    -------
        `ts` if `ts` if is not a single element TimeSeries sequence, else `ts[0]`

    """

    return ts[0] if isinstance(ts, Sequence) and len(ts) == 1 else ts


def get_single_series(
    ts: Optional[Union[TimeSeries, Sequence[TimeSeries]]]
) -> Optional[TimeSeries]:
    """Returns a single (first) TimeSeries or `None` from `ts`. Returns `ts` if  `ts` is a TimeSeries, `ts[0]` if
    `ts` is a Sequence of TimeSeries. Otherwise, returns `None`.

    Parameters
    ----------
    ts
        None, a single TimeSeries, or a sequence of TimeSeries.

    Returns
    -------
        `ts` if  `ts` is a TimeSeries, `ts[0]` if `ts` is a Sequence of TimeSeries. Otherwise, returns `None`

    """
    if isinstance(ts, TimeSeries) or ts is None:
        return ts
    else:
        return ts[0]
