"""
Additional util functions
-------------------------
"""

from typing import Any, List, Optional, Sequence, Union

import numpy as np

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
    ts: Optional[Union[Any, Sequence[Any], Sequence[Sequence[Any]]]],
    seq_type_out: int = 1,
    is_numeric: bool = False,
    nested: bool = False,
) -> Optional[Union[Any, Sequence[Any], Sequence[Sequence[Any]]]]:
    """If possible, converts `ts` into the desired sequence type `seq_type_out`. Otherwise, returns the
    original `ts`.

    Parameters
    ----------
    ts
        None, a single TimeSeries, a sequence of TimeSeries, or a sequence of sequences of TimeSeries.
    seq_type_out
        The output sequence type:

        - 0: `TimeSeries` (e.g. a single series)
        - 1: sequence of `TimeSeries` (e.g. multiple series)
        - 2: sequence of sequences of `TimeSeries` (e.g. historical forecasts output)
    is_numeric
        Whether to look for a numeric value as the innermost value. If `False`, looks
        for a `TimeSeries` object.

    Raises
    ------
    ValueError
        If there is an invalid `seq_type_out` value.
    """
    if ts is None:
        return ts

    if not isinstance(seq_type_out, int) or not 0 <= seq_type_out <= 2:
        raise_log(
            ValueError(
                f"Invalid parameter `seq_type_out={seq_type_out}`. Must be one of `(0, 1, 2)`"
            ),
            logger=logger,
        )

    seq_type_in = get_series_seq_type(ts, is_numeric=is_numeric)

    if seq_type_out == seq_type_in:
        return ts

    n_series = 1 if seq_type_in == 0 else len(ts)

    if seq_type_in == 0 and seq_type_out == 1:
        # ts -> [ts]
        return [ts]
    elif seq_type_in == 0 and seq_type_out == 2:
        # ts -> [[ts]]
        return [[ts]]
    elif seq_type_in == 1 and seq_type_out == 0 and n_series == 1:
        # [ts] -> ts
        return ts[0]
    elif seq_type_in == 1 and seq_type_out == 2:
        if not nested:
            # [ts1, ts2] -> [[ts1, ts2]]
            return [ts]
        else:
            # [ts1, ts2] -> [[ts1], [ts2]]
            return [[ts_] for ts_ in ts]
    elif seq_type_in == 2 and seq_type_out == 0 and n_series == 1:
        if not nested:
            # [[ts]] -> [ts]
            return ts[0]
        else:
            # either [[ts]] -> ts or [[ts1, ts2]] -> [ts1, ts2]
            return series2seq(ts[0], seq_type_out=seq_type_out, is_numeric=is_numeric)
    elif seq_type_in == 2 and seq_type_out == 1 and n_series == 1:
        if not nested:
            # [[ts1, ts2]] -> [[ts1, ts2]]
            return ts
        else:
            # seq_type_in `1` and `2` are both representing time series axis
            # [[ts1, ts2]] -> [ts1, ts2]
            return ts[0]
    else:
        # ts -> ts
        return ts


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


def get_series_seq_type(
    ts: Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]],
    is_numeric: bool = False,
) -> int:
    """Returns the sequence type of `ts`.

    - 0: `TimeSeries` (e.g. a single series)
    - 1: sequence of `TimeSeries` (e.g. multiple series)
    - 2: sequence of sequences of `TimeSeries` (e.g. historical forecasts output)

    Parameters
    ----------
    ts
        The input series to get the sequence type from.
    is_numeric
        Whether to look for a numeric value as the innermost value. If `False`, looks
        for a `TimeSeries` object.

    Raises
    ------
    ValueError
        If `ts` does not have one of the expected sequence types.
    """

    def is_inner_value(val) -> bool:
        if not is_numeric:
            return isinstance(val, TimeSeries)
        else:
            return np.issubdtype(type(val), np.number)

    if is_inner_value(ts):
        return 0
    elif is_inner_value(ts[0]):
        return 1
    elif is_inner_value(ts[0][0]):
        return 2
    else:
        ts_type = "TimeSeries" if not is_numeric else "np.number"
        raise_log(
            ValueError(
                f"`ts` must be of type `{ts_type}`, `Sequence[{ts_type}]`, or "
                f"`Sequence[Sequence[{ts_type}]]`"
            ),
            logger=logger,
        )
