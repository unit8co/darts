"""
Additional util functions
-------------------------
"""

from collections.abc import Sequence
from enum import Enum
from functools import total_ordering
from typing import Optional, Union

from darts import TimeSeries
from darts.logging import get_logger, raise_log

try:
    from IPython import get_ipython
except ModuleNotFoundError:
    get_ipython = None

logger = get_logger(__name__)

_SEQ_TYPE_NAMES = {
    0: "`TimeSeries`",
    1: "`Sequence[TimeSeries]`",
    2: "`Sequence[Sequence[TimeSeries]]`",
}


@total_ordering
class SeriesType(Enum):
    """An Enum for different `TimeSeries` sequence types."""

    NONE = -1  # `None`
    SINGLE = 0  # `TimeSeries`
    SEQ = 1  # `Sequence[TimeSeries]`
    SEQ_SEQ = 2  # `Sequence[Sequence[TimeSeries]]`

    def _check_member(self, other):
        if self.__class__ is not other.__class__:
            raise_log(ValueError("`other` must be a `SeriesType` enum."), logger=logger)

    def __eq__(self, other):
        self._check_member(other)
        return super().__eq__(other)

    def __lt__(self, other):
        self._check_member(other)
        return self.value < other.value

    def __add__(self, other: int):
        if not isinstance(other, int):
            raise_log(ValueError("`other` must be of type `int`."), logger=logger)
        new_val = self.value + other
        if new_val > 2:
            raise_log(
                ValueError("Cannot go higher than `SeriesType.SEQ_SEQ`."), logger=logger
            )
        return SeriesType(new_val)

    def __str__(self):
        return _SEQ_TYPE_NAMES[self.value]


def series2seq(
    ts: Optional[
        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
    ],
    seq_type_out: SeriesType = SeriesType.SEQ,
    nested: bool = False,
) -> Optional[Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]]:
    """If possible, converts `ts` into the desired sequence type `seq_type_out`. Otherwise, returns the
    original `ts`.

    Parameters
    ----------
    ts
        None, a single TimeSeries, a sequence of TimeSeries, or a sequence of sequences of TimeSeries.
    seq_type_out
        The output sequence type:

        - SeriesType.SINGLE: `TimeSeries` (e.g. a single series)
        - SeriesType.SEQ: sequence of `TimeSeries` (e.g. multiple series)
        - SeriesType.SEQ_SEQ: sequence of sequences of `TimeSeries` (e.g. historical forecasts output)
    nested
        Only applies with `seq_type_out=SeriesType.SEQ_SEQ` and `ts` having a sequence type `SeriesType.SEQ`.
        In this case, wrap each element in `ts` in a list ([ts1, ts2] -> [[ts1], [ts2]]).

    Raises
    ------
    ValueError
        If there is an invalid `seq_type_out` value.
    """
    if ts is None:
        return ts

    if not isinstance(seq_type_out, SeriesType):
        raise_log(
            ValueError(
                f"Invalid parameter `seq_type_out={seq_type_out}`. Must be one of `(0, 1, 2)`"
            ),
            logger=logger,
        )

    seq_type_in = get_series_seq_type(ts)

    if seq_type_out == seq_type_in:
        return ts

    n_series = 1 if seq_type_in == SeriesType.SINGLE else len(ts)

    if seq_type_in == SeriesType.SINGLE and seq_type_out == SeriesType.SEQ:
        # ts -> [ts]
        return [ts]
    elif seq_type_in == SeriesType.SINGLE and seq_type_out == SeriesType.SEQ_SEQ:
        # ts -> [[ts]]
        return [[ts]]
    elif (
        seq_type_in == SeriesType.SEQ
        and seq_type_out == SeriesType.SINGLE
        and n_series == 1
    ):
        # [ts] -> ts
        return ts[0]
    elif seq_type_in == SeriesType.SEQ and seq_type_out == SeriesType.SEQ_SEQ:
        if not nested:
            # [ts1, ts2] -> [[ts1, ts2]]
            return [ts]
        else:
            # [ts1, ts2] -> [[ts1], [ts2]]
            return [[ts_] for ts_ in ts]
    elif (
        seq_type_in == SeriesType.SEQ_SEQ
        and seq_type_out == SeriesType.SINGLE
        and n_series == 1
    ):
        # [[ts]] -> [ts]
        return ts[0]
    elif (
        seq_type_in == SeriesType.SEQ_SEQ
        and seq_type_out == SeriesType.SEQ
        and n_series == 1
    ):
        # [[ts1, ts2]] -> [[ts1, ts2]]
        return ts
    else:
        # ts -> ts
        return ts


def seq2series(
    ts: Optional[Union[TimeSeries, Sequence[TimeSeries]]],
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
    return series2seq(ts, seq_type_out=SeriesType.SINGLE)


def get_single_series(
    ts: Optional[
        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
    ],
) -> Optional[TimeSeries]:
    """Returns a single (first) TimeSeries or `None` from `ts`. Returns `ts` if  `ts` is a TimeSeries, `ts[0]` if
    `ts` is a `Sequence[TimeSeries]`, and `ts[0][0]` if `ts` is a `Sequence[Sequence[TimeSeries]]`.
    Otherwise, returns `None`.

    Parameters
    ----------
    ts
        None, a single `TimeSeries`, a sequence of `TimeSeries`, or a sequence of sequences of `TimeSeries`.

    Returns
    -------
    TimeSeries
        `ts` if  `ts` is a TimeSeries, `ts[0]` if `ts` is a Sequence of TimeSeries. Otherwise, returns `None`

    """
    seq_type = get_series_seq_type(ts)
    if seq_type <= SeriesType.SINGLE:
        # `None` and `TimeSeries`
        return ts
    elif seq_type == SeriesType.SEQ:
        return ts[0]
    else:
        return ts[0][0]


def get_series_seq_type(
    ts: Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]],
) -> SeriesType:
    """Returns the sequence type of `ts`.

    - SeriesType.SINGLE: `TimeSeries` (e.g. a single series)
    - SeriesType.SEQ: sequence of `TimeSeries` (e.g. multiple series)
    - SeriesType.SEQ_SEQ: sequence of sequences of `TimeSeries` (e.g. historical forecasts output)

    Parameters
    ----------
    ts
        The input series to get the sequence type from.

    Raises
    ------
    ValueError
        If `ts` does not have one of the expected sequence types.
    """
    if ts is None:
        return SeriesType.NONE
    elif isinstance(ts, TimeSeries):
        return SeriesType.SINGLE
    elif isinstance(ts[0], TimeSeries):
        return SeriesType.SEQ
    else:
        try:
            if isinstance(ts[0][0], TimeSeries):
                return SeriesType.SEQ_SEQ
            else:
                raise_log(
                    ValueError(
                        "input series must be of type `TimeSeries`, `Sequence[TimeSeries]`, or "
                        "`Sequence[Sequence[TimeSeries]]`."
                    ),
                    logger=logger,
                )
        except Exception as err:
            raise_log(
                ValueError(
                    "input series must be of type `TimeSeries`, `Sequence[TimeSeries]`, or "
                    f"`Sequence[Sequence[TimeSeries]]`. Raised: `{type(err).__name__}('{str(err)}')`"
                ),
                logger=logger,
            )


# TODO: we do not check the time index here
def retain_period_common_to_all(series: list[TimeSeries]) -> list[TimeSeries]:
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
