import pytest

from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)


class TestTsUtils:
    def test_series_type(self):
        assert SeriesType.NONE.value == -1
        assert SeriesType.SINGLE.value == 0
        assert SeriesType.SEQ.value == 1
        assert SeriesType.SEQ_SEQ.value == 2

        # equality works with members
        assert SeriesType.NONE == SeriesType.NONE
        assert SeriesType.SINGLE == SeriesType.SINGLE
        assert SeriesType.SEQ == SeriesType.SEQ
        assert SeriesType.SEQ_SEQ == SeriesType.SEQ_SEQ

        # inequality works with members
        assert SeriesType.SINGLE != SeriesType.SEQ
        assert SeriesType.SEQ != SeriesType.SEQ_SEQ

        # equality does not work with non-members
        with pytest.raises(ValueError) as err:
            _ = SeriesType.SINGLE == 0
        assert str(err.value).startswith("`other` must be a `SeriesType` enum.")

        # single series order is < sequence of series order < sequence of sequences of series order
        assert SeriesType.NONE < SeriesType.SINGLE < SeriesType.SEQ < SeriesType.SEQ_SEQ
        assert SeriesType.SEQ_SEQ > SeriesType.SEQ > SeriesType.SINGLE > SeriesType.NONE

    def test_get_series_seq_type(self):
        ts = linear_timeseries(length=3)
        assert get_series_seq_type(None) == SeriesType.NONE
        assert get_series_seq_type(ts) == SeriesType.SINGLE
        assert get_series_seq_type([ts]) == SeriesType.SEQ
        assert get_series_seq_type([[ts]]) == SeriesType.SEQ_SEQ

        # unknown sequence type
        with pytest.raises(ValueError) as err:
            _ = get_series_seq_type([[[ts]]])
        assert str(err.value).startswith(
            "input series must be of type `TimeSeries`, `Sequence[TimeSeries]`"
        )

        # sequence with elements different from `TimeSeries`
        with pytest.raises(ValueError) as err:
            _ = get_series_seq_type([[0.0, 1.0, 2]])
        assert str(err.value).startswith(
            "input series must be of type `TimeSeries`, `Sequence[TimeSeries]`"
        )

    def test_series2seq(self):
        ts = linear_timeseries(length=3)

        # `None` to different sequence types
        assert series2seq(None, seq_type_out=SeriesType.SINGLE) is None
        assert series2seq(None, seq_type_out=SeriesType.SEQ) is None
        assert series2seq(None, seq_type_out=SeriesType.SEQ_SEQ) is None

        # `TimeSeries` to different sequence types
        assert series2seq(ts, seq_type_out=SeriesType.SINGLE) == ts
        assert series2seq(ts, seq_type_out=SeriesType.SEQ) == [ts]
        assert series2seq(ts, seq_type_out=SeriesType.SEQ_SEQ) == [[ts]]

        # Sequence[`TimeSeries`] to different sequence types
        assert series2seq([ts], seq_type_out=SeriesType.SINGLE) == ts
        assert series2seq([ts], seq_type_out=SeriesType.SEQ) == [ts]
        assert series2seq([ts], seq_type_out=SeriesType.SEQ_SEQ) == [[ts]]

        # Sequence[`TimeSeries`, `TimeSeries`] to different sequence types
        # cannot reduce dimension since there is more than one element in SEQ
        assert series2seq([ts, ts], seq_type_out=SeriesType.SINGLE) == [ts, ts]
        assert series2seq([ts, ts], seq_type_out=SeriesType.SEQ) == [ts, ts]
        assert series2seq([ts, ts], seq_type_out=SeriesType.SEQ_SEQ) == [[ts, ts]]
        assert series2seq([ts, ts], seq_type_out=SeriesType.SEQ_SEQ, nested=True) == [
            [ts],
            [ts],
        ]

        # Sequence[Sequence[`TimeSeries`]] to different sequence types
        # SEQ_SEQ represents historical forecasts (and downstream tasks) output
        # the outer sequence represents the series axis, therefore reducing to SINGLE
        # actually returns a Sequence[`TimeSeries`]
        assert series2seq([[ts]], seq_type_out=SeriesType.SINGLE) == [ts]
        assert series2seq([[ts]], seq_type_out=SeriesType.SEQ) == [[ts]]
        assert series2seq([[ts]], seq_type_out=SeriesType.SEQ_SEQ) == [[ts]]

        # Sequence[`TimeSeries`, `TimeSeries`] to different sequence types
        # cannot reduce dimension since there is more than one element in SEQ_SEQ
        assert series2seq([[ts], [ts]], seq_type_out=SeriesType.SINGLE) == [[ts], [ts]]
        assert series2seq([[ts], [ts]], seq_type_out=SeriesType.SEQ) == [[ts], [ts]]
        assert series2seq([[ts], [ts]], seq_type_out=SeriesType.SEQ_SEQ) == [[ts], [ts]]

    def test_get_single_series(self):
        ts = linear_timeseries(length=3)
        assert get_single_series(None) is None
        assert get_single_series(ts) == ts
        assert get_single_series([ts]) == ts
        assert get_single_series([ts, ts]) == ts
