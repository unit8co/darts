from typing import Sequence, Optional, Union
from ..timeseries import TimeSeries


def walk_forward_split(
        data: Sequence[TimeSeries],
        input_chunk_length: int,
        output_chunk_length: int,
        test_size: Optional[Union[float, int]] = 0.25) -> Sequence[Sequence[TimeSeries]]:

    train_set = []
    test_set = []

    for ts in data:
        ts_length = len(ts)

        if 0 < test_size < 1:
            train_end_index = int((ts_length - output_chunk_length) * (1 - test_size))
        else:
            train_end_index = ts_length - output_chunk_length - test_size

        if train_end_index < input_chunk_length:
            raise AttributeError("Test size is too small: training timeseries is of 0 size")

        test_start_index = train_end_index - input_chunk_length

        train_set.append(ts[:train_end_index])
        test_set.append(ts[test_start_index:])

    return train_set, test_set


def train_test_split(
        data: Sequence[TimeSeries],
        test_size: Optional[Union[float, int]] = 0.25) -> Sequence[Sequence[TimeSeries]]:

    if 0 < test_size < 1:
        index = int(len(data) * (1 - test_size))
    else:
        index = test_size

    return data[:index], data[index:]
