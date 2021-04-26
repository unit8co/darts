from warnings import warn
from typing import Sequence, Optional, Union, Tuple
from ..timeseries import TimeSeries


def train_test_split(
        data: Union[TimeSeries, Sequence[TimeSeries]],
        test_size: Optional[Union[float, int]] = 0.25,
        axis: Optional[int] = 0,
        n: Optional[int] = 0,
        horizon: Optional[int] = 0,
        ) -> Union[Tuple[TimeSeries], Tuple[Sequence[TimeSeries]]]:

    """
    Splits the dataset into training and test dataset. Supports splitting along the sample axis and time axis.

    When splitting across the time axis, splitter tries to greedy satisfy the requested test set size, i.e. when one of
    the timeseries in sequence is too small, all samples will go to the test set and the warning will be issued.

    Parameters
    ----------
    data
        original dataset to split into training and test

    test_size
        size of the test set. If the value is between 0 and 1, parameter is treated as a split proportion. Otherwise
        it is treated as a absolute number of samples from each timeseries that will be in the test set. [default = 0.25]

    axis
        Axis to split the dataset on. When 0 (default) it is split on samples. Otherwise, if axis = 1,
        timeseries are split along time axis (columns).

    n
        size of the input

    horizon
        forecast horizon

    Returns
    -------
    tuple of two Sequence[TimeSeries], or tuple of two Timeseries
        Training and test datasets tuple.
    """

    # TODO: support splitting covatiates at the same time
    if not data:
        raise AttributeError('The `data` parameter cannot be empty list.')

    if axis == 0:

        if 0 < test_size < 1:
            index = int(len(data) * (1 - test_size))
        else:
            index = test_size

        return data[:index], data[index:]

    elif axis == 1:

        if horizon == 0 or n == 0:
            raise AttributeError("You need to provide non-zero `horizon` and `n` parameters when axis=1")

        train_set = []
        test_set = []

        for ts in data:
            ts_length = len(ts)
            train_end_index = ts_length - horizon

            if 0 < test_size < 1:
                test_size = int((ts_length - horizon) * (test_size))
            print(train_end_index)
            if train_end_index < n:
                warn("Training timeseries is of 0 size")
            else:
                train_set.append(ts[:train_end_index])

            test_start_index = ts_length - horizon - n - test_size - 1

            if test_start_index < 0:
                test_start_index = 0
                warn("Not enough timesteps to create testset")

            test_set.append(ts[test_start_index:])

        return train_set, test_set

    else:
        raise AttributeError('Wrong value for `axis` parameter. Can be either 0 or 1')
