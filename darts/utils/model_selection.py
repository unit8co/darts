from warnings import warn
from typing import Sequence, Optional, Union, Tuple
from ..timeseries import TimeSeries


class SplitTimeSeriesSequence(Sequence):
    def __init__(self,
                 type: str,
                 data: Union[TimeSeries, Sequence[TimeSeries]],
                 test_size: Optional[Union[float, int]] = 0.25,
                 axis: Optional[int] = 0,
                 input_size: Optional[int] = 0,
                 horizon: Optional[int] = 0,
                 vertical_split_type: Optional[str] = 'simple'):

        if type not in ['train', 'test']:
            raise AttributeError('Value for type parameter should be either `train` or `test`')
        self.type = type

        if not data:
            raise AttributeError('The `data` parameter cannot be empty.')

        if not isinstance(data, Sequence):
            axis = 1
            self.data = [data]  # convert to sequence for unified processing later
            self.single_timeseries = True
        else:
            self.data = data
            self.single_timeseries = False

        self.test_size = test_size
        self.axis = axis
        self.input_size = input_size
        self.horizon = horizon

        if vertical_split_type not in ['simple', 'model-aware']:
            self.vertical_split_type = vertical_split_type

    def _get_horizontal_split_index(self):
        if 0 < self.test_size < 1:
            return int(len(self.data) * (1 - self.test_size))
        else:
            return self.test_size  # TODO: len(self.data) - self.test_size

    def __getitem__(self, i: int) -> TimeSeries:
        if self.axis == 0:
            split_index = self._get_horizontal_split_index()
            if self.type == 'train':
                if i >= split_index: #
                    raise IndexError('Exceeded the size of the sequence.')
                return self.data[i]
            else:
                if i + split_index > len(self.data):
                    raise IndexError('Exceeded the size of the sequence.')
                return self.data[split_index + i]
        else: # axis == 1
            if self.type == 'train':
                return None # TODO
            else:
                return None # TODO

    def __len__(self):
        if self.axis == 0:
            split_index = self._get_horizontal_split_index()
            if self.type == 'train':
                return split_index
            else:
                return len(self.data) - split_index
        else:
            return len(self.data)

    @classmethod
    def make_splitter(cls, data, test_size, axis, input_size, horizon, vertical_split_type):
        return (cls(type='train', data=data, test_size=test_size, axis=axis, input_size=input_size, horizon=horizon, vertical_split_type=vertical_split_type),
                cls(type='test', data=data, test_size=test_size, axis=axis, input_size=input_size, horizon=horizon, vertical_split_type=vertical_split_type))


def train_test_split(
        data: Union[TimeSeries, Sequence[TimeSeries]],
        test_size: Optional[Union[float, int]] = 0.25,
        axis: Optional[int] = 0,
        input_size: Optional[int] = 0,
        horizon: Optional[int] = 0,
        vertical_split_type: Optional[str] = 'simple'
        ) -> Union[Tuple[TimeSeries], Tuple[Sequence[TimeSeries]]]:

    return SplitTimeSeriesSequence.make_splitter(data, test_size, axis, input_size, horizon, vertical_split_type)

def train_test_split_2(
        data: Union[TimeSeries, Sequence[TimeSeries]],
        test_size: Optional[Union[float, int]] = 0.25,
        axis: Optional[int] = 0,
        input_size: Optional[int] = 0,
        horizon: Optional[int] = 0,
        vertical_split_type: Optional[str] = 'simple'
        ) -> Union[Tuple[TimeSeries], Tuple[Sequence[TimeSeries]]]:

    """
    Splits the dataset into training and test dataset. Supports splitting along the sample axis and time axis.

    If the input type is single TimeSeries, then only splitting over time axis is available, thus ``n`` and ``horizon``
    have to be provided.

    When splitting over the time axis, splitter tries to greedy satisfy the requested test set size, i.e. when one of
    the timeseries in sequence is too small, all samples will go to the test set and the warning will be issued.

    Parameters
    ----------
    data
        original dataset to split into training and test

    test_size
        size of the test set. If the value is between 0 and 1, parameter is treated as a split proportion. Otherwise
        it is treated as a absolute number of samples from each timeseries that will be in the test set. [default = 0.25]

    axis
        Axis to split the dataset on. When 0 (default) it is split on samples. Otherwise, if ``axis = 1``,
        timeseries are split along time axis (columns). Note that for single timeseries the default option is 1 (0 makes
        no sense). [default: 0 for sequence of timeseries, 1 for timeseries]

    input_size
        size of the input [default: 0]

    horizon
        forecast horizon [default: 0]

    vertical_split_type
        can be either ``simple``, where the exact number from test size will be deducted from timeseries for test set and
        remaining will go to training set; or ``model-aware``, where you have to provide ``input_size`` and ``horizon``
        as well. Note, that second option is more efficient timestep-wise, since training and test sets will be
        partially overlapping. [default: ``simple``]

    Returns
    -------
    tuple of two Sequence[TimeSeries], or tuple of two Timeseries
        Training and test datasets tuple.
    """

    # TODO: support splitting covatiates at the same time
    if not data:
        raise AttributeError('The `data` parameter cannot be empty.')

    if not isinstance(data, Sequence):
        axis = 1
        data = [data] # convert to sequence for unified processing later
        single_timeseries = True
    else:
        single_timeseries = False

    if axis == 0:

        if 0 < test_size < 1:
            index = int(len(data) * (1 - test_size))
        else:
            index = test_size

        return data[:index], data[index:]

    elif axis == 1:

        train_set = list()
        test_set = list()

        if vertical_split_type == 'simple':

            for ts in data:
                ts_length = len(ts)

                if 0 < test_size < 1:
                    test_size = int(ts_length * test_size)

                test_start_index = ts_length - test_size
                train_end_index =  test_start_index

                train_set.append(ts[:train_end_index])
                test_set.append(ts[test_start_index:])

        elif vertical_split_type == 'model-aware':

            if horizon == 0 or input_size == 0:
                raise AttributeError("You need to provide non-zero `horizon` and `n` parameters when axis=1")

            for ts in data:
                ts_length = len(ts)
                train_end_index = ts_length - horizon

                if 0 < test_size < 1:
                    test_size = int((ts_length - horizon) * (test_size))

                if train_end_index < input_size:
                    warn("Training timeseries is of 0 size")
                else:
                    train_set.append(ts[:train_end_index])

                test_start_index = ts_length - horizon - input_size - test_size - 1

                if test_start_index < 0:
                    test_start_index = 0
                    warn("Not enough timesteps to create testset")
                else:
                    test_set.append(ts[test_start_index:])

        else:
            raise AttributeError('`vertical_split_type` can be eiter `simple` or `model-aware`.')

        if single_timeseries:
            return train_set[0], test_set[0]
        else:
            return train_set, test_set

    else:
        raise AttributeError('Wrong value for `axis` parameter. Can be either 0 or 1')
