from ..base_test_class import DartsBaseTestClass
from darts.utils.model_selection import train_test_split
from darts.utils.timeseries_generation import constant_timeseries


def make_dataset(rows, cols):
    return [constant_timeseries(i, cols) for i in range(rows)]


def verify_shape(dataset, rows, cols):
    return (
        len(dataset) == rows and
        all(len(row) == cols for row in dataset)
    )


class ClassTrainTestSplitTestCase(DartsBaseTestClass):
    # test 1
    # test parameters for axis = 0
    def test_parameters_for_axis_0(self):
        train_test_split(make_dataset(2, 10), axis=0, test_size=1)

        # expecting no exception
        self.assertTrue(True)

    # test 2
    # test parameteres for axis = 1
    def test_parameters_for_axis_1_no_n(self):
        with self.assertRaises(AttributeError):
            train_test_split(make_dataset(1, 10), axis=1, horizon=1)

    def test_parameters_for_axis_1_no_horizon(self):
        with self.assertRaises(AttributeError):
            train_test_split(make_dataset(1, 10), axis=1, n=1)

    # test 3
    # test empty data input
    def test_empty_dataset(self):
        with self.assertRaises(AttributeError):
            train_test_split([])

    # test 4
    # number of samples too small, horizontal split
    # input is 1 timeseries of length 10, axis = 0, default split
    def test_horiz_number_of_samples_too_small(self):
        with self.assertRaises(AttributeError):
            train_test_split(make_dataset(1, 10))

    # test 5
    # sunny day horizontal split
    # input is 4 timeseries of length 10, axis = 0 default split
    def test_sunny_day_horiz_split(self):
        train_set, test_set = train_test_split(make_dataset(4, 10))

        self.assertTrue(
            verify_shape(train_set, 2, 10) and
            verify_shape(train_set, 2, 10)
        )

    # test 6
    # timeseries too short, vertical split
    # input is 2 timeseries of length 1, axis = 1, n = 10, horizon = 10, default split .25
    def test_timeseries_too_short_vertical_split(self):
        self.assertTrue(False)

    # test 7
    # sunny day vertical split
    # input is 2 timeseries of length 250, axis = 1, n = 70, horizon = 50, default split .25
    def test_sunny_day_vertical_split(self):
        train_set, test_set = train_test_split(make_dataset(2, 250), axis=1, n=70, horizon=50)

        self.assertTrue(
            verify_shape(train_set, 2, 200) and
            verify_shape(train_set, 2, 150)
        )

    # test 8
    # training set insufficient size
    # input is 2 timeseries of lenght 250
    def test_training_set_insufficient_size(self):
        with self.assertRaises(AttributeError):
            pass

    # test 9
    # test_split absolute number horizontal split
    # input is 4 timeseries of length 10, axis = 0, test_split = 2
    def test_test_split_absolute_number_horiz(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=0, test_size=2)

        self.assertTrue(
            verify_shape(train_set, 2, 10) and
            verify_shape(train_set, 2, 10)
        )

    # test 10
    # test_split absolute number vertical split
    # input is 4 timeseries of length 10, axis = 1, test_split = 2
    def test_test_split_absolute_number_vertical(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=1, test_size=2, n=1, horizon=2)

        self.assertTrue(
            verify_shape(train_set, 4, 8) and
            verify_shape(train_set, 4, 4)
        )
