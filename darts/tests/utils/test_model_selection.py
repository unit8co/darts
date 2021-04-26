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
    def test_parameters_for_axis_0(self):
        train_test_split(make_dataset(2, 10), axis=0, test_size=1)

        # expecting no exception
        self.assertTrue(True)

    # test 2
    def test_parameters_for_axis_1_no_n(self):
        with self.assertRaises(AttributeError):
            train_test_split(make_dataset(1, 10), axis=1, horizon=1)

    def test_parameters_for_axis_1_no_horizon(self):
        with self.assertRaises(AttributeError):
            train_test_split(make_dataset(1, 10), axis=1, n=1)

    # test 3
    def test_empty_dataset(self):
        with self.assertRaises(AttributeError):
            train_test_split([])

    # test 4
    def test_horiz_number_of_samples_too_small(self):
        with self.assertWarns(UserWarning, msg="Training timeseries is of 0 size"):
            train_test_split(make_dataset(1, 10), axis=1, n=4, horizon=7, test_size=1)

    # test 5
    def test_sunny_day_horiz_split(self):
        train_set, test_set = train_test_split(make_dataset(4, 10))

        self.assertTrue(
            verify_shape(train_set, 3, 10) and
            verify_shape(test_set, 1, 10)
        )

    # test 6
    def test_sunny_day_vertical_split(self):
        train_set, test_set = train_test_split(make_dataset(2, 250), axis=1, n=70, horizon=50)

        self.assertTrue(
            verify_shape(train_set, 2, 200) and
            verify_shape(test_set, 2, 171),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    # test 7
    def test_test_split_absolute_number_horiz(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=0, test_size=2)

        self.assertTrue(
            verify_shape(train_set, 2, 10) and
            verify_shape(test_set, 2, 10)
        )

    # test 8
    def test_test_split_absolute_number_vertical(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=1, test_size=2, n=1, horizon=2)

        self.assertTrue(
            verify_shape(train_set, 4, 8) and
            verify_shape(test_set, 4, 6),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_negative_test_start_index(self):
        with self.assertWarns(UserWarning, msg="Not enough timesteps to create testset"):
            train_test_split(make_dataset(1, 10), axis=1, n=2, horizon=8, test_size=1)

    def test_horiz_split_horizon_equal_to_ts_length(self):
        with self.assertWarns(UserWarning, msg="Not enough timesteps to create testset"):
            train_test_split(make_dataset(1, 10), axis=1, n=2, horizon=10, test_size=1)

    # def test_single_timeseries(self):
    #     train_test_split(constant_timeseries(123, 10), axis=1, horizon=2, n=3, test_size=2)