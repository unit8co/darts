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
        with self.assertRaises(AttributeError,
                               msg="You need to provide non-zero `horizon` and `n` parameters when axis=1"):
            train_test_split(make_dataset(1, 10), axis=1, horizon=1, vertical_split_type='model-aware')

    def test_parameters_for_axis_1_no_horizon(self):
        with self.assertRaises(AttributeError,
                               msg="You need to provide non-zero `horizon` and `n` parameters when axis=1"):
            train_test_split(make_dataset(1, 10), axis=1, input_size=1, vertical_split_type='model-aware')

    # test 3
    def test_empty_dataset(self):
        with self.assertRaises(AttributeError):
            train_test_split([])

    # test 4
    def test_horiz_number_of_samples_too_small(self):
        train_set, test_set = train_test_split(make_dataset(1, 10), axis=1, input_size=4, horizon=7, test_size=1,
                                               vertical_split_type='model-aware')
        with self.assertWarns(UserWarning, msg="Training timeseries is of 0 size"):
            # since dataset is lazy loading, only accessing bad element will trigger the warning
            train_set[0]

    # test 5
    def test_sunny_day_horiz_split(self):
        train_set, test_set = train_test_split(make_dataset(8, 10))

        self.assertTrue(
            verify_shape(train_set, 6, 10) and
            verify_shape(test_set, 2, 10),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_sunny_day_horiz_split_absolute(self):
        train_set, test_set = train_test_split(make_dataset(8, 10), test_size=2)

        self.assertTrue(
            verify_shape(train_set, 6, 10) and
            verify_shape(test_set, 2, 10),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_horiz_split_overindexing_train_set(self):
        train_set, test_set = train_test_split(make_dataset(8, 10))

        with self.assertRaises(IndexError):
            train_set[6]

    def test_horiz_split_last_index_train_set(self):
        train_set, test_set = train_test_split(make_dataset(8, 10))
        # no IndexError is thrown
        train_set[5]

    def test_horiz_split_overindexing_test_set(self):
        train_set, test_set = train_test_split(make_dataset(8, 10))

        with self.assertRaises(IndexError):
            test_set[2]

    def test_horiz_split_last_index_test_set(self):
        train_set, test_set = train_test_split(make_dataset(8, 10))
        # no IndexError is thrown
        train_set[1]

    # test 6
    def test_sunny_day_vertical_split(self):
        train_set, test_set = train_test_split(make_dataset(2, 250), axis=1, input_size=70, horizon=50,
                                               vertical_split_type='model-aware')

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
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=1, test_size=2, input_size=1, horizon=2,
                                               vertical_split_type='model-aware')

        self.assertTrue(
            verify_shape(train_set, 4, 8) and
            verify_shape(test_set, 4, 6),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_negative_test_start_index(self):
        train_set, test_set = train_test_split(make_dataset(1, 10), axis=1, input_size=2, horizon=8, test_size=1,
                                               vertical_split_type='model-aware')
        with self.assertWarns(UserWarning, msg="Not enough timesteps to create testset"):
            test_set[0] # since dataset is lazy loading, only accessing bad element will trigger the warning

    def test_horiz_split_horizon_equal_to_ts_length(self):
        train_set, test_set = train_test_split(make_dataset(1, 10), axis=1, input_size=2, horizon=10, test_size=1,
                                               vertical_split_type='model-aware')
        with self.assertWarns(UserWarning, msg="Not enough timesteps to create testset"):
            test_set[0]  # since dataset is lazy loading, only accessing bad element will trigger the warning

    def test_single_timeseries_no_horizon_no_n(self):
        with self.assertRaises(AttributeError):
            # even if the default axis is 0, but since it is a single timeseries, default axis is 1
            train_test_split(constant_timeseries(123, 10), test_size=2, vertical_split_type='model-aware')

    def test_single_timeseries_sunny_day(self):
        train_set, test_set = train_test_split(constant_timeseries(123, 10), test_size=2, input_size=1, horizon=2,
                                               vertical_split_type = 'model-aware'
                                               )

        self.assertTrue(
            len(train_set) == 8 and len(test_set) == 6,
            "Wrong shapes: training set shape: {}; test set shape {}".format(
                len(train_set), len(test_set))
        )

    def test_multi_timeseries_variable_ts_length_sunny_day(self):
        data = [
            constant_timeseries(123, 10),
            constant_timeseries(123, 100),
            constant_timeseries(123, 1000)
        ]
        train_set, test_set = train_test_split(data, axis=1, test_size=2, input_size=1, horizon=2,
                                               vertical_split_type='model-aware')
        train_lengths = [len(ts) for ts in train_set]
        test_lengths = [len(ts) for ts in test_set]

        self.assertTrue(
            train_lengths == [8, 98, 998] and test_lengths == [6, 6, 6],
            "Wrong shapes: training set shape: {}; test set shape {}".format(
                train_lengths, test_lengths)
        )

    def test_multi_timeseries_variable_ts_length_one_ts_too_small(self):
        data = [
            constant_timeseries(123, 21),
            constant_timeseries(123, 100),
            constant_timeseries(123, 1000)
        ]
        train_set, test_set = train_test_split(data, axis=1, test_size=2, input_size=1, horizon=20,
                                               vertical_split_type='model-aware')

        with self.assertWarns((UserWarning, UserWarning),
                              msg=("Training timeseries is of 0 size", "Not enough timesteps to create testset")):
            # since dataset is lazy loading, only accessing bad element will trigger the warning
            train_set[0]
            test_set[0]

        train_lengths = [len(ts) for ts in train_set]
        test_lengths = [len(ts) for ts in test_set]

        self.assertTrue(
            train_lengths == [1, 80, 980] and test_lengths == [21, 24, 24],
            "Wrong shapes: training set shape: {}; test set shape {}".format(
                train_lengths, test_lengths)
        )

    def test_simple_vertical_split_sunny_day(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=1,
                                               vertical_split_type='simple', test_size=0.2)

        self.assertTrue(
            verify_shape(train_set, 4, 8) and
            verify_shape(test_set, 4, 2),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_simple_vertical_split_sunny_day_absolute_split(self):
        train_set, test_set = train_test_split(make_dataset(4, 10), axis=1,
                                               vertical_split_type='simple', test_size=2)

        self.assertTrue(
            verify_shape(train_set, 4, 8) and
            verify_shape(test_set, 4, 2),
            "Wrong shapes: training set shape: ({}, {}); test set shape ({}, {})".format(
                len(train_set), len(train_set[0]), len(test_set), len(test_set[0]))
        )

    def test_simple_vertical_split_exception_on_bad_param(self):
        # bad value for vertical_split_type
        with self.assertRaises(AttributeError):
            train_set, test_set = train_test_split(make_dataset(4, 10), axis=1,
                                                   vertical_split_type='WRONG_VALUE', test_size=2)

    def test_simple_vertical_split_test_size_too_large(self):

        with self.assertRaises(AttributeError, msg="`test_size` is bigger then timeseries length"):
            train_set, test_set = train_test_split(make_dataset(4, 10), axis=1,
                                                   vertical_split_type='simple', test_size=11)

            train_set[0]
