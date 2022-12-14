from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from darts import TimeSeries
from darts.ad.aggregators.and_aggregator import AndAggregator
from darts.ad.aggregators.ensemble_sklearn_aggregator import EnsembleSklearnAggregator
from darts.ad.aggregators.or_aggregator import OrAggregator
from darts.models import MovingAverage
from darts.tests.base_test_class import DartsBaseTestClass

list_NonFittableAggregator = [
    OrAggregator(),
    AndAggregator(),
]

list_FittableAggregator = [
    EnsembleSklearnAggregator(model=GradientBoostingClassifier())
]


class ADAggregatorsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_anomalies1 = np.random.choice(a=[0, 1], size=100, p=[0.6, 0.4])
    anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies1)

    np_anomalies2 = np.random.choice(a=[0, 1], size=100, p=[0.6, 0.4])
    anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies2)

    np_anomalies3 = np.random.choice(a=[0, 1], size=100, p=[0.6, 0.4])
    anomalies3 = TimeSeries.from_times_and_values(train._time_index, np_anomalies2)

    np_real_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.6, 0.4])
    real_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_real_anomalies
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 1, 5])
    probabilistic = TimeSeries.from_values(np_probabilistic)

    # multivariate series
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies3 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_mts_real_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    mts_real_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_mts_real_anomalies
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 2, 5])
    mts_probabilistic = TimeSeries.from_values(np_probabilistic)

    def test_DetectNonFittableAggregator(self):

        aggregator = OrAggregator()

        # Check return types
        # Check if return type is TimeSeries when input is Sequence of univariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.anomalies1, self.anomalies2]), TimeSeries
            )
        )
        # Check if return type is TimeSeries when input is Sequence of multivariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.mts_anomalies1, self.mts_anomalies2]),
                TimeSeries,
            )
        )

    def test_DetectFittableAggregator(self):
        aggregator = EnsembleSklearnAggregator(model=GradientBoostingClassifier())

        # Check return types
        aggregator.fit(self.real_anomalies, [self.anomalies1, self.anomalies2])
        # Check if return type is TimeSeries when input is Sequence of univariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.anomalies1, self.anomalies2]), TimeSeries
            )
        )

        aggregator.fit(
            self.mts_real_anomalies, [self.mts_anomalies1, self.mts_anomalies2]
        )
        # Check if return type is TimeSeries when input is Sequence of multivariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.mts_anomalies1, self.mts_anomalies2]),
                TimeSeries,
            )
        )

    def test_eval_accuracy(self):

        aggregator = AndAggregator()

        # Check return types
        # Check if return type is float when input is a Sequence of series
        self.assertTrue(
            isinstance(
                aggregator.eval_accuracy(
                    self.real_anomalies, [self.anomalies1, self.anomalies2]
                ),
                float,
            )
        )
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                aggregator.eval_accuracy(
                    self.mts_real_anomalies, [self.mts_anomalies1, self.mts_anomalies2]
                ),
                Sequence,
            )
        )

        with self.assertRaises(ValueError):
            # intersection between 'actual_anomalies' and the series in the sequence 'list_series'
            # must be non empty (univariate)
            aggregator.eval_accuracy(
                self.np_real_anomalies[60:],
                [self.anomalies1[:40], self.anomalies1[:40]],
            )

        with self.assertRaises(ValueError):
            # intersection between 'actual_anomalies' and the series in the sequence 'list_series'
            # must be non empty (multivariate)
            aggregator.eval_accuracy(
                self.mts_real_anomalies[60:],
                [self.mts_anomalies1[:40], self.mts_anomalies1[:40]],
            )

        with self.assertRaises(ValueError):
            # window parameter must be smaller than the length of the input (len = 100)
            aggregator.eval_accuracy(
                self.np_real_anomalies, [self.anomalies1, self.anomalies2], window=101
            )

    def test_NonFittableAggregator(self):

        for aggregator in list_NonFittableAggregator:
            # Check if trainable is False, being a NonFittableAggregator
            self.assertTrue(not aggregator.trainable)

            with self.assertRaises(ValueError):
                # predict on sequence with only 1 series
                aggregator.predict([self.anomalies1])

            with self.assertRaises(ValueError):
                # predict on sequence on 1 series rather than a sequence
                aggregator.predict(self.anomalies1)

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.predict([self.anomalies1, self.train])

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.eval_accuracy(self.train, [self.anomalies1, self.anomalies1])

            with self.assertRaises(ValueError):
                # input a probabilistic series (univariate)
                aggregator.predict([self.anomalies1, self.probabilistic])

            with self.assertRaises(ValueError):
                # input a probabilistic series (multivariate)
                aggregator.predict([self.mts_anomalies1, self.mts_probabilistic])

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.predict([self.mts_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.predict([self.mts_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty
                aggregator.predict([self.anomalies1[:40], self.anomalies1[60:]])

            # Check width return
            # Check if return type is the same number of width as 'actual_anomalies' (multivariate)
            self.assertTrue(
                len(
                    aggregator.eval_accuracy(
                        self.mts_real_anomalies,
                        [self.mts_anomalies1, self.mts_anomalies2],
                    )
                ),
                self.mts_real_anomalies.width,
            )

    def test_FittableAggregator(self):

        for aggregator in list_FittableAggregator:

            # Need to call fit() before calling predict()
            with self.assertRaises(ValueError):
                aggregator.predict([self.anomalies1, self.anomalies2])

            # Check if trainable is True, being a FittableAggregator
            self.assertTrue(aggregator.trainable)

            # Check if _fit_called is False
            self.assertTrue(not aggregator._fit_called)

            with self.assertRaises(ValueError):
                # fit on sequence with series that have different width
                aggregator.fit(
                    self.real_anomalies, [self.anomalies1, self.mts_anomalies1]
                )

            with self.assertRaises(ValueError):
                # fit on sequence with only 1 series
                aggregator.fit(self.real_anomalies, [self.anomalies1])

            with self.assertRaises(ValueError):
                # fit on 1 series rather than a sequence
                aggregator.fit(self.real_anomalies, self.anomalies1)

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.fit(self.real_anomalies, [self.anomalies1, self.train])

            with self.assertRaises(ValueError):
                # input a probabilistic series (univariate)
                aggregator.fit(
                    self.real_anomalies, [self.anomalies1, self.probabilistic]
                )

            with self.assertRaises(ValueError):
                # input a probabilistic series (multivariate)
                aggregator.fit(
                    self.mts_real_anomalies,
                    [self.mts_anomalies1, self.mts_probabilistic],
                )

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.fit(self.mts_real_anomalies, [self.mts_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.fit(self.mts_real_anomalies, [self.mts_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (univariate)
                aggregator.fit(
                    self.real_anomalies, [self.anomalies1[:40], self.anomalies1[60:]]
                )

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (multivariate)
                aggregator.fit(
                    self.mts_real_anomalies,
                    [self.mts_anomalies1[:40], self.mts_anomalies1[60:]],
                )

            # case1: fit on UTS
            aggregator1 = aggregator
            aggregator1.fit(self.real_anomalies, [self.anomalies1, self.anomalies2])

            # Check if _fit_called is True after being fitted
            self.assertTrue(aggregator1._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.mts_anomalies1, self.mts_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.anomalies1, self.mts_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.mts_anomalies1, self.anomalies2])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator1.predict([self.anomalies1])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator1.predict([self.anomalies1, self.anomalies2, self.anomalies3])

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.predict([self.anomalies1, self.train])

            with self.assertRaises(ValueError):
                # input a probabilistic series (univariate)
                aggregator.predict([self.anomalies1, self.probabilistic])

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.predict([self.anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.predict([self.anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (univariate)
                aggregator.predict([self.anomalies1[:40], self.anomalies1[60:]])

            # case2: fit on MTS
            aggregator2 = aggregator
            aggregator2.fit(
                self.mts_real_anomalies, [self.mts_anomalies1, self.mts_anomalies2]
            )

            # Check if _fit_called is True after being fitted
            self.assertTrue(aggregator2._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.anomalies1, self.anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.anomalies1, self.mts_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.mts_anomalies1, self.anomalies2])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator2.predict([self.mts_anomalies1])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator2.predict(
                    [self.mts_anomalies1, self.mts_anomalies2, self.mts_anomalies3]
                )

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.predict([self.mts_anomalies1, self.mts_train])

            with self.assertRaises(ValueError):
                # input a probabilistic series (univariate)
                aggregator.predict([self.anomalies1, self.mts_probabilistic])

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.predict([self.mts_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.predict([self.mts_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (univariate)
                aggregator.predict([self.mts_anomalies1[:40], self.mts_anomalies1[60:]])

            # Check width return
            # Check if return type is the same number of width as 'actual_anomalies' (multivariate)
            self.assertTrue(
                len(
                    aggregator2.eval_accuracy(
                        self.mts_real_anomalies,
                        [self.mts_anomalies1, self.mts_anomalies2],
                    )
                ),
                self.mts_real_anomalies.width,
            )

    def test_OrAggregator(self):

        aggregator = OrAggregator()

        # univariate case
        # aggregator must found 71 anomalies in the input [anomalies1, anomalies2]
        self.assertEqual(
            aggregator.predict([self.anomalies1, self.anomalies2])
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            71,
        )

        # aggregator must have an accuracy of 0.49 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies,
                [self.anomalies1, self.anomalies2],
                metric="accuracy",
            ),
            0.49,
            delta=1e-05,
        )
        # aggregator must have an recall of 0.717391 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies, [self.anomalies1, self.anomalies2], metric="recall"
            ),
            0.717391,
            delta=1e-05,
        )
        # aggregator must have an f1 of 0.56410 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies, [self.anomalies1, self.anomalies2], metric="f1"
            ),
            0.56410,
            delta=1e-05,
        )
        # aggregator must have an precision of 0.46478 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies,
                [self.anomalies1, self.anomalies2],
                metric="precision",
            ),
            0.46478,
            delta=1e-05,
        )

        # multivariate case
        binary_detection = aggregator.predict(
            [self.mts_anomalies1, self.mts_anomalies2]
        )

        # aggregator must found 76 anomalies on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertEqual(
            binary_detection["0"].sum(axis=0).all_values().flatten()[0], 76
        )
        # aggregator must found 81 anomalies on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertEqual(
            binary_detection["1"].sum(axis=0).all_values().flatten()[0], 81
        )

        acc = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="accuracy",
        )
        # aggregator must have an accuracy of 0.25 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(acc[0], 0.25, delta=1e-05)
        # aggregator must have an accuracy of 0.26 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(acc[1], 0.26, delta=1e-05)

        precision = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="precision",
        )
        # aggregator must have an precision of 0.06578 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(precision[0], 0.06578, delta=1e-05)
        # aggregator must have an precision of 0.08641 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(precision[1], 0.08641, delta=1e-05)

        recall = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="recall",
        )
        # aggregator must have an recall of 0.55555 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(recall[0], 0.55555, delta=1e-05)
        # aggregator must have an recall of 1.0 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(recall[1], 1.0, delta=1e-05)

        f1 = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="f1",
        )
        # aggregator must have an f1 of 0.11764 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(f1[0], 0.11764, delta=1e-05)
        # aggregator must have an recall of 0.15909 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(f1[1], 0.15909, delta=1e-05)

    def test_AndAggregator(self):

        aggregator = AndAggregator()

        # univariate case
        # aggregator must found 14 anomalies in the input [anomalies1, anomalies2]
        self.assertEqual(
            aggregator.predict([self.anomalies1, self.anomalies2])
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            14,
        )

        # aggregator must have an accuracy of 0.5 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies,
                [self.anomalies1, self.anomalies2],
                metric="accuracy",
            ),
            0.5,
            delta=1e-05,
        )
        # aggregator must have an recall of 0.108695 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies, [self.anomalies1, self.anomalies2], metric="recall"
            ),
            0.108695,
            delta=1e-05,
        )
        # aggregator must have an f1 of 0.166666 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies, [self.anomalies1, self.anomalies2], metric="f1"
            ),
            0.166666,
            delta=1e-05,
        )
        # aggregator must have an precision of 0.35714 for the input [anomalies1, anomalies2]
        self.assertAlmostEqual(
            aggregator.eval_accuracy(
                self.real_anomalies,
                [self.anomalies1, self.anomalies2],
                metric="precision",
            ),
            0.35714,
            delta=1e-05,
        )

        # multivariate case
        binary_detection = aggregator.predict(
            [self.mts_anomalies1, self.mts_anomalies2]
        )

        # aggregator must found 23 anomalies on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertEqual(
            binary_detection["0"].sum(axis=0).all_values().flatten()[0], 23
        )
        # aggregator must found 28 anomalies on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertEqual(
            binary_detection["1"].sum(axis=0).all_values().flatten()[0], 28
        )

        acc = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="accuracy",
        )
        # aggregator must have an accuracy of 0.72 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(acc[0], 0.72, delta=1e-05)
        # aggregator must have an accuracy of 0.69 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(acc[1], 0.69, delta=1e-05)

        precision = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="recall",
        )
        # aggregator must have an recall of 0.22222 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(precision[0], 0.22222, delta=1e-05)
        # aggregator must have an recall of 0.28571 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(precision[1], 0.28571, delta=1e-05)

        recall = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="precision",
        )
        # aggregator must have an recall of 0.08695 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(recall[0], 0.08695, delta=1e-05)
        # aggregator must have an recall of 0.07142 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(recall[1], 0.07142, delta=1e-05)

        f1 = aggregator.eval_accuracy(
            self.mts_real_anomalies,
            [self.mts_anomalies1, self.mts_anomalies2],
            metric="f1",
        )
        # aggregator must have an f1 of 0.125 on the first width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(f1[0], 0.125, delta=1e-05)
        # aggregator must have an recall of 0.11428 on the second width of the results on
        # [mts_anomalies1, mts_anomalies2]
        self.assertAlmostEqual(f1[1], 0.11428, delta=1e-05)

    def test_EnsembleSklearn(self):

        # Need to input an EnsembleSklearn model
        with self.assertRaises(ValueError):
            EnsembleSklearnAggregator(model=MovingAverage(window=10))
