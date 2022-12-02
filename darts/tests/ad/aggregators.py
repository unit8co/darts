from typing import Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from darts import TimeSeries
from darts.ad import aggregators as Agg
from darts.models import MovingAverage
from darts.tests.base_test_class import DartsBaseTestClass

list_NonFittableAggregator = [
    Agg.OrAggregator(),
    Agg.AndAggregator(),
]

list_FittableAggregator = [
    Agg.EnsembleSklearnAggregator(model=GradientBoostingClassifier())
]


class ADAggregatorsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_anomalies1 = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies1)

    np_anomalies2 = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies2)

    np_anomalies3 = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies3 = TimeSeries.from_times_and_values(train._time_index, np_anomalies2)

    np_anomalies4 = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies4 = TimeSeries.from_times_and_values(train._time_index, np_anomalies2)

    np_real_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    real_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_real_anomalies
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 1, 5])
    probabilistic = TimeSeries.from_values(np_probabilistic)

    # multivariate series
    np_MTS_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    MTS_train = TimeSeries.from_values(np_MTS_train)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    MTS_anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    MTS_anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    MTS_anomalies3 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_MTS_real_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    MTS_real_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_MTS_real_anomalies
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 2, 5])
    MTS_probabilistic = TimeSeries.from_values(np_probabilistic)

    def test_DetectNonFittableAggregator(self):

        aggregator = Agg.OrAggregator()

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
                aggregator.predict([self.MTS_anomalies1, self.MTS_anomalies2]),
                TimeSeries,
            )
        )

    def test_DetectFittableAggregator(self):
        aggregator = Agg.EnsembleSklearnAggregator(model=GradientBoostingClassifier())

        # Check return types
        aggregator.fit(self.real_anomalies, [self.anomalies1, self.anomalies2])
        # Check if return type is TimeSeries when input is Sequence of univariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.anomalies1, self.anomalies2]), TimeSeries
            )
        )

        aggregator.fit(
            self.MTS_real_anomalies, [self.MTS_anomalies1, self.MTS_anomalies2]
        )
        # Check if return type is TimeSeries when input is Sequence of multivariate series
        self.assertTrue(
            isinstance(
                aggregator.predict([self.MTS_anomalies1, self.MTS_anomalies2]),
                TimeSeries,
            )
        )

    def test_eval_accuracy(self):

        aggregator = Agg.AndAggregator()

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
                    self.MTS_real_anomalies, [self.MTS_anomalies1, self.MTS_anomalies2]
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
                self.MTS_real_anomalies[60:],
                [self.MTS_anomalies1[:40], self.MTS_anomalies1[:40]],
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
                aggregator.predict([self.MTS_anomalies1, self.MTS_probabilistic])

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.predict([self.MTS_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.predict([self.MTS_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty
                aggregator.predict([self.anomalies1[:40], self.anomalies1[60:]])

            # Check width return
            # Check if return type is the same number of width as 'actual_anomalies' (multivariate)
            self.assertTrue(
                len(
                    aggregator.eval_accuracy(
                        self.MTS_real_anomalies,
                        [self.MTS_anomalies1, self.MTS_anomalies2],
                    )
                ),
                self.MTS_real_anomalies.width,
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
                    self.real_anomalies, [self.anomalies1, self.MTS_anomalies1]
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
                    self.MTS_real_anomalies,
                    [self.MTS_anomalies1, self.MTS_probabilistic],
                )

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.fit(self.MTS_real_anomalies, [self.MTS_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.fit(self.MTS_real_anomalies, [self.MTS_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (univariate)
                aggregator.fit(
                    self.real_anomalies, [self.anomalies1[:40], self.anomalies1[60:]]
                )

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (multivariate)
                aggregator.fit(
                    self.MTS_real_anomalies,
                    [self.MTS_anomalies1[:40], self.MTS_anomalies1[60:]],
                )

            # case1: fit on UTS
            aggregator1 = aggregator
            aggregator1.fit(self.real_anomalies, [self.anomalies1, self.anomalies2])

            # Check if _fit_called is True after being fitted
            self.assertTrue(aggregator1._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.MTS_anomalies1, self.MTS_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.anomalies1, self.MTS_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator1.predict([self.MTS_anomalies1, self.anomalies2])

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
                self.MTS_real_anomalies, [self.MTS_anomalies1, self.MTS_anomalies2]
            )

            # Check if _fit_called is True after being fitted
            self.assertTrue(aggregator2._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.anomalies1, self.anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.anomalies1, self.MTS_anomalies2])

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                aggregator2.predict([self.MTS_anomalies1, self.anomalies2])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator2.predict([self.MTS_anomalies1])

            with self.assertRaises(ValueError):
                # series must be same length as series used for training
                aggregator2.predict(
                    [self.MTS_anomalies1, self.MTS_anomalies2, self.MTS_anomalies3]
                )

            with self.assertRaises(ValueError):
                # input a non binary series
                aggregator.predict([self.MTS_anomalies1, self.MTS_train])

            with self.assertRaises(ValueError):
                # input a probabilistic series (univariate)
                aggregator.predict([self.anomalies1, self.MTS_probabilistic])

            with self.assertRaises(ValueError):
                # input an element that is not a series (string)
                aggregator.predict([self.MTS_anomalies1, "random"])

            with self.assertRaises(ValueError):
                # input an element that is not a series (number)
                aggregator.predict([self.MTS_anomalies1, 1])

            with self.assertRaises(ValueError):
                # intersection between inputs must be non empty (univariate)
                aggregator.predict([self.MTS_anomalies1[:40], self.MTS_anomalies1[60:]])

            # Check width return
            # Check if return type is the same number of width as 'actual_anomalies' (multivariate)
            self.assertTrue(
                len(
                    aggregator2.eval_accuracy(
                        self.MTS_real_anomalies,
                        [self.MTS_anomalies1, self.MTS_anomalies2],
                    )
                ),
                self.MTS_real_anomalies.width,
            )

    def test_EnsembleSklearn(self):

        # Need to inout a EnsembleSklearn model
        with self.assertRaises(ValueError):
            Agg.EnsembleSklearnAggregator(model=MovingAverage(window=10))
