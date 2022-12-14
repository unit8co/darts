from typing import Sequence

import numpy as np

from darts import TimeSeries
from darts.ad.detectors.quantile_detector import QuantileDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector
from darts.tests.base_test_class import DartsBaseTestClass

list_NonFittableDetector = [ThresholdDetector(low=0.2)]

list_FittableDetector = [QuantileDetector(low=0.2)]


class ADDetectorsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_test = np.random.normal(loc=10, scale=1, size=100)
    test = TimeSeries.from_times_and_values(train._time_index, np_test)

    np_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    # multivariate series
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)

    np_mts_test = np.random.normal(loc=[10, 5], scale=[1, 1.5], size=[100, 2])
    mts_test = TimeSeries.from_times_and_values(mts_train._time_index, np_mts_test)

    np_mts_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    mts_anomalies = TimeSeries.from_times_and_values(
        mts_train._time_index, np_mts_anomalies
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 1, 5])
    probabilistic = TimeSeries.from_values(np_probabilistic)

    def test_DetectNonFittableDetector(self):

        detector = ThresholdDetector(low=0.2)

        # Check return types
        # Check if return TimeSeries is float when input is a series
        self.assertTrue(isinstance(detector.detect(self.test), TimeSeries))

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(isinstance(detector.detect([self.test]), Sequence))

        # Check if return TimeSeries is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect(self.mts_test), TimeSeries))

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect([self.mts_test]), Sequence))

        with self.assertRaises(ValueError):
            # Input cannot be probabilistic
            detector.detect(self.probabilistic)

    def test_DetectFittableDetector(self):
        detector = QuantileDetector(low=0.2)

        # Check return types

        detector.fit(self.train)
        # Check if return type is float when input is a series
        self.assertTrue(isinstance(detector.detect(self.test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of series
        self.assertTrue(isinstance(detector.detect([self.test]), Sequence))

        detector.fit(self.mts_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect(self.mts_test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of multivariate series
        self.assertTrue(isinstance(detector.detect([self.mts_test]), Sequence))

        with self.assertRaises(ValueError):
            # Input cannot be probabilistic
            detector.detect(self.probabilistic)

    def test_eval_accuracy(self):

        detector = ThresholdDetector(low=0.2)

        # Check return types
        # Check if return type is float when input is a series
        self.assertTrue(
            isinstance(detector.eval_accuracy(self.anomalies, self.test), float)
        )

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(
            isinstance(detector.eval_accuracy(self.anomalies, [self.test]), Sequence)
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                detector.eval_accuracy(self.mts_anomalies, self.mts_test), Sequence
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                detector.eval_accuracy(self.mts_anomalies, [self.mts_test]), Sequence
            )
        )

        with self.assertRaises(ValueError):
            # Input cannot be probabilistic
            detector.eval_accuracy(self.anomalies, self.probabilistic)

    def test_NonFittableDetector(self):

        for detector in list_NonFittableDetector:
            # Check if trainable is False, being a NonFittableDetector
            self.assertTrue(not detector.trainable)

    def test_FittableDetector(self):

        for detector in list_FittableDetector:

            # Need to call fit() before calling detect()
            with self.assertRaises(ValueError):
                detector.detect(self.test)

            # Check if trainable is True, being a FittableDetector
            self.assertTrue(detector.trainable)

            # Check if _fit_called is False
            self.assertTrue(not detector._fit_called)

            with self.assertRaises(ValueError):
                # fit on sequence with series that have different width
                detector.fit([self.train, self.mts_train])

            with self.assertRaises(ValueError):
                # Input cannot be probabilistic
                detector.fit(self.probabilistic)

            # case1: fit on UTS
            detector1 = detector
            detector1.fit(self.train)

            # Check if _fit_called is True after being fitted
            self.assertTrue(detector1._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                detector1.detect(self.mts_test)

            # case2: fit on MTS
            detector2 = detector
            detector2.fit(self.mts_test)

            # Check if _fit_called is True after being fitted
            self.assertTrue(detector2._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                detector2.detect(self.train)

    def test_QuantileDetector(self):

        # Need to have at least one parameter (low, high) not None
        with self.assertRaises(ValueError):
            QuantileDetector()
        with self.assertRaises(ValueError):
            QuantileDetector(low=None, high=None)

        # Parameter low must be int or float
        with self.assertRaises(ValueError):
            QuantileDetector(low="0.5")
        with self.assertRaises(ValueError):
            QuantileDetector(low=[0.5])

        # Parameter high must be int or float
        with self.assertRaises(ValueError):
            QuantileDetector(high="0.5")
        with self.assertRaises(ValueError):
            QuantileDetector(high=[0.5])

        # Parameter must be bewteen 0 and 1
        with self.assertRaises(ValueError):
            QuantileDetector(high=1.1)
        with self.assertRaises(ValueError):
            QuantileDetector(high=-0.2)
        with self.assertRaises(ValueError):
            QuantileDetector(low=1.1)
        with self.assertRaises(ValueError):
            QuantileDetector(low=-0.2)

        detector = QuantileDetector(low=0.05, high=0.95)
        detector.fit(self.train)

        binary_detection = detector.detect(self.test)

        # Return of .detect() must be binary
        self.assertTrue(
            np.array_equal(
                binary_detection.values(copy=False),
                binary_detection.values(copy=False).astype(bool),
            )
        )

        # Return of .detect() must be same len as input
        self.assertTrue(len(binary_detection) == len(self.test))

        # univariate test
        # detector parameter 'abs_low_' must be equal to 9.13658 when trained on the series 'train'
        self.assertAlmostEqual(detector.abs_low_.flatten()[0], 9.13658, delta=1e-05)

        # detector parameter 'abs_high_' must be equal to 10.74007 when trained on the series 'train'
        self.assertAlmostEqual(detector.abs_high_.flatten()[0], 10.74007, delta=1e-05)

        # detector must found 10 anomalies in the series 'train'
        self.assertEqual(
            detector.detect(self.train).sum(axis=0).all_values().flatten()[0], 10
        )

        # detector must found 42 anomalies in the series 'test'
        self.assertEqual(binary_detection.sum(axis=0).all_values().flatten()[0], 42)

        # detector must have an accuracy of 0.57 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="accuracy"),
            0.57,
            delta=1e-05,
        )
        # detector must have an recall of 0.4 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="recall"),
            0.4,
            delta=1e-05,
        )
        # detector must have an f1 of 0.08510 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="f1"),
            0.08510,
            delta=1e-05,
        )
        # detector must have an precision of 0.04761 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="precision"),
            0.04761,
            delta=1e-05,
        )

        # multivariate test
        detector = QuantileDetector(low=0.05, high=0.95)
        detector.fit(self.mts_train)
        binary_detection = detector.detect(self.mts_test)

        # width of output must be equal to 2 (same dimension as input)
        self.assertEqual(binary_detection.width, 2)
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="accuracy"
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="recall"
                )
            ),
            2,
        )
        self.assertEqual(
            len(detector.eval_accuracy(self.mts_anomalies, self.mts_test, metric="f1")),
            2,
        )
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="precision"
                )
            ),
            2,
        )

        abs_low_ = detector.abs_low_.flatten()
        abs_high_ = detector.abs_high_.flatten()

        # detector parameter 'abs_high_' must be equal to 10.83047 when trained on the series 'train' for the 1st width
        self.assertAlmostEqual(abs_high_[0], 10.83047, delta=1e-05)
        # detector parameter 'abs_high_' must be equal to 6.47822 when trained on the series 'train' for the 2nd width
        self.assertAlmostEqual(abs_high_[1], 6.47822, delta=1e-05)

        # detector parameter 'abs_low_' must be equal to 9.20248 when trained on the series 'train' for the 1st width
        self.assertAlmostEqual(abs_low_[0], 9.20248, delta=1e-05)
        # detector parameter 'abs_low_' must be equal to 3.61853 when trained on the series 'train' for the 2nd width
        self.assertAlmostEqual(abs_low_[1], 3.61853, delta=1e-05)

        # detector must found 37 anomalies on the first width of the series 'test'
        self.assertEqual(
            binary_detection["0"].sum(axis=0).all_values().flatten()[0], 37
        )
        # detector must found 38 anomalies on the second width of the series 'test'
        self.assertEqual(
            binary_detection["1"].sum(axis=0).all_values().flatten()[0], 38
        )

        acc = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="accuracy"
        )
        # detector must have an accuracy of 0.58 on the first width of the series 'mts_test'
        self.assertAlmostEqual(acc[0], 0.58, delta=1e-05)
        # detector must have an accuracy of 0.58 on the second width of the series 'mts_test'
        self.assertAlmostEqual(acc[1], 0.58, delta=1e-05)

        precision = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="precision"
        )
        # detector must have an precision of 0.08108 on the first width of the series 'mts_test'
        self.assertAlmostEqual(precision[0], 0.08108, delta=1e-05)
        # detector must have an precision of 0.07894 on the second width of the series 'mts_test'
        self.assertAlmostEqual(precision[1], 0.07894, delta=1e-05)

        recall = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="recall"
        )
        # detector must have an recall of 0.2727 on the first width of the series 'mts_test'
        self.assertAlmostEqual(recall[0], 0.27272, delta=1e-05)
        # detector must have an recall of 0.3 on the second width of the series 'mts_test'
        self.assertAlmostEqual(recall[1], 0.3, delta=1e-05)

        f1 = detector.eval_accuracy(self.mts_anomalies, self.mts_test, metric="f1")
        # detector must have an f1 of 0.125 on the first width of the series 'mts_test'
        self.assertAlmostEqual(f1[0], 0.125, delta=1e-05)
        # detector must have an recall of 0.125 on the second width of the series 'mts_test'
        self.assertAlmostEqual(f1[1], 0.125, delta=1e-05)

    def test_ThresholdDetector(self):

        # Parameters
        # Need to have at least one parameter (low, high) not None
        with self.assertRaises(ValueError):
            ThresholdDetector()
        with self.assertRaises(ValueError):
            ThresholdDetector(low=None, high=None)

        # Parameter low must be int or float
        with self.assertRaises(ValueError):
            ThresholdDetector(low="1")
        with self.assertRaises(ValueError):
            ThresholdDetector(low=[1])

        # Parameter high must be int or float
        with self.assertRaises(ValueError):
            ThresholdDetector(high="1")
        with self.assertRaises(ValueError):
            ThresholdDetector(high=[1])

        detector = ThresholdDetector(low=9.5, high=10.5)
        binary_detection = detector.detect(self.test)

        # Return of .detect() must be binary
        self.assertTrue(
            np.array_equal(
                binary_detection.values(copy=False),
                binary_detection.values(copy=False).astype(bool),
            )
        )

        # Return of .detect() must be same len as input
        self.assertTrue(len(binary_detection) == len(self.test))

        # univariate test
        # detector must found 42 anomalies in the series 'test'
        self.assertEqual(binary_detection.sum(axis=0).all_values().flatten()[0], 58)
        # detector must have an accuracy of 0.41 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="accuracy"),
            0.41,
            delta=1e-05,
        )
        # detector must have an recall of 0.4 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="recall"),
            0.4,
            delta=1e-05,
        )
        # detector must have an f1 of 0.06349 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="f1"),
            0.06349,
            delta=1e-05,
        )
        # detector must have an precision of 0.03448 for the series 'test'
        self.assertAlmostEqual(
            detector.eval_accuracy(self.anomalies, self.test, metric="precision"),
            0.03448,
            delta=1e-05,
        )

        # multivariate test
        detector = ThresholdDetector(low=4.8, high=10.5)
        binary_detection = detector.detect(self.mts_test)

        # width of output must be equal to 2 (same dimension as input)
        self.assertEqual(binary_detection.width, 2)
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="accuracy"
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="recall"
                )
            ),
            2,
        )
        self.assertEqual(
            len(detector.eval_accuracy(self.mts_anomalies, self.mts_test, metric="f1")),
            2,
        )
        self.assertEqual(
            len(
                detector.eval_accuracy(
                    self.mts_anomalies, self.mts_test, metric="precision"
                )
            ),
            2,
        )

        # detector must found 28 anomalies on the first width of the series 'test'
        self.assertEqual(
            binary_detection["0"].sum(axis=0).all_values().flatten()[0], 28
        )
        # detector must found 28 anomalies on the second width of the series 'test'
        self.assertEqual(
            binary_detection["1"].sum(axis=0).all_values().flatten()[0], 52
        )

        acc = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="accuracy"
        )
        # detector must have an accuracy of 0.41 on the first width of the series 'mts_test'
        self.assertAlmostEqual(acc[0], 0.71, delta=1e-05)
        # detector must have an accuracy of 0.41 on the second width of the series 'mts_test'
        self.assertAlmostEqual(acc[1], 0.48, delta=1e-05)

        precision = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="precision"
        )
        # detector must have an precision of 0.41 on the first width of the series 'mts_test'
        self.assertAlmostEqual(precision[0], 0.17857, delta=1e-05)
        # detector must have an precision of 0.41 on the second width of the series 'mts_test'
        self.assertAlmostEqual(precision[1], 0.096153, delta=1e-05)

        recall = detector.eval_accuracy(
            self.mts_anomalies, self.mts_test, metric="recall"
        )
        # detector must have an recall of 0.41 on the first width of the series 'mts_test'
        self.assertAlmostEqual(recall[0], 0.45454, delta=1e-05)
        # detector must have an recall of 0.41 on the second width of the series 'mts_test'
        self.assertAlmostEqual(recall[1], 0.5, delta=1e-05)

        f1 = detector.eval_accuracy(self.mts_anomalies, self.mts_test, metric="f1")
        # detector must have an f1 of 0.41 on the first width of the series 'mts_test'
        self.assertAlmostEqual(f1[0], 0.25641, delta=1e-05)
        # detector must have an recall of 0.41 on the second width of the series 'mts_test'
        self.assertAlmostEqual(f1[1], 0.16129, delta=1e-05)
