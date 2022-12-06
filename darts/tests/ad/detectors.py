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
    np_MTS_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    MTS_train = TimeSeries.from_values(np_MTS_train)

    np_MTS_test = np.random.normal(loc=[10, 5], scale=[1, 1.5], size=[100, 2])
    MTS_test = TimeSeries.from_times_and_values(MTS_train._time_index, np_MTS_test)

    np_MTS_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    MTS_anomalies = TimeSeries.from_times_and_values(
        MTS_train._time_index, np_MTS_anomalies
    )

    def test_DetectNonFittableDetector(self):

        detector = ThresholdDetector(low=0.2)

        # Check return types
        # Check if return TimeSeries is float when input is a series
        self.assertTrue(isinstance(detector.detect(self.test), TimeSeries))

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(isinstance(detector.detect([self.test]), Sequence))

        # Check if return TimeSeries is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect(self.MTS_test), TimeSeries))

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect([self.MTS_test]), Sequence))

    def test_DetectFittableDetector(self):
        detector = QuantileDetector(low=0.2)

        # Check return types

        detector.fit(self.train)
        # Check if return type is float when input is a series
        self.assertTrue(isinstance(detector.detect(self.test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of series
        self.assertTrue(isinstance(detector.detect([self.test]), Sequence))

        detector.fit(self.MTS_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(detector.detect(self.MTS_test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of multivariate series
        self.assertTrue(isinstance(detector.detect([self.MTS_test]), Sequence))

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
                detector.eval_accuracy(self.MTS_anomalies, self.MTS_test), Sequence
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                detector.eval_accuracy(self.MTS_anomalies, [self.MTS_test]), Sequence
            )
        )

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
                detector.fit([self.train, self.MTS_train])

            # case1: fit on UTS
            detector1 = detector
            detector1.fit(self.train)

            # Check if _fit_called is True after being fitted
            self.assertTrue(detector1._fit_called)

            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                detector1.detect(self.MTS_test)

            # case2: fit on MTS
            detector2 = detector
            detector2.fit(self.MTS_test)

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

        detector = QuantileDetector(low=0.2)
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

        detector = ThresholdDetector(low=0.2)
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
