from itertools import product
from typing import Sequence

import numpy as np
import pytest

from darts import TimeSeries
from darts.ad.detectors.detectors import FittableDetector
from darts.ad.detectors.quantile_detector import QuantileDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector

list_Detector = [(ThresholdDetector, {"low_threshold": 0.2})]

list_FittableDetector = [(QuantileDetector, {"low_quantile": 0.2})]

list_detectors = list_Detector + list_FittableDetector

metric_func = ["accuracy", "recall", "f1", "precision"]

delta = 1e-05


class TestAnomalyDetectionDetector:

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

    @pytest.mark.parametrize(
        "detector_config,series",
        product(list_detectors, [(train, test), (mts_train, mts_test)]),
    )
    def test_detect_return_type(self, detector_config, series):
        """Check that detect() behave as expected"""
        detector_cls, detector_kwargs = detector_config
        ts_train, ts_test = series
        detector = detector_cls(**detector_kwargs)
        if isinstance(detector, FittableDetector):
            detector.fit(ts_train)

        # Check if return type is TimeSeries when input is a single series
        assert isinstance(detector.detect(ts_test), TimeSeries)
        # Check if return type is Sequence when input is a Sequence of series
        assert isinstance(detector.detect([ts_test]), Sequence)

        # Input cannot be probabilistic
        with pytest.raises(ValueError):
            detector.detect(self.probabilistic)

    @pytest.mark.parametrize("detector_config", list_detectors)
    def test_eval_metric_return_type(self, detector_config):
        """Check that eval_metric() behave as expected"""
        detector_cls, detector_kwargs = detector_config
        detector = detector_cls(**detector_kwargs)

        # univariate
        if isinstance(detector, FittableDetector):
            detector.fit(self.train)
        # Check if return type is float when input is a series
        assert isinstance(
            detector.eval_metric(
                actual_anomalies=self.anomalies, pred_scores=self.test
            ),
            float,
        )
        # Check if return type is Sequence when input is a Sequence of series
        assert isinstance(
            detector.eval_metric(
                actual_anomalies=self.anomalies, pred_scores=[self.test]
            ),
            Sequence,
        )

        # multivariate
        if isinstance(detector, FittableDetector):
            detector.fit(self.mts_train)
        # Check if return type is Sequence when input is a multivariate series
        assert isinstance(
            detector.eval_metric(
                actual_anomalies=self.mts_anomalies, pred_scores=self.mts_test
            ),
            Sequence,
        )
        # Check if return type is Sequence when input is a multivariate series
        assert isinstance(
            detector.eval_metric(
                actual_anomalies=self.mts_anomalies, pred_scores=[self.mts_test]
            ),
            Sequence,
        )

        # Input cannot be probabilistic
        with pytest.raises(ValueError):
            detector.eval_metric(
                actual_anomalies=self.anomalies, pred_scores=self.probabilistic
            )

    @pytest.mark.parametrize(
        "config",
        [
            (
                ThresholdDetector,
                {"low_threshold": 4.8, "high_threshold": 10.5},
                {"low_threshold": [4.8, 4.8], "high_threshold": [10.5, 10.5]},
            ),
            (
                QuantileDetector,
                {"low_quantile": 0.05, "high_quantile": 0.95},
                {"low_quantile": [0.05, 0.05], "high_quantile": [0.95, 0.95]},
            ),
        ],
    )
    def test_bounded_detectors_parameters_broadcasting(self, config):
        """If two values are given for low and high, and a series of width 2 is given,
        then the results must be the same as a detector that was given only one value
        for low and high (will duplicate the value for each width)"""
        detector_cls, kwargs_1param, kwargs_2params = config

        # detector that should broadcast the parameters to match series' width
        detector = detector_cls(**kwargs_1param)
        # detector created with a number of parameters matching the series' width
        detector_2param = detector_cls(**kwargs_2params)
        if isinstance(detector, FittableDetector):
            detector.fit(self.mts_train)
            detector_2param.fit(self.mts_train)

        binary_detection = detector.detect(self.mts_test)
        binary_detection_2param = detector_2param.detect(self.mts_test)
        assert binary_detection == binary_detection_2param

    @pytest.mark.parametrize("detector_config", list_FittableDetector)
    def test_fit_detect_series_width(self, detector_config):
        detector_cls, detector_kwargs = detector_config
        detector = detector_cls(**detector_kwargs)

        # Need to call fit() before calling detect()
        with pytest.raises(ValueError):
            detector.detect(self.test)

        # Check if _fit_called is False
        assert not detector._fit_called

        with pytest.raises(ValueError):
            # fit on sequence with series that have different width
            detector.fit([self.train, self.mts_train])

        with pytest.raises(ValueError):
            # Input cannot be probabilistic
            detector.fit(self.probabilistic)

        # case1: fit on UTS
        detector1 = detector
        detector1.fit(self.train)

        # Check if _fit_called is True after being fitted
        assert detector1._fit_called

        with pytest.raises(ValueError):
            # series must be same width as series used for training
            detector1.detect(self.mts_test)

        # case2: fit on MTS
        detector2 = detector
        detector2.fit(self.mts_test)

        # Check if _fit_called is True after being fitted
        assert detector2._fit_called

        with pytest.raises(ValueError):
            # series must be same width as series used for training
            detector2.detect(self.train)

    def test_QuantileDetector_constructor(self):

        # Need to have at least one parameter (low, high) not None
        with pytest.raises(ValueError):
            QuantileDetector()
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=None, high_quantile=None)

        # Parameter low must be float or Sequence of float
        with pytest.raises(TypeError):
            QuantileDetector(low_quantile="0.5")
        with pytest.raises(TypeError):
            QuantileDetector(low_quantile=[0.2, "0.1"])

        # Parameter high must be float or Sequence of float
        with pytest.raises(TypeError):
            QuantileDetector(high_quantile="0.5")
        with pytest.raises(TypeError):
            QuantileDetector(high_quantile=[0.2, "0.1"])

        # if high and low are both sequences of length>1, they must be of the same size
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[0.2, 0.1], high_quantile=[0.95, 0.8, 0.9])
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[0.2, 0.1, 0.7], high_quantile=[0.95, 0.8])

        # Parameter must be between 0 and 1
        with pytest.raises(ValueError):
            QuantileDetector(high_quantile=1.1)
        with pytest.raises(ValueError):
            QuantileDetector(high_quantile=-0.2)
        with pytest.raises(ValueError):
            QuantileDetector(high_quantile=[-0.1, 0.9])
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=1.1)
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=-0.2)
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[-0.2, 0.3])

        # Parameter high must be higher or equal than parameter low
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=0.7, high_quantile=0.2)
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[0.2, 0.9], high_quantile=[0.95, 0.1])
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=0.2, high_quantile=[0.95, 0.1])
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[0.2, 0.9], high_quantile=0.8)

        # Parameter high/low cannot be sequence of only None
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[None, None, None])
        with pytest.raises(ValueError):
            QuantileDetector(high_quantile=[None, None, None])
        with pytest.raises(ValueError):
            QuantileDetector(low_quantile=[None], high_quantile=[None, None, None])

        # check that low_threshold and high_threshold are the same and no errors are raised
        detector = QuantileDetector(low_quantile=0.5, high_quantile=0.5)
        detector.fit(self.train)
        assert detector.low_threshold == detector.high_threshold

    @pytest.mark.parametrize(
        "detector_kwargs",
        [
            {"low_quantile": 0.1, "high_quantile": [0.8, 0.7]},
            {"low_quantile": [0.1, 0.2], "high_quantile": [0.8, 0.9]},
            {"low_quantile": [0.1, 0.2], "high_quantile": 0.8},
            {"low_quantile": [0.1, 0.2]},
            {"high_quantile": [0.1, 0.2]},
        ],
    )
    def test_quantile_detector_fit_detect_matching_width(self, detector_kwargs):
        """Widths of series should match the number of values given for high or/and low,
        if more than one value is provided for either of them.

        `self.train` series has only one component whereas model is created with 2 values for at
        least one of the model"""
        detector = QuantileDetector(**detector_kwargs)

        # during training
        with pytest.raises(ValueError):
            detector.fit(self.train)
        with pytest.raises(ValueError):
            detector.fit([self.train, self.mts_train])

        # during detection
        detector.fit(self.mts_train)
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

    def test_ThresholdDetector_constructor(self):
        # Need to have at least one parameter (low, high) not None
        with pytest.raises(ValueError):
            ThresholdDetector()
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=None, high_threshold=None)

        # if high and low are both sequences of length>1, they must be of the same size
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[0.2, 0.1], high_threshold=[0.95, 0.8, 0.9])
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[0.2, 0.1, 0.7], high_threshold=[0.95, 0.8])

        # Parameter high must be higher or equal than parameter low
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=0.7, high_threshold=0.2)
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[0.2, 0.9], high_threshold=[0.95, 0.1])
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=0.2, high_threshold=[0.95, 0.1])
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[0.2, 0.9], high_threshold=0.8)
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[0.2, 0.9, None], high_threshold=0.8)

        # Parameter high/low cannot be sequence of only None
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[None, None, None])
        with pytest.raises(ValueError):
            ThresholdDetector(high_threshold=[None, None, None])
        with pytest.raises(ValueError):
            ThresholdDetector(low_threshold=[None], high_threshold=[None, None, None])

        # widths of series used for scoring must match the number of values given for high or/and low,
        # if high and low have a length higher than 1

        detector = ThresholdDetector(low_threshold=0.1, high_threshold=[0.8, 0.7])
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

        detector = ThresholdDetector(
            low_threshold=[0.1, 0.2], high_threshold=[0.8, 0.9]
        )
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

        detector = ThresholdDetector(low_threshold=[0.1, 0.2], high_threshold=0.8)
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

        detector = ThresholdDetector(low_threshold=[0.1, 0.2])
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

        detector = ThresholdDetector(high_threshold=[0.1, 0.2])
        with pytest.raises(ValueError):
            detector.detect(self.train)
        with pytest.raises(ValueError):
            detector.detect([self.train, self.mts_train])

    @pytest.mark.parametrize(
        "config",
        [
            (
                ThresholdDetector,
                {"low_threshold": 9.5, "high_threshold": 10.5},
                {
                    "anomalies": 58,
                    "accuracy": 0.41,
                    "recall": 0.40,
                    "f1": 0.06349,
                    "precision": 0.03448,
                },
                None,
            ),
            (
                QuantileDetector,
                {"low_quantile": 0.05, "high_quantile": 0.95},
                {
                    "anomalies": 42,
                    "accuracy": 0.57,
                    "recall": 0.40,
                    "f1": 0.08510,
                    "precision": 0.04761,
                },
                (9.13658, 10.74007),
            ),
        ],
    )
    def test_bounded_detector_eval_metric_univariate(self, config):
        """Verifying the performance of the bounded detectors on an univariate example"""
        detector_cls, detector_kwargs, expected_values, fitted_params = config
        detector = detector_cls(**detector_kwargs)
        if isinstance(detector, FittableDetector):
            detector.fit(self.train)
        binary_detection = detector.detect(self.test)

        # Return of .detect() must be binary
        np.testing.assert_array_equal(
            binary_detection.values(copy=False),
            binary_detection.values(copy=False).astype(bool),
        )

        # Return of .detect() must be same len as input
        assert len(binary_detection) == len(self.test)

        assert (
            binary_detection.sum(axis=0).all_values().flatten()[0]
            == expected_values["anomalies"]
        )

        for m_func in metric_func:
            assert (
                np.abs(
                    expected_values[m_func]
                    - detector.eval_metric(self.anomalies, self.test, metric=m_func),
                )
                < delta
            )

        # check the fitted parameters
        if isinstance(detector, QuantileDetector):
            assert np.abs(fitted_params[0] - detector.low_threshold[0]) < delta
            assert np.abs(fitted_params[1] - detector.high_threshold[0]) < delta

    @pytest.mark.parametrize(
        "config",
        [
            (
                ThresholdDetector,
                {"low_threshold": [4.8, 4.8], "high_threshold": [10.5, 10.5]},
                {
                    "anomalies": [28, 52],
                    "accuracy": (0.71, 0.48),
                    "recall": (0.45454, 0.5),
                    "f1": (0.25641, 0.16129),
                    "precision": (0.17857, 0.09615),
                },
            ),
            (
                ThresholdDetector,
                {"low_threshold": [10, None], "high_threshold": [None, 5]},
                {
                    "anomalies": [48, 43],
                    "accuracy": (0.51, 0.57),
                    "recall": (0.45454, 0.5),
                    "f1": (0.16949, 0.18867),
                    "precision": (0.10416, 0.11627),
                },
            ),
            (
                QuantileDetector,
                {"low_quantile": [0.05, 0.05], "high_quantile": [0.95, 0.95]},
                {
                    "anomalies": [37, 38],
                    "accuracy": (0.58, 0.58),
                    "recall": (0.27272, 0.3),
                    "f1": (0.125, 0.125),
                    "precision": (0.08108, 0.07894),
                },
            ),
            (
                QuantileDetector,
                {"low_quantile": [0.05, None], "high_quantile": [None, 0.95]},
                {
                    "anomalies": [20, 19],
                    "accuracy": (0.69, 0.75),
                    "recall": (0.0, 0.2),
                    "f1": (0.0, 0.13793),
                    "precision": (0.0, 0.10526),
                },
            ),
        ],
    )
    def test_bounded_detector_performance_multivariate(self, config):
        """
        TODO: improve these tests to introduce some correlation between actual and detected anomalies
        """
        detector_cls, detector_kwargs, expected_values = config
        detector = detector_cls(**detector_kwargs)

        if isinstance(detector, FittableDetector):
            detector.fit(self.mts_train)
        binary_detection = detector.detect(self.mts_test)

        # output must have the same width as the input
        expected_width = self.mts_test.n_components
        assert binary_detection.width == expected_width
        for m_func in metric_func:
            assert (
                len(
                    detector.eval_metric(
                        self.mts_anomalies, self.mts_test, metric=m_func
                    )
                )
                == expected_width
            )

        # check number of anomalies detected in the first component
        assert (
            binary_detection["0"].sum(axis=0).all_values().flatten()[0]
            == expected_values["anomalies"][0]
        )
        # check number of anomalies detected in the second component
        assert (
            binary_detection["1"].sum(axis=0).all_values().flatten()[0]
            == expected_values["anomalies"][1]
        )

        # check each metric on each component of the series
        for m_func in metric_func:
            metric_vals = detector.eval_metric(
                self.mts_anomalies, self.mts_test, metric=m_func
            )
            assert np.abs(expected_values[m_func][0] - metric_vals[0]) < delta
            assert np.abs(expected_values[m_func][1] - metric_vals[1]) < delta

    def test_fit_detect(self):
        """Calling fit() then detect() and fit_detect() should yield the same results"""
        detector1 = QuantileDetector(low_quantile=0.05, high_quantile=0.95)
        detector1.fit(self.train)
        prediction1 = detector1.detect(self.train)

        detector2 = QuantileDetector(low_quantile=0.05, high_quantile=0.95)
        prediction2 = detector2.fit_detect(self.train)

        assert prediction1 == prediction2
