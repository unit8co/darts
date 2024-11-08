from collections.abc import Sequence

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from darts import TimeSeries
from darts.ad.aggregators import (
    AndAggregator,
    EnsembleSklearnAggregator,
    FittableAggregator,
    OrAggregator,
)
from darts.models import MovingAverageFilter

# element shape : (model_cls, model_kwargs, expected metrics)
list_NonFittableAggregator = [
    (
        OrAggregator,
        {},
        {
            "only_ones": {"accuracy": 1, "recall": 1, "f1": 1, "precision": 1},
            "multivariate": {"accuracy": 0, "recall": 0, "f1": 0, "precision": 0},
            "synthetic": {
                "accuracy": 0.56,
                "recall": 0.72549,
                "f1": 0.62711,
                "precision": 0.55223,
                "total": 67,
            },
            "multiple_series": {
                "accuracy": [0.56, 0.52],
                "recall": [0.72549, 0.764706],
                "f1": [0.627119, 0.619048],
                "precision": [0.552239, 0.52],
                "total": [67, 75],
            },
        },
    ),
    (
        AndAggregator,
        {},
        {
            "only_ones": {"accuracy": 1, "recall": 1, "f1": 1, "precision": 1},
            "multivariate": {"accuracy": 1, "recall": 0, "f1": 0, "precision": 0},
            "synthetic": {
                "accuracy": 0.44,
                "recall": 0.21568,
                "f1": 0.28205,
                "precision": 0.40740,
                "total": 27,
            },
            "multiple_series": {
                "accuracy": [0.44, 0.53],
                "recall": [0.215686, 0.27451],
                "f1": [0.282051, 0.373333],
                "precision": [0.407407, 0.583333],
                "total": [27, 24],
            },
        },
    ),
]

# expected metrics values are declared in the test
list_FittableAggregator = [
    (EnsembleSklearnAggregator, {"model": GradientBoostingClassifier()}, {})
]


list_Aggregator = list_NonFittableAggregator + list_FittableAggregator

delta = 1e-05


class TestAnomalyDetectionAggregator:
    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_real_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
    real_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_real_anomalies
    )

    # multivariate series
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies_w3 = np.random.choice(a=[0, 1], size=[100, 3], p=[0.5, 0.5])
    mts_anomalies3 = TimeSeries.from_times_and_values(
        train._time_index, np_anomalies_w3
    )

    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 2, 5])
    mts_probabilistic = TimeSeries.from_values(np_probabilistic)

    # simple case
    np_anomalies_1 = np.random.choice(a=[1], size=100, p=[1])
    onlyones = TimeSeries.from_times_and_values(train._time_index, np_anomalies_1)

    np_anomalies = np.random.choice(a=[1], size=[100, 2], p=[1])
    mts_onlyones = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_anomalies_0 = np.random.choice(a=[0], size=100, p=[1])
    onlyzero = TimeSeries.from_times_and_values(train._time_index, np_anomalies_0)

    series_1_and_0 = TimeSeries.from_values(
        np.dstack((np_anomalies_1, np_anomalies_0))[0],
        columns=["component 1", "component 2"],
    )

    # series has 3 components, and real_anomalies_3w is equal to
    # - component 1 when component 3 is 1
    # - component 2 when component 3 is 0
    np_real_anomalies_3w = [
        elem[0] if elem[2] == 1 else elem[1] for elem in np_anomalies_w3
    ]
    real_anomalies_3w = TimeSeries.from_times_and_values(
        train._time_index, np_real_anomalies_3w
    )

    @staticmethod
    def helper_eval_metric_single_series(
        aggregator,
        series: TimeSeries,
        pred_series: TimeSeries,
        expected_vals: dict[str, float],
    ):
        """Evaluate model on given series, for all 4 supported metric functions"""
        for m_func in ["accuracy", "recall", "f1", "precision"]:
            assert (
                np.abs(
                    expected_vals[m_func]
                    - aggregator.eval_metric(
                        series,
                        pred_series,
                        metric=m_func,
                    )
                )
                < delta
            )

    @staticmethod
    def helper_eval_metric_multiple_series(
        aggregator,
        series: Sequence[TimeSeries],
        pred_series: Sequence[TimeSeries],
        expected_vals: dict[str, list[float]],
    ):
        """Evaluate model on multiple series, for all 4 supported metric functions"""
        for m_func in ["accuracy", "recall", "f1", "precision"]:
            np.testing.assert_array_almost_equal(
                np.array(
                    aggregator.eval_metric(
                        series,
                        pred_series,
                        metric=m_func,
                    )
                ),
                np.array(expected_vals[m_func]),
                decimal=1,
            )

    @pytest.mark.parametrize("config", list_Aggregator)
    def test_predict_return_type(self, config):
        """Check that predict's output are properly unpacked depending on input type"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        if isinstance(aggregator, FittableAggregator):
            aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # single TimeSeries
        assert isinstance(aggregator.predict(self.mts_anomalies1), TimeSeries)

        # Sequence of one TimeSeries
        assert isinstance(
            aggregator.predict([self.mts_anomalies1]),
            Sequence,
        )

        # Sequence of several TimeSeries
        assert isinstance(
            aggregator.predict([self.mts_anomalies1, self.mts_anomalies2]),
            Sequence,
        )

    @pytest.mark.parametrize("config", list_Aggregator)
    def test_eval_metric_return_type(self, config):
        """Check that eval_metric's output are properly unpacked depending on input type"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        if isinstance(aggregator, FittableAggregator):
            aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # Check return types
        assert isinstance(
            aggregator.eval_metric(self.real_anomalies, self.mts_anomalies1),
            float,
        )

        assert isinstance(
            aggregator.eval_metric([self.real_anomalies], [self.mts_anomalies1]),
            Sequence,
        )

        assert isinstance(
            aggregator.eval_metric(self.real_anomalies, [self.mts_anomalies1]),
            Sequence,
        )

        assert isinstance(
            aggregator.eval_metric(
                [self.real_anomalies, self.real_anomalies],
                [self.mts_anomalies1, self.mts_anomalies2],
            ),
            Sequence,
        )

        # Check if return type is the same number of series in input
        assert (
            len(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                )
            )
            == 2
        )

        # intersection between 'anomalies' and the series in the sequence 'list_series'
        # must be non empty
        with pytest.raises(ValueError):
            aggregator.eval_metric(self.real_anomalies[:30], self.mts_anomalies1[40:])
        with pytest.raises(ValueError):
            aggregator.eval_metric(
                [self.real_anomalies, self.real_anomalies[:30]],
                [self.mts_anomalies1, self.mts_anomalies1[40:]],
            )

        # window parameter must be smaller than the length of the input (len = 100)
        with pytest.raises(ValueError):
            aggregator.eval_metric(self.real_anomalies, self.mts_anomalies1, window=101)

    @pytest.mark.parametrize("config", list_Aggregator)
    def test_aggregator_predict_wrong_inputs(self, config):
        """Check that exception is raised when predict() arguments are incorrects."""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        # fit aggregator on series with 2 components
        if isinstance(aggregator, FittableAggregator):
            aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # predict on (sequence of) univariate series
        with pytest.raises(ValueError):
            aggregator.predict([self.real_anomalies])
        with pytest.raises(ValueError):
            aggregator.predict(self.real_anomalies)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.real_anomalies])

        # input a (sequence of) non binary series
        expected_msg = "Input series `series` must have binary values only."
        with pytest.raises(ValueError) as err:
            aggregator.predict(self.mts_train)
        assert str(err.value) == expected_msg
        with pytest.raises(ValueError) as err:
            aggregator.predict([self.mts_anomalies1, self.mts_train])
        assert str(err.value) == expected_msg

        # input a (sequence of) probabilistic series
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_probabilistic)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_probabilistic])

        # input an element that is not a series
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, "random"])
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, 1])

    @pytest.mark.parametrize("config", list_NonFittableAggregator)
    def test_NonFittableAggregator_predict(self, config):
        """Check that predict() works as intended"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        # name must be of type str
        assert isinstance(aggregator.__str__(), str)

        assert not isinstance(aggregator, FittableAggregator)

        # Check that predict can be called when series is appropriate
        pred = aggregator.predict(self.mts_anomalies1)

        # Check that the aggregated result has only one component
        assert pred.width == 1

    @pytest.mark.parametrize("config", list_FittableAggregator)
    def test_FittableAggregator_fit_wrong_inputs(self, config):
        """Check that exception is raised when fit() arguments are incorrects"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        # fit on sequence with series that have different width
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies],
                [self.mts_anomalies1, self.mts_anomalies3],
            )

        # fit on a (sequence of) univariate series
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, self.real_anomalies)
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, [self.real_anomalies])
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies],
                [self.mts_anomalies1, self.real_anomalies],
            )

        # fit on a (sequence of) non binary series
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, self.mts_train)
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, [self.mts_train])
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies],
                [self.mts_anomalies1, self.mts_train],
            )

        # fit on a (sequence of) probabilistic series
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, self.mts_probabilistic)
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, [self.mts_probabilistic])
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies],
                [self.mts_anomalies1, self.mts_probabilistic],
            )

        # input an element that is not a series
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, "random")
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, [self.mts_anomalies1, "random"])
        with pytest.raises(ValueError):
            aggregator.fit(self.real_anomalies, [self.mts_anomalies1, 1])

        # fit on a (sequence of) multivariate anomalies
        with pytest.raises(ValueError):
            aggregator.fit(self.mts_anomalies1, self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit([self.mts_anomalies1], [self.mts_anomalies1])
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.mts_anomalies1],
                [self.mts_anomalies1, self.mts_anomalies1],
            )

        # fit on a (sequence of) non binary anomalies
        with pytest.raises(ValueError):
            aggregator.fit(self.train, self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit([self.train], self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.train],
                [self.mts_anomalies1, self.mts_anomalies1],
            )

        # fit on a (sequence of) probabilistic anomalies
        with pytest.raises(ValueError):
            aggregator.fit(self.mts_probabilistic, self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit([self.mts_probabilistic], self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.mts_probabilistic],
                [self.mts_anomalies1, self.mts_anomalies1],
            )

        # input an element that is not a anomalies
        with pytest.raises(ValueError):
            aggregator.fit("random", self.mts_anomalies1)
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, "random"],
                [self.mts_anomalies1, self.mts_anomalies1],
            )
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, 1], [self.mts_anomalies1, self.mts_anomalies1]
            )

        # nbr of anomalies must match nbr of input series
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies], self.mts_anomalies1
            )
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies, self.real_anomalies], [self.mts_anomalies1]
            )
        with pytest.raises(ValueError):
            aggregator.fit(
                [self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies1]
            )

    @pytest.mark.parametrize("config", list_FittableAggregator)
    def test_FittableAggregator_predict_wrong_inputs(self, config):
        """Check that exception specific to FittableAggregator are properly raised"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # series must be same width as series used for training
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_anomalies3)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies3])
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_anomalies3])

    @pytest.mark.parametrize("config", list_FittableAggregator)
    def test_FittableAggregator_fit_predict(self, config):
        """Check that consecutive calls to fit() and predict() work as intended"""
        aggregator_cls, cls_kwargs, _ = config
        aggregator = aggregator_cls(**cls_kwargs)

        # name must be of type str
        assert isinstance(
            aggregator.__str__(),
            str,
        )

        # Need to call fit() before calling predict()
        with pytest.raises(ValueError) as err:
            aggregator.predict([self.mts_anomalies1, self.mts_anomalies1])
        assert (
            str(err.value)
            == "The `Aggregator` has not been fitted yet. Call `Aggregator.fit()` first."
        )

        # Check if _fit_called is False before calling fit
        assert not aggregator._fit_called

        aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # Check if _fit_called is True after calling fit
        assert aggregator._fit_called

        # Check that predict can be called when series is appropriate
        pred = aggregator.predict(self.mts_anomalies1)

        # Check that the aggregated result has only one component
        assert pred.width == 1

    @pytest.mark.parametrize("config", list_NonFittableAggregator)
    def test_aggregator_performance_single_series(self, config):
        aggregator_cls, cls_kwargs, metrics = config
        aggregator = aggregator_cls(**cls_kwargs)

        # both actual and pred contain only 1
        self.helper_eval_metric_single_series(
            aggregator=aggregator,
            series=self.onlyones,
            pred_series=self.mts_onlyones,
            expected_vals=metrics["only_ones"],
        )

        # input with 2 components (only 1 and only 0) and ground truth is only 0
        self.helper_eval_metric_single_series(
            aggregator=aggregator,
            series=self.onlyzero,
            pred_series=self.series_1_and_0,
            expected_vals=metrics["multivariate"],
        )

        # synthetic example
        self.helper_eval_metric_single_series(
            aggregator=aggregator,
            series=self.real_anomalies,
            pred_series=self.mts_anomalies1,
            expected_vals=metrics["synthetic"],
        )

        # number of detected anomalies in synthetic example
        assert (
            aggregator.predict(self.mts_anomalies1)
            .sum(axis=0)
            .all_values()
            .flatten()[0]
            == metrics["synthetic"]["total"]
        )

    @pytest.mark.parametrize("config", list_NonFittableAggregator)
    def test_aggregator_performance_multiple_series(self, config):
        aggregator_cls, cls_kwargs, metrics = config
        aggregator = aggregator_cls(**cls_kwargs)

        self.helper_eval_metric_multiple_series(
            aggregator=aggregator,
            series=[self.real_anomalies, self.real_anomalies],
            pred_series=[self.mts_anomalies1, self.mts_anomalies2],
            expected_vals=metrics["multiple_series"],
        )

        # number of detected anomalies
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal(
            [v.sum(axis=0).all_values().flatten()[0] for v in values],
            metrics["multiple_series"]["total"],
            decimal=1,
        )

    def test_ensemble_aggregator_constructor(self):
        # Need to input an EnsembleSklearn model
        with pytest.raises(ValueError):
            EnsembleSklearnAggregator(model=MovingAverageFilter(window=10))

    @pytest.mark.parametrize(
        "config",
        [
            (
                real_anomalies_3w,
                mts_anomalies3,
                {
                    "accuracy": 0.92,
                    "recall": 0.86666,
                    "f1": 0.92857,
                    "precision": 1.0,
                    "total": 52,
                },
            ),
            (
                real_anomalies,
                mts_anomalies1,
                {
                    "accuracy": 0.51,
                    "recall": 1.0,
                    "f1": 0.67549,
                    "precision": 0.51,
                    "total": 100,
                },
            ),
        ],
    )
    def test_ensemble_aggregator_single_series(self, config):
        """Check performance of ensemble aggregator on single series cases"""
        series, pred_series, expected_metrics = config

        aggregator = EnsembleSklearnAggregator(
            model=GradientBoostingClassifier(
                n_estimators=50, learning_rate=1.0, max_depth=1
            )
        )

        aggregator.fit(series, pred_series)

        self.helper_eval_metric_single_series(
            aggregator=aggregator,
            series=series,
            pred_series=pred_series,
            expected_vals=expected_metrics,
        )

        assert (
            aggregator.predict(pred_series).sum(axis=0).all_values().flatten()[0]
            == expected_metrics["total"]
        )

    def test_ensemble_aggregator_multiple_series(self):
        """Ensemble aggregator is fitted on one series, evaluated on two."""
        aggregator = EnsembleSklearnAggregator(
            model=GradientBoostingClassifier(
                n_estimators=50, learning_rate=1.0, max_depth=1
            )
        )
        aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        self.helper_eval_metric_multiple_series(
            aggregator=aggregator,
            series=[self.real_anomalies, self.real_anomalies],
            pred_series=[self.mts_anomalies1, self.mts_anomalies2],
            expected_vals={
                "accuracy": [0.51, 0.51],
                "recall": [1, 1],
                "f1": [0.68, 0.68],
                "precision": [0.51, 0.51],
            },
        )

        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal(
            [v.sum(axis=0).all_values().flatten()[0] for v in values],
            [100, 100],
            decimal=1,
        )
