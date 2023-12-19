from typing import Sequence

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

list_NonFittableAggregator = [
    (OrAggregator, {}),
    (AndAggregator, {}),
]

list_FittableAggregator = [
    (EnsembleSklearnAggregator, {"model": GradientBoostingClassifier()})
]

list_Aggregator = list_NonFittableAggregator + list_FittableAggregator

delta = 1e-05

metric_func = ["accuracy", "recall", "f1", "precision"]


class TestAnomalyDetectionAggregators:

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

    np_real_anomalies_3w = [
        elem[0] if elem[2] == 1 else elem[1] for elem in np_anomalies_w3
    ]
    real_anomalies_3w = TimeSeries.from_times_and_values(
        train._time_index, np_real_anomalies_3w
    )

    @pytest.mark.parametrize("config", list_Aggregator)
    def test_predict_return_type(self, config):
        """Check that predictions are properly unpacked depending on input type"""
        aggregator_cls, cls_kwargs = config
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

    def test_eval_metric_return_type(self):

        aggregator = AndAggregator()

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

        # intersection between 'actual_anomalies' and the series in the sequence 'list_series'
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

    @pytest.mark.parametrize("config", list_NonFittableAggregator)
    def test_NonFittableAggregator(self, config):
        aggregator_cls, cls_kwargs = config
        aggregator = aggregator_cls(**cls_kwargs)

        # name must be of type str
        assert isinstance(aggregator.__str__(), str)

        assert not isinstance(aggregator, FittableAggregator)

        # predict on (sequence of) univariate series
        with pytest.raises(ValueError):
            aggregator.predict([self.real_anomalies])
        with pytest.raises(ValueError):
            aggregator.predict(self.real_anomalies)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.real_anomalies])

        # input a (sequence of) non binary series
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_train)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_train])

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

    @pytest.mark.parametrize("config", list_FittableAggregator)
    def test_FittableAggregator(self, config):
        aggregator_cls, cls_kwargs = config
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
            == f"The Aggregator {aggregator.__str__()} has not been fitted yet. Call `fit()` first."
        )

        # Check if _fit_called is False before calling fit()
        assert not aggregator._fit_called

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

        # case1: fit
        aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # Check if _fit_called is True after being fitted
        assert aggregator._fit_called

        # series must be same width as series used for training
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_anomalies3)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies3])
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_anomalies3])

        # predict on (sequence of) univariate series
        with pytest.raises(ValueError):
            aggregator.predict([self.real_anomalies])
        with pytest.raises(ValueError):
            aggregator.predict(self.real_anomalies)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.real_anomalies])

        # input a (sequence of) non binary series
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_train)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_train])

        # input a (sequence of) probabilistic series
        with pytest.raises(ValueError):
            aggregator.predict(self.mts_probabilistic)
        with pytest.raises(ValueError):
            aggregator.predict([self.mts_anomalies1, self.mts_probabilistic])

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

    def test_OrAggregator(self):

        aggregator = OrAggregator()

        # simple case
        # aggregator must have an accuracy of 0 for input with 2 components
        # (only 1 and only 0) and ground truth is only 0
        assert (
            np.abs(
                0
                - aggregator.eval_metric(
                    self.onlyzero,
                    self.series_1_and_0,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for input with 2 components
        # (only 1 and only 0) and ground truth is only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.series_1_and_0,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="recall",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="precision",
                )
            )
            < delta
        )

        # single series case (random example)
        # aggregator must found 67 anomalies in the input mts_anomalies1
        assert (
            aggregator.predict(self.mts_anomalies1)
            .sum(axis=0)
            .all_values()
            .flatten()[0]
            == 67
        )

        # aggregator must have an accuracy of 0.56 for the input mts_anomalies1
        assert (
            np.abs(
                0.56
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an recall of 0.72549 for the input mts_anomalies1
        assert (
            np.abs(
                0.72549
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="recall"
                )
            )
            < delta
        )

        # aggregator must have an f1 of 0.62711 for the input mts_anomalies1
        assert (
            np.abs(
                0.62711
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="f1"
                )
            )
            < delta
        )

        # aggregator must have an precision of 0.55223 for the input mts_anomalies1
        assert (
            np.abs(
                0.55223
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="precision",
                )
            )
            < delta
        )

        # multiple series case (random example)
        # aggregator must found [67,75] anomalies in the input [mts_anomalies1, mts_anomalies2]
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal(
            [v.sum(axis=0).all_values().flatten()[0] for v in values],
            [67, 75],
            decimal=1,
        )

        # aggregator must have an accuracy of [0.56,0.52] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="accuracy",
                )
            ),
            np.array([0.56, 0.52]),
            decimal=1,
        )

        # aggregator must have an recall of [0.72549,0.764706] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="recall",
                )
            ),
            np.array([0.72549, 0.764706]),
            decimal=1,
        )
        # aggregator must have an f1 of [0.627119,0.619048] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="f1",
                )
            ),
            np.array([0.627119, 0.619048]),
            decimal=1,
        )
        # aggregator must have an precision of [0.552239,0.52] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="precision",
                )
            ),
            np.array([0.552239, 0.52]),
            decimal=1,
        )

    # @pytest.mark.parametrize("metric_func,expected_vals", [])
    def test_AndAggregator(self):

        aggregator = AndAggregator()

        # simple case
        # aggregator must have an accuracy of 0 for input with 2 components
        # (only 1 and only 0) and ground truth is only 1
        assert (
            np.abs(
                0
                - aggregator.eval_metric(
                    self.onlyones,
                    self.series_1_and_0,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 0 for input with 2 components
        # (only 1 and only 0) and ground truth is only 0
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyzero,
                    self.series_1_and_0,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an recall of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="recall",
                )
            )
            < delta
        )

        # aggregator must have an accuracy of 1 for the input containing only 1
        assert (
            np.abs(
                1
                - aggregator.eval_metric(
                    self.onlyones,
                    self.mts_onlyones,
                    metric="precision",
                )
            )
            < delta
        )

        # single series case (random example)
        # aggregator must found 27 anomalies in the input mts_anomalies1
        assert (
            aggregator.predict(self.mts_anomalies1)
            .sum(axis=0)
            .all_values()
            .flatten()[0]
            == 27
        )

        # aggregator must have an accuracy of 0.44 for the input mts_anomalies1
        assert (
            np.abs(
                0.44
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an recall of 0.21568 for the input mts_anomalies1
        assert (
            np.abs(
                0.21568
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="recall"
                )
            )
            < delta
        )

        # aggregator must have an f1 of 0.28205 for the input mts_anomalies1
        assert (
            np.abs(
                0.28205
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="f1"
                )
            )
            < delta
        )

        # aggregator must have an precision of 0.40740 for the input mts_anomalies1
        assert (
            np.abs(
                0.40740
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="precision",
                )
            )
            < delta
        )

        # multiple series case (random example)
        # aggregator must found [27,24] anomalies in the input [mts_anomalies1, mts_anomalies2]
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal(
            [v.sum(axis=0).all_values().flatten()[0] for v in values],
            [27, 24],
            decimal=1,
        )

        # aggregator must have an accuracy of [0.44,0.53] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="accuracy",
                )
            ),
            np.array([0.44, 0.53]),
            decimal=1,
        )
        # aggregator must have an recall of [0.215686,0.27451] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="recall",
                )
            ),
            np.array([0.215686, 0.27451]),
            decimal=1,
        )
        # aggregator must have an f1 of [0.282051,0.373333] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="f1",
                )
            ),
            np.array([0.282051, 0.373333]),
            decimal=1,
        )
        # aggregator must have an precision of [0.407407, 0.583333] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="precision",
                )
            ),
            np.array([0.407407, 0.583333]),
            decimal=1,
        )

    def test_EnsembleSklearn(self):

        # Need to input an EnsembleSklearn model
        with pytest.raises(ValueError):
            EnsembleSklearnAggregator(model=MovingAverageFilter(window=10))

        # simple case
        # series has 3 components, and real_anomalies_3w is equal to
        # - component 1 when component 3 is 1
        # - component 2 when component 3 is 0
        # must have a high accuracy (here 0.92)
        aggregator = EnsembleSklearnAggregator(
            model=GradientBoostingClassifier(
                n_estimators=50, learning_rate=1.0, max_depth=1
            )
        )
        aggregator.fit(self.real_anomalies_3w, self.mts_anomalies3)

        assert (
            np.abs(
                0.92
                - aggregator.eval_metric(
                    self.real_anomalies_3w,
                    self.mts_anomalies3,
                    metric="accuracy",
                )
            )
            < delta
        )

        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies_3w, self.real_anomalies_3w],
                    [self.mts_anomalies3, self.mts_anomalies3],
                    metric="accuracy",
                )
            ),
            np.array([0.92, 0.92]),
            decimal=1,
        )

        # single series case (random example)
        aggregator = EnsembleSklearnAggregator(
            model=GradientBoostingClassifier(
                n_estimators=50, learning_rate=1.0, max_depth=1
            )
        )
        aggregator.fit(self.real_anomalies, self.mts_anomalies1)

        # aggregator must found 100 anomalies in the input mts_anomalies1
        assert (
            aggregator.predict(self.mts_anomalies1)
            .sum(axis=0)
            .all_values()
            .flatten()[0]
            == 100
        )

        # aggregator must have an accuracy of 0.51 for the input mts_anomalies1
        assert (
            np.abs(
                0.51
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="accuracy",
                )
            )
            < delta
        )

        # aggregator must have an recall 1.0 for the input mts_anomalies1
        assert (
            np.abs(
                1.0
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="recall"
                )
            )
            < delta
        )

        # aggregator must have an f1 of 0.67549 for the input mts_anomalies1
        assert (
            np.abs(
                0.67549
                - aggregator.eval_metric(
                    self.real_anomalies, self.mts_anomalies1, metric="f1"
                )
            )
            < delta
        )

        # aggregator must have an precision of 0.51 for the input mts_anomalies1
        assert (
            np.abs(
                0.51
                - aggregator.eval_metric(
                    self.real_anomalies,
                    self.mts_anomalies1,
                    metric="precision",
                )
            )
            < delta
        )

        # multiple series case (random example)
        # aggregator must found [100,100] anomalies in the input [mts_anomalies1, mts_anomalies2]
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal(
            [v.sum(axis=0).all_values().flatten()[0] for v in values],
            [100, 100.0],
            decimal=1,
        )

        # aggregator must have an accuracy of [0.51, 0.51] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="accuracy",
                )
            ),
            np.array([0.51, 0.51]),
            decimal=1,
        )
        # aggregator must have an recall of [1,1] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="recall",
                )
            ),
            np.array([1, 1]),
            decimal=1,
        )
        # aggregator must have an f1 of [0.675497, 0.675497] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="f1",
                )
            ),
            np.array([0.675497, 0.675497]),
            decimal=1,
        )
        # aggregator must have an precision of [0.51, 0.51] for the input [mts_anomalies1, mts_anomalies2]
        np.testing.assert_array_almost_equal(
            np.array(
                aggregator.eval_metric(
                    [self.real_anomalies, self.real_anomalies],
                    [self.mts_anomalies1, self.mts_anomalies2],
                    metric="precision",
                )
            ),
            np.array([0.51, 0.51]),
            decimal=1,
        )
