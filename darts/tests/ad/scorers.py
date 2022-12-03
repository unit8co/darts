from typing import Sequence

import numpy as np
from pyod.models.knn import KNN

from darts import TimeSeries
from darts.ad import scorers as S
from darts.models import MovingAverage
from darts.tests.base_test_class import DartsBaseTestClass

list_NonFittableAnomalyScorer = [
    S.Norm(),
    S.Difference(),
    S.GaussianNLLScorer(),
    S.ExponentialNLLScorer(),
    S.PoissonNLLScorer(),
    S.LaplaceNLLScorer(),
    S.CauchyNLLScorer(),
    S.GammaNLLScorer(),
]

list_FittableAnomalyScorer = [
    S.PyODScorer(model=KNN()),
    S.KMeansScorer(),
    S.WassersteinScorer(),
]

list_NLLScorer = [
    S.GaussianNLLScorer(),
    S.ExponentialNLLScorer(),
    S.PoissonNLLScorer(),
    S.LaplaceNLLScorer(),
    S.CauchyNLLScorer(),
    S.GammaNLLScorer(),
]


class ADAnomalyScorerTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_test = np.random.normal(loc=10, scale=1, size=100)
    test = TimeSeries.from_times_and_values(train._time_index, np_test)

    np_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.9, 0.1])
    anomalies = TimeSeries.from_times_and_values(train._time_index, np_anomalies)

    np_only_1_anomalies = np.random.choice(a=[0, 1], size=100, p=[0, 1])
    only_1_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_only_1_anomalies
    )

    np_only_0_anomalies = np.random.choice(a=[0, 1], size=100, p=[1, 0])
    only_0_anomalies = TimeSeries.from_times_and_values(
        train._time_index, np_only_0_anomalies
    )

    modified_train = MovingAverage(window=10).filter(train)
    modified_test = MovingAverage(window=10).filter(test)

    np_probabilistic = np.random.normal(loc=10, scale=1, size=[100, 1, 20])
    probabilistic = TimeSeries.from_times_and_values(
        train._time_index, np_probabilistic
    )

    # multivariate series
    np_MTS_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    MTS_train = TimeSeries.from_values(np_MTS_train)

    np_MTS_test = np.random.normal(loc=[10, 5], scale=[1, 1.5], size=[100, 2])
    MTS_test = TimeSeries.from_times_and_values(MTS_train._time_index, np_MTS_test)

    np_MTS_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    MTS_anomalies = TimeSeries.from_times_and_values(
        MTS_train._time_index, np_MTS_anomalies
    )

    modified_MTS_train = MovingAverage(window=10).filter(MTS_train)
    modified_MTS_test = MovingAverage(window=10).filter(MTS_test)

    np_MTS_probabilistic = np.random.normal(
        loc=[[10], [5]], scale=[[1], [1.5]], size=[100, 2, 20]
    )
    MTS_probabilistic = TimeSeries.from_times_and_values(
        MTS_train._time_index, np_MTS_probabilistic
    )

    def test_ScoreNonFittableAnomalyScorer(self):
        scorer = S.Norm()

        # Check return types for score_from_prediction()
        # Check if return type is float when input is a series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction(self.test, self.modified_test), TimeSeries
            )
        )

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.test], [self.modified_test]),
                Sequence,
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction(self.MTS_test, self.modified_MTS_test),
                TimeSeries,
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.MTS_test], [self.modified_MTS_test]),
                Sequence,
            )
        )

    def test_ScoreFittableAnomalyScorer(self):
        scorer = S.KMeansScorer()

        # Check return types for score()
        scorer.fit(self.train)
        # Check if return type is float when input is a series
        self.assertTrue(isinstance(scorer.score(self.test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of series
        self.assertTrue(isinstance(scorer.score([self.test]), Sequence))

        scorer.fit(self.MTS_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(scorer.score(self.MTS_test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of multivariate series
        self.assertTrue(isinstance(scorer.score([self.MTS_test]), Sequence))

        # Check return types for score_from_prediction()
        scorer.fit_from_prediction(self.train, self.modified_train)
        # Check if return type is float when input is a series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction(self.test, self.modified_test), TimeSeries
            )
        )

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.test], [self.modified_test]),
                Sequence,
            )
        )

        scorer.fit_from_prediction(self.MTS_train, self.modified_MTS_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction(self.MTS_test, self.modified_MTS_test),
                TimeSeries,
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.MTS_test], [self.modified_MTS_test]),
                Sequence,
            )
        )

    def test_eval_accuracy_from_prediction(self):

        scorer = S.Norm(component_wise=False)
        # Check return types
        # Check if return type is float when input is a series
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, self.test, self.modified_test
                ),
                float,
            )
        )

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, [self.test], self.modified_test
                ),
                Sequence,
            )
        )

        # Check if return type is a float when input is a multivariate series and component_wise is set to False
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, self.MTS_test, self.modified_MTS_test
                ),
                float,
            )
        )

        # Check if return type is Sequence when input is a multivariate series and component_wise is set to False
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, [self.MTS_test], self.modified_MTS_test
                ),
                Sequence,
            )
        )

        scorer = S.Norm(component_wise=True)
        # Check return types
        # Check if return type is float when input is a series
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, self.test, self.modified_test
                ),
                float,
            )
        )

        # Check if return type is Sequence when input is a Sequence of series
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, [self.test], self.modified_test
                ),
                Sequence,
            )
        )

        # Check if return type is a float when input is a multivariate series and component_wise is set to True
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.MTS_anomalies, self.MTS_test, self.modified_MTS_test
                ),
                Sequence,
            )
        )

        # Check if return type is Sequence when input is a multivariate series and component_wise is set to True
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.MTS_anomalies, [self.MTS_test], self.modified_MTS_test
                ),
                Sequence,
            )
        )

        Non_fittable_scorer = S.Norm(component_wise=False)
        Fittable_scorer = S.KMeansScorer(component_wise=False)
        Fittable_scorer.fit(self.train)

        # if component_wise set to False, 'actual_anomalies' must have widths of 1
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.MTS_anomalies, series=self.test
            )
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.MTS_anomalies],
                series=[self.test, self.test],
            )

        # 'metric' must be str and "AUC_ROC" or "AUC_PR"
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric=1
            )
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric="auc_roc"
            )
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric=["AUC_ROC"]
            )

        # 'actual_anomalies' must be binary
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(actual_anomalies=self.test, series=self.test)

        # 'actual_anomalies' must contain anomalies (at least one)
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.only_0_anomalies, series=self.test
            )

        # 'actual_anomalies' cannot contain only anomalies
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.only_1_anomalies, series=self.test
            )

        # 'actual_anomalies' must match the number of series
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=[self.test, self.test]
            )
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.anomalies], series=self.test
            )

        # 'actual_anomalies' must have non empty intersection with 'series'
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies[:20], series=self.test[30:]
            )
        with self.assertRaises(ValueError):
            Fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.anomalies[:20]],
                series=[self.test, self.test[40:]],
            )

        for scorer in [Non_fittable_scorer, Fittable_scorer]:

            # 'metric' must be str and "AUC_ROC" or "AUC_PR"
            with self.assertRaises(ValueError):
                Fittable_scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                    metric=1,
                )
            with self.assertRaises(ValueError):
                Fittable_scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                    metric="auc_roc",
                )
            with self.assertRaises(ValueError):
                Fittable_scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                    metric=["AUC_ROC"],
                )

            # 'actual_anomalies' must be binary
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.test,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                )

            # 'actual_anomalies' must contain anomalies (at least one)
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.only_0_anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                )

            # 'actual_anomalies' cannot contain only anomalies
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.only_1_anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                )

            # 'actual_anomalies' must match the number of series
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=[self.test, self.test],
                    pred_series=[self.modified_test, self.modified_test],
                )
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=[self.anomalies, self.anomalies],
                    actual_series=self.test,
                    pred_series=self.modified_test,
                )

            # 'actual_anomalies' must have non empty intersection with 'actual_series' and 'pred_series'
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies[:20],
                    actual_series=self.test[30:],
                    pred_series=self.modified_test[30:],
                )
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=[self.anomalies, self.anomalies[:20]],
                    actual_series=[self.test, self.test[40:]],
                    pred_series=[self.modified_test, self.modified_test[40:]],
                )

    def test_NonFittableAnomalyScorer(self):

        for scorer in list_NonFittableAnomalyScorer:
            # Check if trainable is False, being a NonFittableAnomalyScorer
            self.assertTrue(not scorer.trainable)

            # checks for score_from_prediction()
            # input must be Timeseries or sequence of Timeseries
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train, "str")
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train], [self.modified_train, "str"]
                )
            # score on sequence with series that have different width
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train, self.modified_MTS_train)
            # input sequences have different length
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train], [self.modified_train]
                )
            # two inputs must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train[:50], self.train[55:])
            # every pairwise element must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train[:50]], [self.train, self.train[55:]]
                )

    def test_FittableAnomalyScorer(self):

        for scorer in list_FittableAnomalyScorer:

            # Need to call fit() before calling score()
            with self.assertRaises(ValueError):
                scorer.score(self.test)

            # Need to call fit() before calling score_from_prediction()
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.test, self.modified_test)

            # Check if trainable is True, being a FittableAnomalyScorer
            self.assertTrue(scorer.trainable)

            # Check if _fit_called is False
            self.assertTrue(not scorer._fit_called)

            # fit on sequence with series that have different width
            with self.assertRaises(ValueError):
                scorer.fit([self.train, self.MTS_train])

            # fit on sequence with series that have different width
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.MTS_train],
                    [self.modified_train, self.modified_MTS_train],
                )

            # checks for fit_from_prediction()
            # input must be Timeseries or sequence of Timeseries
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train, "str")
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train], [self.modified_train, "str"]
                )
            # two inputs must have the same length
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.train], [self.modified_train]
                )
            # two inputs must have the same width
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction([self.train], [self.modified_MTS_train])
            # every element must have the same width
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.MTS_train],
                    [self.modified_train, self.modified_MTS_train],
                )
            # two inputs must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(self.train[:50], self.train[55:])
            # every pairwise element must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.train[:50]], [self.train, self.train[55:]]
                )

            # checks for fit()
            # input must be Timeseries or sequence of Timeseries
            with self.assertRaises(ValueError):
                scorer.fit("str")
            with self.assertRaises(ValueError):
                scorer.fit([self.modified_train, "str"])

            # checks for score_from_prediction()
            scorer.fit_from_prediction(self.train, self.modified_train)
            # input must be Timeseries or sequence of Timeseries
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train, "str")
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train], [self.modified_train, "str"]
                )
            # two inputs must have the same length
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train], [self.modified_train]
                )
            # two inputs must have the same width
            with self.assertRaises(ValueError):
                scorer.score_from_prediction([self.train], [self.modified_MTS_train])
            # every element must have the same width
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.MTS_train],
                    [self.modified_train, self.modified_MTS_train],
                )
            # two inputs must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(self.train[:50], self.train[55:])
            # every pairwise element must have a non zero intersection
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.train[:50]], [self.train, self.train[55:]]
                )

            # checks for score()
            # input must be Timeseries or sequence of Timeseries
            with self.assertRaises(ValueError):
                scorer.score("str")
            with self.assertRaises(ValueError):
                scorer.score([self.modified_train, "str"])

            # caseA: fit with fit()
            # case1: fit on UTS
            scorerA1 = scorer
            scorerA1.fit(self.train)
            # Check if _fit_called is True after being fitted
            self.assertTrue(scorerA1._fit_called)
            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                scorerA1.score(self.MTS_test)
            # case2: fit on MTS
            scorerA2 = scorer
            scorerA2.fit(self.MTS_train)
            # Check if _fit_called is True after being fitted
            self.assertTrue(scorerA2._fit_called)
            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                scorerA2.score(self.test)

            # caseB: fit with fit_from_prediction()
            # case1: fit on UTS
            scorerB1 = scorer
            scorerB1.fit_from_prediction(self.train, self.modified_train)
            # Check if _fit_called is True after being fitted
            self.assertTrue(scorerB1._fit_called)
            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                scorerB1.score_from_prediction(self.MTS_test, self.modified_MTS_test)
            # case2: fit on MTS
            scorerB2 = scorer
            scorerB2.fit_from_prediction(self.MTS_train, self.modified_MTS_train)
            # Check if _fit_called is True after being fitted
            self.assertTrue(scorerB2._fit_called)
            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                scorerB2.score_from_prediction(self.test, self.modified_test)

    def test_Norm(self):

        # component_wise must be bool
        with self.assertRaises(ValueError):
            S.Norm(component_wise=1)
        with self.assertRaises(ValueError):
            S.Norm(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = S.Norm(component_wise=False)
        self.assertTrue(
            scorer.score_from_prediction(self.test, self.modified_test).width == 1
        )
        self.assertTrue(
            scorer.score_from_prediction(self.MTS_test, self.modified_MTS_test).width
            == 1
        )
        # if component_wise=True must always return the same width as the input
        scorer = S.Norm(component_wise=True)
        self.assertTrue(
            scorer.score_from_prediction(self.test, self.modified_test).width == 1
        )
        self.assertTrue(
            scorer.score_from_prediction(self.MTS_test, self.modified_MTS_test).width
            == self.MTS_test.width
        )

        scorer = S.Norm()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)

    def test_Difference(self):

        scorer = S.Difference()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)

    def test_WassersteinScorer(self):

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            S.WassersteinScorer(component_wise=1)
        with self.assertRaises(ValueError):
            S.WassersteinScorer(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = S.WassersteinScorer(component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = S.WassersteinScorer(component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == self.MTS_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            S.WassersteinScorer(window=True)
        with self.assertRaises(ValueError):
            S.WassersteinScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            S.WassersteinScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            S.WassersteinScorer(window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            S.WassersteinScorer(diff_fn="random")
        with self.assertRaises(ValueError):
            S.WassersteinScorer(diff_fn=1)

        scorer = S.WassersteinScorer()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = S.WassersteinScorer(window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = S.WassersteinScorer(window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

    def test_KMeansScorer(self):

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            S.KMeansScorer(component_wise=1)
        with self.assertRaises(ValueError):
            S.KMeansScorer(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = S.KMeansScorer(component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = S.KMeansScorer(component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == self.MTS_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            S.KMeansScorer(window=True)
        with self.assertRaises(ValueError):
            S.KMeansScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            S.KMeansScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            S.KMeansScorer(window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            S.KMeansScorer(diff_fn="random")
        with self.assertRaises(ValueError):
            S.KMeansScorer(diff_fn=1)

        scorer = S.KMeansScorer()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = S.KMeansScorer(window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = S.KMeansScorer(window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

    def test_PyODScorer(self):

        # model parameter must be pyod.models typy BaseDetector
        with self.assertRaises(ValueError):
            S.PyODScorer(model=MovingAverage(window=10))

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), component_wise=1)
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = S.PyODScorer(model=KNN(), component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = S.PyODScorer(model=KNN(), component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.MTS_train)
        self.assertTrue(scorer.score(self.MTS_test).width == self.MTS_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), window=True)
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), diff_fn="random")
        with self.assertRaises(ValueError):
            S.PyODScorer(model=KNN(), diff_fn=1)

        scorer = S.PyODScorer(model=KNN())

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = S.PyODScorer(model=KNN(), window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = S.PyODScorer(model=KNN(), window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

    def test_NLLScorer(self):

        for s in list_NLLScorer:
            # expects for 'actual_series' a deterministic input and for 'pred_series' a probabilistic input
            with self.assertRaises(ValueError):
                s.score_from_prediction(actual_series=self.test, pred_series=self.test)
            with self.assertRaises(ValueError):
                s.score_from_prediction(
                    actual_series=self.probabilistic, pred_series=self.probabilistic
                )
            with self.assertRaises(ValueError):
                s.score_from_prediction(
                    actual_series=self.probabilistic, pred_series=self.train
                )

    def test_GaussianNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            S.GaussianNLLScorer(window=True)
        with self.assertRaises(ValueError):
            S.GaussianNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            S.GaussianNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            S.GaussianNLLScorer(window=0)

        scorer = S.GaussianNLLScorer(window=101)
        # window must be smaller than the input of score()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100
