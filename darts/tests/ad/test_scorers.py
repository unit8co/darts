from typing import Sequence

import numpy as np
from pyod.models.knn import KNN
from scipy.stats import cauchy, expon, gamma, laplace, norm, poisson

from darts import TimeSeries
from darts.ad.scorers import CauchyNLLScorer
from darts.ad.scorers import DifferenceScorer as Difference
from darts.ad.scorers import (
    ExponentialNLLScorer,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
)
from darts.ad.scorers import NormScorer as Norm
from darts.ad.scorers import PoissonNLLScorer, PyODScorer, WassersteinScorer
from darts.models import MovingAverage
from darts.tests.base_test_class import DartsBaseTestClass

list_NonFittableAnomalyScorer = [
    Norm(),
    Difference(),
    GaussianNLLScorer(),
    ExponentialNLLScorer(),
    PoissonNLLScorer(),
    LaplaceNLLScorer(),
    CauchyNLLScorer(),
    GammaNLLScorer(),
]

list_FittableAnomalyScorer = [
    PyODScorer(model=KNN()),
    KMeansScorer(),
    WassersteinScorer(),
]

list_NLLScorer = [
    GaussianNLLScorer(),
    ExponentialNLLScorer(),
    PoissonNLLScorer(),
    LaplaceNLLScorer(),
    CauchyNLLScorer(),
    GammaNLLScorer(),
]


class ADAnomalyScorerTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_test = np.random.normal(loc=10, scale=2, size=100)
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

    np_probabilistic = np.random.normal(loc=10, scale=2, size=[100, 1, 20])
    probabilistic = TimeSeries.from_times_and_values(
        train._time_index, np_probabilistic
    )

    # multivariate series
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)

    np_mts_test = np.random.normal(loc=[10, 5], scale=[1, 1.5], size=[100, 2])
    mts_test = TimeSeries.from_times_and_values(mts_train._time_index, np_mts_test)

    np_mts_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    mts_anomalies = TimeSeries.from_times_and_values(
        mts_train._time_index, np_mts_anomalies
    )

    modified_mts_train = MovingAverage(window=10).filter(mts_train)
    modified_mts_test = MovingAverage(window=10).filter(mts_test)

    np_mts_probabilistic = np.random.normal(
        loc=[[10], [5]], scale=[[1], [1.5]], size=[100, 2, 20]
    )
    mts_probabilistic = TimeSeries.from_times_and_values(
        mts_train._time_index, np_mts_probabilistic
    )

    def test_ScoreNonFittableAnomalyScorer(self):
        scorer = Norm()

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
                scorer.score_from_prediction(self.mts_test, self.modified_mts_test),
                TimeSeries,
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.mts_test], [self.modified_mts_test]),
                Sequence,
            )
        )

    def test_ScoreFittableAnomalyScorer(self):
        scorer = KMeansScorer()

        # Check return types for score()
        scorer.fit(self.train)
        # Check if return type is float when input is a series
        self.assertTrue(isinstance(scorer.score(self.test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of series
        self.assertTrue(isinstance(scorer.score([self.test]), Sequence))

        scorer.fit(self.mts_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(isinstance(scorer.score(self.mts_test), TimeSeries))

        # Check if return type is Sequence when input is a sequence of multivariate series
        self.assertTrue(isinstance(scorer.score([self.mts_test]), Sequence))

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

        scorer.fit_from_prediction(self.mts_train, self.modified_mts_train)
        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction(self.mts_test, self.modified_mts_test),
                TimeSeries,
            )
        )

        # Check if return type is Sequence when input is a multivariate series
        self.assertTrue(
            isinstance(
                scorer.score_from_prediction([self.mts_test], [self.modified_mts_test]),
                Sequence,
            )
        )

    def test_eval_accuracy_from_prediction(self):

        scorer = Norm(component_wise=False)
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
                    self.anomalies, self.mts_test, self.modified_mts_test
                ),
                float,
            )
        )

        # Check if return type is Sequence when input is a multivariate series and component_wise is set to False
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.anomalies, [self.mts_test], self.modified_mts_test
                ),
                Sequence,
            )
        )

        scorer = Norm(component_wise=True)
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
                    self.mts_anomalies, self.mts_test, self.modified_mts_test
                ),
                Sequence,
            )
        )

        # Check if return type is Sequence when input is a multivariate series and component_wise is set to True
        self.assertTrue(
            isinstance(
                scorer.eval_accuracy_from_prediction(
                    self.mts_anomalies, [self.mts_test], self.modified_mts_test
                ),
                Sequence,
            )
        )

        non_fittable_scorer = Norm(component_wise=False)
        fittable_scorer = KMeansScorer(component_wise=False)
        fittable_scorer.fit(self.train)

        # if component_wise set to False, 'actual_anomalies' must have widths of 1
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.mts_anomalies, series=self.test
            )
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.mts_anomalies],
                series=[self.test, self.test],
            )

        # 'metric' must be str and "AUC_ROC" or "AUC_PR"
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric=1
            )
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric="auc_roc"
            )
        with self.assertRaises(TypeError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies, series=self.test, metric=["AUC_ROC"]
            )

        # 'actual_anomalies' must be binary
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(actual_anomalies=self.test, series=self.test)

        # 'actual_anomalies' must contain anomalies (at least one)
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.only_0_anomalies, series=self.test
            )

        # 'actual_anomalies' cannot contain only anomalies
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.only_1_anomalies, series=self.test
            )

        # 'actual_anomalies' must match the number of series if length higher than 1
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.anomalies], series=self.test
            )
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.anomalies],
                series=[self.test, self.test, self.test],
            )

        # 'actual_anomalies' must have non empty intersection with 'series'
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=self.anomalies[:20], series=self.test[30:]
            )
        with self.assertRaises(ValueError):
            fittable_scorer.eval_accuracy(
                actual_anomalies=[self.anomalies, self.anomalies[:20]],
                series=[self.test, self.test[40:]],
            )

        for scorer in [non_fittable_scorer, fittable_scorer]:

            # name must be of type str
            self.assertEqual(
                type(scorer.__str__()),
                str,
            )

            # 'metric' must be str and "AUC_ROC" or "AUC_PR"
            with self.assertRaises(ValueError):
                fittable_scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                    metric=1,
                )
            with self.assertRaises(ValueError):
                fittable_scorer.eval_accuracy_from_prediction(
                    actual_anomalies=self.anomalies,
                    actual_series=self.test,
                    pred_series=self.modified_test,
                    metric="auc_roc",
                )
            with self.assertRaises(TypeError):
                fittable_scorer.eval_accuracy_from_prediction(
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

            # 'actual_anomalies' must match the number of series if length higher than 1
            with self.assertRaises(ValueError):
                scorer.eval_accuracy_from_prediction(
                    actual_anomalies=[self.anomalies, self.anomalies],
                    actual_series=[self.test, self.test, self.test],
                    pred_series=[
                        self.modified_test,
                        self.modified_test,
                        self.modified_test,
                    ],
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
                scorer.score_from_prediction(self.train, self.modified_mts_train)
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
                scorer.fit([self.train, self.mts_train])

            # fit on sequence with series that have different width
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.mts_train],
                    [self.modified_train, self.modified_mts_train],
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
                scorer.fit_from_prediction([self.train], [self.modified_mts_train])
            # every element must have the same width
            with self.assertRaises(ValueError):
                scorer.fit_from_prediction(
                    [self.train, self.mts_train],
                    [self.modified_train, self.modified_mts_train],
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
                scorer.score_from_prediction([self.train], [self.modified_mts_train])
            # every element must have the same width
            with self.assertRaises(ValueError):
                scorer.score_from_prediction(
                    [self.train, self.mts_train],
                    [self.modified_train, self.modified_mts_train],
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
                scorerA1.score(self.mts_test)
            # case2: fit on MTS
            scorerA2 = scorer
            scorerA2.fit(self.mts_train)
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
                scorerB1.score_from_prediction(self.mts_test, self.modified_mts_test)
            # case2: fit on MTS
            scorerB2 = scorer
            scorerB2.fit_from_prediction(self.mts_train, self.modified_mts_train)
            # Check if _fit_called is True after being fitted
            self.assertTrue(scorerB2._fit_called)
            with self.assertRaises(ValueError):
                # series must be same width as series used for training
                scorerB2.score_from_prediction(self.test, self.modified_test)

    def test_Norm(self):

        # component_wise must be bool
        with self.assertRaises(ValueError):
            Norm(component_wise=1)
        with self.assertRaises(ValueError):
            Norm(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = Norm(component_wise=False)
        self.assertTrue(
            scorer.score_from_prediction(self.test, self.modified_test).width == 1
        )
        self.assertTrue(
            scorer.score_from_prediction(self.mts_test, self.modified_mts_test).width
            == 1
        )
        # if component_wise=True must always return the same width as the input
        scorer = Norm(component_wise=True)
        self.assertTrue(
            scorer.score_from_prediction(self.test, self.modified_test).width == 1
        )
        self.assertTrue(
            scorer.score_from_prediction(self.mts_test, self.modified_mts_test).width
            == self.mts_test.width
        )

        scorer = Norm(component_wise=True)
        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)

        # univariate case (equivalent to abs diff)
        self.assertEqual(
            scorer.score_from_prediction(self.test, self.test + 1)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            len(self.test),
        )
        self.assertEqual(
            scorer.score_from_prediction(self.test + 1, self.test)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            len(self.test),
        )

        # multivariate case with component_wise set to True (equivalent to abs diff)
        # abs(a - 2a) =  a
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["0"],
            self.mts_test["0"],
        )
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["1"],
            self.mts_test["1"],
        )
        # abs(2a - a) =  a
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["0"],
            self.mts_test["0"],
        )
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["1"],
            self.mts_test["1"],
        )

        scorer = Norm(component_wise=False)

        # always expects a deterministic input

        # univariate case (equivalent to abs diff)
        self.assertEqual(
            scorer.score_from_prediction(self.test, self.test + 1)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            len(self.test),
        )
        self.assertEqual(
            scorer.score_from_prediction(self.test + 1, self.test)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            len(self.test),
        )

        # multivariate case with component_wise set to False
        # norm(a - a + sqrt(2)) = 2 * len(a) with a being series of dim=2
        self.assertAlmostEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test + np.sqrt(2))
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            2 * len(self.mts_test),
            delta=1e-05,
        )

        self.assertFalse(scorer.is_probabilistic)

    def test_Difference(self):

        scorer = Difference()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)

        # univariate case
        self.assertEqual(
            scorer.score_from_prediction(self.test, self.test + 1)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            -len(self.test),
        )
        self.assertEqual(
            scorer.score_from_prediction(self.test + 1, self.test)
            .sum(axis=0)
            .all_values()
            .flatten()[0],
            len(self.test),
        )

        # multivariate case
        # output of score() must be the same width as the width of the input
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test).width,
            self.mts_test.width,
        )

        # a - 2a = - a
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["0"],
            -self.mts_test["0"],
        )
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["1"],
            -self.mts_test["1"],
        )
        # 2a - a =  a
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["0"],
            self.mts_test["0"],
        )
        self.assertEqual(
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["1"],
            self.mts_test["1"],
        )

        self.assertFalse(scorer.is_probabilistic)

    def test_WassersteinScorer(self):

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            WassersteinScorer(component_wise=1)
        with self.assertRaises(ValueError):
            WassersteinScorer(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = WassersteinScorer(component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = WassersteinScorer(component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == self.mts_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            WassersteinScorer(window=True)
        with self.assertRaises(ValueError):
            WassersteinScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            WassersteinScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            WassersteinScorer(window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            WassersteinScorer(diff_fn="random")
        with self.assertRaises(ValueError):
            WassersteinScorer(diff_fn=1)

        # test _diff_series() directly
        with self.assertRaises(ValueError):
            s_tmp = WassersteinScorer()
            s_tmp.diff_fn = "random"
            s_tmp._diff_series(self.train, self.test)
        WassersteinScorer(diff_fn="diff")._diff_series(self.train, self.test)
        WassersteinScorer()._diff_series(self.train, self.test)

        scorer = WassersteinScorer()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = WassersteinScorer(window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = WassersteinScorer(window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

        # test plotting (just call the functions)
        scorer = WassersteinScorer(window=2)
        scorer.fit(self.train)
        scorer.show_anomalies(self.test, self.anomalies)
        with self.assertRaises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies([self.test, self.test], self.anomalies)
        scorer.show_anomalies_from_prediction(
            actual_series=self.test,
            pred_series=self.test + 1,
            actual_anomalies=self.anomalies,
        )
        with self.assertRaises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies_from_prediction(
                actual_series=[self.test, self.test],
                pred_series=self.test + 1,
                actual_anomalies=self.anomalies,
            )
        with self.assertRaises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies_from_prediction(
                actual_series=self.test,
                pred_series=[self.test + 1, self.test + 2],
                actual_anomalies=self.anomalies,
            )

        self.assertFalse(scorer.is_probabilistic)

    def test_univariate_Wasserstein(self):

        # univariate example
        np.random.seed(42)

        np_train_wasserstein = np.abs(np.random.normal(loc=0, scale=0.1, size=100))
        train_wasserstein = TimeSeries.from_times_and_values(
            self.train._time_index, np_train_wasserstein
        )

        np_test_wasserstein = np.abs(np.random.normal(loc=0, scale=0.1, size=100))
        np_first_anomaly = np.abs(np.random.normal(loc=0, scale=0.25, size=10))
        np_second_anomaly = np.abs(np.random.normal(loc=0.25, scale=0.05, size=5))
        np_third_anomaly = np.abs(np.random.normal(loc=0, scale=0.15, size=15))

        np_test_wasserstein[10:20] = np_first_anomaly
        np_test_wasserstein[40:45] = np_second_anomaly
        np_test_wasserstein[70:85] = np_third_anomaly
        test_wasserstein = TimeSeries.from_times_and_values(
            self.train._time_index, np_test_wasserstein
        )

        # create the anomaly series
        np_anomalies = np.zeros(len(test_wasserstein))
        np_anomalies[10:17] = 1
        np_anomalies[40:42] = 1
        np_anomalies[70:85] = 1
        anomalies_wasserstein = TimeSeries.from_times_and_values(
            test_wasserstein.time_index, np_anomalies, columns=["is_anomaly"]
        )

        # test model with window of 10
        scorer_10 = WassersteinScorer(window=10)
        scorer_10.fit(train_wasserstein)
        auc_roc_w10 = scorer_10.eval_accuracy(
            anomalies_wasserstein, test_wasserstein, metric="AUC_ROC"
        )
        auc_pr_w10 = scorer_10.eval_accuracy(
            anomalies_wasserstein, test_wasserstein, metric="AUC_PR"
        )

        # test model with window of 20
        scorer_20 = WassersteinScorer(window=20)
        scorer_20.fit(train_wasserstein)
        auc_roc_w20 = scorer_20.eval_accuracy(
            anomalies_wasserstein, test_wasserstein, metric="AUC_ROC"
        )
        auc_pr_w20 = scorer_20.eval_accuracy(
            anomalies_wasserstein, test_wasserstein, metric="AUC_PR"
        )

        self.assertAlmostEqual(auc_roc_w10, 0.80637, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w10, 0.83390, delta=1e-05)
        self.assertAlmostEqual(auc_roc_w20, 0.77828, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w20, 0.93934, delta=1e-05)

    def test_multivariate_componentwise_Wasserstein(self):

        # example multivariate WassersteinScorer component wise (True and False)
        np.random.seed(3)
        np_mts_train_wasserstein = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        mts_train_wasserstein = TimeSeries.from_times_and_values(
            self.train._time_index, np_mts_train_wasserstein
        )

        np_mts_test_wasserstein = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        np_first_anomaly_width1 = np.abs(np.random.normal(loc=0.5, scale=0.4, size=10))
        np_first_anomaly_width2 = np.abs(np.random.normal(loc=0, scale=0.5, size=10))
        np_first_commmon_anomaly = np.abs(
            np.random.normal(loc=0.5, scale=0.5, size=[10, 2])
        )

        np_mts_test_wasserstein[5:15, 0] = np_first_anomaly_width1
        np_mts_test_wasserstein[35:45, 1] = np_first_anomaly_width2
        np_mts_test_wasserstein[65:75, :] = np_first_commmon_anomaly

        mts_test_wasserstein = TimeSeries.from_times_and_values(
            mts_train_wasserstein._time_index, np_mts_test_wasserstein
        )

        # create the anomaly series width 1
        np_anomalies_width1 = np.zeros(len(mts_test_wasserstein))
        np_anomalies_width1[5:15] = 1
        np_anomalies_width1[65:75] = 1

        # create the anomaly series width 2
        np_anomaly_width2 = np.zeros(len(mts_test_wasserstein))
        np_anomaly_width2[35:45] = 1
        np_anomaly_width2[65:75] = 1

        anomalies_wasserstein_per_width = TimeSeries.from_times_and_values(
            mts_test_wasserstein.time_index,
            list(zip(*[np_anomalies_width1, np_anomaly_width2])),
            columns=["is_anomaly_0", "is_anomaly_1"],
        )

        # create the anomaly series for the entire series
        np_commmon_anomaly = np.zeros(len(mts_test_wasserstein))
        np_commmon_anomaly[5:15] = 1
        np_commmon_anomaly[35:45] = 1
        np_commmon_anomaly[65:75] = 1
        anomalies_common_wasserstein = TimeSeries.from_times_and_values(
            mts_test_wasserstein.time_index, np_commmon_anomaly, columns=["is_anomaly"]
        )

        # test scorer with component_wise=False
        scorer_w10_cwfalse = WassersteinScorer(window=10, component_wise=False)
        scorer_w10_cwfalse.fit(mts_train_wasserstein)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_accuracy(
            anomalies_common_wasserstein, mts_test_wasserstein, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = WassersteinScorer(window=10, component_wise=True)
        scorer_w10_cwtrue.fit(mts_train_wasserstein)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_accuracy(
            anomalies_wasserstein_per_width, mts_test_wasserstein, metric="AUC_ROC"
        )

        self.assertAlmostEqual(auc_roc_cwfalse, 0.94637, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[0], 0.98606, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[1], 0.96722, delta=1e-05)

    def test_kmeansScorer(self):

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            KMeansScorer(component_wise=1)
        with self.assertRaises(ValueError):
            KMeansScorer(component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = KMeansScorer(component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = KMeansScorer(component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == self.mts_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            KMeansScorer(window=True)
        with self.assertRaises(ValueError):
            KMeansScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            KMeansScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            KMeansScorer(window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            KMeansScorer(diff_fn="random")
        with self.assertRaises(ValueError):
            KMeansScorer(diff_fn=1)

        scorer = KMeansScorer()

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = KMeansScorer(window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = KMeansScorer(window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

        self.assertFalse(scorer.is_probabilistic)

    def test_univariate_kmeans(self):

        # univariate example

        np.random.seed(40)

        # create the train set
        np_width1 = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
        np_width2 = (np_width1 == 0).astype(float)
        KMeans_mts_train = TimeSeries.from_values(
            np.dstack((np_width1, np_width2))[0], columns=["component 1", "component 2"]
        )

        # create the test set
        # inject anomalies in the test timeseries
        np.random.seed(3)
        np_width1 = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
        np_width2 = (np_width1 == 0).astype(int)

        # 2 anomalies per type
        # type 1: random values for only one width
        np_width1[20:21] = 2
        np_width2[30:32] = 2

        # type 2: shift both widths values (+/- 1 for both widths)
        np_width1[45:47] = np_width1[45:47] + 1
        np_width2[45:47] = np_width2[45:47] + 1
        np_width1[60:64] = np_width1[65:69] - 1
        np_width2[60:64] = np_width2[65:69] - 1

        # type 3: switch one state to another for only one width (1 to 0 for one width)
        np_width1[75:82] = (np_width1[75:82] != 1).astype(int)
        np_width2[90:96] = (np_width2[90:96] != 1).astype(int)

        KMeans_mts_test = TimeSeries.from_values(
            np.dstack((np_width1, np_width2))[0], columns=["component 1", "component 2"]
        )

        # create the anomaly series
        anomalies_index = [
            20,
            30,
            31,
            45,
            46,
            60,
            61,
            62,
            63,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            90,
            91,
            92,
            93,
            94,
            95,
        ]
        np_anomalies = np.zeros(len(KMeans_mts_test))
        np_anomalies[anomalies_index] = 1
        KMeans_mts_anomalies = TimeSeries.from_times_and_values(
            KMeans_mts_test.time_index, np_anomalies, columns=["is_anomaly"]
        )

        kmeans_scorer = KMeansScorer(k=2, window=1, component_wise=False)
        kmeans_scorer.fit(KMeans_mts_train)

        metric_AUC_ROC = kmeans_scorer.eval_accuracy(
            KMeans_mts_anomalies, KMeans_mts_test, metric="AUC_ROC"
        )
        metric_AUC_PR = kmeans_scorer.eval_accuracy(
            KMeans_mts_anomalies, KMeans_mts_test, metric="AUC_PR"
        )

        self.assertEqual(metric_AUC_ROC, 1.0)
        self.assertEqual(metric_AUC_PR, 1.0)

    def test_multivariate_window_kmeans(self):

        # multivariate example with different windows

        np.random.seed(1)

        # create the train set
        np_series = np.zeros(100)
        np_series[0] = 2

        for i in range(1, len(np_series)):
            np_series[i] = np_series[i - 1] + np.random.choice(a=[-1, 1], p=[0.5, 0.5])
            if np_series[i] > 3:
                np_series[i] = 3
            if np_series[i] < 0:
                np_series[i] = 0

        ts_train = TimeSeries.from_values(np_series, columns=["series"])

        # create the test set
        np.random.seed(3)
        np_series = np.zeros(100)
        np_series[0] = 1

        for i in range(1, len(np_series)):
            np_series[i] = np_series[i - 1] + np.random.choice(a=[-1, 1], p=[0.5, 0.5])
            if np_series[i] > 3:
                np_series[i] = 3
            if np_series[i] < 0:
                np_series[i] = 0

        # 3 anomalies per type
        # type 1: sudden shift between state 0 to state 2 without passing by state 1
        np_series[23] = 3
        np_series[44] = 3
        np_series[91] = 0

        # type 2: having consecutive timestamps at state 1 or 2
        np_series[3:5] = 2
        np_series[17:19] = 1
        np_series[62:65] = 2

        ts_test = TimeSeries.from_values(np_series, columns=["series"])

        anomalies_index = [4, 23, 18, 44, 63, 64, 91]
        np_anomalies = np.zeros(100)
        np_anomalies[anomalies_index] = 1
        ts_anomalies = TimeSeries.from_times_and_values(
            ts_test.time_index, np_anomalies, columns=["is_anomaly"]
        )

        kmeans_scorer_w1 = KMeansScorer(k=4, window=1)
        kmeans_scorer_w1.fit(ts_train)

        kmeans_scorer_w2 = KMeansScorer(k=8, window=2)
        kmeans_scorer_w2.fit(ts_train)

        auc_roc_w1 = kmeans_scorer_w1.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w1 = kmeans_scorer_w1.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_PR"
        )

        auc_roc_w2 = kmeans_scorer_w2.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w2 = kmeans_scorer_w2.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_PR"
        )

        self.assertAlmostEqual(auc_roc_w1, 0.41551, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w1, 0.064761, delta=1e-05)
        self.assertAlmostEqual(auc_roc_w2, 0.957513, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w2, 0.88584, delta=1e-05)

    def test_multivariate_componentwise_kmeans(self):

        # example multivariate KMeans component wise (True and False)
        np.random.seed(1)

        np_mts_train_kmeans = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        mts_train_kmeans = TimeSeries.from_times_and_values(
            self.train._time_index, np_mts_train_kmeans
        )

        np_mts_test_kmeans = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        np_first_anomaly_width1 = np.abs(np.random.normal(loc=0.5, scale=0.4, size=10))
        np_first_anomaly_width2 = np.abs(np.random.normal(loc=0, scale=0.5, size=10))
        np_first_commmon_anomaly = np.abs(
            np.random.normal(loc=0.5, scale=0.5, size=[10, 2])
        )

        np_mts_test_kmeans[5:15, 0] = np_first_anomaly_width1
        np_mts_test_kmeans[35:45, 1] = np_first_anomaly_width2
        np_mts_test_kmeans[65:75, :] = np_first_commmon_anomaly

        mts_test_kmeans = TimeSeries.from_times_and_values(
            mts_train_kmeans._time_index, np_mts_test_kmeans
        )

        # create the anomaly series width 1
        np_anomalies_width1 = np.zeros(len(mts_test_kmeans))
        np_anomalies_width1[5:15] = 1
        np_anomalies_width1[65:75] = 1

        # create the anomaly series width 2
        np_anomaly_width2 = np.zeros(len(mts_test_kmeans))
        np_anomaly_width2[35:45] = 1
        np_anomaly_width2[65:75] = 1

        anomalies_kmeans_per_width = TimeSeries.from_times_and_values(
            mts_test_kmeans.time_index,
            list(zip(*[np_anomalies_width1, np_anomaly_width2])),
            columns=["is_anomaly_0", "is_anomaly_1"],
        )

        # create the anomaly series for the entire series
        np_commmon_anomaly = np.zeros(len(mts_test_kmeans))
        np_commmon_anomaly[5:15] = 1
        np_commmon_anomaly[35:45] = 1
        np_commmon_anomaly[65:75] = 1
        anomalies_common_kmeans = TimeSeries.from_times_and_values(
            mts_test_kmeans.time_index, np_commmon_anomaly, columns=["is_anomaly"]
        )

        # test scorer with component_wise=False
        scorer_w10_cwfalse = KMeansScorer(window=10, component_wise=False)
        scorer_w10_cwfalse.fit(mts_train_kmeans)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_accuracy(
            anomalies_common_kmeans, mts_test_kmeans, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = KMeansScorer(window=10, component_wise=True)
        scorer_w10_cwtrue.fit(mts_train_kmeans)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_accuracy(
            anomalies_kmeans_per_width, mts_test_kmeans, metric="AUC_ROC"
        )

        self.assertAlmostEqual(auc_roc_cwfalse, 0.9851, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[0], 1.0, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[1], 0.97666, delta=1e-05)

    def test_PyODScorer(self):

        # model parameter must be pyod.models typy BaseDetector
        with self.assertRaises(ValueError):
            PyODScorer(model=MovingAverage(window=10))

        # component_wise parameter
        # component_wise must be bool
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), component_wise=1)
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), component_wise="string")
        # if component_wise=False must always return a univariate anomaly score
        scorer = PyODScorer(model=KNN(), component_wise=False)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == 1)
        # if component_wise=True must always return the same width as the input
        scorer = PyODScorer(model=KNN(), component_wise=True)
        scorer.fit(self.train)
        self.assertTrue(scorer.score(self.test).width == 1)
        scorer.fit(self.mts_train)
        self.assertTrue(scorer.score(self.mts_test).width == self.mts_test.width)

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), window=True)
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), window=0)

        # diff_fn paramter
        # must be None, 'diff' or 'abs_diff'
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), diff_fn="random")
        with self.assertRaises(ValueError):
            PyODScorer(model=KNN(), diff_fn=1)

        scorer = PyODScorer(model=KNN())

        # always expects a deterministic input
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.train, self.probabilistic)
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(self.probabilistic, self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.probabilistic)

        # window must be smaller than the input of score()
        scorer = PyODScorer(model=KNN(), window=101)
        with self.assertRaises(ValueError):
            scorer.fit(self.train)  # len(self.train)=100

        scorer = PyODScorer(model=KNN(), window=80)
        scorer.fit(self.train)
        with self.assertRaises(ValueError):
            scorer.score(self.test[:50])  # len(self.test)=100

        self.assertFalse(scorer.is_probabilistic)

    def test_univariate_PyODScorer(self):

        # univariate test
        np.random.seed(40)

        # create the train set
        np_width1 = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
        np_width2 = (np_width1 == 0).astype(float)
        pyod_mts_train = TimeSeries.from_values(
            np.dstack((np_width1, np_width2))[0], columns=["component 1", "component 2"]
        )

        # create the test set
        # inject anomalies in the test timeseries
        np.random.seed(3)
        np_width1 = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
        np_width2 = (np_width1 == 0).astype(int)

        # 2 anomalies per type
        # type 1: random values for only one width
        np_width1[20:21] = 2
        np_width2[30:32] = 2

        # type 2: shift both widths values (+/- 1 for both widths)
        np_width1[45:47] = np_width1[45:47] + 1
        np_width2[45:47] = np_width2[45:47] + 1
        np_width1[60:64] = np_width1[65:69] - 1
        np_width2[60:64] = np_width2[65:69] - 1

        # type 3: switch one state to another for only one width (1 to 0 for one width)
        np_width1[75:82] = (np_width1[75:82] != 1).astype(int)
        np_width2[90:96] = (np_width2[90:96] != 1).astype(int)

        pyod_mts_test = TimeSeries.from_values(
            np.dstack((np_width1, np_width2))[0], columns=["component 1", "component 2"]
        )

        # create the anomaly series
        anomalies_index = [
            20,
            30,
            31,
            45,
            46,
            60,
            61,
            62,
            63,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            90,
            91,
            92,
            93,
            94,
            95,
        ]
        np_anomalies = np.zeros(len(pyod_mts_test))
        np_anomalies[anomalies_index] = 1
        pyod_mts_anomalies = TimeSeries.from_times_and_values(
            pyod_mts_test.time_index, np_anomalies, columns=["is_anomaly"]
        )

        pyod_scorer = PyODScorer(
            model=KNN(n_neighbors=10), component_wise=False, window=1
        )
        pyod_scorer.fit(pyod_mts_train)

        metric_AUC_ROC = pyod_scorer.eval_accuracy(
            pyod_mts_anomalies, pyod_mts_test, metric="AUC_ROC"
        )
        metric_AUC_PR = pyod_scorer.eval_accuracy(
            pyod_mts_anomalies, pyod_mts_test, metric="AUC_PR"
        )

        self.assertEqual(metric_AUC_ROC, 1.0)
        self.assertEqual(metric_AUC_PR, 1.0)

    def test_multivariate_window_PyODScorer(self):

        # multivariate example (with different window)

        np.random.seed(1)

        # create the train set
        np_series = np.zeros(100)
        np_series[0] = 2

        for i in range(1, len(np_series)):
            np_series[i] = np_series[i - 1] + np.random.choice(a=[-1, 1], p=[0.5, 0.5])
            if np_series[i] > 3:
                np_series[i] = 3
            if np_series[i] < 0:
                np_series[i] = 0

        ts_train = TimeSeries.from_values(np_series, columns=["series"])

        # create the test set
        np.random.seed(3)
        np_series = np.zeros(100)
        np_series[0] = 1

        for i in range(1, len(np_series)):
            np_series[i] = np_series[i - 1] + np.random.choice(a=[-1, 1], p=[0.5, 0.5])
            if np_series[i] > 3:
                np_series[i] = 3
            if np_series[i] < 0:
                np_series[i] = 0

        # 3 anomalies per type
        # type 1: sudden shift between state 0 to state 2 without passing by state 1
        np_series[23] = 3
        np_series[44] = 3
        np_series[91] = 0

        # type 2: having consecutive timestamps at state 1 or 2
        np_series[3:5] = 2
        np_series[17:19] = 1
        np_series[62:65] = 2

        ts_test = TimeSeries.from_values(np_series, columns=["series"])

        anomalies_index = [4, 23, 18, 44, 63, 64, 91]
        np_anomalies = np.zeros(100)
        np_anomalies[anomalies_index] = 1
        ts_anomalies = TimeSeries.from_times_and_values(
            ts_test.time_index, np_anomalies, columns=["is_anomaly"]
        )

        pyod_scorer_w1 = PyODScorer(
            model=KNN(n_neighbors=10), component_wise=False, window=1
        )
        pyod_scorer_w1.fit(ts_train)

        pyod_scorer_w2 = PyODScorer(
            model=KNN(n_neighbors=10), component_wise=False, window=2
        )
        pyod_scorer_w2.fit(ts_train)

        auc_roc_w1 = pyod_scorer_w1.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w1 = pyod_scorer_w1.eval_accuracy(ts_anomalies, ts_test, metric="AUC_PR")

        auc_roc_w2 = pyod_scorer_w2.eval_accuracy(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w2 = pyod_scorer_w2.eval_accuracy(ts_anomalies, ts_test, metric="AUC_PR")

        self.assertAlmostEqual(auc_roc_w1, 0.5, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w1, 0.07, delta=1e-05)
        self.assertAlmostEqual(auc_roc_w2, 0.957513, delta=1e-05)
        self.assertAlmostEqual(auc_pr_w2, 0.88584, delta=1e-05)

    def test_multivariate_componentwise_PyODScorer(self):

        # multivariate example with component wise (True and False)

        np.random.seed(1)

        np_mts_train_PyOD = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        mts_train_PyOD = TimeSeries.from_times_and_values(
            self.train._time_index, np_mts_train_PyOD
        )

        np_mts_test_PyOD = np.abs(
            np.random.normal(loc=[0, 0], scale=[0.1, 0.2], size=[100, 2])
        )
        np_first_anomaly_width1 = np.abs(np.random.normal(loc=0.5, scale=0.4, size=10))
        np_first_anomaly_width2 = np.abs(np.random.normal(loc=0, scale=0.5, size=10))
        np_first_commmon_anomaly = np.abs(
            np.random.normal(loc=0.5, scale=0.5, size=[10, 2])
        )

        np_mts_test_PyOD[5:15, 0] = np_first_anomaly_width1
        np_mts_test_PyOD[35:45, 1] = np_first_anomaly_width2
        np_mts_test_PyOD[65:75, :] = np_first_commmon_anomaly

        mts_test_PyOD = TimeSeries.from_times_and_values(
            mts_train_PyOD._time_index, np_mts_test_PyOD
        )

        # create the anomaly series width 1
        np_anomalies_width1 = np.zeros(len(mts_test_PyOD))
        np_anomalies_width1[5:15] = 1
        np_anomalies_width1[65:75] = 1

        # create the anomaly series width 2
        np_anomaly_width2 = np.zeros(len(mts_test_PyOD))
        np_anomaly_width2[35:45] = 1
        np_anomaly_width2[65:75] = 1

        anomalies_pyod_per_width = TimeSeries.from_times_and_values(
            mts_test_PyOD.time_index,
            list(zip(*[np_anomalies_width1, np_anomaly_width2])),
            columns=["is_anomaly_0", "is_anomaly_1"],
        )

        # create the anomaly series for the entire series
        np_commmon_anomaly = np.zeros(len(mts_test_PyOD))
        np_commmon_anomaly[5:15] = 1
        np_commmon_anomaly[35:45] = 1
        np_commmon_anomaly[65:75] = 1
        anomalies_common_PyOD = TimeSeries.from_times_and_values(
            mts_test_PyOD.time_index, np_commmon_anomaly, columns=["is_anomaly"]
        )

        # test scorer with component_wise=False
        scorer_w10_cwfalse = PyODScorer(
            model=KNN(n_neighbors=10), component_wise=False, window=10
        )
        scorer_w10_cwfalse.fit(mts_train_PyOD)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_accuracy(
            anomalies_common_PyOD, mts_test_PyOD, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = PyODScorer(
            model=KNN(n_neighbors=10), component_wise=True, window=10
        )
        scorer_w10_cwtrue.fit(mts_train_PyOD)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_accuracy(
            anomalies_pyod_per_width, mts_test_PyOD, metric="AUC_ROC"
        )

        self.assertAlmostEqual(auc_roc_cwfalse, 0.990566, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[0], 1.0, delta=1e-05)
        self.assertAlmostEqual(auc_roc_cwtrue[1], 0.98311, delta=1e-05)

    def test_NLLScorer(self):

        for s in list_NLLScorer:
            # expects for 'actual_series' a deterministic input and for 'pred_series' a probabilistic input
            with self.assertRaises(ValueError):
                s.score_from_prediction(actual_series=self.test, pred_series=self.test)
            with self.assertRaises(ValueError):
                s.score_from_prediction(
                    actual_series=self.probabilistic, pred_series=self.train
                )

    def test_GaussianNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            GaussianNLLScorer(window=True)
        with self.assertRaises(ValueError):
            GaussianNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            GaussianNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            GaussianNLLScorer(window=0)

        scorer = GaussianNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)
        scorer = GaussianNLLScorer()

        # test 1 univariate (len=1 and window=1)
        gaussian_samples_1 = np.random.normal(loc=0, scale=2, size=10000)
        distribution_series = TimeSeries.from_values(
            gaussian_samples_1.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(
            value_test1, -np.log(norm.pdf(3, loc=0, scale=2)), delta=1e-01
        )

        # test 2 univariate (len=1 and window=1)
        gaussian_samples_2 = np.random.normal(loc=0, scale=2, size=10000)
        distribution_series = TimeSeries.from_values(
            gaussian_samples_2.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([-2]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(
            value_test2, -np.log(norm.pdf(-2, loc=0, scale=2)), delta=1e-01
        )

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array(
                [gaussian_samples_1.reshape(1, -1), gaussian_samples_2.reshape(1, -1)]
            )
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = GaussianNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = GaussianNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([gaussian_samples_1, gaussian_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertEqual(value_multivariate.all_values().flatten()[0], value_test1)
        self.assertEqual(value_multivariate.all_values().flatten()[1], value_test2)

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = GaussianNLLScorer(window=1)
        scorer_w2 = GaussianNLLScorer(window=2)

        gaussian_samples_3 = np.random.normal(loc=0, scale=2, size=10000)
        gaussian_samples_4 = np.random.normal(loc=0, scale=2, size=10000)

        distribution_series = TimeSeries.from_values(
            np.array(
                [
                    gaussian_samples_1,
                    gaussian_samples_2,
                    gaussian_samples_3,
                    gaussian_samples_4,
                ]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(
            np.array([1.5, 2.1, 0.1, 0.001]).reshape(2, -1)
        )

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0],
            -np.log(norm.pdf(1.5, loc=0, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[1],
            -np.log(norm.pdf(2.1, loc=0, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2],
            -np.log(norm.pdf(0.1, loc=0, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3],
            -np.log(norm.pdf(0.001, loc=0, scale=2)),
            delta=1e-01,
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (
                -np.log(norm.pdf(1.5, loc=0, scale=2))
                - np.log(norm.pdf(0.1, loc=0, scale=2))
            )
            / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[1],
            (
                -np.log(norm.pdf(2.1, loc=0, scale=2))
                - np.log(norm.pdf(0.001, loc=0, scale=2))
            )
            / 2,
            delta=1e-01,
        )

        self.assertTrue(scorer.is_probabilistic)

    def test_LaplaceNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            LaplaceNLLScorer(window=True)
        with self.assertRaises(ValueError):
            LaplaceNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            LaplaceNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            LaplaceNLLScorer(window=0)

        scorer = LaplaceNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)

        scorer = LaplaceNLLScorer()

        # test 1 univariate (len=1 and window=1)
        laplace_samples_1 = np.random.laplace(loc=0, scale=2, size=1000)
        distribution_series = TimeSeries.from_values(
            laplace_samples_1.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples
            value_test1,
            -np.log(laplace.pdf(3, loc=0, scale=2)),
            delta=1e-01,
        )

        # test 2 univariate (len=1 and window=1)
        laplace_samples_2 = np.random.laplace(loc=0, scale=2, size=1000)
        distribution_series = TimeSeries.from_values(
            laplace_samples_2.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([-2]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples
            value_test2,
            -np.log(laplace.pdf(-2, loc=0, scale=2)),
            delta=1e-01,
        )

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array(
                [laplace_samples_1.reshape(1, -1), laplace_samples_2.reshape(1, -1)]
            )
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertAlmostEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertAlmostEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = LaplaceNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = LaplaceNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([laplace_samples_1, laplace_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertAlmostEqual(
            value_multivariate.all_values().flatten()[0], value_test1
        )
        self.assertAlmostEqual(
            value_multivariate.all_values().flatten()[1], value_test2
        )

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = LaplaceNLLScorer(window=1)
        scorer_w2 = LaplaceNLLScorer(window=2)

        laplace_samples_3 = np.random.laplace(loc=0, scale=2, size=1000)
        laplace_samples_4 = np.random.laplace(loc=0, scale=2, size=1000)

        distribution_series = TimeSeries.from_values(
            np.array(
                [
                    laplace_samples_1,
                    laplace_samples_2,
                    laplace_samples_3,
                    laplace_samples_4,
                ]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(
            np.array([1.5, 2, 0.1, 0.001]).reshape(2, -1)
        )

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0],
            -np.log(laplace.pdf(1.5, loc=0, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples
            score_w1.all_values().flatten()[1],
            -np.log(laplace.pdf(2, loc=0, scale=2)),
            delta=0.5,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2],
            -np.log(laplace.pdf(0.1, loc=0, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3],
            -np.log(laplace.pdf(0.001, loc=0, scale=2)),
            delta=1e-01,
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (
                -np.log(laplace.pdf(1.5, loc=0, scale=2))
                - np.log(laplace.pdf(0.1, loc=0, scale=2))
            )
            / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples
            score_w2.all_values().flatten()[1],
            (
                -np.log(laplace.pdf(2, loc=0, scale=2))
                - np.log(laplace.pdf(0.001, loc=0, scale=2))
            )
            / 2,
            delta=0.5,
        )

        self.assertTrue(scorer.is_probabilistic)

    def test_ExponentialNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            ExponentialNLLScorer(window=True)
        with self.assertRaises(ValueError):
            ExponentialNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            ExponentialNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            ExponentialNLLScorer(window=0)

        scorer = ExponentialNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)
        scorer = ExponentialNLLScorer()

        # test 1 univariate (len=1 and window=1)
        exponential_samples_1 = np.random.exponential(scale=2.0, size=1000)
        distribution_series = TimeSeries.from_values(
            exponential_samples_1.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples and also uses loc
            value_test1,
            -np.log(expon.pdf(3, scale=2.0)),
            delta=1e-01,
        )

        # test 2 univariate (len=1 and window=1)
        exponential_samples_2 = np.random.exponential(scale=2.0, size=1000)
        distribution_series = TimeSeries.from_values(
            exponential_samples_2.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([10]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples and also uses loc
            value_test2,
            -np.log(expon.pdf(10, scale=2)),
            delta=1e-01,
        )

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array(
                [
                    exponential_samples_1.reshape(1, -1),
                    exponential_samples_2.reshape(1, -1),
                ]
            )
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = ExponentialNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = ExponentialNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([exponential_samples_1, exponential_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertEqual(value_multivariate.all_values().flatten()[0], value_test1)
        self.assertEqual(value_multivariate.all_values().flatten()[1], value_test2)

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = ExponentialNLLScorer(window=1)
        scorer_w2 = ExponentialNLLScorer(window=2)

        exponential_samples_3 = np.random.exponential(scale=2, size=1000)
        exponential_samples_4 = np.random.exponential(scale=2, size=1000)

        distribution_series = TimeSeries.from_values(
            np.array(
                [
                    exponential_samples_1,
                    exponential_samples_2,
                    exponential_samples_3,
                    exponential_samples_4,
                ]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(
            np.array([1.5, 2, 0.1, 0.001]).reshape(2, -1)
        )

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0],
            -np.log(expon.pdf(1.5, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[1],
            -np.log(expon.pdf(2, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2],
            -np.log(expon.pdf(0.1, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3],
            -np.log(expon.pdf(0.001, scale=2)),
            delta=1e-01,
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (-np.log(expon.pdf(1.5, scale=2)) - np.log(expon.pdf(0.1, scale=2))) / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[1],
            (-np.log(expon.pdf(2, scale=2)) - np.log(expon.pdf(0.001, scale=2))) / 2,
            delta=1e-01,
        )

        self.assertTrue(scorer.is_probabilistic)

    def test_GammaNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            GammaNLLScorer(window=True)
        with self.assertRaises(ValueError):
            GammaNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            GammaNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            GammaNLLScorer(window=0)

        scorer = GammaNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)
        scorer = GammaNLLScorer()

        # test 1 univariate (len=1 and window=1)
        gamma_samples_1 = np.random.gamma(shape=2, scale=2, size=10000)
        distribution_series = TimeSeries.from_values(gamma_samples_1.reshape(1, 1, -1))
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples and also uses loc
            value_test1,
            -np.log(gamma.pdf(3, 2, scale=2)),
            delta=1e-01,
        )

        # test 2 univariate (len=1 and window=1)
        gamma_samples_2 = np.random.gamma(2, scale=2, size=10000)
        distribution_series = TimeSeries.from_values(gamma_samples_2.reshape(1, 1, -1))
        actual_series = TimeSeries.from_values(np.array([10]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(
            # This is approximate because our NLL scorer is fit from samples and also uses loc
            value_test2,
            -np.log(gamma.pdf(10, 2, scale=2)),
            delta=1e-01,
        )

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array([gamma_samples_1.reshape(1, -1), gamma_samples_2.reshape(1, -1)])
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = GammaNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = GammaNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([gamma_samples_1, gamma_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertEqual(value_multivariate.all_values().flatten()[0], value_test1)
        self.assertEqual(value_multivariate.all_values().flatten()[1], value_test2)

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = GammaNLLScorer(window=1)
        scorer_w2 = GammaNLLScorer(window=2)

        gamma_samples_3 = np.random.gamma(2, scale=2, size=10000)
        gamma_samples_4 = np.random.gamma(2, scale=2, size=10000)

        distribution_series = TimeSeries.from_values(
            np.array(
                [gamma_samples_1, gamma_samples_2, gamma_samples_3, gamma_samples_4]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(
            np.array([1.5, 2, 0.5, 0.9]).reshape(2, -1)
        )

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0],
            -np.log(gamma.pdf(1.5, 2, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[1],
            -np.log(gamma.pdf(2, 2, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2],
            -np.log(gamma.pdf(0.5, 2, scale=2)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3],
            -np.log(gamma.pdf(0.9, 2, scale=2)),
            delta=1e-01,
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (-np.log(gamma.pdf(1.5, 2, scale=2)) - np.log(gamma.pdf(0.5, 2, scale=2)))
            / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[1],
            (-np.log(gamma.pdf(2, 2, scale=2)) - np.log(gamma.pdf(0.9, 2, scale=2)))
            / 2,
            delta=1e-01,
        )

        self.assertTrue(scorer.is_probabilistic)

    def test_CauchyNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            CauchyNLLScorer(window=True)
        with self.assertRaises(ValueError):
            CauchyNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            CauchyNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            CauchyNLLScorer(window=0)

        scorer = CauchyNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)
        scorer = CauchyNLLScorer()

        # test 1 univariate (len=1 and window=1)
        cauchy_samples_1 = np.random.standard_cauchy(size=10000)
        distribution_series = TimeSeries.from_values(cauchy_samples_1.reshape(1, 1, -1))
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(value_test1, -np.log(cauchy.pdf(3)), delta=1e-01)

        # test 2 univariate (len=1 and window=1)
        cauchy_samples_2 = np.random.standard_cauchy(size=10000)
        distribution_series = TimeSeries.from_values(cauchy_samples_2.reshape(1, 1, -1))
        actual_series = TimeSeries.from_values(np.array([-2]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(value_test2, -np.log(cauchy.pdf(-2)), delta=1e-01)

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array([cauchy_samples_1.reshape(1, -1), cauchy_samples_2.reshape(1, -1)])
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = CauchyNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = CauchyNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([cauchy_samples_1, cauchy_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, -2]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertEqual(value_multivariate.all_values().flatten()[0], value_test1)
        self.assertEqual(value_multivariate.all_values().flatten()[1], value_test2)

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = CauchyNLLScorer(window=1)
        scorer_w2 = CauchyNLLScorer(window=2)

        cauchy_samples_3 = np.random.standard_cauchy(size=10000)
        cauchy_samples_4 = np.random.standard_cauchy(size=10000)

        distribution_series = TimeSeries.from_values(
            np.array(
                [cauchy_samples_1, cauchy_samples_2, cauchy_samples_3, cauchy_samples_4]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(
            np.array([1.5, 2, 0.5, 0.9]).reshape(2, -1)
        )

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0], -np.log(cauchy.pdf(1.5)), delta=1e-01
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[1], -np.log(cauchy.pdf(2)), delta=1e-01
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2], -np.log(cauchy.pdf(0.5)), delta=1e-01
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3], -np.log(cauchy.pdf(0.9)), delta=1e-01
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (-np.log(cauchy.pdf(1.5)) - np.log(cauchy.pdf(0.5))) / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[1],
            (-np.log(cauchy.pdf(2)) - np.log(cauchy.pdf(0.9))) / 2,
            delta=1e-01,
        )

        self.assertTrue(scorer.is_probabilistic)

    def test_PoissonNLLScorer(self):

        # window parameter
        # window must be int
        with self.assertRaises(ValueError):
            PoissonNLLScorer(window=True)
        with self.assertRaises(ValueError):
            PoissonNLLScorer(window="string")
        # window must be non negative
        with self.assertRaises(ValueError):
            PoissonNLLScorer(window=-1)
        # window must be different from 0
        with self.assertRaises(ValueError):
            PoissonNLLScorer(window=0)

        scorer = PoissonNLLScorer(window=101)
        # window must be smaller than the input of score_from_prediction()
        with self.assertRaises(ValueError):
            scorer.score_from_prediction(
                actual_series=self.test, pred_series=self.probabilistic
            )  # len(self.test)=100

        np.random.seed(4)
        scorer = PoissonNLLScorer()

        # test 1 univariate (len=1 and window=1)
        poisson_samples_1 = np.random.poisson(size=10000, lam=1)
        distribution_series = TimeSeries.from_values(
            poisson_samples_1.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3]))
        value_test1 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test1 is the - log likelihood
        self.assertAlmostEqual(value_test1, -np.log(poisson.pmf(3, mu=1)), delta=1e-02)

        # test 2 univariate (len=1 and window=1)
        poisson_samples_2 = np.random.poisson(size=10000, lam=1)
        distribution_series = TimeSeries.from_values(
            poisson_samples_2.reshape(1, 1, -1)
        )
        actual_series = TimeSeries.from_values(np.array([10]))
        value_test2 = (
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0]
        )

        # check if value_test2 is the - log likelihood
        self.assertAlmostEqual(value_test2, -np.log(poisson.pmf(10, mu=1)), delta=1e-01)

        # test window univariate (len=2 and window=2)
        distribution_series = TimeSeries.from_values(
            np.array(
                [poisson_samples_1.reshape(1, -1), poisson_samples_2.reshape(1, -1)]
            )
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]))
        value_window = scorer.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(value_window), 2)
        # check width
        self.assertEqual(value_window.width, 1)

        # check equal value_test1 and value_test2
        self.assertEqual(value_window.all_values().flatten()[0], value_test1)
        self.assertEqual(value_window.all_values().flatten()[1], value_test2)

        scorer = PoissonNLLScorer(window=2)
        # check avg of two values
        self.assertEqual(
            scorer.score_from_prediction(actual_series, distribution_series)
            .all_values()
            .flatten()[0],
            (value_test1 + value_test2) / 2,
        )

        # test window multivariate (n_samples=2, len=1, window=1)
        scorer = PoissonNLLScorer(window=1)
        distribution_series = TimeSeries.from_values(
            np.array([poisson_samples_1, poisson_samples_2]).reshape(1, 2, -1)
        )
        actual_series = TimeSeries.from_values(np.array([3, 10]).reshape(1, -1))
        value_multivariate = scorer.score_from_prediction(
            actual_series, distribution_series
        )

        # check length
        self.assertEqual(len(value_multivariate), 1)
        # check width
        self.assertEqual(value_multivariate.width, 2)

        # check equal value_test1 and value_test2
        self.assertEqual(value_multivariate.all_values().flatten()[0], value_test1)
        self.assertEqual(value_multivariate.all_values().flatten()[1], value_test2)

        # test window multivariate (n_samples=2, len=2, window=1 and 2)
        scorer_w1 = PoissonNLLScorer(window=1)
        scorer_w2 = PoissonNLLScorer(window=2)

        poisson_samples_3 = np.random.poisson(size=10000, lam=1)
        poisson_samples_4 = np.random.poisson(size=10000, lam=1)

        distribution_series = TimeSeries.from_values(
            np.array(
                [
                    poisson_samples_1,
                    poisson_samples_2,
                    poisson_samples_3,
                    poisson_samples_4,
                ]
            ).reshape(2, 2, -1)
        )

        actual_series = TimeSeries.from_values(np.array([1, 2, 3, 4]).reshape(2, -1))

        score_w1 = scorer_w1.score_from_prediction(actual_series, distribution_series)
        score_w2 = scorer_w2.score_from_prediction(actual_series, distribution_series)

        # check length
        self.assertEqual(len(score_w1), 2)
        self.assertEqual(len(score_w2), 1)
        # check width
        self.assertEqual(score_w1.width, 2)
        self.assertEqual(score_w2.width, 2)

        # check values for window=1
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[0],
            -np.log(poisson.pmf(1, mu=1)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[1],
            -np.log(poisson.pmf(2, mu=1)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[2],
            -np.log(poisson.pmf(3, mu=1)),
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w1.all_values().flatten()[3],
            -np.log(poisson.pmf(4, mu=1)),
            delta=1e-01,
        )

        # check values for window=2 (must be equal to the mean of the past 2 values)
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[0],
            (-np.log(poisson.pmf(1, mu=1)) - np.log(poisson.pmf(3, mu=1))) / 2,
            delta=1e-01,
        )
        self.assertAlmostEqual(
            score_w2.all_values().flatten()[1],
            (-np.log(poisson.pmf(2, mu=1)) - np.log(poisson.pmf(4, mu=1))) / 2,
            delta=1e-01,
        )

        self.assertTrue(scorer.is_probabilistic)
