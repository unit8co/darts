from collections.abc import Sequence
from itertools import product

import numpy as np
import pytest
from pyod.models.knn import KNN
from scipy.stats import cauchy, expon, gamma, laplace, norm, poisson

from darts import TimeSeries, metrics
from darts.ad.scorers import (
    CauchyNLLScorer,
    ExponentialNLLScorer,
    FittableAnomalyScorer,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
    PoissonNLLScorer,
    PyODScorer,
    WassersteinScorer,
)
from darts.ad.scorers import DifferenceScorer as Difference
from darts.ad.scorers import NormScorer as Norm
from darts.ad.scorers.scorers import NLLScorer
from darts.models import MovingAverageFilter
from darts.utils.timeseries_generation import linear_timeseries

list_NonFittableAnomalyScorer = [
    Norm(component_wise=False),
    Norm(component_wise=True),
    Difference(),
    GaussianNLLScorer(),
    ExponentialNLLScorer(),
    PoissonNLLScorer(),
    LaplaceNLLScorer(),
    CauchyNLLScorer(),
    GammaNLLScorer(),
]

list_FittableAnomalyScorer = [
    (PyODScorer, {"model": KNN(), "component_wise": False}),
    (KMeansScorer, {"component_wise": False}),
    (WassersteinScorer, {"window_agg": False, "component_wise": False}),
]

# (scorer_cls, values, distribution, distribution_kwargs, prob_density_func, prob_density_func)
list_NLLScorer = [
    (
        CauchyNLLScorer,
        [3, 2, 0.5, 0.9],
        np.random.standard_cauchy,
        {},
        cauchy.pdf,
        None,
    ),
    (
        ExponentialNLLScorer,
        [3, 0.1, 2, 0.01],
        np.random.exponential,
        {"scale": 2.0},
        expon.pdf,
        None,
    ),
    (
        GammaNLLScorer,
        [3, 0.1, 2, 0.5],
        np.random.gamma,
        {"shape": 2, "scale": 2},
        gamma.pdf,
        {"a": 2, "scale": 2},
    ),
    (
        GaussianNLLScorer,
        [3, 0.1, -2, 0.01],
        np.random.normal,
        {"loc": 0, "scale": 2},
        norm.pdf,
        None,
    ),
    (
        LaplaceNLLScorer,
        [3, 10, -2, 0.01],
        np.random.laplace,
        {"loc": 0, "scale": 2},
        laplace.pdf,
        None,
    ),
    (
        PoissonNLLScorer,
        [3, 2, 10, 1],
        np.random.poisson,
        {"lam": 1},
        poisson.pmf,
        {"mu": 1},
    ),
]

delta = 1e-05


class TestAnomalyDetectionScorer:
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

    modified_train = MovingAverageFilter(window=10).filter(train)
    modified_test = MovingAverageFilter(window=10).filter(test)

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

    modified_mts_train = MovingAverageFilter(window=10).filter(mts_train)
    modified_mts_test = MovingAverageFilter(window=10).filter(mts_test)

    np_mts_probabilistic = np.random.normal(
        loc=[[10], [5]], scale=[[1], [1.5]], size=[100, 2, 20]
    )
    mts_probabilistic = TimeSeries.from_times_and_values(
        mts_train._time_index, np_mts_probabilistic
    )

    @pytest.mark.parametrize("scorer", list_NonFittableAnomalyScorer)
    def test_score_from_pred_non_fittable_scorer(self, scorer):
        # NLLScorer require deterministic `series`
        if isinstance(scorer, NLLScorer):
            # series and pred_series are both deterministic
            with pytest.raises(ValueError):
                scorer.score_from_prediction(series=self.test, pred_series=self.test)
            # series is probabilistic, pred_series is deterministic
            with pytest.raises(ValueError):
                scorer.score_from_prediction(
                    series=self.probabilistic, pred_series=self.train
                )

            score = scorer.score_from_prediction(
                series=self.train, pred_series=self.probabilistic
            )
            assert isinstance(score, TimeSeries)
            assert score.all_values().shape == (len(self.train), 1, 1)
        else:
            # Check if return type is float when input is a series
            assert isinstance(
                scorer.score_from_prediction(self.test, self.modified_test), TimeSeries
            )

            # Check if return type is Sequence when input is a Sequence of series
            assert isinstance(
                scorer.score_from_prediction([self.test], [self.modified_test]),
                Sequence,
            )

            # Check if return type is Sequence when input is a multivariate series
            assert isinstance(
                scorer.score_from_prediction(self.mts_test, self.modified_mts_test),
                TimeSeries,
            )

            # Check if return type is Sequence when input is a multivariate series
            assert isinstance(
                scorer.score_from_prediction([self.mts_test], [self.modified_mts_test]),
                Sequence,
            )

    @pytest.mark.parametrize("scorer_config", list_FittableAnomalyScorer)
    def test_score_return_type(self, scorer_config):
        scorer_cls, scorer_kwargs = scorer_config
        scorer = scorer_cls(**scorer_kwargs)

        scorer.fit(self.train)
        # Check if return type is float when input is a series
        assert isinstance(scorer.score(self.test), TimeSeries)

        # Check if return type is Sequence when input is a sequence of series
        assert isinstance(scorer.score([self.test]), Sequence)

        scorer.fit(self.mts_train)
        # Check if return type is Sequence when input is a multivariate series
        assert isinstance(scorer.score(self.mts_test), TimeSeries)

        # Check if return type is Sequence when input is a sequence of multivariate series
        assert isinstance(scorer.score([self.mts_test]), Sequence)

        # Check return types for score_from_prediction()
        scorer.fit_from_prediction(self.train, self.modified_train)
        # Check if return type is float when input is a series
        assert isinstance(
            scorer.score_from_prediction(self.test, self.modified_test), TimeSeries
        )

        # Check if return type is Sequence when input is a Sequence of series
        assert isinstance(
            scorer.score_from_prediction([self.test], [self.modified_test]),
            Sequence,
        )

        scorer.fit_from_prediction(self.mts_train, self.modified_mts_train)
        # Check if return type is Sequence when input is a multivariate series
        assert isinstance(
            scorer.score_from_prediction(self.mts_test, self.modified_mts_test),
            TimeSeries,
        )

        # Check if return type is Sequence when input is a multivariate series
        assert isinstance(
            scorer.score_from_prediction([self.mts_test], [self.modified_mts_test]),
            Sequence,
        )

    def test_eval_metric_from_prediction_return_type(self):
        scorer = Norm(component_wise=False)
        # Check if return type is float when input is a series
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.anomalies, self.test, self.modified_test
            ),
            float,
        )
        # Check if return type is Sequence when input is a Sequence of series
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.anomalies, [self.test], self.modified_test
            ),
            Sequence,
        )
        # Check if return type is a float when input is a multivariate series and component_wise is set to `False`
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.anomalies, self.mts_test, self.modified_mts_test
            ),
            float,
        )
        # Check if return type is Sequence when input is a multivariate series and component_wise is set to `False`
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.anomalies, [self.mts_test], self.modified_mts_test
            ),
            Sequence,
        )

        scorer = Norm(component_wise=True)
        # Check if return type is a float when input is a multivariate series and component_wise is set to `True`
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.mts_anomalies, self.mts_test, self.modified_mts_test
            ),
            Sequence,
        )
        # Check if return type is Sequence when input is a multivariate series and component_wise is set to `True`
        assert isinstance(
            scorer.eval_metric_from_prediction(
                self.mts_anomalies, [self.mts_test], self.modified_mts_test
            ),
            Sequence,
        )

    @pytest.mark.parametrize("scorer_config", list_FittableAnomalyScorer)
    def test_eval_metric_fittable_scorer(self, scorer_config):
        scorer_cls, scorer_kwargs = scorer_config
        fittable_scorer = scorer_cls(**scorer_kwargs)
        fittable_scorer.fit(self.train)

        # if component_wise set to False, 'anomalies' must have widths of 1
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(anomalies=self.mts_anomalies, series=self.test)
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=[self.anomalies, self.mts_anomalies],
                series=[self.test, self.test],
            )

        # 'metric' must be str and "AUC_ROC" or "AUC_PR"
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=self.anomalies, series=self.test, metric=1
            )
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=self.anomalies,
                series=self.test,
                metric="auc_roc",
            )
        with pytest.raises(TypeError):
            fittable_scorer.eval_metric(
                anomalies=self.anomalies,
                series=self.test,
                metric=["AUC_ROC"],
            )

        # 'anomalies' must be binary
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(anomalies=self.test, series=self.test)

        # 'anomalies' must contain anomalies (at least one)
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=self.only_0_anomalies, series=self.test
            )

        # 'anomalies' cannot contain only anomalies
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=self.only_1_anomalies, series=self.test
            )

        # 'anomalies' must match the number of series if length higher than 1
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=[self.anomalies, self.anomalies],
                series=self.test,
            )
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=[self.anomalies, self.anomalies],
                series=[self.test, self.test, self.test],
            )

        # 'anomalies' must have non empty intersection with 'series'
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=self.anomalies[:20], series=self.test[30:]
            )
        with pytest.raises(ValueError):
            fittable_scorer.eval_metric(
                anomalies=[self.anomalies, self.anomalies[:20]],
                series=[self.test, self.test[40:]],
            )

    @pytest.mark.parametrize(
        "scorer", [Norm(component_wise=False), KMeansScorer(component_wise=False)]
    )
    def test_eval_metric_from_prediction(self, scorer):
        if isinstance(scorer, FittableAnomalyScorer):
            scorer.fit(self.train)

        # name must be of type str
        assert isinstance(scorer.__str__(), str)

        # 'metric' must be str and "AUC_ROC" or "AUC_PR"
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.anomalies,
                series=self.test,
                pred_series=self.modified_test,
                metric=1,
            )
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.anomalies,
                series=self.test,
                pred_series=self.modified_test,
                metric="auc_roc",
            )
        with pytest.raises(TypeError):
            scorer.eval_metric_from_prediction(
                anomalies=self.anomalies,
                series=self.test,
                pred_series=self.modified_test,
                metric=["AUC_ROC"],
            )

        # 'anomalies' must be binary
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.test,
                series=self.test,
                pred_series=self.modified_test,
            )

        # 'anomalies' must contain anomalies (at least one)
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.only_0_anomalies,
                series=self.test,
                pred_series=self.modified_test,
            )

        # 'anomalies' cannot contain only anomalies
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.only_1_anomalies,
                series=self.test,
                pred_series=self.modified_test,
            )

        # 'anomalies' must match the number of series if length higher than 1
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=[self.anomalies, self.anomalies],
                series=[self.test, self.test, self.test],
                pred_series=[
                    self.modified_test,
                    self.modified_test,
                    self.modified_test,
                ],
            )
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=[self.anomalies, self.anomalies],
                series=self.test,
                pred_series=self.modified_test,
            )

        # 'anomalies' must have non empty intersection with 'series' and 'pred_series'
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=self.anomalies[:20],
                series=self.test[30:],
                pred_series=self.modified_test[30:],
            )
        with pytest.raises(ValueError):
            scorer.eval_metric_from_prediction(
                anomalies=[self.anomalies, self.anomalies[:20]],
                series=[self.test, self.test[40:]],
                pred_series=[self.modified_test, self.modified_test[40:]],
            )

    @pytest.mark.parametrize("scorer", list_NonFittableAnomalyScorer)
    def test_NonFittableAnomalyScorer(self, scorer):
        # Check if trainable is False, being a NonFittableAnomalyScorer
        assert not scorer.is_trainable

        # checks for score_from_prediction()
        # input must be Timeseries or sequence of Timeseries
        with pytest.raises(ValueError):
            scorer.score_from_prediction(self.train, "str")
        with pytest.raises(ValueError):
            scorer.score_from_prediction(
                [self.train, self.train], [self.modified_train, "str"]
            )
        # score on sequence with series that have different width
        with pytest.raises(ValueError):
            scorer.score_from_prediction(self.train, self.modified_mts_train)
        # input sequences have different length
        with pytest.raises(ValueError):
            scorer.score_from_prediction(
                [self.train, self.train], [self.modified_train]
            )
        # two inputs must have a non zero intersection
        with pytest.raises(ValueError):
            scorer.score_from_prediction(self.train[:50], self.train[55:])
        # every pairwise element must have a non zero intersection
        with pytest.raises(ValueError):
            scorer.score_from_prediction(
                [self.train, self.train[:50]], [self.train, self.train[55:]]
            )

    @pytest.mark.parametrize("scorer_config", list_FittableAnomalyScorer)
    def test_FittableAnomalyScorer(self, scorer_config):
        scorer_cls, scorer_kwargs = scorer_config
        fittable_scorer = scorer_cls(**scorer_kwargs)

        # Need to call fit() before calling score()
        with pytest.raises(ValueError):
            fittable_scorer.score(self.test)

        # Need to call fit() before calling score_from_prediction()
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(self.test, self.modified_test)

        # Check if _fit_called is False
        assert not fittable_scorer._fit_called

        # fit on sequence with series that have different width
        with pytest.raises(ValueError):
            fittable_scorer.fit([self.train, self.mts_train])

        # fit on sequence with series that have different width
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction(
                [self.train, self.mts_train],
                [self.modified_train, self.modified_mts_train],
            )

        # checks for fit_from_prediction()
        # input must be Timeseries or sequence of Timeseries
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(self.train, "str")
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train, self.train], [self.modified_train, "str"]
            )
        # two inputs must have the same length
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction(
                [self.train, self.train], [self.modified_train]
            )
        # two inputs must have the same width
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction([self.train], [self.modified_mts_train])
        # every element must have the same width
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction(
                [self.train, self.mts_train],
                [self.modified_train, self.modified_mts_train],
            )
        # two inputs must have a non zero intersection
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction(self.train[:50], self.train[55:])
        # every pairwise element must have a non zero intersection
        with pytest.raises(ValueError):
            fittable_scorer.fit_from_prediction(
                [self.train, self.train[:50]], [self.train, self.train[55:]]
            )

        # checks for fit()
        # input must be Timeseries or sequence of Timeseries
        with pytest.raises(ValueError):
            fittable_scorer.fit("str")
        with pytest.raises(ValueError):
            fittable_scorer.fit([self.modified_train, "str"])

        # checks for score_from_prediction()
        fittable_scorer.fit_from_prediction(self.train, self.modified_train)
        # input must be Timeseries or sequence of Timeseries
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(self.train, "str")
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train, self.train], [self.modified_train, "str"]
            )
        # two inputs must have the same length
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train, self.train], [self.modified_train]
            )
        # two inputs must have the same width
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train], [self.modified_mts_train]
            )
        # every element must have the same width
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train, self.mts_train],
                [self.modified_train, self.modified_mts_train],
            )
        # two inputs must have a non zero intersection
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(self.train[:50], self.train[55:])
        # every pairwise element must have a non zero intersection
        with pytest.raises(ValueError):
            fittable_scorer.score_from_prediction(
                [self.train, self.train[:50]], [self.train, self.train[55:]]
            )

        # checks for score()
        # input must be Timeseries or sequence of Timeseries
        with pytest.raises(ValueError):
            fittable_scorer.score("str")
        with pytest.raises(ValueError):
            fittable_scorer.score([self.modified_train, "str"])

        # caseA: fit with fit()
        # case1: fit on UTS
        fittable_scorerA1 = fittable_scorer
        fittable_scorerA1.fit(self.train)
        # Check if _fit_called is True after being fitted
        assert fittable_scorerA1._fit_called
        with pytest.raises(ValueError):
            # series must be same width as series used for training
            fittable_scorerA1.score(self.mts_test)
        # case2: fit on MTS
        fittable_scorerA2 = fittable_scorer
        fittable_scorerA2.fit(self.mts_train)
        # Check if _fit_called is True after being fitted
        assert fittable_scorerA2._fit_called
        with pytest.raises(ValueError):
            # series must be same width as series used for training
            fittable_scorerA2.score(self.test)

        # caseB: fit with fit_from_prediction()
        # case1: fit on UTS
        fittable_scorerB1 = fittable_scorer
        fittable_scorerB1.fit_from_prediction(self.train, self.modified_train)
        # Check if _fit_called is True after being fitted
        assert fittable_scorerB1._fit_called
        with pytest.raises(ValueError):
            # series must be same width as series used for training
            fittable_scorerB1.score_from_prediction(
                self.mts_test, self.modified_mts_test
            )
        # case2: fit on MTS
        fittable_scorerB2 = fittable_scorer
        fittable_scorerB2.fit_from_prediction(self.mts_train, self.modified_mts_train)
        # Check if _fit_called is True after being fitted
        assert fittable_scorerB2._fit_called
        with pytest.raises(ValueError):
            # series must be same width as series used for training
            fittable_scorerB2.score_from_prediction(self.test, self.modified_test)

    def test_Norm(self):
        # Check parameters
        self.expects_deterministic_input(Norm)

        # if component_wise=False must always return a univariate anomaly score
        scorer = Norm(component_wise=False)
        assert scorer.score_from_prediction(self.test, self.modified_test).width == 1

        assert (
            scorer.score_from_prediction(self.mts_test, self.modified_mts_test).width
            == 1
        )

        # if component_wise=True must always return the same width as the input
        scorer = Norm(component_wise=True)
        assert scorer.score_from_prediction(self.test, self.modified_test).width == 1
        assert (
            scorer.score_from_prediction(self.mts_test, self.modified_mts_test).width
            == self.mts_test.width
        )

        scorer = Norm(component_wise=True)
        # univariate case (equivalent to abs diff)
        assert scorer.score_from_prediction(self.test, self.test + 1).sum(
            axis=0
        ).all_values().flatten()[0] == len(self.test)
        assert scorer.score_from_prediction(self.test + 1, self.test).sum(
            axis=0
        ).all_values().flatten()[0] == len(self.test)

        # multivariate case with component_wise set to True (equivalent to abs diff)
        # abs(a - 2a) =  a
        assert (
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["0"]
            == self.mts_test["0"]
        )
        assert (
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["1"]
            == self.mts_test["1"]
        )

        # abs(2a - a) =  a
        assert (
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["0"]
            == self.mts_test["0"]
        )
        assert (
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["1"]
            == self.mts_test["1"]
        )

        scorer = Norm(component_wise=False)

        # univariate case (equivalent to abs diff)
        assert scorer.score_from_prediction(self.test, self.test + 1).sum(
            axis=0
        ).all_values().flatten()[0] == len(self.test)
        assert scorer.score_from_prediction(self.test + 1, self.test).sum(
            axis=0
        ).all_values().flatten()[0] == len(self.test)

        # multivariate case with component_wise set to False
        # norm(a - a + sqrt(2)) = 2 * len(a) with a being series of dim=2
        assert (
            np.abs(
                2 * len(self.mts_test)
                - scorer.score_from_prediction(
                    self.mts_test, self.mts_test + np.sqrt(2)
                )
                .sum(axis=0)
                .all_values()
                .flatten()[0]
            )
            < delta
        )

        assert not scorer.is_probabilistic

    def test_Difference(self):
        self.expects_deterministic_input(Difference)

        scorer = Difference()

        # univariate case
        assert scorer.score_from_prediction(self.test, self.test + 1).sum(
            axis=0
        ).all_values().flatten()[0] == -len(self.test)

        assert (
            scorer.score_from_prediction(self.test + 1, self.test)
            .sum(axis=0)
            .all_values()
            .flatten()[0]
        ) == len(self.test)

        # multivariate case
        # output of score() must be the same width as the width of the input
        assert (
            scorer.score_from_prediction(self.mts_test, self.mts_test).width
        ) == self.mts_test.width

        # a - 2a = - a
        assert (
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["0"]
            == -self.mts_test["0"]
        )
        assert (
            scorer.score_from_prediction(self.mts_test, self.mts_test * 2)["1"]
            == -self.mts_test["1"]
        )
        # 2a - a =  a
        assert (
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["0"]
            == self.mts_test["0"]
        )
        assert (
            scorer.score_from_prediction(self.mts_test * 2, self.mts_test)["1"]
            == self.mts_test["1"]
        )

        assert not scorer.is_probabilistic

    @staticmethod
    def helper_check_type_window(scorer, **kwargs):
        # window must be non-negative
        with pytest.raises(ValueError):
            scorer(window=-1, **kwargs)
        # window must be different from 0
        with pytest.raises(ValueError):
            scorer(window=0, **kwargs)

    def helper_window_parameter(self, scorer_to_test, **kwargs):
        self.helper_check_type_window(scorer_to_test, **kwargs)

        if scorer_to_test(**kwargs).is_trainable:
            # window must be smaller than the input of score()
            scorer = scorer_to_test(window=len(self.train) + 1, **kwargs)
            with pytest.raises(ValueError):
                scorer.fit(self.train)

            scorer = scorer_to_test(window=len(self.train) - 20, **kwargs)
            scorer.fit(self.train)
            with pytest.raises(ValueError):
                scorer.score(self.test[: len(self.train) // 2])

        else:
            # case only NLL scorers for now

            scorer = scorer_to_test(window=101)
            # window must be smaller than the input of score_from_prediction()
            with pytest.raises(ValueError):
                scorer.score_from_prediction(
                    series=self.test, pred_series=self.probabilistic
                )  # len(self.test)=100

    def diff_fn_parameter(self, scorer, **kwargs):
        # must be one of Darts per time step metrics (e.g. ae, err, ...)
        with pytest.raises(ValueError):
            scorer(diff_fn="abs_diff", **kwargs)
        # absolute error / absolute difference
        s_tmp = scorer(diff_fn=metrics.ae, **kwargs)
        diffs = s_tmp._diff_series([self.train], [self.test])
        assert diffs == [abs(self.train - self.test)]
        # error / difference
        s_tmp = scorer(diff_fn=metrics.err, **kwargs)
        diffs = s_tmp._diff_series([self.train], [self.test])
        assert diffs == [self.train - self.test]

    def component_wise_parameter(self, scorer_to_test, **kwargs):
        # if component_wise=False must always return a univariate anomaly score
        scorer = scorer_to_test(component_wise=False, **kwargs)
        scorer.fit(self.train)
        assert scorer.score(self.test).width == 1
        scorer.fit(self.mts_train)
        assert scorer.score(self.mts_test).width == 1

        # if component_wise=True must always return the same width as the input
        scorer = scorer_to_test(component_wise=True, **kwargs)
        scorer.fit(self.train)
        assert scorer.score(self.test).width == 1
        scorer.fit(self.mts_train)
        assert scorer.score(self.mts_test).width == self.mts_test.width

    def check_diff_series(self, scorer, **kwargs):
        # test _diff_series() directly: parameter must by "abs_diff" or "diff"
        with pytest.raises(ValueError):
            s_tmp = scorer(**kwargs)
            s_tmp.diff_fn = "random"
            s_tmp._diff_series(self.train, self.test)

    def expects_deterministic_input(self, scorer, **kwargs):
        scorer = scorer(**kwargs)
        if scorer.is_trainable:
            scorer.fit(self.train)
            np.testing.assert_warns(scorer.score(self.probabilistic))

        # always expects a deterministic input
        np.testing.assert_warns(
            scorer.score_from_prediction(self.train, self.probabilistic)
        )
        np.testing.assert_warns(
            scorer.score_from_prediction(self.probabilistic, self.train)
        )

    def test_WassersteinScorer(self):
        # Check parameters and inputs
        self.component_wise_parameter(WassersteinScorer)
        self.helper_window_parameter(WassersteinScorer)
        self.diff_fn_parameter(WassersteinScorer)
        self.expects_deterministic_input(WassersteinScorer)

        # test plotting (just call the functions)
        scorer = WassersteinScorer(window=2, window_agg=False)
        scorer.fit(self.train)
        scorer.show_anomalies(self.test, self.anomalies)
        with pytest.raises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies([self.test, self.test], self.anomalies)
        scorer.show_anomalies_from_prediction(
            series=self.test,
            pred_series=self.test + 1,
            anomalies=self.anomalies,
        )
        with pytest.raises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies_from_prediction(
                series=[self.test, self.test],
                pred_series=self.test + 1,
                anomalies=self.anomalies,
            )
        with pytest.raises(ValueError):
            # should fail for a sequence of series
            scorer.show_anomalies_from_prediction(
                series=self.test,
                pred_series=[self.test + 1, self.test + 2],
                anomalies=self.anomalies,
            )

        assert not scorer.is_probabilistic

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
        scorer_10 = WassersteinScorer(window=10, window_agg=False)
        scorer_10.fit(train_wasserstein)
        auc_roc_w10 = scorer_10.eval_metric(
            anomalies_wasserstein, test_wasserstein, metric="AUC_ROC"
        )
        auc_pr_w10 = scorer_10.eval_metric(
            anomalies_wasserstein, test_wasserstein, metric="AUC_PR"
        )

        # test model with window of 20
        scorer_20 = WassersteinScorer(window=20, window_agg=False)
        scorer_20.fit(train_wasserstein)
        auc_roc_w20 = scorer_20.eval_metric(
            anomalies_wasserstein, test_wasserstein, metric="AUC_ROC"
        )
        auc_pr_w20 = scorer_20.eval_metric(
            anomalies_wasserstein, test_wasserstein, metric="AUC_PR"
        )

        assert np.abs(0.80637 - auc_roc_w10) < delta
        assert np.abs(0.83390 - auc_pr_w10) < delta
        assert np.abs(0.77828 - auc_roc_w20) < delta
        assert np.abs(0.93934 - auc_pr_w20) < delta

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
        scorer_w10_cwfalse = WassersteinScorer(
            window=10, component_wise=False, window_agg=False
        )
        scorer_w10_cwfalse.fit(mts_train_wasserstein)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_metric(
            anomalies_common_wasserstein, mts_test_wasserstein, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = WassersteinScorer(
            window=10, component_wise=True, window_agg=False
        )
        scorer_w10_cwtrue.fit(mts_train_wasserstein)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_metric(
            anomalies_wasserstein_per_width, mts_test_wasserstein, metric="AUC_ROC"
        )

        assert np.abs(0.94637 - auc_roc_cwfalse) < delta
        assert np.abs(0.98606 - auc_roc_cwtrue[0]) < delta
        assert np.abs(0.96722 - auc_roc_cwtrue[1]) < delta

    def test_kmeansScorer(self):
        # Check parameters and inputs
        self.component_wise_parameter(KMeansScorer)
        self.helper_window_parameter(KMeansScorer)
        self.diff_fn_parameter(KMeansScorer)
        self.expects_deterministic_input(KMeansScorer)
        assert not KMeansScorer().is_probabilistic

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

        metric_AUC_ROC = kmeans_scorer.eval_metric(
            KMeans_mts_anomalies, KMeans_mts_test, metric="AUC_ROC"
        )
        metric_AUC_PR = kmeans_scorer.eval_metric(
            KMeans_mts_anomalies, KMeans_mts_test, metric="AUC_PR"
        )

        assert metric_AUC_ROC == 1.0
        assert metric_AUC_PR == 1.0

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

        kmeans_scorer_w2 = KMeansScorer(k=8, window=2, window_agg=False)
        kmeans_scorer_w2.fit(ts_train)

        auc_roc_w1 = kmeans_scorer_w1.eval_metric(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w1 = kmeans_scorer_w1.eval_metric(ts_anomalies, ts_test, metric="AUC_PR")

        auc_roc_w2 = kmeans_scorer_w2.eval_metric(
            ts_anomalies, ts_test, metric="AUC_ROC"
        )
        auc_pr_w2 = kmeans_scorer_w2.eval_metric(ts_anomalies, ts_test, metric="AUC_PR")

        assert np.abs(0.41551 - auc_roc_w1) < delta
        assert np.abs(0.064761 - auc_pr_w1) < delta
        assert np.abs(0.957513 - auc_roc_w2) < delta
        assert np.abs(0.88584 - auc_pr_w2) < delta

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
        scorer_w10_cwfalse = KMeansScorer(
            window=10, component_wise=False, n_init=10, window_agg=False
        )
        scorer_w10_cwfalse.fit(mts_train_kmeans)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_metric(
            anomalies_common_kmeans, mts_test_kmeans, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = KMeansScorer(
            window=10, component_wise=True, n_init=10, window_agg=False
        )
        scorer_w10_cwtrue.fit(mts_train_kmeans)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_metric(
            anomalies_kmeans_per_width, mts_test_kmeans, metric="AUC_ROC"
        )

        assert np.abs(1.0 - auc_roc_cwtrue[0]) < delta
        assert np.abs(0.97666 - auc_roc_cwtrue[1]) < delta
        assert np.abs(0.99007 - auc_roc_cwfalse) < delta

    def test_PyODScorer(self):
        # Check parameters and inputs
        self.component_wise_parameter(PyODScorer, model=KNN())
        self.helper_window_parameter(PyODScorer, model=KNN())
        self.diff_fn_parameter(PyODScorer, model=KNN())
        self.expects_deterministic_input(PyODScorer, model=KNN())
        assert not PyODScorer(model=KNN()).is_probabilistic

        # model parameter must be pyod.models type BaseDetector
        with pytest.raises(ValueError):
            PyODScorer(model=MovingAverageFilter(window=10))

        # component_wise parameter
        # if component_wise=False must always return a univariate anomaly score
        scorer = PyODScorer(model=KNN(), component_wise=False)
        scorer.fit(self.train)
        assert scorer.score(self.test).width == 1
        scorer.fit(self.mts_train)
        assert scorer.score(self.mts_test).width == 1
        # if component_wise=True must always return the same width as the input
        scorer = PyODScorer(model=KNN(), component_wise=True)
        scorer.fit(self.train)
        assert scorer.score(self.test).width == 1
        scorer.fit(self.mts_train)
        assert scorer.score(self.mts_test).width == self.mts_test.width

        # window parameter
        # window must be non-negative
        with pytest.raises(ValueError):
            PyODScorer(model=KNN(), window=-1)
        # window must be different from 0
        with pytest.raises(ValueError):
            PyODScorer(model=KNN(), window=0)

        # diff_fn parameter
        # must be None, 'diff' or 'abs_diff'
        with pytest.raises(ValueError):
            PyODScorer(model=KNN(), diff_fn="random")
        with pytest.raises(ValueError):
            PyODScorer(model=KNN(), diff_fn=1)

        scorer = PyODScorer(model=KNN())

        # model parameter must be pyod.models type BaseDetector
        with pytest.raises(ValueError):
            PyODScorer(model=MovingAverageFilter(window=10))

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

        metric_AUC_ROC = pyod_scorer.eval_metric(
            pyod_mts_anomalies, pyod_mts_test, metric="AUC_ROC"
        )
        metric_AUC_PR = pyod_scorer.eval_metric(
            pyod_mts_anomalies, pyod_mts_test, metric="AUC_PR"
        )

        assert metric_AUC_ROC == 1.0
        assert metric_AUC_PR == 1.0

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
            model=KNN(n_neighbors=10),
            component_wise=False,
            window=2,
            window_agg=False,
        )
        pyod_scorer_w2.fit(ts_train)

        auc_roc_w1 = pyod_scorer_w1.eval_metric(ts_anomalies, ts_test, metric="AUC_ROC")
        auc_pr_w1 = pyod_scorer_w1.eval_metric(ts_anomalies, ts_test, metric="AUC_PR")

        auc_roc_w2 = pyod_scorer_w2.eval_metric(ts_anomalies, ts_test, metric="AUC_ROC")
        auc_pr_w2 = pyod_scorer_w2.eval_metric(ts_anomalies, ts_test, metric="AUC_PR")

        assert np.abs(0.5 - auc_roc_w1) < delta
        assert np.abs(0.07 - auc_pr_w1) < delta
        assert np.abs(0.957513 - auc_roc_w2) < delta
        assert np.abs(0.88584 - auc_pr_w2) < delta

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
            model=KNN(n_neighbors=10),
            component_wise=False,
            window=10,
            window_agg=False,
        )
        scorer_w10_cwfalse.fit(mts_train_PyOD)
        auc_roc_cwfalse = scorer_w10_cwfalse.eval_metric(
            anomalies_common_PyOD, mts_test_PyOD, metric="AUC_ROC"
        )

        # test scorer with component_wise=True
        scorer_w10_cwtrue = PyODScorer(
            model=KNN(n_neighbors=10),
            component_wise=True,
            window=10,
            window_agg=False,
        )
        scorer_w10_cwtrue.fit(mts_train_PyOD)
        auc_roc_cwtrue = scorer_w10_cwtrue.eval_metric(
            anomalies_pyod_per_width, mts_test_PyOD, metric="AUC_ROC"
        )

        assert np.abs(0.990566 - auc_roc_cwfalse) < delta
        assert np.abs(1.0 - auc_roc_cwtrue[0]) < delta
        assert np.abs(0.98311 - auc_roc_cwtrue[1]) < delta

    @staticmethod
    def helper_evaluate_nll_scorer(
        NLLscorer_to_test,
        distribution_arrays,
        deterministic_values,
        real_NLL_values,
    ):
        NLLscorer_w1 = NLLscorer_to_test(window=1)
        NLLscorer_w2 = NLLscorer_to_test(window=2)

        assert NLLscorer_w1.is_probabilistic

        # create timeseries
        distribution_series = TimeSeries.from_values(
            distribution_arrays.reshape(2, 2, -1)
        )
        series = TimeSeries.from_values(
            np.array(deterministic_values).reshape(2, 2, -1)
        )

        # compute the NLL values with score_from_prediction for scorer with window=1 and 2
        # t -> timestamp, c -> component and w -> window used in scorer
        value_t1_c1_w1 = NLLscorer_w1.score_from_prediction(
            series[0]["0"], distribution_series[0]["0"]
        )
        value_t2_c1_w1 = NLLscorer_w1.score_from_prediction(
            series[1]["0"], distribution_series[1]["0"]
        )
        value_t1_2_c1_w1 = NLLscorer_w1.score_from_prediction(
            series["0"], distribution_series["0"]
        )
        value_t1_2_c1_w2 = NLLscorer_w2.score_from_prediction(
            series["0"], distribution_series["0"]
        )

        # check length
        assert len(value_t1_2_c1_w1) == 2
        # check width
        assert value_t1_2_c1_w1.width == 1

        # check equal value_test1 and value_test2
        assert value_t1_2_c1_w1[0] == value_t1_c1_w1
        assert value_t1_2_c1_w1[1] == value_t2_c1_w1

        # check if value_t1_2_c1_w1 is the - log likelihood
        np.testing.assert_array_almost_equal(
            # This is approximate because our NLL scorer is fit from samples
            value_t1_2_c1_w1.all_values().reshape(-1),
            real_NLL_values[::2],
            decimal=1,
        )

        # check if result is equal to avg of two values when window is equal to 2
        assert (
            value_t1_2_c1_w2.all_values().reshape(-1)[0]
            == value_t1_2_c1_w1.mean(axis=0).all_values().reshape(-1)[0]
        )

        # multivariate case
        # compute the NLL values with score_from_prediction for scorer with window=1 and window=2
        value_t1_2_c1_2_w1 = NLLscorer_w1.score_from_prediction(
            series, distribution_series
        )
        value_t1_2_c1_2_w2 = NLLscorer_w2.score_from_prediction(
            series, distribution_series
        )

        # check length
        assert len(value_t1_2_c1_2_w1) == 2
        assert len(value_t1_2_c1_2_w2) == 1
        # check width
        assert value_t1_2_c1_2_w1.width == 2
        assert value_t1_2_c1_2_w2.width == 2

        # check if value_t1_2_c1_2_w1 is the - log likelihood
        np.testing.assert_array_almost_equal(
            # This is approximate because our NLL scorer is fit from samples
            value_t1_2_c1_2_w1.all_values().reshape(-1),
            real_NLL_values,
            decimal=1,
        )

        # check if result is equal to avg of two values when window is equal to 2
        assert value_t1_2_c1_w2.all_values().reshape(-1) == value_t1_2_c1_w1.mean(
            axis=0
        ).all_values().reshape(-1)

    @pytest.mark.parametrize("config", list_NLLScorer)
    def test_nll_scorer(self, config):
        np.random.seed(4)

        (
            scorer_cls,
            values,
            distribution,
            dist_kwargs,
            prob_dens_func,
            pdf_kwargs,
        ) = config
        # some pdf don't have the same parameters as the corresponding distribution
        if pdf_kwargs is None:
            pdf_kwargs = dist_kwargs
        self.helper_window_parameter(scorer_cls)

        distribution = np.array([
            distribution(size=10000, **dist_kwargs) for _ in range(len(values))
        ])
        real_values = [-np.log(prob_dens_func(value, **pdf_kwargs)) for value in values]

        self.helper_evaluate_nll_scorer(scorer_cls, distribution, values, real_values)

    @pytest.mark.parametrize(
        "model,series",
        product(
            [(KMeansScorer, {"random_state": 42}), (PyODScorer, {"model": KNN()})],
            [(train, test), (mts_train, mts_test)],
        ),
    )
    def test_window_equal_one(self, model, series):
        """Check that model, created with window=1 generate the same score regardless of window_agg value."""
        ts_train, ts_test = series
        model_cls, model_kwargs = model

        scorer_T = model_cls(window=1, window_agg=True, **model_kwargs)
        scorer_F = model_cls(window=1, window_agg=False, **model_kwargs)

        scorer_T.fit(ts_train)
        scorer_F.fit(ts_train)

        auc_roc_T = scorer_T.eval_metric(anomalies=self.anomalies, series=ts_test)
        auc_roc_F = scorer_F.eval_metric(anomalies=self.anomalies, series=ts_test)

        assert auc_roc_T == auc_roc_F

    @pytest.mark.parametrize(
        "window,model,series",
        product(
            [2, 10, 39],
            [
                (KMeansScorer, {"random_state": 42}),
                (WassersteinScorer, {}),
                (PyODScorer, {"model": KNN()}),
            ],
            [(train, test), (mts_train, mts_test)],
        ),
    )
    def test_window_greater_than_one(self, window, model, series):
        """Check scorer with same window greater than 1 and different values of window_agg produce correct scores"""
        ts_train, ts_test = series
        model_cls, model_kwargs = model
        scorer_T = model_cls(window=window, window_agg=True, **model_kwargs)
        scorer_F = model_cls(window=window, window_agg=False, **model_kwargs)

        scorer_T.fit(ts_train)
        scorer_F.fit(ts_train)

        score_T = scorer_T.score(ts_test)
        score_F = scorer_F.score(ts_test)

        # same length
        assert len(score_T) == len(score_F)

        # same width
        assert score_T.width == score_F.width

        # same first time index
        assert score_T.time_index[0] == score_F.time_index[0]

        # same last time index
        assert score_T.time_index[-1] == score_F.time_index[-1]

        # same last value (by definition)
        assert score_T[-1] == score_F[-1]

    def test_fun_window_agg(self):
        """Verify that the anomaly score aggregation works as intended"""
        # window = 2, alternating anomaly scores
        window = 2
        scorer = KMeansScorer(window=window)
        anomaly_scores = TimeSeries.from_values(np.resize([1, -1], 10))
        aggreg_scores = scorer._fun_window_agg([anomaly_scores], window=window)[0]
        # in the last window, the score is not zeroed
        np.testing.assert_array_almost_equal(
            aggreg_scores.values(), np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]).T
        )
        assert aggreg_scores.time_index.equals(anomaly_scores.time_index)

        # window = 3, increment of 2 anomaly scores
        window = 3
        scorer = KMeansScorer(window=window)
        anomaly_scores = linear_timeseries(length=10, start_value=2, end_value=20)
        aggreg_scores = scorer._fun_window_agg([anomaly_scores], window=window)[0]
        # on the last "window" values, difference of only 1 between consecutive aggregated scores
        np.testing.assert_array_almost_equal(
            aggreg_scores.values(), np.array([[4, 6, 8, 10, 12, 14, 16, 18, 19, 20]]).T
        )
        assert aggreg_scores.time_index.equals(anomaly_scores.time_index)

        # window = 6, increment of 2 anomaly scores
        window = 6
        scorer = KMeansScorer(window=window)
        anomaly_scores = linear_timeseries(length=10, start_value=2, end_value=20)
        aggreg_scores = scorer._fun_window_agg([anomaly_scores], window=window)[0]
        # on the last "window" values, difference of only 1 between consecutive aggregated scores
        np.testing.assert_array_almost_equal(
            aggreg_scores.values(), np.array([[7, 9, 11, 13, 15, 16, 17, 18, 19, 20]]).T
        )
        assert aggreg_scores.time_index.equals(anomaly_scores.time_index)

        # window = 7, increment of 2 anomaly scores
        window = 7
        scorer = KMeansScorer(window=window)
        anomaly_scores = linear_timeseries(length=10, start_value=2, end_value=20)
        aggreg_scores = scorer._fun_window_agg([anomaly_scores], window=window)[0]
        # on the last "window" values, difference of only 1 between consecutive aggregated scores
        np.testing.assert_array_almost_equal(
            aggreg_scores.values(),
            np.array([[8, 10, 12, 14, 15, 16, 17, 18, 19, 20]]).T,
        )
        assert aggreg_scores.time_index.equals(anomaly_scores.time_index)
