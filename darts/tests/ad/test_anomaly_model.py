from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from pyod.models.knn import KNN

from darts import TimeSeries
from darts.ad.anomaly_model.filtering_am import FilteringAnomalyModel
from darts.ad.anomaly_model.forecasting_am import ForecastingAnomalyModel
from darts.ad.scorers import CauchyNLLScorer
from darts.ad.scorers import DifferenceScorer as Difference
from darts.ad.scorers import (
    ExponentialNLLScorer,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
    NormScorer,
    PoissonNLLScorer,
    PyODScorer,
    WassersteinScorer,
)
from darts.ad.utils import eval_accuracy_from_scores
from darts.models import MovingAverage, NaiveSeasonal, RegressionModel
from darts.tests.base_test_class import DartsBaseTestClass


class ADAnomalyModelTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # univariate series
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)

    np_covariates = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
    covariates = TimeSeries.from_times_and_values(train._time_index, np_covariates)

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

    def test_Scorer(self):

        list_NonFittableAnomalyScorer = [
            NormScorer(),
            Difference(),
            GaussianNLLScorer(),
            ExponentialNLLScorer(),
            PoissonNLLScorer(),
            LaplaceNLLScorer(),
            CauchyNLLScorer(),
            GammaNLLScorer(),
        ]

        for scorers in list_NonFittableAnomalyScorer:
            for anomaly_model in [
                ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=scorers),
                FilteringAnomalyModel(model=MovingAverage(window=20), scorer=scorers),
            ]:

                # scorer are trainable
                self.assertTrue(anomaly_model.scorers_are_trainable is False)

        list_FittableAnomalyScorer = [
            PyODScorer(model=KNN()),
            KMeansScorer(),
            WassersteinScorer(),
        ]

        for scorers in list_FittableAnomalyScorer:
            for anomaly_model in [
                ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=scorers),
                FilteringAnomalyModel(model=MovingAverage(window=20), scorer=scorers),
            ]:

                # scorer are not trainable
                self.assertTrue(anomaly_model.scorers_are_trainable is True)

    def test_Score(self):

        am1 = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=NormScorer()
        )
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=NormScorer())

        for am in [am1, am2]:
            # Parameter return_model_prediction
            # parameter return_model_prediction must be bool
            with self.assertRaises(ValueError):
                am.score(self.test, return_model_prediction=1)
            with self.assertRaises(ValueError):
                am.score(self.test, return_model_prediction="True")

            # if return_model_prediction set to true, output must be tuple
            self.assertTrue(
                isinstance(am.score(self.test, return_model_prediction=True), Tuple)
            )

            # if return_model_prediction set to false output must be
            # Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
            self.assertTrue(
                not isinstance(
                    am.score(self.test, return_model_prediction=False), Tuple
                )
            )

    def test_FitFilteringAnomalyModelInput(self):

        for anomaly_model in [
            FilteringAnomalyModel(model=MovingAverage(window=20), scorer=NormScorer()),
            FilteringAnomalyModel(
                model=MovingAverage(window=20), scorer=[NormScorer(), KMeansScorer()]
            ),
            FilteringAnomalyModel(
                model=MovingAverage(window=20), scorer=KMeansScorer()
            ),
        ]:

            # filter must be fittable if allow_filter_training is set to True
            with self.assertRaises(ValueError):
                anomaly_model.fit(self.train, allow_filter_training=True)

            # input 'series' must be a series or Sequence of series
            with self.assertRaises(ValueError):
                anomaly_model.fit([self.train, "str"], allow_filter_training=True)
            with self.assertRaises(ValueError):
                anomaly_model.fit(
                    [[self.train, self.train]], allow_filter_training=True
                )
            with self.assertRaises(ValueError):
                anomaly_model.fit("str", allow_filter_training=True)
            with self.assertRaises(ValueError):
                anomaly_model.fit([1, 2, 3], allow_filter_training=True)

            # allow_model_training must be a bool
            with self.assertRaises(ValueError):
                anomaly_model.fit(self.train, allow_filter_training=1)
            with self.assertRaises(ValueError):
                anomaly_model.fit(self.train, allow_filter_training="True")

    def test_FitForecastingAnomalyModelInput(self):

        for anomaly_model in [
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=NormScorer()
            ),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[NormScorer(), KMeansScorer()]
            ),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=KMeansScorer()
            ),
        ]:

            # input 'series' must be a series or Sequence of series
            with self.assertRaises(ValueError):
                anomaly_model.fit([self.train, "str"], allow_model_training=True)
            with self.assertRaises(ValueError):
                anomaly_model.fit([[self.train, self.train]], allow_model_training=True)
            with self.assertRaises(ValueError):
                anomaly_model.fit("str", allow_model_training=True)
            with self.assertRaises(ValueError):
                anomaly_model.fit([1, 2, 3], allow_model_training=True)

            # allow_model_training must be a bool
            with self.assertRaises(ValueError):
                anomaly_model.fit(self.train, allow_model_training=1)
            with self.assertRaises(ValueError):
                anomaly_model.fit(self.train, allow_model_training="True")

            # 'allow_model_training' must be set to True if forecasting model is not fitted
            if anomaly_model.scorers_are_trainable:
                with self.assertRaises(ValueError):
                    anomaly_model.fit(self.train, allow_model_training=False)
                    anomaly_model.score(self.train)

            with self.assertRaises(ValueError):
                # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
                anomaly_model.fit(
                    series=[self.train, self.train],
                    past_covariates=self.covariates,
                    allow_model_training=True,
                )

            with self.assertRaises(ValueError):
                # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
                anomaly_model.fit(
                    series=self.train,
                    past_covariates=[self.covariates, self.covariates],
                    allow_model_training=True,
                )

            with self.assertRaises(ValueError):
                # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
                anomaly_model.fit(
                    series=[self.train, self.train],
                    future_covariates=self.covariates,
                    allow_model_training=True,
                )

            with self.assertRaises(ValueError):
                # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
                anomaly_model.fit(
                    series=self.train,
                    future_covariates=[self.covariates, self.covariates],
                    allow_model_training=True,
                )

        fitted_model = RegressionModel(lags=10).fit(self.train)
        # Fittable scorer must be fitted before calling .score(), even if forecasting model is fitted
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=fitted_model, scorer=KMeansScorer()).score(
                series=self.test
            )
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=fitted_model, scorer=[NormScorer(), KMeansScorer()]
            ).score(series=self.test)

        # forecasting model that do not accept past/future covariates
        # with self.assertRaises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=NormScorer()).fit(
        #           series=self.train, past_covariates=self.covariates, allow_model_training=True
        #       )
        # with self.assertRaises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=NormScorer()).fit(
        #           series=self.train, future_covariates=self.covariates, allow_model_training=True
        #       )

        # check window size
        # max window size is len(series.drop_before(series.get_timestamp_at_point(start))) + 1
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=KMeansScorer(window=50)
            ).fit(series=self.train, start=0.9)

        # forecasting model that cannot be trained on a list of series
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=NaiveSeasonal(), scorer=NormScorer()).fit(
                series=[self.train, self.train], allow_model_training=True
            )

    def test_ScoreForecastingAnomalyModelInput(self):

        for anomaly_model in [
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=NormScorer()
            ),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[NormScorer(), KMeansScorer()]
            ),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=KMeansScorer()
            ),
        ]:

            anomaly_model.fit(self.train, allow_model_training=True)

            # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
            with self.assertRaises(ValueError):
                anomaly_model.score(
                    series=[self.train, self.train], past_covariates=self.covariates
                )

            # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
            with self.assertRaises(ValueError):
                anomaly_model.score(
                    series=self.train,
                    past_covariates=[self.covariates, self.covariates],
                )

            # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
            with self.assertRaises(ValueError):
                anomaly_model.score(
                    series=[self.train, self.train], future_covariates=self.covariates
                )

            # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
            with self.assertRaises(ValueError):
                anomaly_model.score(
                    series=self.train,
                    future_covariates=[self.covariates, self.covariates],
                )

        # check window size
        # max window size is len(series.drop_before(series.get_timestamp_at_point(start))) + 1 for score()
        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=KMeansScorer(window=30)
        )
        anomaly_model.fit(self.train, allow_model_training=True)
        with self.assertRaises(ValueError):
            anomaly_model.score(series=self.train, start=0.9)

    def test_ScoreFilteringAnomalyModelInput(self):

        for anomaly_model in [
            FilteringAnomalyModel(model=MovingAverage(window=10), scorer=NormScorer()),
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=[NormScorer(), KMeansScorer()]
            ),
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=KMeansScorer()
            ),
        ]:

            if anomaly_model.scorers_are_trainable:
                anomaly_model.fit(self.train)

    def test_show_anomalies(self):

        am1 = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=NormScorer()
        )
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=NormScorer())

        for am in [am1, am2]:
            # input 'series' must be a series and not a Sequence of series
            with self.assertRaises(ValueError):
                am.show_anomalies([self.train, self.train])

    def test_eval_accuracy(self):

        am1 = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=NormScorer()
        )
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=NormScorer())

        am3 = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=[NormScorer(), WassersteinScorer()]
        )
        am3.fit(self.train, allow_model_training=True)

        am4 = FilteringAnomalyModel(
            model=MovingAverage(window=20), scorer=[NormScorer(), WassersteinScorer()]
        )
        am4.fit(self.train)

        for am in [am1, am2, am3, am4]:

            # if the anomaly_model have scorers that have the parameter returns_UTS set to True,
            # 'actual_anomalies' must have widths of 1
            if am.univariate_scoring:
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=self.MTS_anomalies, series=self.test
                    )
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=self.MTS_anomalies, series=self.MTS_test
                    )
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=[self.anomalies, self.MTS_anomalies],
                        series=[self.test, self.MTS_test],
                    )

            # 'metric' must be str and "AUC_ROC" or "AUC_PR"
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.anomalies, series=self.test, metric=1
                )
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.anomalies, series=self.test, metric="auc_roc"
                )
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.anomalies,
                    series=self.test,
                    metric=["AUC_ROC"],
                )

            # 'actual_anomalies' must be binary
            with self.assertRaises(ValueError):
                am.eval_accuracy(actual_anomalies=self.test, series=self.test)

            # 'actual_anomalies' must contain anomalies (at least one)
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.only_0_anomalies, series=self.test
                )

            # 'actual_anomalies' cannot contain only anomalies
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.only_1_anomalies, series=self.test
                )

            # 'actual_anomalies' must match the number of series
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.anomalies, series=[self.test, self.test]
                )
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=[self.anomalies, self.anomalies], series=self.test
                )

            # 'actual_anomalies' must have non empty intersection with 'series'
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=self.anomalies[:20], series=self.test[30:]
                )
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    actual_anomalies=[self.anomalies, self.anomalies[:20]],
                    series=[self.test, self.test[40:]],
                )

            # Check input type
            # 'actual_anomalies' and 'series' must be of same length
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies], [self.test, self.test])
            with self.assertRaises(ValueError):
                am.eval_accuracy(self.anomalies, [self.test, self.test])
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies, self.anomalies], [self.test])
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies, self.anomalies], self.test)

            # 'actual_anomalies' and 'series' must be of type Timeseries
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies], [2, 3, 4])
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies], "str")
            with self.assertRaises(ValueError):
                am.eval_accuracy([2, 3, 4], self.test)
            with self.assertRaises(ValueError):
                am.eval_accuracy("str", self.test)
            with self.assertRaises(ValueError):
                am.eval_accuracy(
                    [self.anomalies, self.anomalies], [self.test, [3, 2, 1]]
                )
            with self.assertRaises(ValueError):
                am.eval_accuracy([self.anomalies, [3, 2, 1]], [self.test, self.test])

            # Check return types
            # Check if return type is float when input is a series
            self.assertTrue(
                isinstance(
                    am.eval_accuracy(self.anomalies, self.test),
                    Dict,
                )
            )

            # Check if return type is Sequence when input is a Sequence of series
            self.assertTrue(
                isinstance(
                    am.eval_accuracy(self.anomalies, [self.test]),
                    Sequence,
                )
            )
            self.assertTrue(
                isinstance(
                    am.eval_accuracy(
                        [self.anomalies, self.anomalies], [self.test, self.test]
                    ),
                    Sequence,
                )
            )

    def test_ForecastingAnomalyModelInput(self):

        # model input
        # model input must be of type ForecastingModel
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model="str", scorer=NormScorer())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=1, scorer=NormScorer())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=MovingAverage(window=10), scorer=NormScorer())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=[RegressionModel(lags=10), RegressionModel(lags=5)],
                scorer=NormScorer(),
            )

        # scorer input
        # scorer input must be of type AnomalyScorer
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=1)
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer="str")
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=RegressionModel(lags=10)
            )
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[NormScorer(), "str"]
            )

    def test_FilteringAnomalyModelInput(self):

        # model input
        # model input must be of type FilteringModel
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model="str", scorer=NormScorer())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=1, scorer=NormScorer())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=RegressionModel(lags=10), scorer=NormScorer())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(
                model=[MovingAverage(window=10), MovingAverage(window=10)],
                scorer=NormScorer(),
            )

        # scorer input
        # scorer input must be of type AnomalyScorer
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=MovingAverage(window=10), scorer=1)
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=MovingAverage(window=10), scorer="str")
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=MovingAverage(window=10)
            )
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=[NormScorer(), "str"]
            )

    def test_univariate_ForecastingAnomalyModel(self):

        np.random.seed(40)

        np_train_slope = np.array(range(0, 100, 1))
        np_test_slope = np.array(range(0, 100, 1))

        np_test_slope[30:32] = 29
        np_test_slope[50:65] = np_test_slope[50:65] + 1
        np_test_slope[75:80] = np_test_slope[75:80] * 0.98

        train_series_slope = TimeSeries.from_values(np_train_slope)
        test_series_slope = TimeSeries.from_values(np_test_slope)

        np_anomalies = np.zeros(100)
        np_anomalies[30:32] = 1
        np_anomalies[50:55] = 1
        np_anomalies[70:80] = 1
        TS_anomalies = TimeSeries.from_times_and_values(
            test_series_slope.time_index, np_anomalies, columns=["is_anomaly"]
        )

        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=5),
            scorer=[
                NormScorer(),
                Difference(),
                WassersteinScorer(),
                KMeansScorer(),
                KMeansScorer(window=10),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10),
                WassersteinScorer(window=15),
            ],
        )

        anomaly_model.fit(train_series_slope, allow_model_training=True, start=0.1)
        score, model_output = anomaly_model.score(
            test_series_slope, return_model_prediction=True, start=0.1
        )

        # check that NormScorer is the abs difference of model_output and test_series_slope
        self.assertEqual(
            (model_output - test_series_slope.slice_intersect(model_output)).map(
                lambda x: np.abs(x)
            ),
            NormScorer().score_from_prediction(model_output, test_series_slope),
        )

        # check that Difference is the difference of model_output and test_series_slope
        self.assertEqual(
            model_output - test_series_slope.slice_intersect(model_output),
            Difference().score_from_prediction(model_output, test_series_slope),
        )

        dict_AUC_ROC = anomaly_model.eval_accuracy(
            TS_anomalies, test_series_slope, metric="AUC_ROC", start=0.1
        )
        dict_AUC_PR = anomaly_model.eval_accuracy(
            TS_anomalies, test_series_slope, metric="AUC_PR", start=0.1
        )

        AUC_ROC_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        AUC_PR_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(AUC_ROC_from_scores), len(dict_AUC_ROC))
        self.assertEqual(len(AUC_PR_from_scores), len(dict_AUC_PR))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        self.assertEqual(AUC_ROC_from_scores, list(dict_AUC_ROC.values()))
        self.assertEqual(AUC_PR_from_scores, list(dict_AUC_PR.values()))

        true_AUC_ROC = [
            0.773449920508744,
            0.40659777424483307,
            0.9153708133971291,
            0.7702702702702702,
            0.9135765550239234,
            0.7603338632750397,
            0.9153708133971292,
            0.9006591337099811,
        ]

        true_AUC_PR = [
            0.4818991248542174,
            0.20023033665128342,
            0.9144135170539835,
            0.47953161438253644,
            0.9127969832903458,
            0.47039678636225957,
            0.9147124232933175,
            0.9604714100445533,
        ]

        # check value of results
        self.assertEqual(AUC_ROC_from_scores, true_AUC_ROC)
        self.assertEqual(AUC_PR_from_scores, true_AUC_PR)

    def test_univariate_FilteringAnomalyModel(self):

        np.random.seed(40)

        np_series_train = np.array(range(0, 100, 1)) + np.random.normal(
            loc=0, scale=1, size=100
        )
        np_series_test = np.array(range(0, 100, 1)) + np.random.normal(
            loc=0, scale=1, size=100
        )

        np_series_test[30:35] = np_series_test[30:35] + np.random.normal(
            loc=0, scale=10, size=5
        )
        np_series_test[50:60] = np_series_test[50:60] + np.random.normal(
            loc=0, scale=4, size=10
        )
        np_series_test[75:80] = np_series_test[75:80] + np.random.normal(
            loc=0, scale=3, size=5
        )

        train_series_noise = TimeSeries.from_values(np_series_train)
        test_series_noise = TimeSeries.from_values(np_series_test)

        np_anomalies = np.zeros(100)
        np_anomalies[30:35] = 1
        np_anomalies[50:60] = 1
        np_anomalies[75:80] = 1
        TS_anomalies = TimeSeries.from_times_and_values(
            test_series_noise.time_index, np_anomalies, columns=["is_anomaly"]
        )

        anomaly_model = FilteringAnomalyModel(
            model=MovingAverage(window=5),
            scorer=[
                NormScorer(),
                Difference(),
                WassersteinScorer(),
                KMeansScorer(),
                KMeansScorer(window=10),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10),
                WassersteinScorer(window=15),
            ],
        )
        anomaly_model.fit(train_series_noise)
        score, model_output = anomaly_model.score(
            test_series_noise, return_model_prediction=True
        )

        # check that NormScorer is the abs difference of model_output and test_series_noise
        self.assertEqual(
            (model_output - test_series_noise.slice_intersect(model_output)).map(
                lambda x: np.abs(x)
            ),
            NormScorer().score_from_prediction(model_output, test_series_noise),
        )

        # check that Difference is the difference of model_output and test_series_noise
        self.assertEqual(
            model_output - test_series_noise.slice_intersect(model_output),
            Difference().score_from_prediction(model_output, test_series_noise),
        )

        dict_AUC_ROC = anomaly_model.eval_accuracy(
            TS_anomalies, test_series_noise, metric="AUC_ROC"
        )
        dict_AUC_PR = anomaly_model.eval_accuracy(
            TS_anomalies, test_series_noise, metric="AUC_PR"
        )

        AUC_ROC_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        AUC_PR_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(AUC_ROC_from_scores), len(dict_AUC_ROC))
        self.assertEqual(len(AUC_PR_from_scores), len(dict_AUC_PR))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        self.assertEqual(AUC_ROC_from_scores, list(dict_AUC_ROC.values()))
        self.assertEqual(AUC_PR_from_scores, list(dict_AUC_PR.values()))

        true_AUC_ROC = [
            0.875625,
            0.5850000000000001,
            0.952127659574468,
            0.814375,
            0.9598646034816247,
            0.88125,
            0.9666344294003868,
            0.9731182795698925,
        ]

        true_AUC_PR = [
            0.7691407907338141,
            0.5566414178265074,
            0.9720504927710986,
            0.741298584352156,
            0.9744855592642071,
            0.7808056518442923,
            0.9800621192517156,
            0.9911842778990486,
        ]

        # check value of results
        self.assertEqual(AUC_ROC_from_scores, true_AUC_ROC)
        self.assertEqual(AUC_PR_from_scores, true_AUC_PR)

    def test_univariate_covariate_ForecastingAnomalyModel(self):

        np.random.seed(40)

        day_week = [0, 1, 2, 3, 4, 5, 6]
        np_day_week = np.array(day_week * 10)

        np_train_series = 0.5 * np_day_week
        np_test_series = 0.5 * np_day_week

        np_test_series[30:35] = np_test_series[30:35] + np.random.normal(
            loc=0, scale=2, size=5
        )
        np_test_series[50:60] = np_test_series[50:60] + np.random.normal(
            loc=0, scale=1, size=10
        )

        covariates = TimeSeries.from_times_and_values(
            pd.date_range(start="1949-01-01", end="1949-03-11"), np_day_week
        )
        series_train = TimeSeries.from_times_and_values(
            pd.date_range(start="1949-01-01", end="1949-03-11"), np_train_series
        )
        series_test = TimeSeries.from_times_and_values(
            pd.date_range(start="1949-01-01", end="1949-03-11"), np_test_series
        )

        np_anomalies = np.zeros(70)
        np_anomalies[30:35] = 1
        np_anomalies[50:60] = 1
        TS_anomalies = TimeSeries.from_times_and_values(
            series_test.time_index, np_anomalies, columns=["is_anomaly"]
        )

        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=2, lags_future_covariates=[0]),
            scorer=[
                NormScorer(),
                Difference(),
                WassersteinScorer(),
                KMeansScorer(),
                KMeansScorer(window=10),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10),
                WassersteinScorer(window=15),
            ],
        )

        anomaly_model.fit(
            series_train,
            allow_model_training=True,
            future_covariates=covariates,
            start=0.2,
        )

        score, model_output = anomaly_model.score(
            series_test,
            return_model_prediction=True,
            future_covariates=covariates,
            start=0.2,
        )

        # check that NormScorer is the abs difference of model_output and series_test
        self.assertEqual(
            (model_output - series_test.slice_intersect(model_output)).map(
                lambda x: np.abs(x)
            ),
            NormScorer().score_from_prediction(model_output, series_test),
        )

        # check that Difference is the difference of model_output and series_test
        self.assertEqual(
            model_output - series_test.slice_intersect(model_output),
            Difference().score_from_prediction(model_output, series_test),
        )

        dict_AUC_ROC = anomaly_model.eval_accuracy(
            TS_anomalies, series_test, metric="AUC_ROC", start=0.2
        )
        dict_AUC_PR = anomaly_model.eval_accuracy(
            TS_anomalies, series_test, metric="AUC_PR", start=0.2
        )

        AUC_ROC_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        AUC_PR_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[TS_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(AUC_ROC_from_scores), len(dict_AUC_ROC))
        self.assertEqual(len(AUC_PR_from_scores), len(dict_AUC_PR))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        self.assertEqual(AUC_ROC_from_scores, list(dict_AUC_ROC.values()))
        self.assertEqual(AUC_PR_from_scores, list(dict_AUC_PR.values()))

        true_AUC_ROC = [1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        true_AUC_PR = [
            1.0,
            0.6914399076961142,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.9999999999999999,
        ]

        # check value of results
        self.assertEqual(AUC_ROC_from_scores, true_AUC_ROC)
        self.assertEqual(AUC_PR_from_scores, true_AUC_PR)
