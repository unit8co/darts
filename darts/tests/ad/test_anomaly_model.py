from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from pyod.models.knn import KNN

from darts import TimeSeries

# anomaly aggregators
# import everything in darts.ad (also for testing imports)
from darts.ad import AndAggregator  # noqa: F401
from darts.ad import EnsembleSklearnAggregator  # noqa: F401
from darts.ad import OrAggregator  # noqa: F401
from darts.ad import QuantileDetector  # noqa: F401
from darts.ad import ThresholdDetector  # noqa: F401
from darts.ad import CauchyNLLScorer
from darts.ad import DifferenceScorer as Difference
from darts.ad import (
    ExponentialNLLScorer,
    FilteringAnomalyModel,
    ForecastingAnomalyModel,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
    NormScorer,
    PoissonNLLScorer,
    PyODScorer,
    WassersteinScorer,
)
from darts.ad.utils import eval_accuracy_from_scores, show_anomalies_from_scores
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
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)

    np_mts_test = np.random.normal(loc=[10, 5], scale=[1, 1.5], size=[100, 2])
    mts_test = TimeSeries.from_times_and_values(mts_train._time_index, np_mts_test)

    np_mts_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.9, 0.1])
    mts_anomalies = TimeSeries.from_times_and_values(
        mts_train._time_index, np_mts_anomalies
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
                anomaly_model.fit(self.train, allow_model_training=True)

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

            # if the anomaly_model have scorers that have the parameter univariate_scorer set to True,
            # 'actual_anomalies' must have widths of 1
            if am.univariate_scoring:
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=self.mts_anomalies, series=self.test
                    )
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=self.mts_anomalies, series=self.mts_test
                    )
                with self.assertRaises(ValueError):
                    am.eval_accuracy(
                        actual_anomalies=[self.anomalies, self.mts_anomalies],
                        series=[self.test, self.mts_test],
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
            with self.assertRaises(TypeError):
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
        ts_anomalies = TimeSeries.from_times_and_values(
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
            (model_output - test_series_slope.slice_intersect(model_output)).__abs__(),
            NormScorer().score_from_prediction(test_series_slope, model_output),
        )

        # check that Difference is the difference of model_output and test_series_slope
        self.assertEqual(
            test_series_slope.slice_intersect(model_output) - model_output,
            Difference().score_from_prediction(test_series_slope, model_output),
        )

        dict_auc_roc = anomaly_model.eval_accuracy(
            ts_anomalies, test_series_slope, metric="AUC_ROC", start=0.1
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            ts_anomalies, test_series_slope, metric="AUC_PR", start=0.1
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            0.773449920508744,
            0.40659777424483307,
            0.9153708133971291,
            0.7702702702702702,
            0.9135765550239234,
            0.7603338632750397,
            0.9153708133971292,
            0.9006591337099811,
        ]

        true_auc_pr = [
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
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

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
        ts_anomalies = TimeSeries.from_times_and_values(
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

        # check that Difference is the difference of model_output and test_series_noise
        self.assertEqual(
            test_series_noise.slice_intersect(model_output) - model_output,
            Difference().score_from_prediction(test_series_noise, model_output),
        )

        # check that NormScorer is the abs difference of model_output and test_series_noise
        self.assertEqual(
            (test_series_noise.slice_intersect(model_output) - model_output).__abs__(),
            NormScorer().score_from_prediction(test_series_noise, model_output),
        )

        dict_auc_roc = anomaly_model.eval_accuracy(
            ts_anomalies, test_series_noise, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            ts_anomalies, test_series_noise, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            0.875625,
            0.5850000000000001,
            0.952127659574468,
            0.814375,
            0.9598646034816247,
            0.88125,
            0.9666344294003868,
            0.9731182795698925,
        ]

        true_auc_pr = [
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
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

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
        ts_anomalies = TimeSeries.from_times_and_values(
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
            (series_test.slice_intersect(model_output) - model_output).__abs__(),
            NormScorer().score_from_prediction(series_test, model_output),
        )

        # check that Difference is the difference of model_output and series_test
        self.assertEqual(
            series_test.slice_intersect(model_output) - model_output,
            Difference().score_from_prediction(series_test, model_output),
        )

        dict_auc_roc = anomaly_model.eval_accuracy(
            ts_anomalies, series_test, metric="AUC_ROC", start=0.2
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            ts_anomalies, series_test, metric="AUC_PR", start=0.2
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        true_auc_pr = [
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
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

    def test_multivariate__FilteringAnomalyModel(self):

        np.random.seed(40)

        data_1 = np.random.normal(0, 0.1, 100)
        data_2 = np.random.normal(0, 0.1, 100)

        mts_series_train = TimeSeries.from_values(
            np.dstack((data_1, data_2))[0], columns=["component 1", "component 2"]
        )

        data_1[15:20] = data_1[15:20] + np.random.normal(0, 0.9, 5)
        data_1[35:40] = data_1[35:40] + np.random.normal(0, 0.4, 5)

        data_2[50:55] = data_2[50:55] + np.random.normal(0, 0.7, 5)
        data_2[65:70] = data_2[65:70] + np.random.normal(0, 0.4, 5)

        data_1[80:85] = data_1[80:85] + np.random.normal(0, 0.6, 5)
        data_2[80:85] = data_2[80:85] + np.random.normal(0, 0.6, 5)

        data_1[93:98] = data_1[93:98] + np.random.normal(0, 0.6, 5)
        data_2[93:98] = data_2[93:98] + np.random.normal(0, 0.6, 5)
        mts_series_test = TimeSeries.from_values(
            np.dstack((data_1, data_2))[0], columns=["component 1", "component 2"]
        )

        np1_anomalies = np.zeros(len(data_1))
        np1_anomalies[15:20] = 1
        np1_anomalies[35:40] = 1
        np1_anomalies[80:85] = 1
        np1_anomalies[93:98] = 1

        np2_anomalies = np.zeros(len(data_2))
        np2_anomalies[50:55] = 1
        np2_anomalies[67:70] = 1
        np2_anomalies[80:85] = 1
        np2_anomalies[93:98] = 1

        np_anomalies = np.zeros(len(data_2))
        np_anomalies[15:20] = 1
        np_anomalies[35:40] = 1
        np_anomalies[50:55] = 1
        np_anomalies[67:70] = 1
        np_anomalies[80:85] = 1
        np_anomalies[93:98] = 1

        ts_anomalies = TimeSeries.from_times_and_values(
            mts_series_train.time_index,
            np.dstack((np1_anomalies, np2_anomalies))[0],
            columns=["is_anomaly_1", "is_anomaly_2"],
        )

        mts_anomalies = TimeSeries.from_times_and_values(
            mts_series_train.time_index, np_anomalies, columns=["is_anomaly"]
        )

        # first case: scorers that return univariate scores
        anomaly_model = FilteringAnomalyModel(
            model=MovingAverage(window=10),
            scorer=[
                NormScorer(component_wise=False),
                WassersteinScorer(),
                WassersteinScorer(window=12),
                KMeansScorer(),
                KMeansScorer(window=5),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=5),
            ],
        )
        anomaly_model.fit(mts_series_train)

        scores, model_output = anomaly_model.score(
            mts_series_test, return_model_prediction=True
        )

        # model_output must be multivariate (same width as input)
        self.assertEqual(model_output.width, mts_series_test.width)

        # scores must be of the same length as the number of scorers
        self.assertEqual(len(scores), len(anomaly_model.scorers))

        dict_auc_roc = anomaly_model.eval_accuracy(
            mts_anomalies, mts_series_test, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            mts_anomalies, mts_series_test, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[mts_anomalies] * 7,
            anomaly_score=scores,
            window=[1, 10, 12, 1, 5, 1, 5],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[mts_anomalies] * 7,
            anomaly_score=scores,
            window=[1, 10, 12, 1, 5, 1, 5],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            0.8695436507936507,
            0.9737678855325913,
            0.9930555555555555,
            0.857638888888889,
            0.9639130434782609,
            0.8690476190476191,
            0.9630434782608696,
        ]

        true_auc_pr = [
            0.814256917602188,
            0.9945160041091712,
            0.9992086070916503,
            0.8054288542539664,
            0.9777504211642852,
            0.8164636240285442,
            0.9763049418985656,
        ]

        # check value of results
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

        # second case: scorers that return scorers that have the same width as the input
        anomaly_model = FilteringAnomalyModel(
            model=MovingAverage(window=10),
            scorer=[
                NormScorer(component_wise=True),
                Difference(),
                WassersteinScorer(component_wise=True),
                WassersteinScorer(window=12, component_wise=True),
                KMeansScorer(component_wise=True),
                KMeansScorer(window=5, component_wise=True),
                PyODScorer(model=KNN(), component_wise=True),
                PyODScorer(model=KNN(), window=5, component_wise=True),
            ],
        )
        anomaly_model.fit(mts_series_train)

        scores, model_output = anomaly_model.score(
            mts_series_test, return_model_prediction=True
        )

        # model_output must be multivariate (same width as input)
        self.assertEqual(model_output.width, mts_series_test.width)

        # scores must be of the same length as the number of scorers
        self.assertEqual(len(scores), len(anomaly_model.scorers))

        dict_auc_roc = anomaly_model.eval_accuracy(
            ts_anomalies, mts_series_test, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            ts_anomalies, mts_series_test, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=scores,
            window=[1, 1, 10, 12, 1, 5, 1, 5],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=scores,
            window=[1, 1, 10, 12, 1, 5, 1, 5],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            [0.859375, 0.9200542005420054],
            [0.49875, 0.513550135501355],
            [0.997093023255814, 0.9536231884057971],
            [0.998960498960499, 0.9739795918367344],
            [0.8143750000000001, 0.8218157181571816],
            [0.9886148007590132, 0.94677734375],
            [0.830625, 0.9369918699186992],
            [0.9909867172675522, 0.94580078125],
        ]

        true_auc_pr = [
            [0.7213314465376244, 0.8191331553279771],
            [0.4172305056124696, 0.49249755343619195],
            [0.9975245098039216, 0.9741870252257915],
            [0.9992877492877493, 0.9865792868871687],
            [0.7095552075210219, 0.7591858780309868],
            [0.9827224901431558, 0.9402925739221939],
            [0.7095275592303261, 0.8313668186652059],
            [0.9858096294704315, 0.9391783485106905],
        ]

        # check value of results
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

    def test_multivariate__ForecastingAnomalyModel(self):

        np.random.seed(40)

        data_sin = np.array([np.sin(x) for x in np.arange(0, 20 * np.pi, 0.2)])
        data_cos = np.array([np.cos(x) for x in np.arange(0, 20 * np.pi, 0.2)])

        mts_series_train = TimeSeries.from_values(
            np.dstack((data_sin, data_cos))[0], columns=["component 1", "component 2"]
        )

        data_sin[10:20] = 0
        data_cos[60:80] = 0

        data_sin[100:110] = 1
        data_cos[150:155] = 1

        data_sin[200:240] = 0.9 * data_cos[200:240]
        data_cos[200:240] = 0.9 * data_sin[200:240]

        data_sin[275:295] = data_sin[275:295] + np.random.normal(0, 0.1, 20)
        data_cos[275:295] = data_cos[275:295] + np.random.normal(0, 0.1, 20)

        mts_series_test = TimeSeries.from_values(
            np.dstack((data_sin, data_cos))[0], columns=["component 1", "component 2"]
        )

        np1_anomalies = np.zeros(len(data_sin))
        np1_anomalies[10:20] = 1
        np1_anomalies[100:110] = 1
        np1_anomalies[200:240] = 1
        np1_anomalies[275:295] = 1

        np2_anomalies = np.zeros(len(data_cos))
        np2_anomalies[60:80] = 1
        np2_anomalies[150:155] = 1
        np2_anomalies[200:240] = 1
        np2_anomalies[275:295] = 1

        np_anomalies = np.zeros(len(data_cos))
        np_anomalies[10:20] = 1
        np_anomalies[60:80] = 1
        np_anomalies[100:110] = 1
        np_anomalies[150:155] = 1
        np_anomalies[200:240] = 1
        np_anomalies[275:295] = 1

        ts_anomalies = TimeSeries.from_times_and_values(
            mts_series_train.time_index,
            np.dstack((np1_anomalies, np2_anomalies))[0],
            columns=["is_anomaly_1", "is_anomaly_2"],
        )

        mts_anomalies = TimeSeries.from_times_and_values(
            mts_series_train.time_index, np_anomalies, columns=["is_anomaly"]
        )

        # first case: scorers that return univariate scores
        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10),
            scorer=[
                NormScorer(component_wise=False),
                WassersteinScorer(),
                WassersteinScorer(window=20),
                KMeansScorer(),
                KMeansScorer(window=20),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10),
            ],
        )
        anomaly_model.fit(mts_series_train, allow_model_training=True, start=0.1)

        scores, model_output = anomaly_model.score(
            mts_series_test, return_model_prediction=True, start=0.1
        )

        # model_output must be multivariate (same width as input)
        self.assertEqual(model_output.width, mts_series_test.width)

        # scores must be of the same length as the number of scorers
        self.assertEqual(len(scores), len(anomaly_model.scorers))

        dict_auc_roc = anomaly_model.eval_accuracy(
            mts_anomalies, mts_series_test, start=0.1, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            mts_anomalies, mts_series_test, start=0.1, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[mts_anomalies] * 7,
            anomaly_score=scores,
            window=[1, 10, 20, 1, 20, 1, 10],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[mts_anomalies] * 7,
            anomaly_score=scores,
            window=[1, 10, 20, 1, 20, 1, 10],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            0.9252575884154831,
            0.9130158730158731,
            0.9291228070175439,
            0.9252575884154832,
            0.9211929824561403,
            0.9252575884154831,
            0.915873015873016,
        ]

        true_auc_pr = [
            0.8389462532437767,
            0.9151621069238896,
            0.9685249535885079,
            0.8389462532437765,
            0.9662153835545242,
            0.8389462532437764,
            0.9212725256428517,
        ]

        # check value of results
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

        # second case: scorers that return scorers that have the same width as the input
        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10),
            scorer=[
                NormScorer(component_wise=True),
                Difference(),
                WassersteinScorer(component_wise=True),
                WassersteinScorer(window=20, component_wise=True),
                KMeansScorer(component_wise=True),
                KMeansScorer(window=20, component_wise=True),
                PyODScorer(model=KNN(), component_wise=True),
                PyODScorer(model=KNN(), window=10, component_wise=True),
            ],
        )
        anomaly_model.fit(mts_series_train, allow_model_training=True, start=0.1)

        scores, model_output = anomaly_model.score(
            mts_series_test, return_model_prediction=True, start=0.1
        )

        # model_output must be multivariate (same width as input)
        self.assertEqual(model_output.width, mts_series_test.width)

        # scores must be of the same length as the number of scorers
        self.assertEqual(len(scores), len(anomaly_model.scorers))

        dict_auc_roc = anomaly_model.eval_accuracy(
            ts_anomalies, mts_series_test, start=0.1, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_accuracy(
            ts_anomalies, mts_series_test, start=0.1, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=scores,
            window=[1, 1, 10, 20, 1, 20, 1, 10],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_accuracy_from_scores(
            actual_anomalies=[ts_anomalies] * 8,
            anomaly_score=scores,
            window=[1, 1, 10, 20, 1, 20, 1, 10],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        self.assertEqual(len(auc_roc_from_scores), len(dict_auc_roc))
        self.assertEqual(len(auc_pr_from_scores), len(dict_auc_pr))

        # function eval_accuracy_from_scores and eval_accuracy must return the same values
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, list(dict_auc_roc.values()), decimal=1
        )
        np.testing.assert_array_almost_equal(
            auc_pr_from_scores, list(dict_auc_pr.values()), decimal=1
        )

        true_auc_roc = [
            [0.8803738317757009, 0.912267218445167],
            [0.48898531375166887, 0.5758202778598878],
            [0.8375999073323295, 0.9162283996994741],
            [0.7798128494807715, 0.8739249880554228],
            [0.8803738317757008, 0.912267218445167],
            [0.7787287458632889, 0.8633540372670807],
            [0.8803738317757009, 0.9122672184451671],
            [0.8348777945094406, 0.9137061285821616],
        ]

        true_auc_pr = [
            [0.7123114333965317, 0.7579757115620807],
            [0.4447973021706103, 0.596776950584551],
            [0.744325434474558, 0.8984960888744328],
            [0.7653561450296187, 0.9233662817550338],
            [0.7123114333965317, 0.7579757115620807],
            [0.7852553779986415, 0.9185701347601994],
            [0.7123114333965319, 0.7579757115620807],
            [0.757208451057927, 0.8967178983419622],
        ]

        # check value of results
        np.testing.assert_array_almost_equal(
            auc_roc_from_scores, true_auc_roc, decimal=1
        )
        np.testing.assert_array_almost_equal(auc_pr_from_scores, true_auc_pr, decimal=1)

    def test_show_anomalies(self):

        forecasting_anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=NormScorer()
        )
        forecasting_anomaly_model.fit(self.train, allow_model_training=True)

        filtering_anomaly_model = FilteringAnomalyModel(
            model=MovingAverage(window=10), scorer=NormScorer()
        )

        for anomaly_model in [forecasting_anomaly_model, filtering_anomaly_model]:

            # must input only one series
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(series=[self.train, self.train])

            # input must be a series
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(series=[1, 2, 4])

            # metric must be "AUC_ROC" or "AUC_PR"
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train, actual_anomalies=self.anomalies, metric="str"
                )
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train, actual_anomalies=self.anomalies, metric="auc_roc"
                )
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train, actual_anomalies=self.anomalies, metric=1
                )

            # actual_anomalies must be not none if metric is given
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(series=self.train, metric="AUC_ROC")

            # actual_anomalies must be binary
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train, actual_anomalies=self.test, metric="AUC_ROC"
                )

            # actual_anomalies must contain at least 1 anomaly if metric is given
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train,
                    actual_anomalies=self.only_0_anomalies,
                    metric="AUC_ROC",
                )

            # actual_anomalies must contain at least 1 non-anomoulous timestamp
            # if metric is given
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train,
                    actual_anomalies=self.only_1_anomalies,
                    metric="AUC_ROC",
                )

            # names_of_scorers must be str
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(series=self.train, names_of_scorers=2)
            # nbr of names_of_scorers must match the nbr of scores (only 1 here)
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(
                    series=self.train, names_of_scorers=["scorer1", "scorer2"]
                )

            # title must be str
            with self.assertRaises(ValueError):
                anomaly_model.show_anomalies(series=self.train, title=1)

    def test_show_anomalies_from_scores(self):

        # must input only one series
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(series=[self.train, self.train])

        # input must be a series
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(series=[1, 2, 4])

        # must input only one model_output
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, model_output=[self.test, self.train]
            )

        # metric must be "AUC_ROC" or "AUC_PR"
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.anomalies,
                metric="str",
            )
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.anomalies,
                metric="auc_roc",
            )
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.anomalies,
                metric=1,
            )

        # actual_anomalies must be not none if metric is given
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, metric="AUC_ROC"
            )

        # actual_anomalies must be binary
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.test,
                metric="AUC_ROC",
            )

        # actual_anomalies must contain at least 1 anomaly if metric is given
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.only_0_anomalies,
                metric="AUC_ROC",
            )

        # actual_anomalies must contain at least 1 non-anomoulous timestamp
        # if metric is given
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                actual_anomalies=self.only_1_anomalies,
                metric="AUC_ROC",
            )

        # window must be int
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, window="1"
            )
        # window must be an int positive
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, window=-1
            )
        # window must smaller than the score series
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, window=200
            )

        # must have the same nbr of windows than scores
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, window=[1, 2]
            )
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=[self.test, self.test],
                window=[1, 2, 1],
            )

        # names_of_scorers must be str
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train, anomaly_scores=self.test, names_of_scorers=2
            )
        # nbr of names_of_scorers must match the nbr of scores
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=self.test,
                names_of_scorers=["scorer1", "scorer2"],
            )
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(
                series=self.train,
                anomaly_scores=[self.test, self.test],
                names_of_scorers=["scorer1", "scorer2", "scorer3"],
            )

        # title must be str
        with self.assertRaises(ValueError):
            show_anomalies_from_scores(series=self.train, title=1)
