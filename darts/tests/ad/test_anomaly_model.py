from typing import Dict, Sequence, Tuple

import numpy as np
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
)
from darts.ad.scorers import NormScorer as Norm
from darts.ad.scorers import PoissonNLLScorer, PyODScorer, WassersteinScorer
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
            Norm(),
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

        am1 = ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm())
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=Norm())

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
            FilteringAnomalyModel(model=MovingAverage(window=20), scorer=Norm()),
            FilteringAnomalyModel(
                model=MovingAverage(window=20), scorer=[Norm(), KMeansScorer()]
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
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm()),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[Norm(), KMeansScorer()]
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
                model=fitted_model, scorer=[Norm(), KMeansScorer()]
            ).score(series=self.test)

        # forecasting model that do not accept past/future covariates
        # with self.assertRaises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=Norm()).fit(series=self.train, past_covariates=self.covariates, allow_model_training=True)
        # with self.assertRaises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=Norm()).fit(series=self.train, future_covariates=self.covariates, allow_model_training=True)

        # check window size
        # max window size is len(series.drop_before(series.get_timestamp_at_point(start))) + 1
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=KMeansScorer(window=50)
            ).fit(series=self.train, start=0.9)

        # forecasting model that cannot be trained on a list of series
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=NaiveSeasonal(), scorer=Norm()).fit(
                series=[self.train, self.train], allow_model_training=True
            )

    def test_ScoreForecastingAnomalyModelInput(self):

        for anomaly_model in [
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm()),
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[Norm(), KMeansScorer()]
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
            FilteringAnomalyModel(model=MovingAverage(window=10), scorer=Norm()),
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=[Norm(), KMeansScorer()]
            ),
            FilteringAnomalyModel(
                model=MovingAverage(window=10), scorer=KMeansScorer()
            ),
        ]:

            if anomaly_model.scorers_are_trainable:
                anomaly_model.fit(self.train)

    def test_show_anomalies(self):

        am1 = ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm())
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=Norm())

        for am in [am1, am2]:
            # input 'series' must be a series and not a Sequence of series
            with self.assertRaises(ValueError):
                am.show_anomalies([self.train, self.train])

    def test_eval_accuracy(self):

        am1 = ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm())
        am1.fit(self.train, allow_model_training=True)

        am2 = FilteringAnomalyModel(model=MovingAverage(window=20), scorer=Norm())

        am3 = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=[Norm(), WassersteinScorer()]
        )
        am3.fit(self.train, allow_model_training=True)

        am4 = FilteringAnomalyModel(
            model=MovingAverage(window=20), scorer=[Norm(), WassersteinScorer()]
        )
        am4.fit(self.train)

        for am in [am1, am2, am3, am4]:

            # if the anomaly_model have scorers that have the parameter returns_UTS set to True,
            # 'actual_anomalies' must have widths of 1
            if am.scorers_are_returns_UTS:
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
            ForecastingAnomalyModel(model="str", scorer=Norm())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=1, scorer=Norm())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(model=MovingAverage(window=10), scorer=Norm())
        with self.assertRaises(ValueError):
            ForecastingAnomalyModel(
                model=[RegressionModel(lags=10), RegressionModel(lags=5)],
                scorer=Norm(),
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
                model=RegressionModel(lags=10), scorer=[Norm(), "str"]
            )

    def test_FilteringAnomalyModelInput(self):

        # model input
        # model input must be of type FilteringModel
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model="str", scorer=Norm())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=1, scorer=Norm())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(model=RegressionModel(lags=10), scorer=Norm())
        with self.assertRaises(ValueError):
            FilteringAnomalyModel(
                model=[MovingAverage(window=10), MovingAverage(window=10)],
                scorer=Norm(),
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
                model=MovingAverage(window=10), scorer=[Norm(), "str"]
            )
