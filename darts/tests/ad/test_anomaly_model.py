from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
import pytest
from pyod.models.knn import KNN

from darts import TimeSeries
from darts.ad import (
    AndAggregator,  # noqa: F401
    CauchyNLLScorer,
    EnsembleSklearnAggregator,  # noqa: F401
    ExponentialNLLScorer,
    FilteringAnomalyModel,
    ForecastingAnomalyModel,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
    OrAggregator,  # noqa: F401
    PoissonNLLScorer,
    PyODScorer,
    QuantileDetector,  # noqa: F401
    ThresholdDetector,  # noqa: F401
    WassersteinScorer,
)
from darts.ad import DifferenceScorer as Difference
from darts.ad import NormScorer as Norm
from darts.ad.utils import eval_metric_from_scores, show_anomalies_from_scores
from darts.models import MovingAverageFilter, NaiveSeasonal, RegressionModel

filtering_am = [
    (
        FilteringAnomalyModel,
        {"model": MovingAverageFilter(window=10), "scorer": Norm()},
    ),
    (
        FilteringAnomalyModel,
        {"model": MovingAverageFilter(window=10), "scorer": [Norm(), KMeansScorer()]},
    ),
    (
        FilteringAnomalyModel,
        {"model": MovingAverageFilter(window=10), "scorer": KMeansScorer()},
    ),
]

forecasting_am = [
    (ForecastingAnomalyModel, {"model": RegressionModel(lags=10), "scorer": Norm()}),
    (
        ForecastingAnomalyModel,
        {"model": RegressionModel(lags=10), "scorer": [Norm(), KMeansScorer()]},
    ),
    (
        ForecastingAnomalyModel,
        {"model": RegressionModel(lags=10), "scorer": KMeansScorer()},
    ),
]


class TestAnomalyDetectionModel:
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

    modified_train = MovingAverageFilter(window=10).filter(train)
    modified_test = MovingAverageFilter(window=10).filter(test)

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

    @pytest.mark.parametrize(
        "scorer,anomaly_model_config",
        product(
            [
                Norm(),
                Difference(),
                GaussianNLLScorer(),
                ExponentialNLLScorer(),
                PoissonNLLScorer(),
                LaplaceNLLScorer(),
                CauchyNLLScorer(),
                GammaNLLScorer(),
            ],
            [
                (ForecastingAnomalyModel, {"model": RegressionModel(lags=10)}),
                (FilteringAnomalyModel, {"model": MovingAverageFilter(window=20)}),
            ],
        ),
    )
    def test_non_fittable_scorer(self, scorer, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(scorer=scorer, **am_kwargs)
        assert not anomaly_model.scorers_are_trainable

    @pytest.mark.parametrize(
        "scorer,anomaly_model_config",
        product(
            [
                PyODScorer(model=KNN()),
                KMeansScorer(),
                WassersteinScorer(window_agg=False),
            ],
            [
                (ForecastingAnomalyModel, {"model": RegressionModel(lags=10)}),
                (FilteringAnomalyModel, {"model": MovingAverageFilter(window=20)}),
            ],
        ),
    )
    def test_fittable_scorer(self, scorer, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(scorer=scorer, **am_kwargs)
        assert anomaly_model.scorers_are_trainable

    def test_no_local_model(self):
        with pytest.raises(ValueError) as err:
            _ = ForecastingAnomalyModel(model=NaiveSeasonal(), scorer=KMeansScorer())
        assert str(err.value) == "`model` must be a Darts `GlobalForecastingModel`."

    @pytest.mark.parametrize(
        "anomaly_model,fit_model",
        [
            (
                ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm()),
                True,
            ),
            (
                FilteringAnomalyModel(
                    model=MovingAverageFilter(window=20), scorer=Norm()
                ),
                False,
            ),
        ],
    )
    def test_score(self, anomaly_model, fit_model):
        if fit_model:
            anomaly_model.fit(self.train, allow_model_training=True)

        # if return_model_prediction set to true, output must be tuple
        assert isinstance(
            anomaly_model.score(self.test, return_model_prediction=True), tuple
        )

        # if return_model_prediction set to false output must be
        # Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
        assert not isinstance(
            anomaly_model.score(self.test, return_model_prediction=False), tuple
        )

    @pytest.mark.parametrize("anomaly_model_config", filtering_am)
    def test_FitFilteringAnomalyModelInput(self, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(**am_kwargs)
        # `allow_model_training=True` has no effect if filter model has no `fit()` method
        anomaly_model.fit(self.train, allow_model_training=True)

        # input 'series' must be a series or Sequence of series
        with pytest.raises(ValueError):
            anomaly_model.fit([self.train, "str"], allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit([[self.train, self.train]], allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit("str", allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit([1, 2, 3], allow_model_training=True)

    @pytest.mark.parametrize("anomaly_model_config", forecasting_am)
    def test_FitForecastingAnomalyModelInput(self, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(**am_kwargs)

        # input 'series' must be a series or Sequence of series
        with pytest.raises(ValueError):
            anomaly_model.fit([self.train, "str"], allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit([[self.train, self.train]], allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit("str", allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.fit([1, 2, 3], allow_model_training=True)

        # 'allow_model_training' must be set to True if forecasting model is not fitted
        if anomaly_model.scorers_are_trainable:
            with pytest.raises(ValueError):
                anomaly_model.fit(self.train, allow_model_training=False)
                anomaly_model.score(self.train)

        with pytest.raises(ValueError):
            # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
            anomaly_model.fit(
                series=[self.train, self.train],
                past_covariates=self.covariates,
                allow_model_training=True,
            )

        with pytest.raises(ValueError):
            # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
            anomaly_model.fit(
                series=self.train,
                past_covariates=[self.covariates, self.covariates],
                allow_model_training=True,
            )

        with pytest.raises(ValueError):
            # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
            anomaly_model.fit(
                series=[self.train, self.train],
                future_covariates=self.covariates,
                allow_model_training=True,
            )

        with pytest.raises(ValueError):
            # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
            anomaly_model.fit(
                series=self.train,
                future_covariates=[self.covariates, self.covariates],
                allow_model_training=True,
            )

    def test_pretrain_forecasting_model(self):
        fitted_model = RegressionModel(lags=10).fit(self.train)
        # Fittable scorer must be fitted before calling .score(), even if forecasting model is fitted
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=fitted_model, scorer=KMeansScorer()).score(
                series=self.test
            )
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(
                model=fitted_model, scorer=[Norm(), KMeansScorer()]
            ).score(series=self.test)

        # forecasting model that do not accept past/future covariates
        # with pytest.raises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=NormScorer()).fit(
        #           series=self.train, past_covariates=self.covariates, allow_model_training=True
        #       )
        # with pytest.raises(ValueError):
        #    ForecastingAnomalyModel(model=ExponentialSmoothing(),
        #       scorer=NormScorer()).fit(
        #           series=self.train, future_covariates=self.covariates, allow_model_training=True
        #       )

        # check window size
        # max window size is len(series.drop_before(series.get_timestamp_at_point(start))) + 1
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10),
                scorer=KMeansScorer(window=50, window_agg=False),
            ).fit(series=self.train, start=0.9)

        # forecasting model that cannot be trained on a list of series
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=NaiveSeasonal(), scorer=Norm()).fit(
                series=[self.train, self.train], allow_model_training=True
            )

    @pytest.mark.parametrize("anomaly_model_config", forecasting_am)
    def test_ScoreForecastingAnomalyModelInput(self, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(**am_kwargs)
        anomaly_model.fit(self.train, allow_model_training=True)

        # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
        with pytest.raises(ValueError):
            anomaly_model.score(
                series=[self.train, self.train], past_covariates=self.covariates
            )

        # number of 'past_covariates' must be the same as the number of Timeseries in 'series'
        with pytest.raises(ValueError):
            anomaly_model.score(
                series=self.train,
                past_covariates=[self.covariates, self.covariates],
            )

        # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
        with pytest.raises(ValueError):
            anomaly_model.score(
                series=[self.train, self.train], future_covariates=self.covariates
            )

        # number of 'future_covariates' must be the same as the number of Timeseries in 'series'
        with pytest.raises(ValueError):
            anomaly_model.score(
                series=self.train,
                future_covariates=[self.covariates, self.covariates],
            )

    def test_window_size(self):
        # max window size is len(series.drop_before(series.get_timestamp_at_point(start))) + 1 for score()
        anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10),
            scorer=KMeansScorer(window=30, window_agg=False),
        )
        anomaly_model.fit(self.train, allow_model_training=True)
        with pytest.raises(ValueError):
            anomaly_model.score(series=self.train, start=0.9)

    @pytest.mark.parametrize("anomaly_model_config", filtering_am)
    def test_ScoreFilteringAnomalyModelInput(self, anomaly_model_config):
        am_cls, am_kwargs = anomaly_model_config
        anomaly_model = am_cls(**am_kwargs)

        if anomaly_model.scorers_are_trainable:
            anomaly_model.fit(self.train)

    @pytest.mark.parametrize(
        "anomaly_model,fit_kwargs",
        [
            (
                ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=Norm()),
                {"series": train, "allow_model_training": True},
            ),
            (
                FilteringAnomalyModel(
                    model=MovingAverageFilter(window=20), scorer=Norm()
                ),
                False,
            ),
            (
                ForecastingAnomalyModel(
                    model=RegressionModel(lags=10),
                    scorer=[Norm(), WassersteinScorer(window_agg=False)],
                ),
                {"series": train, "allow_model_training": True},
            ),
            (
                FilteringAnomalyModel(
                    model=MovingAverageFilter(window=20),
                    scorer=[Norm(), WassersteinScorer(window_agg=False)],
                ),
                {"series": train},
            ),
        ],
    )
    def test_eval_metric(self, anomaly_model, fit_kwargs):
        if fit_kwargs:
            anomaly_model.fit(**fit_kwargs)

        # if the anomaly_model have scorers that have the parameter is_univariate set to True,
        # 'anomalies' must have widths of 1
        if anomaly_model.scorers_are_univariate:
            with pytest.raises(ValueError):
                anomaly_model.eval_metric(
                    anomalies=self.mts_anomalies, series=self.test
                )
            with pytest.raises(ValueError):
                anomaly_model.eval_metric(
                    anomalies=self.mts_anomalies, series=self.mts_test
                )
            with pytest.raises(ValueError):
                anomaly_model.eval_metric(
                    anomalies=[self.anomalies, self.mts_anomalies],
                    series=[self.test, self.mts_test],
                )

        # 'metric' must be str and "AUC_ROC" or "AUC_PR"
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=self.anomalies, series=self.test, metric=1
            )
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=self.anomalies,
                series=self.test,
                metric="auc_roc",
            )
        with pytest.raises(TypeError):
            anomaly_model.eval_metric(
                anomalies=self.anomalies,
                series=self.test,
                metric=["AUC_ROC"],
            )

        # 'anomalies' must be binary
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(anomalies=self.test, series=self.test)

        # 'anomalies' must contain anomalies (at least one)
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(anomalies=self.only_0_anomalies, series=self.test)

        # 'anomalies' cannot contain only anomalies
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(anomalies=self.only_1_anomalies, series=self.test)

        # 'anomalies' must match the number of series
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=self.anomalies, series=[self.test, self.test]
            )
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=[self.anomalies, self.anomalies],
                series=self.test,
            )

        # 'anomalies' must have non empty intersection with 'series'
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=self.anomalies[:20], series=self.test[30:]
            )
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                anomalies=[self.anomalies, self.anomalies[:20]],
                series=[self.test, self.test[40:]],
            )

        # Check input type
        # 'anomalies' and 'series' must be of same length
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([self.anomalies], [self.test, self.test])
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(self.anomalies, [self.test, self.test])
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([self.anomalies, self.anomalies], [self.test])
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([self.anomalies, self.anomalies], self.test)

        # 'anomalies' and 'series' must be of type Timeseries
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([self.anomalies], [2, 3, 4])
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([self.anomalies], "str")
        with pytest.raises(ValueError):
            anomaly_model.eval_metric([2, 3, 4], self.test)
        with pytest.raises(ValueError):
            anomaly_model.eval_metric("str", self.test)
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                [self.anomalies, self.anomalies], [self.test, [3, 2, 1]]
            )
        with pytest.raises(ValueError):
            anomaly_model.eval_metric(
                [self.anomalies, [3, 2, 1]], [self.test, self.test]
            )

        # Check return types
        # Check if return type is float when input is a series
        assert isinstance(
            anomaly_model.eval_metric(self.anomalies, self.test),
            dict,
        )

        # Check if return type is Sequence when input is a Sequence of series
        assert isinstance(
            anomaly_model.eval_metric(self.anomalies, [self.test]),
            Sequence,
        )

        assert isinstance(
            anomaly_model.eval_metric(
                [self.anomalies, self.anomalies], [self.test, self.test]
            ),
            Sequence,
        )

    def test_ForecastingAnomalyModelInput(self):
        # model input
        # model input must be of type ForecastingModel
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model="str", scorer=Norm())
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=1, scorer=Norm())
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=MovingAverageFilter(window=10), scorer=Norm())
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(
                model=[RegressionModel(lags=10), RegressionModel(lags=5)],
                scorer=Norm(),
            )

        # scorer input
        # scorer input must be of type AnomalyScorer
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer=1)
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(model=RegressionModel(lags=10), scorer="str")
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=RegressionModel(lags=10)
            )
        with pytest.raises(ValueError):
            ForecastingAnomalyModel(
                model=RegressionModel(lags=10), scorer=[Norm(), "str"]
            )

    def test_FilteringAnomalyModelInput(self):
        # model input
        # model input must be of type FilteringModel
        with pytest.raises(ValueError):
            FilteringAnomalyModel(model="str", scorer=Norm())
        with pytest.raises(ValueError):
            FilteringAnomalyModel(model=1, scorer=Norm())
        with pytest.raises(ValueError):
            FilteringAnomalyModel(model=RegressionModel(lags=10), scorer=Norm())
        with pytest.raises(ValueError):
            FilteringAnomalyModel(
                model=[MovingAverageFilter(window=10), MovingAverageFilter(window=10)],
                scorer=Norm(),
            )

        # scorer input
        # scorer input must be of type AnomalyScorer
        with pytest.raises(ValueError):
            FilteringAnomalyModel(model=MovingAverageFilter(window=10), scorer=1)
        with pytest.raises(ValueError):
            FilteringAnomalyModel(model=MovingAverageFilter(window=10), scorer="str")
        with pytest.raises(ValueError):
            FilteringAnomalyModel(
                model=MovingAverageFilter(window=10),
                scorer=MovingAverageFilter(window=10),
            )
        with pytest.raises(ValueError):
            FilteringAnomalyModel(
                model=MovingAverageFilter(window=10), scorer=[Norm(), "str"]
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
                Norm(),
                Difference(),
                WassersteinScorer(window_agg=False),
                KMeansScorer(k=5),
                KMeansScorer(window=10, window_agg=False),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10, window_agg=False),
                WassersteinScorer(window=15, window_agg=False),
            ],
        )

        anomaly_model.fit(train_series_slope, allow_model_training=True, start=0.1)
        score, pred_series = anomaly_model.score(
            test_series_slope, return_model_prediction=True, start=0.1
        )

        # check that NormScorer is the abs difference of pred_series and test_series_slope
        assert (
            pred_series - test_series_slope.slice_intersect(pred_series)
        ).__abs__() == Norm().score_from_prediction(test_series_slope, pred_series)

        # check that Difference is the difference of pred_series and test_series_slope
        assert test_series_slope.slice_intersect(
            pred_series
        ) - pred_series == Difference().score_from_prediction(
            test_series_slope, pred_series
        )

        dict_auc_roc = anomaly_model.eval_metric(
            ts_anomalies, test_series_slope, metric="AUC_ROC", start=0.1
        )
        dict_auc_pr = anomaly_model.eval_metric(
            ts_anomalies, test_series_slope, metric="AUC_PR", start=0.1
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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
            model=MovingAverageFilter(window=5),
            scorer=[
                Norm(),
                Difference(),
                WassersteinScorer(window_agg=False),
                KMeansScorer(),
                KMeansScorer(window=10, window_agg=False),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10, window_agg=False),
                WassersteinScorer(window=15, window_agg=False),
            ],
        )
        anomaly_model.fit(train_series_noise)
        score, pred_series = anomaly_model.score(
            test_series_noise, return_model_prediction=True
        )

        # check that Difference is the difference of pred_series and test_series_noise
        assert test_series_noise.slice_intersect(
            pred_series
        ) - pred_series == Difference().score_from_prediction(
            test_series_noise, pred_series
        )

        # check that NormScorer is the abs difference of pred_series and test_series_noise
        assert (
            test_series_noise.slice_intersect(pred_series) - pred_series
        ).__abs__() == Norm().score_from_prediction(test_series_noise, pred_series)

        dict_auc_roc = anomaly_model.eval_metric(
            ts_anomalies, test_series_noise, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_metric(
            ts_anomalies, test_series_noise, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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
                Norm(),
                Difference(),
                WassersteinScorer(window_agg=False),
                KMeansScorer(k=4),
                KMeansScorer(k=7, window=10, window_agg=False),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10, window_agg=False),
                WassersteinScorer(window=15, window_agg=False),
            ],
        )

        anomaly_model.fit(
            series_train,
            allow_model_training=True,
            future_covariates=covariates,
            start=0.2,
        )

        score, pred_series = anomaly_model.score(
            series_test,
            return_model_prediction=True,
            future_covariates=covariates,
            start=0.2,
        )

        # check that NormScorer is the abs difference of pred_series and series_test
        assert (
            series_test.slice_intersect(pred_series) - pred_series
        ).__abs__() == Norm().score_from_prediction(series_test, pred_series)

        # check that Difference is the difference of pred_series and series_test
        assert series_test.slice_intersect(
            pred_series
        ) - pred_series == Difference().score_from_prediction(series_test, pred_series)

        dict_auc_roc = anomaly_model.eval_metric(
            ts_anomalies, series_test, metric="AUC_ROC", start=0.2
        )
        dict_auc_pr = anomaly_model.eval_metric(
            ts_anomalies, series_test, metric="AUC_PR", start=0.2
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=score,
            window=[1, 1, 10, 1, 10, 1, 10, 15],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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

    def test_multivariate_FilteringAnomalyModel(self):
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
            model=MovingAverageFilter(window=10),
            scorer=[
                Norm(component_wise=False),
                WassersteinScorer(window_agg=False),
                WassersteinScorer(window=12, window_agg=False),
                KMeansScorer(),
                KMeansScorer(window=5, window_agg=False),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=5, window_agg=False),
            ],
        )
        anomaly_model.fit(mts_series_train)

        scores, pred_series = anomaly_model.score(
            mts_series_test, return_model_prediction=True
        )

        # pred_series must be multivariate (same width as input)
        assert pred_series.width == mts_series_test.width

        # scores must be of the same length as the number of scorers
        assert len(scores) == len(anomaly_model.scorers)

        dict_auc_roc = anomaly_model.eval_metric(
            mts_anomalies, mts_series_test, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_metric(
            mts_anomalies, mts_series_test, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[mts_anomalies] * 7,
            pred_scores=scores,
            window=[1, 10, 12, 1, 5, 1, 5],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[mts_anomalies] * 7,
            pred_scores=scores,
            window=[1, 10, 12, 1, 5, 1, 5],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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
            model=MovingAverageFilter(window=10),
            scorer=[
                Norm(component_wise=True),
                Difference(),
                WassersteinScorer(component_wise=True, window_agg=False),
                WassersteinScorer(window=12, component_wise=True, window_agg=False),
                KMeansScorer(component_wise=True),
                KMeansScorer(window=5, component_wise=True, window_agg=False),
                PyODScorer(model=KNN(), component_wise=True),
                PyODScorer(
                    model=KNN(), window=5, component_wise=True, window_agg=False
                ),
            ],
        )
        anomaly_model.fit(mts_series_train)

        scores, pred_series = anomaly_model.score(
            mts_series_test, return_model_prediction=True
        )

        # pred_series must be multivariate (same width as input)
        assert pred_series.width == mts_series_test.width

        # scores must be of the same length as the number of scorers
        assert len(scores) == len(anomaly_model.scorers)

        dict_auc_roc = anomaly_model.eval_metric(
            ts_anomalies, mts_series_test, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_metric(
            ts_anomalies, mts_series_test, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=scores,
            window=[1, 1, 10, 12, 1, 5, 1, 5],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=scores,
            window=[1, 1, 10, 12, 1, 5, 1, 5],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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

    def test_multivariate_ForecastingAnomalyModel(self):
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
                Norm(component_wise=False),
                WassersteinScorer(window_agg=False),
                WassersteinScorer(window=20, window_agg=False),
                KMeansScorer(),
                KMeansScorer(window=20, window_agg=False),
                PyODScorer(model=KNN()),
                PyODScorer(model=KNN(), window=10, window_agg=False),
            ],
        )
        anomaly_model.fit(mts_series_train, allow_model_training=True, start=0.1)

        scores, pred_series = anomaly_model.score(
            mts_series_test, return_model_prediction=True, start=0.1
        )

        # pred_series must be multivariate (same width as input)
        assert pred_series.width == mts_series_test.width

        # scores must be of the same length as the number of scorers
        assert len(scores) == len(anomaly_model.scorers)

        dict_auc_roc = anomaly_model.eval_metric(
            mts_anomalies, mts_series_test, start=0.1, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_metric(
            mts_anomalies, mts_series_test, start=0.1, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[mts_anomalies] * 7,
            pred_scores=scores,
            window=[1, 10, 20, 1, 20, 1, 10],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[mts_anomalies] * 7,
            pred_scores=scores,
            window=[1, 10, 20, 1, 20, 1, 10],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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
                Norm(component_wise=True),
                Difference(),
                WassersteinScorer(component_wise=True, window_agg=False),
                WassersteinScorer(window=20, component_wise=True, window_agg=False),
                KMeansScorer(component_wise=True),
                KMeansScorer(window=20, component_wise=True, window_agg=False),
                PyODScorer(model=KNN(), component_wise=True),
                PyODScorer(
                    model=KNN(), window=10, component_wise=True, window_agg=False
                ),
            ],
        )
        anomaly_model.fit(mts_series_train, allow_model_training=True, start=0.1)

        scores, pred_series = anomaly_model.score(
            mts_series_test, return_model_prediction=True, start=0.1
        )

        # pred_series must be multivariate (same width as input)
        assert pred_series.width == mts_series_test.width

        # scores must be of the same length as the number of scorers
        assert len(scores) == len(anomaly_model.scorers)

        dict_auc_roc = anomaly_model.eval_metric(
            ts_anomalies, mts_series_test, start=0.1, metric="AUC_ROC"
        )
        dict_auc_pr = anomaly_model.eval_metric(
            ts_anomalies, mts_series_test, start=0.1, metric="AUC_PR"
        )

        auc_roc_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=scores,
            window=[1, 1, 10, 20, 1, 20, 1, 10],
            metric="AUC_ROC",
        )

        auc_pr_from_scores = eval_metric_from_scores(
            anomalies=[ts_anomalies] * 8,
            pred_scores=scores,
            window=[1, 1, 10, 20, 1, 20, 1, 10],
            metric="AUC_PR",
        )

        # function eval_accuracy_from_scores and eval_accuracy must return an input of same length
        assert len(auc_roc_from_scores) == len(dict_auc_roc)
        assert len(auc_pr_from_scores) == len(dict_auc_pr)

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

    def test_visualization(self):
        # test function show_anomalies() and show_anomalies_from_scores()
        forecasting_anomaly_model = ForecastingAnomalyModel(
            model=RegressionModel(lags=10), scorer=Norm()
        )
        forecasting_anomaly_model.fit(self.train, allow_model_training=True)

        filtering_anomaly_model = FilteringAnomalyModel(
            model=MovingAverageFilter(window=10), scorer=Norm()
        )

        self.show_anomalies_function(
            visualization_function=forecasting_anomaly_model.show_anomalies
        )
        self.show_anomalies_function(
            visualization_function=filtering_anomaly_model.show_anomalies
        )
        self.show_anomalies_function(visualization_function=show_anomalies_from_scores)

    def show_anomalies_function(self, visualization_function):
        # must input only one series
        with pytest.raises(ValueError) as err:
            visualization_function(series=[self.train, self.train])
        assert (
            str(err.value)
            == "`series` must be single `TimeSeries` or a sequence of `TimeSeries` of length `1`."
        )
        # input must be a series
        with pytest.raises(ValueError):
            visualization_function(series=[1, 2, 4])

        if visualization_function != show_anomalies_from_scores:
            # metric must be "AUC_ROC" or "AUC_PR"
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train,
                    anomalies=self.anomalies,
                    metric="str",
                )
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train,
                    anomalies=self.anomalies,
                    metric="auc_roc",
                )
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train, anomalies=self.anomalies, metric=1
                )

            # anomalies must be not none if metric is given
            with pytest.raises(ValueError):
                visualization_function(series=self.train, metric="AUC_ROC")

            # anomalies must be binary
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train,
                    anomalies=self.test,
                    metric="AUC_ROC",
                )

            # anomalies must contain at least 1 anomaly if metric is given
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train,
                    anomalies=self.only_0_anomalies,
                    metric="AUC_ROC",
                )

            # anomalies must contain at least 1 non-anomoulous timestamp
            # if metric is given
            with pytest.raises(ValueError):
                visualization_function(
                    series=self.train,
                    anomalies=self.only_1_anomalies,
                    metric="AUC_ROC",
                )
        else:
            # window must be a positive int
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train, pred_scores=self.test, window=-1
                )
            # window must smaller than the score series
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train, pred_scores=self.test, window=200
                )
            # must have the same nbr of windows than scores
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train, pred_scores=self.test, window=[1, 2]
                )
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train,
                    pred_scores=[self.test, self.test],
                    window=[1, 2, 1],
                )
            # nbr of names_of_scorers must match the nbr of scores
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train,
                    pred_scores=self.test,
                    names_of_scorers=["scorer1", "scorer2"],
                )
            with pytest.raises(ValueError):
                show_anomalies_from_scores(
                    series=self.train,
                    pred_scores=[self.test, self.test],
                    names_of_scorers=["scorer1", "scorer2", "scorer3"],
                )
