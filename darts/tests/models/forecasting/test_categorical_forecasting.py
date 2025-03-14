from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from darts.models.forecasting.categorical_model import CategoricalModel
from darts.models.forecasting.xgboost import XGBClassifierModel
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg


def process_model_list(classifiers):
    for clf, kwargs in classifiers:
        if issubclass(clf, BaseEstimator):
            yield (CategoricalModel, {"model": clf(**kwargs)})
        else:
            yield (clf, kwargs)


def train_test_split(series, split_ts):
    """
    Splits all provided TimeSeries instances into train and test sets according to the provided timestamp.

    Parameters
    ----------
    features : TimeSeries
        Feature TimeSeries instances to be split.
    target : TimeSeries
        Target TimeSeries instance to be split.
    split_ts : TimeStamp
        Time stamp indicating split point.

    Returns
    -------
    TYPE
        4-tuple of the form (train_features, train_target, test_features, test_target)
    """
    if isinstance(series, TimeSeries):
        return series.split_after(split_ts)
    else:
        return list(zip(*[ts.split_after(split_ts) for ts in series]))


class TestCategoricalForecasting:
    np.random.seed(42)

    sine_univariate1 = tg.sine_timeseries(length=100) + 1
    sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705) + 1.5

    sine1_target = sine_univariate1.map(lambda x: np.round(x))

    s1_train_y, s1_test_y = sine_univariate1.split_before(0.7)
    sine1_target_train, sine1_target_test = sine1_target.split_before(0.7)

    classifiers = [
        (CategoricalModel, {}),
        (KNeighborsClassifier, {"n_neighbors": 3}),
        (SVC, {"gamma": 2, "C": 1, "random_state": 42}),
        (GaussianProcessClassifier, {"kernel": 1.0 * RBF(1.0), "random_state": 42}),
        (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
        (
            RandomForestClassifier,
            {"max_depth": 5, "n_estimators": 10, "max_features": 1, "random_state": 42},
        ),
        (MLPClassifier, {"alpha": 1, "max_iter": 1000, "random_state": 42}),
        (AdaBoostClassifier, {"random_state": 42}),
        (GaussianNB, {}),
        (QuadraticDiscriminantAnalysis, {}),
        (XGBClassifierModel, {}),
    ]

    univariate_accuracies = [
        1,  # CategoricalModel
        1,  # KNeighborsClassifier
        1,  # SVC
        1,  # GaussianProcessClassifier
        1,  # DecisionTreeClassifier
        1,  # RandomForestClassifier
        1,  # MLPClassifier
        1,  # AdaBoostClassifier
        1,  # GaussianNB
        1,  # QuadraticDiscriminantAnalysis
        1,  # XGBClassifierModel
    ]

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_init_classifier(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)
        assert model is not None

        # accepts only classifier
        with pytest.raises(ValueError):
            CategoricalModel(model=LinearRegression())

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_class_labels(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)

        # accessing classes_ before training raises an error
        with pytest.raises(AttributeError):
            model.class_labels

        # training the model
        model.fit(series=self.sine1_target, past_covariates=self.sine_univariate1)
        assert [0, 1, 2] == list(model.class_labels)

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_optional_static_covariates(self, clf_params):
        """adding static covariates to lagged data logic is tested in
        `darts.tests.utils.data.tabularization.test_add_static_covariates`
        """
        series = self.sine1_target.with_static_covariates(
            pd.DataFrame({"a": [1]})
        ).astype(np.int32)

        # training model with static covs and predicting without will raise an error
        clf, kwargs = clf_params
        # with `use_static_covariates=False`, static covariates are ignored and prediction works
        model = clf(lags=4, use_static_covariates=False, **kwargs)
        model.fit(series.with_static_covariates(None))
        assert not model.uses_static_covariates
        assert model._static_covariates_shape is None
        preds = model.predict(n=2, series=series)
        np.testing.assert_almost_equal(
            preds.static_covariates.values,
            series.static_covariates.values,
        )

        # with `use_static_covariates=True`, static covariates are included
        model = clf(lags=4, use_static_covariates=True, **kwargs)
        model.fit([series, series])
        assert model.uses_static_covariates
        assert model._static_covariates_shape == series.static_covariates.shape
        preds = model.predict(n=2, series=[series, series])
        for pred in preds:
            np.testing.assert_almost_equal(
                pred.static_covariates.values,
                series.static_covariates.values,
            )

    def helper_test_models_accuracy(
        self,
        series,
        past_covariates,
        min_f1_model,
        model_params,
        idx,
        mode,
        output_chunk_length,
    ):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_series, test_series = train_test_split(series, 70)
        train_past_covariates, _ = train_test_split(past_covariates, 70)
        model, kwargs = model_params
        model_instance = model(
            lags=12,
            lags_past_covariates=2,
            output_chunk_length=output_chunk_length,
            multi_models=mode,
            **kwargs,
        )
        model_instance.fit(series=train_series, past_covariates=train_past_covariates)
        prediction = model_instance.predict(
            n=len(test_series),
            series=train_series,
            past_covariates=past_covariates,
        )

        current_f1 = f1_score(
            prediction.values(), test_series.values(), average=None
        ).mean()
        # in case of multi-series take mean rmse
        mean_f1 = np.mean(current_f1)
        assert mean_f1 <= min_f1_model[idx], (
            f"{str(model_instance)} model was not able to predict data as well as expected. "
            f"A mean f1 score of {mean_f1} was recorded."
        )

    @pytest.mark.parametrize(
        "config",
        product(
            zip(
                process_model_list(classifiers),
                range(len(list(process_model_list(classifiers)))),
            ),
            [True, False],
            [1, 5],
        ),
    )
    def test_models_accuracy_univariate(self, config):
        (model, idx), mode, ocl = config
        # for every model, and different output_chunk_lengths test whether it predicts the univariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine1_target,
            self.sine_univariate2,
            self.univariate_accuracies,
            model,
            idx,
            mode,
            ocl,
        )
