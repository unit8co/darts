import numpy as np
import pytest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from darts.models.forecasting.categorical_model import CategoricalModel
from darts.models.forecasting.nlinear import NLinearClassifierModel
from darts.models.forecasting.xgboost import XGBClassifierModel
from darts.utils import timeseries_generation as tg


def init_models(classifiers, extra_models):
    models = [
        CategoricalModel(model=clf(**kwargs), lags_past_covariates=5)
        for (clf, kwargs) in classifiers
    ]
    models.extend([
        clf(lags_past_covariates=5, **kwargs) for clf, kwargs in extra_models
    ])
    return models


def init_torch_models(torch_models):
    return [clf(**kwargs) for clf, kwargs in torch_models]


class TestCategoricalForecasting:
    np.random.seed(42)

    sine_univariate1 = tg.sine_timeseries(length=100) + 1
    sine1_target = sine_univariate1.map(lambda x: np.round(x))

    s1_train_y, s1_test_y = sine_univariate1.split_before(0.7)
    sine1_target_train, sine1_target_test = sine1_target.split_before(0.7)

    classifiers = [
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
    ]

    extra_models = [(XGBClassifierModel, {})]
    torch_models = [
        (
            NLinearClassifierModel,
            {
                "input_chunk_length": 10,
                "output_chunk_length": 1,
                "output_chunk_shift": 0,
                "loss_fn": nn.CrossEntropyLoss(),
            },
        )
    ]

    @pytest.mark.parametrize("model", init_models(classifiers, extra_models))
    def test_classifier_specific_behavior(self, model):
        # accepts only classifier
        with pytest.raises(ValueError):
            CategoricalModel(model=LinearRegression())

        # accessing classes_ before training raises an error
        with pytest.raises(AttributeError):
            model.class_labels

        # training the model
        model.fit(series=self.sine1_target, past_covariates=self.sine_univariate1)
        assert [0, 1, 2] == list(model.class_labels)

    @pytest.mark.parametrize("model", init_models(classifiers, extra_models))
    def test_classifier_accuracy(self, model):
        model.fit(series=self.sine1_target_train, past_covariates=self.sine_univariate1)
        pred = model.predict(
            n=len(self.s1_test_y), past_covariates=self.sine_univariate1
        )
        assert (
            sum(pred.values() == self.sine1_target_test.values())[0] / len(pred)
        ) >= 1

    @pytest.mark.parametrize("model", init_torch_models(torch_models))
    def test_torch_classifier_accuracy(self, model):
        model.fit(self.sine1_target_train.astype(np.float32), epochs=500, verbose=True)
        pred = model.predict(n=len(self.sine1_target_test))
        assert (
            sum(pred.values() == self.sine1_target_test.values())[0] / len(pred)
        ) >= 1
