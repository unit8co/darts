import numpy as np
import pytest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from darts.logging import get_logger
from darts.models.forecasting.catboost_model import CatBoostClassifierModel
from darts.models.forecasting.lgbm import LightGBMClassifierModel
from darts.models.forecasting.xgboost import XGBClassifierModel
from darts.tests.models.forecasting.test_classifier_model import process_model_list
from darts.utils import timeseries_generation as tg
from darts.utils.utils import NotImportedModule

lgbm_available = not isinstance(LightGBMClassifierModel, NotImportedModule)
cb_available = not isinstance(CatBoostClassifierModel, NotImportedModule)

logger = get_logger(__name__)


class TestProbabilisticClassifierModels:
    np.random.seed(42)
    # shift sines to positive values so that they can be used as target for classification with classes [0, 1, 2]
    sine_univariate1 = tg.sine_timeseries(length=100) + 1
    sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705) + 1
    sine_univariate3 = tg.sine_timeseries(length=100, value_phase=0.78525) + 1
    sine_univariate4 = tg.sine_timeseries(length=100, value_phase=0.392625) + 1
    sine_univariate5 = tg.sine_timeseries(length=100, value_phase=0.1963125) + 1
    sine_univariate6 = tg.sine_timeseries(length=100, value_phase=0.09815625) + 1

    sine_univariate1_cat = sine_univariate1.map(lambda x: np.round(x))
    sine_univariate2_cat = sine_univariate2.map(lambda x: np.round(x))
    sine_univariate3_cat = sine_univariate3.map(lambda x: np.round(x))

    sine_multivariate1_cat = sine_univariate1_cat.stack(sine_univariate2_cat)
    sine_multivariate2 = sine_univariate2.stack(sine_univariate3)

    sine_multiseries1_cat = [
        sine_univariate1_cat,
        sine_univariate2_cat,
        sine_univariate3_cat,
    ]
    sine_multiseries2 = [sine_univariate4, sine_univariate5, sine_univariate6]

    probabilistic_classifiers = [
        (LogisticRegression, {}),
        (KNeighborsClassifier, {"n_neighbors": 3}),
        (SVC, {"gamma": 2, "C": 1, "random_state": 42, "probability": True}),
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
        (
            XGBClassifierModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "max_leaves": 1,
                "random_state": 42,
            },
        ),
    ]

    if lgbm_available:
        probabilistic_classifiers.append((
            LightGBMClassifierModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "num_leaves": 2,
                "verbosity": -1,
                "random_state": 42,
            },
        ))

    if cb_available:
        probabilistic_classifiers.append((
            CatBoostClassifierModel,
            {
                "iterations": 1,
                "depth": 1,
                "verbose": -1,
                "random_state": 42,
            },
        ))

    @pytest.mark.parametrize(
        "clf_params",
        process_model_list(probabilistic_classifiers),
    )
    def test_class_proba_likelihood_median_pred_is_same_than_no_likelihood(
        self, clf_params
    ):
        clf, kwargs = clf_params
        model = clf(lags=2, **kwargs)
        model._likelihood = None  # Hard remove likelihood

        model_likelihood = clf(
            lags=2,
            **kwargs,
        )

        model.fit(self.sine_univariate1_cat)
        # model has no likelihood
        with pytest.raises(ValueError) as err:
            model.predict(5, predict_likelihood_parameters=True)
        assert (
            str(err.value) == "`predict_likelihood_parameters=True` is only"
            " supported for probabilistic models fitted with a likelihood."
        )

        model_likelihood.fit(self.sine_univariate1_cat)
        # model_likelihood has ClassProbability likelihood
        probas = model_likelihood.predict(1, predict_likelihood_parameters=True)
        # Sum of class proba is 1
        assert probas.sum(axis=1).values()[0][0] == pytest.approx(1)
        # As many probabilties as classes
        assert len(probas.components) == 3

        # Without predict_likelihood_parameters model predict same class with and without the likelihood
        # Meaning _get_median_prediction on top on predict_proba produce the same output than the model predict
        assert model_likelihood.predict(5) == model.predict(5)
