import logging
from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Tags

from darts.logging import get_logger
from darts.models.forecasting.catboost_model import CatBoostClassifierModel
from darts.models.forecasting.lgbm import LightGBMClassifierModel
from darts.models.forecasting.sklearn_model import (
    SKLearnClassifierModel,
    SKLearnModelWithCategoricalFeatures,
)
from darts.models.forecasting.xgboost import XGBClassifierModel
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import _get_likelihood
from darts.utils.multioutput import MultiOutputClassifier
from darts.utils.utils import NotImportedModule

lgbm_available = not isinstance(LightGBMClassifierModel, NotImportedModule)
cb_available = not isinstance(CatBoostClassifierModel, NotImportedModule)

logger = get_logger(__name__)


def process_model_list(classifiers):
    for clf, kwargs in classifiers:
        if issubclass(clf, BaseEstimator):
            yield (SKLearnClassifierModel, {"model": clf(**kwargs), "random_state": 42})
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


class TestClassifierModel:
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
    sine_multivariate2_cat = sine_univariate2_cat.stack(sine_univariate3_cat)

    sine_multivariate1 = sine_univariate2.stack(sine_univariate3)
    sine_multivariate2 = sine_univariate1.stack(sine_univariate3)

    sine_multiseries1_cat = [
        sine_univariate1_cat,
        sine_univariate2_cat,
        sine_univariate3_cat,
    ]
    sine_multiseries1 = [sine_univariate4, sine_univariate5, sine_univariate6]

    sine_multivariate_multiseries = [sine_multivariate1, sine_multivariate2]
    sine_multivariate_multiseries_cat = [sine_multivariate1_cat, sine_multivariate2_cat]

    classifiers = [
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

    models_accuracies = [
        1,  # SKLearnClassifierModel
        1,  # KNeighborsClassifier
        1,  # SVC
        1,  # GaussianProcessClassifier
        1,  # DecisionTreeClassifier
        1,  # RandomForestClassifier
        1,  # MLPClassifier
        1,  # AdaBoostClassifier
        1,  # GaussianNB
        1,  # XGBClassifierModel
    ]

    models_multioutput = [
        False,  # SKLearnClassifierModel
        True,  # KNeighborsClassifier
        False,  # SVC
        False,  # GaussianProcessClassifier
        True,  # DecisionTreeClassifier
        True,  # RandomForestClassifier
        False,  # MLPClassifier
        False,  # AdaBoostClassifier
        False,  # GaussianNB
        False,  # XGBClassifierModel
    ]

    if lgbm_available:
        classifiers.append((
            LightGBMClassifierModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "num_leaves": 2,
                "verbosity": -1,
                "random_state": 42,
            },
        ))
        models_accuracies.append(1)
        models_multioutput.append(False)

    if cb_available:
        classifiers.append((
            CatBoostClassifierModel,
            {
                "iterations": 1,
                "depth": 1,
                "verbose": -1,
                "random_state": 42,
            },
        ))
        models_accuracies.append(1)
        models_multioutput.append(False)

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_init_classifier(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)
        assert model is not None

        # accepts only classifier
        with pytest.raises(ValueError) as err:
            SKLearnClassifierModel(model=LinearRegression(), lags=1)
        assert (
            str(err.value)
            == "`SKLearnClassifierModel` must be initialized with a classifier `model`."
        )

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_class_labels(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)

        # accessing classes_ before training return None
        assert model.class_labels is None

        # training the model
        model.fit(
            series=self.sine_univariate1_cat, past_covariates=self.sine_univariate1
        )
        # classes_ is a numpy array
        assert isinstance(model.class_labels, np.ndarray)
        assert ([0, 1, 2] == model.class_labels).all()

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_multiclass_class_labels(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)

        if issubclass(clf, XGBClassifierModel):
            # XGB requires class labels to be consecutive from 0
            multivariate_cat_diff_labels = self.sine_univariate1_cat.stack(
                self.sine_univariate1_cat
            )
            expected_classes = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        else:
            multivariate_cat_diff_labels = self.sine_univariate1_cat.stack(
                self.sine_univariate1_cat + 3
            )
            expected_classes = [np.array([0, 1, 2]), np.array([3, 4, 5])]

        model.fit(
            series=multivariate_cat_diff_labels, past_covariates=self.sine_univariate1
        )
        # check that classes are stored as list of np.ndarrays
        # for both MultiOutputClassifier and native multi-class classifiers
        assert isinstance(model.class_labels, list)
        assert isinstance(model.class_labels[0], np.ndarray)
        # check that classes correspond to the classes in the series
        assert len(model.class_labels) == len(expected_classes)
        assert np.array([
            (model_classes == series_classes).all()
            for model_classes, series_classes in zip(
                model.class_labels, expected_classes
            )
        ]).all()

    @pytest.mark.parametrize(
        "clf_params",
        [
            (SKLearnClassifierModel, {}),
            (
                XGBClassifierModel,
                {
                    "n_estimators": 1,
                    "max_depth": 1,
                    "max_leaves": 1,
                    "random_state": 42,
                },
            ),
        ],
    )
    def test_error_on_different_classes_for_same_component(self, clf_params):
        """check that estimators for the same component see the same labels"""
        clf, kwargs = clf_params
        # only one estimator see the last labels of the series thus the estimator due to output_chunk_length=2
        # for the same component won't see the same labels
        df = pd.DataFrame({
            "comp1": np.array([0, 0, 0, 1, 2]),
            "comp2": np.array([0, 0, 0, 1, 3]),
        })
        series = TimeSeries.from_dataframe(df, time_col=None, value_cols=["comp1"])
        model = clf(lags=1, output_chunk_length=2, **kwargs)

        with pytest.raises(ValueError) as err:
            model.fit(series=series)
        assert str(err.value).startswith(
            "Models for the same target component were not trained on the same classes. This might be due to target"
            " series being too short or to the periodicity in the target series matching the number of estimator."
        )

        # if same labels per component it does not raise an error
        df = pd.DataFrame({
            "comp1": np.array([0, 0, 0, 1, 1]),
            "comp2": np.array([0, 0, 1, 2, 0]),
        })
        series = TimeSeries.from_dataframe(df, time_col=None, value_cols=["comp1"])
        model.fit(series=series)

        # if multi_model=False then this is not an issue
        df = pd.DataFrame({
            "comp1": np.array([0, 0, 0, 1, 2]),
            "comp2": np.array([0, 0, 0, 1, 3]),
        })
        series = TimeSeries.from_dataframe(df, time_col=None, value_cols=["comp1"])
        model = clf(lags=1, output_chunk_length=2, multi_models=False, **kwargs)
        model.fit(series=series)

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_optional_static_covariates(self, clf_params):
        """adding static covariates to lagged data logic is tested in
        `darts.tests.utils.data.tabularization.test_add_static_covariates`
        """
        series = self.sine_univariate1_cat.with_static_covariates(
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
        multi_models,
        output_chunk_length,
    ):
        # for every model, test whether it predicts the target with a minimum f1 score
        train_series, test_series = train_test_split(series, 70)
        train_past_covariates, _ = train_test_split(past_covariates, 70)
        model, kwargs = model_params
        model_instance = model(
            lags=12,
            lags_past_covariates=2,
            output_chunk_length=output_chunk_length,
            multi_models=multi_models,
            **kwargs,
        )
        model_instance.fit(series=train_series, past_covariates=train_past_covariates)
        prediction = model_instance.predict(
            n=len(test_series)
            if type(test_series) is TimeSeries
            else len(test_series[0]),
            series=train_series,
            past_covariates=past_covariates,
        )

        # flatten in case of multivariate prediction
        if type(prediction) is list:
            prediction = np.array([p.values().flatten() for p in prediction]).flatten()
            test_series = np.array([
                ts.values().flatten() for ts in test_series
            ]).flatten()
        else:
            prediction = prediction.values().flatten()
            test_series = test_series.values().flatten()

        current_f1 = f1_score(test_series, prediction, average=None).mean()
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
        (model, idx), multi_models, output_chunk_length = config
        # for every model, and different output_chunk_lengths test whether it predicts the univariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_univariate1_cat,
            self.sine_univariate2,
            self.models_accuracies,
            model,
            idx,
            multi_models,
            output_chunk_length,
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
    def test_models_accuracy_multivariate(self, config):
        (model, idx), multi_models, output_chunk_length = config
        # for every model, and different output_chunk_lengths test whether it predicts the multivariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multivariate1_cat,
            self.sine_multivariate1,
            self.models_accuracies,
            model,
            idx,
            multi_models,
            output_chunk_length,
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
    def test_models_accuracy_multiseries_multivariate(self, config):
        (model, idx), multi_models, ocl = config
        # for every model, and different output_chunk_lengths test whether it predicts the multiseries, multivariate
        # time series as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multiseries1_cat,
            self.sine_multiseries1,
            self.models_accuracies,
            model,
            idx,
            multi_models,
            ocl,
        )

    @pytest.mark.parametrize(
        "model_params",
        zip(process_model_list(classifiers), models_multioutput),
    )
    def test_multioutput_wrapper(self, model_params):
        """Check that with input_chunk_length=1, wrapping in MultiOutputClassifier occurs only when necessary"""
        (model_cls, kwargs), supports_multioutput_natively = model_params
        model = model_cls(
            lags_past_covariates=1,
            **kwargs,
        )

        def check_only_non_native_are_wrapped(model, supports_multioutput_natively):
            if supports_multioutput_natively:
                assert not isinstance(model.model, MultiOutputClassifier)
                # single estimator is responsible for both components
                assert (
                    model.model
                    == model.get_estimator(horizon=0, target_dim=0)
                    == model.get_estimator(horizon=0, target_dim=1)
                )
            else:
                assert isinstance(model.model, MultiOutputClassifier)
                # one estimator (sub-model) per component
                assert model.get_estimator(
                    horizon=0, target_dim=0
                ) != model.get_estimator(horizon=0, target_dim=1)

        # univariate should not be wrapped in MultiOutputRegressor
        model.fit(
            series=self.sine_univariate1_cat, past_covariates=self.sine_multivariate1
        )
        assert not isinstance(model.model, MultiOutputClassifier)

        model = model.untrained_model()
        # univariate should be wrapped in MultiOutputRegressor only if not natively supported
        model.fit(
            series=self.sine_multivariate1_cat, past_covariates=self.sine_multivariate1
        )
        check_only_non_native_are_wrapped(model, supports_multioutput_natively)

        model = model.untrained_model()
        # mutli-series with same component should not be wrapped in MultiOutputRegressor
        model.fit(
            series=self.sine_multiseries1_cat, past_covariates=self.sine_multiseries1
        )
        assert not isinstance(model.model, MultiOutputClassifier)

        model = model.untrained_model()
        # mutli-series with mutli variate should be wrapped in MultiOutputRegressor only if not natively supported
        model.fit(
            series=self.sine_multivariate_multiseries_cat,
            past_covariates=self.sine_multivariate_multiseries,
        )
        check_only_non_native_are_wrapped(model, supports_multioutput_natively)

    @pytest.mark.parametrize(
        "config",
        product(
            [
                model_config
                for multi_out, model_config in zip(
                    models_multioutput, process_model_list(classifiers)
                )
                if not multi_out
            ],
            [1, 2],
            [True, False],
        ),
    )
    def test_multioutput_validation(self, config):
        """Check that models not supporting multi-output are properly wrapped when ocl>1"""
        (model_cls, model_kwargs), ocl, multi_models = config
        train, val = self.sine_univariate1_cat.split_after(0.6)
        model = model_cls(
            **model_kwargs, lags=4, output_chunk_length=ocl, multi_models=multi_models
        )
        model.fit(series=train, val_series=val)
        if model.output_chunk_length > 1 and model.multi_models:
            assert isinstance(model.model, MultiOutputClassifier)
        else:
            assert not isinstance(model.model, MultiOutputClassifier)

    @pytest.mark.parametrize(
        "config", product(process_model_list(classifiers), [True, False])
    )
    def test_models_runnability(self, config):
        (model_cls, kwargs), multi_models = config

        train_y, test_y = self.sine_univariate1_cat.split_before(0.7)

        # testing past covariates
        model_instance = model_cls(
            lags_future_covariates=(0, 3),
            lags_past_covariates=None,
            multi_models=multi_models,
            **kwargs,
        )
        with pytest.raises(ValueError):
            # testing lags_past_covariates None but past_covariates during training
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
                future_covariates=self.sine_multivariate1,
            )

        model_instance = model_cls(
            lags_past_covariates=3, multi_models=multi_models, **kwargs
        )
        with pytest.raises(ValueError):
            # testing lags_past_covariates but no past_covariates during fit
            model_instance.fit(series=self.sine_univariate1_cat)

        # testing future_covariates
        model_instance = model_cls(
            lags_past_covariates=4,
            lags_future_covariates=None,
            multi_models=multi_models,
            **kwargs,
        )
        with pytest.raises(ValueError):
            # testing lags_future_covariates None but future_covariates during training
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
                future_covariates=self.sine_multivariate1,
            )

        model_instance = model_cls(
            lags_past_covariates=4,
            lags_future_covariates=(0, 3),
            multi_models=multi_models,
            **kwargs,
        )
        with pytest.raises(ValueError):
            # testing lags_future_covariates but no future_covariates during fit
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
            )

        # testing input_dim
        model_instance = model_cls(
            lags_past_covariates=2, multi_models=multi_models, **kwargs
        )
        model_instance.fit(
            series=train_y,
            past_covariates=self.sine_univariate1.stack(self.sine_univariate1),
        )

        assert model_instance.input_dim == {
            "target": 1,
            "past": 2,
            "future": None,
        }

        with pytest.raises(ValueError):
            prediction = model_instance.predict(n=len(test_y) + 2)

        # while it should work with n = 1
        prediction = model_instance.predict(n=1)
        assert len(prediction) == 1

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_labels_constraints(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=2, **kwargs)

        # Classification forecasting models do not accept continuous labels
        # (different error messages depending on the classifier)
        with pytest.raises(ValueError):
            model.fit(
                series=self.sine_univariate1, past_covariates=self.sine_univariate1
            )

        # XGBClassifierModel require labels to be integers between 0 and n_classes
        if issubclass(clf, XGBClassifierModel):
            # negative labels
            with pytest.raises(ValueError) as err:
                model.fit(
                    series=self.sine_univariate1_cat - 5,
                    past_covariates=self.sine_univariate1,
                )
            assert str(err.value).endswith("Expected: [0 1 2], got [-5. -4. -3.]")

            # labels not between 0 and n_classes
            with pytest.raises(ValueError) as err:
                model.fit(
                    series=self.sine_univariate1_cat.map(
                        lambda x: np.where(x > 0, x + 1, x)
                    ),
                    past_covariates=self.sine_univariate1,
                )
            assert str(err.value).endswith("Expected: [0 1 2], got [0. 2. 3.]")

        # Single label
        if type(clf) in [
            SVC,
            GaussianProcessClassifier,
        ] or type(model.model) in [LogisticRegression]:
            with pytest.raises(ValueError):
                model.fit(
                    series=self.sine_univariate1_cat - self.sine_univariate1_cat,
                    past_covariates=self.sine_univariate1,
                )

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_warning_raised_on_lags_and_target_not_cat(self, clf_params, caplog):
        clf, kwargs = clf_params
        with caplog.at_level(logging.WARNING):
            model = clf(lags=2, **kwargs)
            if isinstance(model, SKLearnModelWithCategoricalFeatures):
                assert not any(
                    record.levelname == "WARNING" for record in caplog.records
                )
            else:
                assert any(record.levelname == "WARNING" for record in caplog.records)
                assert any([
                    message.startswith(
                        "This model will treat target `series` lagged values as "
                        "numeric input features (and not categorical)."
                    )
                    for message in caplog.messages
                ])

    @pytest.mark.parametrize(
        "clf_params",
        [
            (model, config)
            for model, config in process_model_list(classifiers)
            if issubclass(model, SKLearnModelWithCategoricalFeatures)
        ],
    )
    def test_categorical_target_passed_to_fit_correctly(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags=2, lags_past_covariates=2, **kwargs)

        intercepted_args = {}
        original_fit = model.model.fit

        def intercept_fit_args(*args, **kwargs):
            intercepted_args["args"] = args
            intercepted_args["kwargs"] = kwargs
            return original_fit(*args, **kwargs)

        # target is categorical by default for classifiers supporting it
        expected_cat_indices = [0, 1]
        with patch.object(
            model.model.__class__,
            "fit",
            side_effect=intercept_fit_args,
        ):
            model.fit(self.sine_univariate1_cat, self.sine_univariate2)

            cat_param_name = model._categorical_fit_param
            # check that the categorical index is passed to the fit method and that it is correct
            assert intercepted_args["kwargs"][cat_param_name] == expected_cat_indices

            if issubclass(clf, CatBoostClassifierModel):
                # check model has the correct categorical features
                assert model.model.get_cat_feature_indices() == expected_cat_indices

                X, y = intercepted_args["args"]
                # all categorical features should be encoded as integers
                assert isinstance(X, pd.DataFrame)
                for i, col in enumerate(X.columns):
                    assert X[col].dtype == (int if i in expected_cat_indices else float)
            elif issubclass(clf, LightGBMClassifierModel):
                X, y = intercepted_args["args"]
                assert isinstance(X, np.ndarray)
            else:
                assert False, f"{clf} need to be tested for fit arguments"


@pytest.fixture(autouse=True)
def random():
    np.random.seed(0)


class TestProbabilisticClassifierModels:
    np.random.seed(0)
    # shift sines to positive values so that they can be used as target for classification with classes [0, 1, 2]
    sine_univariate1 = tg.sine_timeseries(length=100) + 1
    sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705) + 1
    sine_univariate3 = tg.sine_timeseries(length=100, value_phase=0.78525) + 1
    sine_univariate4 = tg.sine_timeseries(length=100, value_phase=0.392625) + 1
    sine_univariate5 = tg.sine_timeseries(length=100, value_phase=0.1963125) + 1
    sine_univariate6 = tg.sine_timeseries(length=100, value_phase=0.09815625) + 1

    sine_univariate1_cat = sine_univariate1.map(lambda x: np.round(x))  # [0, 1, 2]
    sine_univariate2_cat = sine_univariate2.map(
        lambda x: np.where(np.round(x) >= 1, 1, 0)
    )  # [0, 1]
    sine_univariate3_cat = sine_univariate3.map(lambda x: np.round(x))

    sine_multivariate1_cat = sine_univariate1_cat.stack(sine_univariate2_cat)
    sine_multivariate2_cat = sine_univariate1_cat.stack(sine_univariate3_cat)
    sine_multivariate3_cat = sine_univariate2_cat.stack(sine_univariate3_cat)

    sine_multivariate1 = sine_univariate2.stack(sine_univariate3)
    sine_multivariate2 = sine_univariate1.stack(sine_univariate3)

    sine_multiseries1_cat = [
        sine_univariate1_cat,
        sine_univariate2_cat,
        sine_univariate3_cat,
    ]
    sine_multiseries1 = [sine_univariate4, sine_univariate5, sine_univariate6]

    sine_multiseries_multivariate_cat = [
        sine_multivariate1_cat,
        sine_multivariate2_cat,
        sine_multivariate3_cat,
    ]

    sine_multivariate_multiseries = [sine_multivariate1, sine_multivariate2]

    probabilistic_classifiers = [
        (LogisticRegression, {}),
        (KNeighborsClassifier, {"n_neighbors": 10}),
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

    rmse_class_proba = [
        0.07,  # LogisticRegression
        0.16,  # KNeighborsClassifier
        0.06,  # SVC
        0.1,  # GaussianProcessClassifier
        0.27,  # DecisionTreeClassifier
        0.11,  # RandomForestClassifier
        0.02,  # MLPClassifier
        0.20,  # AdaBoostClassifier
        0.07,  # GaussianNB
        0.17,  # XGBClassifierModel
    ]

    rmse_class_sample = [
        0.08,  # LogisticRegression
        0.16,  # KNeighborsClassifier
        0.09,  # SVC
        0.05,  # GaussianProcessClassifier
        0.15,  # DecisionTreeClassifier
        0.11,  # RandomForestClassifier
        0.11,  # MLPClassifier
        0.19,  # AdaBoostClassifier
        0.14,  # GaussianNB
        0.16,  # XGBClassifierModel
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
        rmse_class_proba.append(0.04)
        rmse_class_sample.append(0.01)

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
        rmse_class_proba.append(0.13)
        rmse_class_sample.append(0.12)

    @pytest.mark.parametrize(
        "clf_params",
        process_model_list(probabilistic_classifiers),
    )
    def test_wrong_likelihood(self, clf_params):
        clf, kwargs = clf_params
        with pytest.raises(ValueError) as exc:
            _ = clf(lags=1, likelihood="does_not_exist", **kwargs)
        assert (
            str(exc.value)
            == "Invalid `likelihood='does_not_exist'`. Must be one of ['classprobability']"
        )

        with pytest.raises(ValueError) as exc:
            _ = _get_likelihood(
                likelihood="does_not_exist",
                n_outputs=1,
                random_state=None,
                available_likelihoods=[LikelihoodType.ClassProbability],
            )
        assert (
            str(exc.value)
            == "Invalid `likelihood='does_not_exist'`. Must be one of ['classprobability']"
        )

    @pytest.mark.parametrize(
        "clf_params",
        process_model_list(probabilistic_classifiers),
    )
    def test_class_proba_likelihood_median_pred_is_same_than_no_likelihood(
        self, clf_params
    ):
        # check that the model's prediction is the same with and without the likelihood
        # when predict_likelihood_parameters=False
        # Meaning _get_median_prediction on top on predict_proba produce the same output than the model predict
        clf, kwargs = clf_params
        model = clf(lags=2, **kwargs, likelihood=None)
        model_likelihood = clf(
            lags=2,
            **kwargs,
        )

        model.fit(self.sine_univariate1_cat)
        # model has no likelihood
        with pytest.raises(ValueError) as err:
            model.predict(2, predict_likelihood_parameters=True)
        assert (
            str(err.value) == "`predict_likelihood_parameters=True` is only"
            " supported for probabilistic models fitted with a likelihood."
        )
        # univariate series
        model_likelihood.fit(self.sine_univariate1_cat)
        assert model_likelihood.predict(5) == model.predict(5)

        # multivariate series
        model = model.untrained_model()
        model_likelihood = model_likelihood.untrained_model()
        model.fit(self.sine_multivariate1_cat)
        model_likelihood.fit(self.sine_multivariate1_cat)
        assert model_likelihood.predict(5) == model.predict(5)

        # multiple univariate series
        model = model.untrained_model()
        model_likelihood = model_likelihood.untrained_model()
        model.fit(self.sine_multiseries1_cat)
        model_likelihood.fit(self.sine_multiseries1_cat)
        assert model_likelihood.predict(
            n=5, series=self.sine_multiseries1_cat
        ) == model.predict(n=5, series=self.sine_multiseries1_cat)

        # multiple multivariate series
        # 3 series two variates each
        model = model.untrained_model()
        model_likelihood = model_likelihood.untrained_model()
        model.fit(self.sine_multiseries_multivariate_cat)
        model_likelihood.fit(self.sine_multiseries_multivariate_cat)
        pred = model.predict(n=1, series=self.sine_multiseries_multivariate_cat)
        pred_likelihood = model_likelihood.predict(
            n=1, series=self.sine_multiseries_multivariate_cat
        )
        assert pred == pred_likelihood

    def class_probability_check_helper(
        self, model, series_test, true_probas, accepted_rmse
    ):
        if not isinstance(series_test, list):
            series_test = [series_test]

        avg_probas = np.array([
            predicted_ts.values()
            for predicted_ts in model.predict(
                n=1, series=series_test, predict_likelihood_parameters=True
            )
        ])
        for i in range(1, len(series_test[0]) - 1):
            probas = model.predict(
                n=1,
                series=[serie.split_after(i)[0] for serie in series_test],
                predict_likelihood_parameters=True,
            )
            avg_probas += np.array([predicted_ts.values() for predicted_ts in probas])

        avg_probas /= len(series_test[0]) - 1
        rmse = np.mean((avg_probas - true_probas) ** 2, axis=2) ** 0.5
        assert np.all(rmse <= accepted_rmse)

    @pytest.mark.parametrize(
        "clf_params",
        zip(process_model_list(probabilistic_classifiers), rmse_class_proba),
    )
    def test_univariate_class_probabilities_are_valid(self, clf_params):
        """Check class probabilties have correct shape and meaning"""

        (clf, kwargs), rmse_margin = clf_params
        model = clf(
            lags=2,
            **kwargs,
        )

        true_probas = np.array([0.1, 0.3, 0.6])
        true_labels = [0, 1, 2]
        df = pd.DataFrame({
            "random_train": np.random.choice(
                true_labels, size=100, replace=True, p=true_probas
            ),
            "random_test": np.random.choice(
                true_labels, size=100, replace=True, p=true_probas
            ),
        })
        series = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["random_train"]
        )
        model.fit(series)
        # model_likelihood has ClassProbability likelihood
        probas = model.predict(1, predict_likelihood_parameters=True)
        # Sum of class proba is 1
        assert probas.sum(axis=1).values()[0][0] == pytest.approx(1)
        # As many probabilties as classes
        assert len(probas.components) == 3
        # Class porbabilities have the correct ordering
        assert np.all(probas.values()[1:] > probas.values()[:-1])
        assert np.all(model.class_labels == true_labels)

        series_test = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["random_test"]
        )

        self.class_probability_check_helper(
            model=model,
            series_test=series_test,
            true_probas=true_probas,
            accepted_rmse=rmse_margin,
        )

    @pytest.mark.parametrize(
        "clf_params",
        zip(process_model_list(probabilistic_classifiers), rmse_class_proba),
    )
    def test_multi_series_class_probabilities_are_valid(self, clf_params):
        """Check class probabilties have correct shape and meaning"""

        (clf, kwargs), rmse_margin = clf_params
        model = clf(
            lags=2,
            **kwargs,
        )

        true_probas_component1 = np.array([0.1, 0.3, 0.6])
        true_probas_component2 = np.array([0.7, 0.3])

        true_labels_component1 = [0, 1, 2]
        true_labels_component2 = [0, 1]

        df = pd.DataFrame({
            name: np.random.choice(
                true_labels_comp,
                size=100,
                replace=True,
                p=true_probas_comp,
            )
            for name, true_labels_comp, true_probas_comp in zip(
                [
                    "component1_s1",
                    "component1_s1_test",
                    "component2_s1",
                    "component2_s1_test",
                    "component1_s2",
                    "component1_s2_test",
                    "component2_s2",
                    "component2_s2_test",
                ],
                [
                    true_labels_component1,
                    true_labels_component1,
                    true_labels_component2,
                    true_labels_component2,
                ]
                * 2,
                [
                    true_probas_component1,
                    true_probas_component1,
                    true_probas_component2,
                    true_probas_component2,
                ]
                * 2,
            )
        })

        series1 = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component1_s1", "component2_s1"]
        )
        series2 = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component1_s2", "component2_s2"]
        )
        model.fit([series1, series2])

        series1_test = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component1_s1_test", "component2_s1_test"]
        )
        series2_test = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component1_s2_test", "component2_s2_test"]
        )

        series_test = [series1_test, series2_test]

        # model_likelihood has ClassProbability likelihood
        list_of_probas = model.predict(
            n=1, series=series_test, predict_likelihood_parameters=True
        )
        # Sum of class proba is 1 (x2 component must be 2)
        assert np.all([
            p.sum(axis=1).values()[0][0] == pytest.approx(2) for p in list_of_probas
        ])

        # As many probabilties as classes
        assert np.all([len(p.components) == 5 for p in list_of_probas])
        assert np.all(model.class_labels[0] == true_labels_component1)
        assert np.all(model.class_labels[1] == true_labels_component2)

        true_probas = np.concatenate((true_probas_component1, true_probas_component2))

        self.class_probability_check_helper(
            model=model,
            series_test=series_test,
            true_probas=true_probas,
            accepted_rmse=rmse_margin,
        )

    @pytest.mark.parametrize(
        "clf_params",
        zip(process_model_list(probabilistic_classifiers), rmse_class_proba),
    )
    def test_multiivariate_class_probabilities_are_valid(self, clf_params):
        """Check class probabilties have correct shape and meaning"""

        (clf, kwargs), rmse_margin = clf_params
        model = clf(
            lags=2,
            **kwargs,
        )

        true_probas_component1 = np.array([0.1, 0.3, 0.6])
        true_probas_component2 = np.array([0.7, 0.3])

        true_labels_component1 = [0, 1, 2]
        true_labels_component2 = [0, 1]

        df = pd.DataFrame({
            "component_1": np.random.choice(
                true_labels_component1, size=100, replace=True, p=true_probas_component1
            ),
            "component_2": np.random.choice(
                true_labels_component2, size=100, replace=True, p=true_probas_component2
            ),
            "component_1_test": np.random.choice(
                true_labels_component1, size=100, replace=True, p=true_probas_component1
            ),
            "component_2_test": np.random.choice(
                true_labels_component2, size=100, replace=True, p=true_probas_component2
            ),
        })
        series = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component_1", "component_2"]
        )
        model.fit(series)
        # model_likelihood has ClassProbability likelihood
        probas = model.predict(1, predict_likelihood_parameters=True)
        # Sum of class proba is 1 (x2 component must be 2)
        assert probas.sum(axis=1).values()[0][0] == pytest.approx(2)

        # As many probabilties as classes
        assert len(probas.components) == 5
        # Class porbabilities have the correct ordering
        assert np.all(probas.values()[1:] > probas.values()[:-1])
        assert np.all(model.class_labels[0] == true_labels_component1)
        assert np.all(model.class_labels[1] == true_labels_component2)

        series_test = TimeSeries.from_dataframe(
            df, time_col=None, value_cols=["component_1_test", "component_2_test"]
        )
        true_probas = np.concatenate((true_probas_component1, true_probas_component2))

        self.class_probability_check_helper(
            model=model,
            series_test=series_test,
            true_probas=true_probas,
            accepted_rmse=rmse_margin,
        )

    def test_warning_on_no_predict_proba(self, caplog):
        class NoPredictProbaModel:
            def fit(*args):
                pass

            def predict(*args):
                pass

            def __sklearn_tags__(self):
                return Tags(estimator_type="classifier", target_tags=None)

        with caplog.at_level(logging.WARNING):
            SKLearnClassifierModel(model=NoPredictProbaModel(), lags=2)
            assert any([
                message.startswith(
                    "`model` has no method with name `predict_proba()`. "
                    "Probabilistic forecasting support not available."
                )
                for message in caplog.messages
            ])
        caplog.clear()

        with caplog.at_level(logging.WARNING):
            SKLearnClassifierModel(
                model=SVC(gamma=2, C=1, random_state=42, probability=True), lags=2
            )
            assert not any([
                message.startswith(
                    "`model` has no method with name `predict_proba()`. "
                )
                for message in caplog.messages
            ])

            # Without probability=True SVC has no "predict_proba" method
            SKLearnClassifierModel(model=SVC(gamma=2, C=1, random_state=42), lags=2)
            assert any(record.levelname == "WARNING" for record in caplog.records)
            assert any([
                message.startswith(
                    "`model` has no method with name `predict_proba()`. "
                    "Set `probability=True` at `SVC` model creation "
                )
                for message in caplog.messages
            ])

    def test_class_probability_component_names(self):
        component_names = [
            "sine_p_0",
            "sine_p_1",
            "sine_p_2",
            "sine_1_p_0",
            "sine_1_p_1",
        ]

        model = SKLearnClassifierModel(lags=1, output_chunk_length=2)

        # component_names before fit throws an error
        with pytest.raises(ValueError) as err:
            model.likelihood.component_names(self.sine_multivariate1_cat)
        assert (
            str(err.value) == "`component_names` requires the likelihood to be fitted "
            "but `ClassProbabilityLikelihood` is not fitted."
        )

        # once fitted, component_names are correct
        model.fit(self.sine_multivariate1_cat)
        model.likelihood.component_names(self.sine_multivariate1_cat) == component_names

        # predicted component names are correct
        preds = model.predict(n=2, predict_likelihood_parameters=True)

        assert np.all(preds.components == component_names)

    @pytest.mark.parametrize(
        "clf_params",
        zip(process_model_list(probabilistic_classifiers), rmse_class_sample),
    )
    def test_multi_sample_with_class_probabilities(
        self,
        clf_params,
    ):
        (clf, kwargs), model_rmse = clf_params
        model = clf(lags=2, **kwargs)

        true_probas = np.array([0.1, 0.3, 0.6])
        true_labels = [0, 1, 2]
        series = TimeSeries.from_values(
            np.random.choice(true_labels, size=100, replace=True, p=true_probas),
        )

        model.fit(series)
        prediction = model.predict(n=1, num_samples=100)
        count = np.zeros(len(model.class_labels))
        preds = prediction.all_values().flatten()
        for i in preds:
            count[int(i)] += 1
        count /= len(preds)
        rmse = np.mean((count - true_probas) ** 2) ** 0.5
        assert rmse < model_rmse
