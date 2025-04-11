import logging
from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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

import darts
from darts.logging import get_logger
from darts.models.forecasting.catboost_model import CatBoostClassifierModel
from darts.models.forecasting.classifier_model import SklearnClassifierModel
from darts.models.forecasting.lgbm import LightGBMClassifierModel
from darts.models.forecasting.xgboost import XGBClassifierModel
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.multioutput import MultiOutputClassifier
from darts.utils.utils import NotImportedModule

lgbm_available = not isinstance(LightGBMClassifierModel, NotImportedModule)
cb_available = not isinstance(CatBoostClassifierModel, NotImportedModule)

logger = get_logger(__name__)


def process_model_list(classifiers):
    for clf, kwargs in classifiers:
        if issubclass(clf, BaseEstimator):
            yield (SklearnClassifierModel, {"model": clf(**kwargs)})
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
    sine_multivariate2 = sine_univariate2.stack(sine_univariate3)

    sine_multiseries1_cat = [
        sine_univariate1_cat,
        sine_univariate2_cat,
        sine_univariate3_cat,
    ]
    sine_multiseries2 = [sine_univariate4, sine_univariate5, sine_univariate6]

    classifiers = [
        (LogisticRegression, {}),
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
        1,  # SklearnClassifierModel
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

    models_multioutput = [
        False,  # SklearnClassifierModel
        True,  # KNeighborsClassifier
        False,  # SVC
        False,  # GaussianProcessClassifier
        True,  # DecisionTreeClassifier
        True,  # RandomForestClassifier
        False,  # MLPClassifier
        False,  # AdaBoostClassifier
        False,  # GaussianNB
        False,  # QuadraticDiscriminantAnalysis
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
        with pytest.raises(ValueError):
            SklearnClassifierModel(model=LinearRegression())

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_classes_labels(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)

        # accessing classes_ before training raises an error
        with pytest.raises(AttributeError):
            model.classes_

        # training the model
        model.fit(
            series=self.sine_univariate1_cat, past_covariates=self.sine_univariate1
        )
        assert [0, 1, 2] == list(model.classes_)

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
            self.sine_multivariate2,
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
            self.sine_multiseries2,
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
        (model_cls, kwargs), support_multioutput = model_params
        model = model_cls(
            lags_past_covariates=1,
            **kwargs,
        )
        model.fit(
            series=self.sine_multivariate1_cat, past_covariates=self.sine_multivariate2
        )
        if support_multioutput:
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
            assert model.get_estimator(horizon=0, target_dim=0) != model.get_estimator(
                horizon=0, target_dim=1
            )

    @pytest.mark.parametrize(
        "config", product(process_model_list(classifiers), [True, False])
    )
    def test_models_runnability(self, config):
        (model_cls, kwargs), multi_models = config

        train_y, test_y = self.sine_univariate1_cat.split_before(0.7)

        # TODO settings lags should either raise an error or a warning

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
                past_covariates=self.sine_multivariate2,
                future_covariates=self.sine_multivariate2,
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
                past_covariates=self.sine_multivariate2,
                future_covariates=self.sine_multivariate2,
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
                past_covariates=self.sine_multivariate2,
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
        with pytest.raises(ValueError):
            model.fit(
                series=self.sine_univariate1, past_covariates=self.sine_univariate1
            )

        # XGBClassifierModel require labels to be integers between 0 and n_classes
        if type(clf) is XGBClassifierModel:
            # negative labels
            with pytest.raises(ValueError):
                model.fit(
                    series=self.sine_univariate1_cat - 5,
                    past_covariates=self.sine_univariate1,
                )

            # labels not between 0 and n_classes
            with pytest.raises(ValueError):
                model.fit(
                    series=self.sine_univariate1_cat + 1,
                    past_covariates=self.sine_univariate1,
                )

        # Single label
        if type(clf) in [
            SVC,
            GaussianProcessClassifier,
            QuadraticDiscriminantAnalysis,
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
            if model._is_target_categorical:
                assert not any(
                    record.levelname == "WARNING" for record in caplog.records
                )
            else:
                assert any(record.levelname == "WARNING" for record in caplog.records)
                assert any([
                    message.startswith(
                        "This model will treat the target `series` data/label as a numerical feature"
                    )
                    for message in caplog.messages
                ])

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_categorical_target_passed_to_fit_correctly(self, clf_params):
        expected_cat_index = [0, 1]

        clf, kwargs = clf_params
        model = clf(lags=2, lags_past_covariates=2, **kwargs)

        intercepted_args = {}
        original_fit = model.model.fit

        def intercept_fit_args(*args, **kwargs):
            intercepted_args["args"] = args
            intercepted_args["kwargs"] = kwargs
            return original_fit(*args, **kwargs)

        if model._is_target_categorical:
            if clf == CatBoostClassifierModel:
                with patch.object(
                    darts.models.forecasting.catboost_model.CatBoostClassifier,
                    "fit",
                    side_effect=intercept_fit_args,
                ):
                    model.fit(self.sine_univariate1_cat, self.sine_univariate2)

                    model_cat_indices = model.model.get_cat_feature_indices()
                    kwargs_cat_indices = intercepted_args["kwargs"]["cat_features"]

                    assert (
                        len(model_cat_indices)
                        == len(kwargs_cat_indices)
                        == len(expected_cat_index)
                    )

                    for mci, kci, eci in zip(
                        model_cat_indices, kwargs_cat_indices, expected_cat_index
                    ):
                        assert mci == kci == eci

                    X, y = intercepted_args["args"]
                    # all categorical features should be encoded as integers
                    for col in X[model_cat_indices].columns:
                        assert X[col].dtype == int
            elif clf == LightGBMClassifierModel:
                with patch.object(
                    darts.models.forecasting.lgbm.lgb.LGBMClassifier,
                    "fit",
                    side_effect=intercept_fit_args,
                ):
                    model.fit(self.sine_univariate1_cat, self.sine_univariate2)

                    cat_param_name = model._categorical_fit_param
                    assert (
                        intercepted_args["kwargs"][cat_param_name] == expected_cat_index
                    )
            else:
                assert False, f"{clf} need to be tested for fit arguments"
