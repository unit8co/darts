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
from darts.models import (
    CatBoostClassifierModel,
    LightGBMClassifierModel,
    SKLearnClassifierModel,
    XGBClassifierModel,
)
from darts.models.forecasting.sklearn_model import SKLearnModelWithCategoricalFeatures
from darts.tests.conftest import CB_AVAILABLE, LGBM_AVAILABLE, XGB_AVAILABLE
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import (
    ClassProbabilityLikelihood,
    _get_likelihood,
)
from darts.utils.multioutput import (
    MultiOutputClassifier,
    MultiOutputRegressor,
    get_multioutput_estimator_cls,
)
from darts.utils.utils import ModelType

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
    ]

    if XGB_AVAILABLE:
        classifiers.append((
            XGBClassifierModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "max_leaves": 1,
                "random_state": 42,
            },
        ))
        models_accuracies.append(1)
        models_multioutput.append(False)

    if LGBM_AVAILABLE:
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

    if CB_AVAILABLE:
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
        assert model.model is not None
        assert isinstance(model.likelihood, ClassProbabilityLikelihood)

        # accepts only classifier
        with pytest.raises(ValueError) as err:
            SKLearnClassifierModel(model=LinearRegression(), lags=1)
        assert (
            str(err.value)
            == "`SKLearnClassifierModel` must be initialized with a classifier `model`."
        )

    @pytest.mark.parametrize("clf_params", process_model_list(classifiers))
    def test_univariate_class_labels(self, clf_params):
        clf, kwargs = clf_params
        model = clf(lags_past_covariates=5, **kwargs)

        # accessing classes_ before training return None
        assert model.class_labels is None

        # training the model
        model.fit(
            series=self.sine_univariate1_cat, past_covariates=self.sine_univariate1
        )
        # `class_labels` is a list of component-specific numpy array (univariate = length 1)
        assert isinstance(model.class_labels, list)
        assert len(model.class_labels) == 1
        assert isinstance(model.class_labels[0], np.ndarray)
        assert set(np.unique(self.sine_univariate1_cat.values())) == {0, 1, 2}
        assert (model.class_labels[0] == [0, 1, 2]).all()

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGB required")
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
            classes = {0, 1, 2}
        else:
            multivariate_cat_diff_labels = self.sine_univariate1_cat.stack(
                self.sine_univariate1_cat + 3
            )
            expected_classes = [np.array([0, 1, 2]), np.array([3, 4, 5])]
            classes = {0, 1, 2, 3, 4, 5}

        assert set(np.unique(multivariate_cat_diff_labels.values())) == classes
        model.fit(
            series=multivariate_cat_diff_labels, past_covariates=self.sine_univariate1
        )
        # `class_labels` is a list of component-specific numpy array (multivariate = length 2)
        # for both MultiOutputClassifier and native multi-class classifiers
        assert isinstance(model.class_labels, list)
        assert len(model.class_labels) == len(expected_classes)
        assert isinstance(model.class_labels[0], np.ndarray)
        # check that classes correspond to the classes in the series
        assert np.array([
            (c1 == c2).all() for c1, c2 in zip(model.class_labels, expected_classes)
        ]).all()

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGB required")
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
        series = TimeSeries.from_values(np.array([0, 0, 0, 1, 2]), columns=["comp1"])
        model = clf(lags=1, output_chunk_length=2, **kwargs)

        with pytest.raises(ValueError) as err:
            model.fit(series=series)
        assert str(err.value).startswith(
            "Estimators for the same target component received different class labels "
            "during training. In most cases this occurs for shorter target series. "
        )

        # if only one estimator due to multi_models=False, then there are no issue
        single_model = clf(lags=1, output_chunk_length=2, multi_models=False, **kwargs)
        single_model.fit(series=series)

        # if same labels per component it does not raise an error
        series = TimeSeries.from_values(np.array([0, 0, 0, 1, 1]), columns=["comp1"])
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
            pred.static_covariates.equals(series.static_covariates)

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
        multi_series = not isinstance(series, TimeSeries)
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
            n=len(test_series) if not multi_series else len(test_series[0]),
            series=train_series,
            past_covariates=past_covariates,
        )

        # flatten in case of multivariate prediction
        if multi_series:
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
        """Check that with output_chunk_length=1, wrapping in MultiOutputClassifier occurs only when necessary"""
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
        # multivariate should be wrapped in MultiOutputRegressor only if not natively supported
        model.fit(
            series=self.sine_multivariate1_cat, past_covariates=self.sine_multivariate1
        )
        check_only_non_native_are_wrapped(model, supports_multioutput_natively)

        model = model.untrained_model()
        # multi-series with same component should not be wrapped in MultiOutputRegressor
        model.fit(
            series=self.sine_multiseries1_cat, past_covariates=self.sine_multiseries1
        )
        assert not isinstance(model.model, MultiOutputClassifier)

        model = model.untrained_model()
        # multi-series with multivariate should be wrapped in MultiOutputRegressor only if not natively supported
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
        with pytest.raises(ValueError) as err:
            # testing lags_past_covariates None but past_covariates during training
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
                future_covariates=self.sine_multivariate1,
            )
        assert str(err.value).startswith(
            "`past_covariates` not None in `fit()` method call"
        )

        model_instance = model_cls(
            lags_past_covariates=3, multi_models=multi_models, **kwargs
        )
        with pytest.raises(ValueError) as err:
            # testing lags_past_covariates but no past_covariates during fit
            model_instance.fit(series=self.sine_univariate1_cat)
        assert str(err.value).startswith(
            "`past_covariates` is None in `fit()` method call"
        )

        # testing future_covariates
        model_instance = model_cls(
            lags_past_covariates=4,
            lags_future_covariates=None,
            multi_models=multi_models,
            **kwargs,
        )
        with pytest.raises(ValueError) as err:
            # testing lags_future_covariates None but future_covariates during training
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
                future_covariates=self.sine_multivariate1,
            )
        assert str(err.value).startswith(
            "`future_covariates` not None in `fit()` method call"
        )

        model_instance = model_cls(
            lags_past_covariates=4,
            lags_future_covariates=(0, 3),
            multi_models=multi_models,
            **kwargs,
        )
        with pytest.raises(ValueError) as err:
            # testing lags_future_covariates but no future_covariates during fit
            model_instance.fit(
                series=self.sine_univariate1_cat,
                past_covariates=self.sine_multivariate1,
            )
        assert str(err.value).startswith(
            "`future_covariates` is None in `fit()` method call"
        )

        # testing input_dim
        model_instance = model_cls(
            lags_past_covariates=2, multi_models=multi_models, **kwargs
        )
        model_instance.fit(
            series=train_y,
            past_covariates=self.sine_univariate1.stack(self.sine_univariate1)[
                : train_y.end_time()
            ],
        )

        assert model_instance.input_dim == {
            "target": 1,
            "past": 2,
            "future": None,
        }

        with pytest.raises(ValueError) as err:
            _ = model_instance.predict(n=model_instance.output_chunk_length + 1)
        assert str(err.value).startswith("The `past_covariates` are not long enough.")

        # while it should work with n = output_chunk_length
        prediction = model_instance.predict(n=model_instance.output_chunk_length)
        assert len(prediction) == 1

        # test wrong covariates dimensionality
        with pytest.raises(ValueError) as err:
            _ = prediction = model_instance.predict(
                n=model_instance.output_chunk_length,
                past_covariates=self.sine_univariate1,
            )
        assert str(err.value).startswith(
            "The number of components of the target series and the covariates"
        )

    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGB required")
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
        if issubclass(clf, (SVC, GaussianProcessClassifier)) or isinstance(
            model.model, LogisticRegression
        ):
            # Model specific error message
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
                assert not caplog.messages
            else:
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

    def test_get_multioutput_estimator_cls(self):
        assert issubclass(
            get_multioutput_estimator_cls(ModelType.FORECASTING_CLASSIFIER),
            MultiOutputClassifier,
        )
        assert issubclass(
            get_multioutput_estimator_cls(ModelType.FORECASTING_REGRESSOR),
            MultiOutputRegressor,
        )

        with pytest.raises(ValueError) as err:
            get_multioutput_estimator_cls(model_type="invalid_type")
        assert (
            str(err.value)
            == "Model type must be one of `[ModelType.FORECASTING_REGRESSOR, ModelType.FORECASTING_CLASSIFIER]`. "
            "Received: `invalid_type`."
        )


@pytest.fixture(autouse=True)
def random():
    np.random.seed(0)


def generate_random_series_with_probabilities(
    multi_variate: bool, multi_series: bool, length=100
):
    """
    Generate categorical series with specific probabilities for each class
    """
    probas_component1 = np.array([0.1, 0.3, 0.6])
    labels_component1 = [0, 1, 2]

    probas_component2 = np.array([0.7, 0.3]) if multi_variate else None
    labels_component2 = [0, 1] if multi_variate else None

    columns = ["component1", "component2"] if multi_variate else ["component1"]

    def generate_samples():
        array = np.random.choice(
            labels_component1, size=length * 2, replace=True, p=probas_component1
        )

        if multi_variate:
            array = np.vstack((
                array,
                np.random.choice(
                    labels_component2,
                    size=length * 2,
                    replace=True,
                    p=probas_component2,
                ),
            )).T
        return array

    series = TimeSeries.from_values(generate_samples(), columns=columns)
    series_train, series_test = series.split_before(length)

    if multi_series:
        series2 = TimeSeries.from_values(generate_samples(), columns=columns)
        series2_train, series2_test = series2.split_before(length)
        series = [series, series2]
        series_train = [series_train, series2_train]
        series_test = [series_test, series2_test]

    return (
        series,
        series_train,
        series_test,
        probas_component1,
        labels_component1,
        probas_component2,
        labels_component2,
    )


class TestProbabilisticClassifierModels:
    probabilistic_classifiers = [
        (LogisticRegression, {}),
        (KNeighborsClassifier, {"n_neighbors": 10}),
        (GaussianProcessClassifier, {"kernel": 1.0 * RBF(1.0), "random_state": 42}),
        (DecisionTreeClassifier, {"max_depth": 5, "random_state": 42}),
        (
            RandomForestClassifier,
            {"max_depth": 5, "n_estimators": 10, "max_features": 1, "random_state": 42},
        ),
        (
            MLPClassifier,
            {
                # "alpha": 1,
                # "max_iter": 1000,
                "random_state": 42
            },
        ),
        (AdaBoostClassifier, {"random_state": 42}),
        (GaussianNB, {}),
    ]

    rmse_class_proba = [
        0.07,  # LogisticRegression
        0.16,  # KNeighborsClassifier
        0.1,  # GaussianProcessClassifier
        0.27,  # DecisionTreeClassifier
        0.11,  # RandomForestClassifier
        0.05,  # MLPClassifier
        0.20,  # AdaBoostClassifier
        0.07,  # GaussianNB
    ]

    rmse_class_sample = [
        0.08,  # LogisticRegression
        0.16,  # KNeighborsClassifier
        0.16,  # GaussianProcessClassifier
        0.15,  # DecisionTreeClassifier
        0.17,  # RandomForestClassifier
        0.11,  # MLPClassifier
        0.2,  # AdaBoostClassifier
        0.14,  # GaussianNB
    ]

    if XGB_AVAILABLE:
        probabilistic_classifiers.append((
            XGBClassifierModel,
            {
                "n_estimators": 1,
                "max_depth": 1,
                "max_leaves": 1,
                "random_state": 42,
            },
        ))
        rmse_class_proba.append(0.17)
        rmse_class_sample.append(0.18)

    if LGBM_AVAILABLE:
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
        rmse_class_proba.append(0.06)
        rmse_class_sample.append(0.06)

    if CB_AVAILABLE:
        probabilistic_classifiers.append((
            CatBoostClassifierModel,
            {
                "iterations": 1,
                "depth": 1,
                "verbose": -1,
                "random_state": 42,
            },
        ))
        rmse_class_proba.append(0.15)
        rmse_class_sample.append(0.15)

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
                available_likelihoods=[LikelihoodType.ClassProbability],
            )
        assert (
            str(exc.value)
            == "Invalid `likelihood='does_not_exist'`. Must be one of ['classprobability']"
        )

    @pytest.mark.parametrize(
        "clf_params",
        product(
            process_model_list(probabilistic_classifiers), [False, True], [False, True]
        ),
    )
    def test_class_proba_likelihood_median_pred_is_same_than_no_likelihood(
        self, clf_params
    ):
        # check that the model's prediction is the same with and without the likelihood
        # when predict_likelihood_parameters=False
        # Meaning _get_median_prediction on top of predict_proba produce the same output than the model predict
        (clf, kwargs), multi_series, multi_variate = clf_params

        _, series_train, _, _, labels_1, _, labels_2 = (
            generate_random_series_with_probabilities(
                multi_series=multi_series, multi_variate=multi_variate
            )
        )
        ocl = 5
        model = clf(lags=2, output_chunk_length=ocl, likelihood=None, **kwargs)
        assert model.likelihood is None
        model_likelihood = clf(lags=2, output_chunk_length=ocl, **kwargs)
        assert isinstance(model_likelihood.likelihood, ClassProbabilityLikelihood)

        model.fit(series_train)
        model_likelihood.fit(series_train)

        # native class prediction with maximum probability
        pred = model.predict(n=ocl, series=series_train)
        # likelihood class prediction with maximum probability
        pred_likelihood = model_likelihood.predict(
            n=ocl,
            series=series_train,
            predict_likelihood_parameters=False,
        )
        # likelihood class probability prediction
        pred_proba = model_likelihood.predict(
            n=ocl,
            series=series_train,
            predict_likelihood_parameters=True,
        )
        if not multi_series:
            pred = [pred]
            pred_likelihood = [pred_likelihood]
            pred_proba = [pred_proba]

        # compute class prediction with maximum probability
        pred_max_proba = []
        for idx, p in enumerate(pred_proba):
            p = p.values()
            # compute labels for each component
            p_1 = np.argmax(p[:, : len(labels_1)], axis=1, keepdims=True)
            if multi_variate:
                p_2 = np.argmax(p[:, len(labels_1) :], axis=1, keepdims=True)
                p_1 = np.concatenate([p_1, p_2], axis=1)
            pred_max_proba.append(pred[idx].with_values(p_1))

        # all predictions are the same
        assert pred_likelihood == pred
        assert pred_max_proba == pred

        with pytest.raises(ValueError) as err:
            model.predict(
                n=ocl + 1, series=series_train, predict_likelihood_parameters=True
            )
        assert (
            str(err.value) == "`predict_likelihood_parameters=True` is only"
            " supported for probabilistic models fitted with a likelihood."
        )

    @pytest.mark.parametrize(
        "clf_params",
        product(
            zip(process_model_list(probabilistic_classifiers), rmse_class_proba),
            [True, False],
            [True, False],
        ),
    )
    def test_class_probabilities_are_valid(self, clf_params):
        """
        Check class probabilities have correct shape and meaning in case of:
        - single series, univariate
        - single series, multivariate
        - multi series, univariate
        - multi series, multivariate
        Components of multi-variate do not have the same classes
        """

        ((clf, kwargs), rmse_margin), multi_series, multi_variate = clf_params
        model = clf(
            lags=2,
            **kwargs,
        )

        (
            series,
            series_train,
            series_test,
            probas_component1,
            labels_component1,
            probas_component2,
            labels_component2,
        ) = generate_random_series_with_probabilities(
            multi_series=multi_series, multi_variate=multi_variate
        )
        model.fit(series_train)

        # model_likelihood has ClassProbability likelihood
        list_of_probas = model.predict(
            n=1, series=series_test, predict_likelihood_parameters=True
        )

        list_of_probas2 = (
            model.untrained_model()
            .fit(series_train)
            .predict(n=1, series=series_test, predict_likelihood_parameters=True)
        )

        if not multi_series:
            series = [series]
            series_test = [series_test]
            list_of_probas = [list_of_probas]
            list_of_probas2 = [list_of_probas2]

        # Probability are reproducible
        for s, p1, p2 in zip(series, list_of_probas, list_of_probas2):
            vals1 = p1.all_values()
            vals2 = p2.all_values()
            np.testing.assert_allclose(vals1, vals2)

            # As many probability components as classes
            n_classes = len(labels_component1) + (
                len(labels_component2) if multi_variate else 0
            )
            assert p1.n_components == n_classes

            # Sum of class proba is 1 per component (x2 component must be 2)
            assert p1.values().sum() == pytest.approx(s.n_components)

            # component-specific probabilities sum to 1
            assert vals1[:, : len(labels_component1)].sum() == pytest.approx(1.0)
            if multi_variate:
                assert vals1[:, len(labels_component1) :].sum() == pytest.approx(1.0)

            # all labels are present in the class_labels
            assert np.all(model.class_labels[0] == labels_component1)
            if multi_variate:
                assert np.all(model.class_labels[1] == labels_component2)

        # verify that historical_forecasts returns on average approximately the same probabilities
        true_probas = (
            np.concatenate((probas_component1, probas_component2))
            if multi_variate
            else probas_component1
        )

        pred_probas = model.historical_forecasts(
            series=series,
            forecast_horizon=1,
            predict_likelihood_parameters=True,
            retrain=False,
            start=series_test[0].start_time() + 2 * series_test[0].freq,
            overlap_end=True,
        )
        avg_probas = np.array([pred.values().mean(axis=0) for pred in pred_probas])
        rmse = np.mean((avg_probas - true_probas) ** 2, axis=1) ** 0.5
        assert np.all(rmse <= rmse_margin)

    def test_bad_behavior_model_properties(self, caplog):
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
                    "Probabilistic forecasting support deactivated."
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
                message.startswith("`model` has no method with name `predict_proba()`.")
                for message in caplog.messages
            ])

        class NoClassLabelsModel:
            def fit(*args):
                pass

            def predict(*args):
                pass

            def predict_proba(*args):
                pass

            def __sklearn_tags__(self):
                return Tags(estimator_type="classifier", target_tags=None)

        model = NoClassLabelsModel()
        likelihood = ClassProbabilityLikelihood(n_outputs=1)
        with pytest.raises(ValueError) as err:
            likelihood.fit(model)
        assert (
            str(err.value)
            == "The model must have a `class_labels` attribute to fit the likelihood."
        )

        class NotSummingToOneProbasModel:
            def __init__(self, gap_to_one: int):
                self.gap_to_one = gap_to_one

            def fit(*args):
                pass

            def predict(*args):
                pass

            def predict_proba(self, *args):
                return np.array([[0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]) - (
                    self.gap_to_one / 3
                )

            @property
            def class_labels(self):
                return [np.array([0, 1, 2])]

            def __sklearn_tags__(self):
                return Tags(estimator_type="classifier", target_tags=None)

        tolerance = 0.0000001

        # difference is smaller than tolerance, probability will be adjusted to sum up to one
        small_diff_model = NotSummingToOneProbasModel(tolerance / 2)
        small_diff_model.model = small_diff_model
        likelihood.fit(small_diff_model)
        likelihood.predict(
            small_diff_model,
            np.array([0]),
            num_samples=2,
            predict_likelihood_parameters=False,
        )

        # difference is larger than tolerance, error is raised
        large_diff_model = NotSummingToOneProbasModel(tolerance * 2)
        large_diff_model.model = large_diff_model
        likelihood.fit(large_diff_model)
        with pytest.raises(ValueError) as err:
            likelihood.predict(
                large_diff_model,
                np.array([0]),
                num_samples=2,
                predict_likelihood_parameters=False,
            )
        assert (
            str(err.value)
            == "The class probabilities returned by the model do not sum to one"
        )

    def test_class_probability_component_names(self):
        _, series_train, _, _, _, _, _ = generate_random_series_with_probabilities(
            multi_series=False, multi_variate=True
        )

        component_names = [
            "component1_p0",
            "component1_p1",
            "component1_p2",
            "component2_p0",
            "component2_p1",
        ]

        model = SKLearnClassifierModel(lags=1, output_chunk_length=2)

        # component_names before fit throws an error
        with pytest.raises(ValueError) as err:
            model.likelihood.component_names(series_train)
        assert str(err.value) == "The likelihood has not been fitted yet."

        # once fitted, component_names are correct
        model.fit(series_train)
        assert model.likelihood.component_names(series=series_train) == component_names
        assert (
            model.likelihood.component_names(components=series_train.components)
            == component_names
        )

        # predicted component names are correct
        preds = model.predict(n=2, predict_likelihood_parameters=True)

        assert np.all(preds.components == component_names)

        with pytest.raises(ValueError) as err:
            _ = model.likelihood.component_names(
                series=series_train, components=series_train.components
            )
        assert (
            str(err.value) == "Only one of `series` or `components` must be specified."
        )

    @pytest.mark.parametrize(
        "clf_params",
        product(
            zip(process_model_list(probabilistic_classifiers), rmse_class_sample),
            [False, True],
        ),
    )
    def test_multi_sample_with_class_probabilities(self, clf_params):
        """
        The distribution of samples corresponds to the distribution of labels in the TS
        """
        ((clf, kwargs), model_rmse), multi_variate = clf_params
        model = clf(lags=2, **kwargs)

        (
            _,
            series_train,
            _,
            probas_component1,
            labels_component1,
            probas_component2,
            labels_component2,
        ) = generate_random_series_with_probabilities(
            multi_series=False, multi_variate=multi_variate
        )

        model.fit(series_train)

        # predict_likelihood_parameters is not supported with multiple samples
        with pytest.raises(ValueError) as err:
            model.predict(n=1, num_samples=2, predict_likelihood_parameters=True)
        assert (
            str(err.value)
            == "`predict_likelihood_parameters=True` is only supported for `num_samples=1`, received 2."
        )

        # predict with multiple samples and verify the label probability
        prediction = model.predict(n=5, num_samples=1000)
        preds_c = [prediction.all_values()[:, 0]]
        probas_c = [probas_component1]
        labels_c = [labels_component1]
        if multi_variate:
            preds_c.append(prediction.all_values()[:, 1])
            probas_c.append(probas_component2)
            labels_c.append(labels_component2)

        for idx, (preds, probas, labels) in enumerate(zip(preds_c, probas_c, labels_c)):
            pred_probas = pd.value_counts(preds.flatten(), normalize=True).to_dict()
            pred_probas = [pred_probas[label] for label in labels]
            rmse = np.mean((pred_probas - probas) ** 2) ** 0.5
            assert rmse < model_rmse

        # reproducible samples when call order is the same (and also with transferable series)
        model = model.untrained_model().fit(series_train)
        preds_2 = model.predict(n=5, num_samples=1000, series=series_train)
        assert preds_2 == prediction

        # different samples when call order changes
        preds_3 = model.predict(n=5, num_samples=1000)
        assert preds_3 != prediction

        # same samples when random_state is set
        preds_4 = model.predict(
            n=5, num_samples=1000, series=series_train, random_state=42
        )
        preds_5 = model.predict(
            n=5, num_samples=1000, series=series_train, random_state=42
        )
        assert preds_4 == preds_5

    @pytest.mark.parametrize(
        "params",
        product(
            [True, False],  # multi_variate
            [True, False],  # multi_model
        ),
    )
    def test_historical_forecast(self, params):
        multi_model, multi_variate = params

        _, series_train, _, _, _, _, _ = generate_random_series_with_probabilities(
            multi_series=False, multi_variate=multi_variate
        )

        _, series_multi_variate, _, _, _, _, _ = (
            generate_random_series_with_probabilities(
                multi_series=False, multi_variate=True
            )
        )

        # without covariate
        model = SKLearnClassifierModel(
            model=LogisticRegression(), lags=4, multi_models=multi_model
        )
        result = model.historical_forecasts(
            series=series_train,
            future_covariates=None,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

        # With past covariate
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=5,
            lags_past_covariates=5,
            multi_models=multi_model,
        )
        result = model.historical_forecasts(
            series=series_train,
            past_covariates=series_multi_variate,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

        # with past covariate and output_chunk_length
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=5,
            lags_past_covariates=5,
            output_chunk_length=5,
            multi_models=multi_model,
        )
        result = model.historical_forecasts(
            series=series_train,
            past_covariates=series_multi_variate,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

        # forecast_horizon > output_chunk_length
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=5,
            lags_past_covariates=5,
            output_chunk_length=1,
            multi_models=multi_model,
        )
        result = model.historical_forecasts(
            series=series_train,
            past_covariates=series_multi_variate,
            start=0.8,
            forecast_horizon=3,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 19

        # Likelihood parameters
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=5,
            lags_past_covariates=5,
            multi_models=multi_model,
        )
        result = model.historical_forecasts(
            series=series_train,
            past_covariates=series_multi_variate,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
            predict_likelihood_parameters=True,
        )
        assert len(result) == 21

        # Sampled prediction
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=5,
            lags_past_covariates=5,
            multi_models=multi_model,
        )
        result = model.historical_forecasts(
            series=series_train,
            past_covariates=series_multi_variate,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
            num_samples=100,
        )
        assert len(result) == 21
