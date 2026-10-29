import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

from darts.utils.data.tabularization.features import (
    ArrayLaggedFeatures,
    StepwiseLaggedFeatures,
)
from darts.utils.multioutput import MultiOutputClassifier, MultiOutputRegressor
from darts.utils.output_chunk import (
    DirectOutputStrategy,
    RecursiveOutputStrategy,
)


class RecordingRegressor(BaseEstimator, RegressorMixin):
    fit_records = []
    predict_records = []

    def __init__(self):
        pass

    @classmethod
    def reset(cls):
        cls.fit_records = []
        cls.predict_records = []

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_samples=None,
        eval_labels=None,
        eval_weight=None,
    ):
        X = np.asarray(X)
        y = np.asarray(y)
        type(self).fit_records.append({
            "X": X.copy(),
            "y": y.copy(),
            "sample_weight": None
            if sample_weight is None
            else np.asarray(sample_weight).copy(),
            "eval_set": eval_set,
            "eval_samples": eval_samples,
            "eval_labels": eval_labels,
            "eval_weight": eval_weight,
        })
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.fit_sample_weight_ = (
            None if sample_weight is None else np.asarray(sample_weight).copy()
        )
        self.fit_eval_set_ = eval_set
        self.fit_eval_samples_ = eval_samples
        self.fit_eval_labels_ = eval_labels
        self.fit_eval_weight_ = eval_weight
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        type(self).predict_records.append(X.copy())
        return X[:, 0]


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    fit_records = []
    predict_proba_records = []

    def __init__(self):
        pass

    @classmethod
    def reset(cls):
        cls.fit_records = []
        cls.predict_proba_records = []

    def fit(self, X, y, sample_weight=None, **fit_params):
        X = np.asarray(X)
        y = np.asarray(y)
        type(self).fit_records.append({"X": X.copy(), "y": y.copy()})
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X)
        type(self).predict_proba_records.append(X.copy())
        return np.tile([0.5, 0.5], (len(X), 1))


def make_stepwise(n_rows: int = 6):
    array = np.column_stack([
        np.arange(n_rows, dtype=float),
        np.arange(n_rows, dtype=float) + 10.0,
    ])
    step_cols = [0]
    step_values = np.stack([
        array[:, step_cols],
        array[:, step_cols] + 100.0,
    ])
    return (
        StepwiseLaggedFeatures(
            array=array,
            step_values=step_values,
            step_cols=step_cols,
        ),
        array,
        step_values,
    )


def test_direct_strategy_routes_stepwise_horizons():
    X, base, step_values = make_stepwise()
    y = np.column_stack([np.arange(6), np.arange(6) + 100])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y)

    assert len(wrapper.estimators_) == 2
    assert wrapper.estimators_[0].fit_X_[:, 0].tolist() == base[:, 0].tolist()
    assert wrapper.estimators_[1].fit_X_[:, 0].tolist() == step_values[1, :, 0].tolist()
    assert np.array_equal(wrapper.estimators_[0].fit_y_, y[:, 0])
    assert np.array_equal(wrapper.estimators_[1].fit_y_, y[:, 1])

    RecordingRegressor.reset()
    predictions = wrapper.predict(X)

    assert predictions.shape == (6, 2)
    assert len(RecordingRegressor.predict_records) == 2
    assert RecordingRegressor.predict_records[0][:, 0].tolist() == base[:, 0].tolist()
    assert (
        RecordingRegressor.predict_records[1][:, 0].tolist()
        == step_values[1, :, 0].tolist()
    )


def test_features_for_output_use_zero_copy_base_arrays():
    base = np.arange(12, dtype=float).reshape(4, 3)
    X = ArrayLaggedFeatures.wrap(base, n_horizons=2)
    strategy = DirectOutputStrategy(output_chunk_length=2)

    features = MultiOutputRegressor._features_for_output(X, 0, strategy, 1)
    assert type(features) is np.ndarray
    assert np.shares_memory(features, base)
    assert np.array_equal(features, base)

    assert MultiOutputRegressor._features_for_output(X, 0, None, None) is not X
    assert np.shares_memory(
        MultiOutputRegressor._features_for_output(X, 0, None, None), base
    )
    assert MultiOutputRegressor._features_for_output(base, 0, strategy, 1) is base


def test_recursive_strategy_uses_the_same_features_for_all_outputs():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=RecursiveOutputStrategy(output_chunk_length=3),
    )
    wrapper.fit(X, y)

    assert len(wrapper.estimators_) == 2
    for estimator in wrapper.estimators_:
        assert np.array_equal(estimator.fit_X_, X)
    assert np.array_equal(wrapper.estimators_[0].fit_y_, y[:, 0])
    assert np.array_equal(wrapper.estimators_[1].fit_y_, y[:, 1])

    RecordingRegressor.reset()
    predictions = wrapper.predict(X)

    assert predictions.shape == (4, 2)
    assert len(RecordingRegressor.predict_records) == 2
    assert all(
        np.array_equal(record, X) for record in RecordingRegressor.predict_records
    )


def test_direct_strategy_keeps_the_output_order():
    X = np.arange(15, dtype=float).reshape(5, 3)
    y = np.column_stack([np.arange(5) + i for i in range(4)])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y)

    assert len(wrapper.estimators_) == 4
    for output_idx in range(4):
        assert np.array_equal(wrapper.estimators_[output_idx].fit_y_, y[:, output_idx])


def test_sample_weight_is_sliced_per_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    sample_weight = np.column_stack([
        np.arange(4) + 0.5,
        np.arange(4) + 10.5,
    ])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y, sample_weight=sample_weight)

    for i in range(2):
        assert np.array_equal(
            wrapper.estimators_[i].fit_sample_weight_, sample_weight[:, i]
        )


def test_eval_set_is_routed_to_each_horizon():
    X, base, step_values = make_stepwise()
    y = np.column_stack([np.arange(6), np.arange(6) + 100])
    eval_set = [(X, y[:, 0]), (X, y[:, 1])]
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        eval_set_name="eval_set",
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y, eval_set=eval_set)

    first_samples, first_labels = wrapper.estimators_[0].fit_eval_set_[0]
    second_samples, second_labels = wrapper.estimators_[1].fit_eval_set_[0]
    assert first_samples[:, 0].tolist() == base[:, 0].tolist()
    assert second_samples[:, 0].tolist() == step_values[1, :, 0].tolist()
    assert np.array_equal(first_labels, y[:, 0])
    assert np.array_equal(second_labels, y[:, 1])


def test_eval_samples_are_routed_to_each_horizon():
    X, base, step_values = make_stepwise()
    y = np.column_stack([np.arange(6), np.arange(6) + 100])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        eval_set_name=("eval_samples", "eval_labels"),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(
        X,
        y,
        eval_samples=[X, X],
        eval_labels=[y[:, 0], y[:, 1]],
    )

    assert (
        wrapper.estimators_[0].fit_eval_samples_[:, 0].tolist() == base[:, 0].tolist()
    )
    assert (
        wrapper.estimators_[1].fit_eval_samples_[:, 0].tolist()
        == step_values[1, :, 0].tolist()
    )


def test_predict_proba_routes_stepwise_horizons():
    X, base, step_values = make_stepwise()
    y = np.column_stack([
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([1, 0, 1, 0, 1, 0]),
    ])
    RecordingClassifier.reset()

    wrapper = MultiOutputClassifier(
        RecordingClassifier(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y)
    RecordingClassifier.reset()

    probabilities = wrapper.predict_proba(X)

    assert len(probabilities) == 2
    assert all(probability.shape == (6, 2) for probability in probabilities)
    assert len(RecordingClassifier.predict_proba_records) == 2
    assert (
        RecordingClassifier.predict_proba_records[0][:, 0].tolist()
        == base[:, 0].tolist()
    )
    assert (
        RecordingClassifier.predict_proba_records[1][:, 0].tolist()
        == step_values[1, :, 0].tolist()
    )


def test_array_container_with_direct_strategy_is_accepted():
    array = np.arange(12, dtype=float).reshape(4, 3)
    X = ArrayLaggedFeatures(array, n_horizons=2)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y)

    for estimator in wrapper.estimators_:
        assert np.array_equal(estimator.fit_X_, array)


def test_legacy_path_without_strategy_still_works():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(RecordingRegressor())
    wrapper.fit(X, y)

    assert wrapper.strategy_ is None
    assert len(wrapper.estimators_) == 2
    for estimator in wrapper.estimators_:
        assert np.array_equal(estimator.fit_X_, X)

    predictions = wrapper.predict(X)
    assert predictions.shape == (4, 2)


def test_legacy_classifier_without_strategy_predict_proba_still_works():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])])
    RecordingClassifier.reset()

    wrapper = MultiOutputClassifier(RecordingClassifier())
    wrapper.fit(X, y)

    assert wrapper.strategy_ is None
    probabilities = wrapper.predict_proba(X)
    assert len(probabilities) == 2
    assert all(probability.shape == (4, 2) for probability in probabilities)
    assert len(RecordingClassifier.predict_proba_records) == 2
    for record in RecordingClassifier.predict_proba_records:
        assert np.array_equal(record, X)


def test_legacy_pickle_without_strategy_can_be_refitted():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=2),
    )
    wrapper.fit(X, y)

    # simulate the state of a wrapper pickled before the output-chunk strategies existed
    for key in ("strategy", "strategy_", "n_components_"):
        del wrapper.__dict__[key]
    wrapper.__dict__["output_chunk_length"] = 2

    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored.strategy is None
    assert "output_chunk_length" not in restored.__dict__
    assert restored.get_params()["strategy"] is None
    assert clone(restored).strategy is None

    restored.fit(X, y)
    assert restored.strategy_ is None
    assert restored.predict(X).shape == (4, 2)


def test_strategy_must_match_the_number_of_outputs():
    X = np.zeros((4, 3))
    y = np.zeros((4, 2))

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        strategy=DirectOutputStrategy(output_chunk_length=3),
    )

    with pytest.raises(ValueError):
        wrapper.fit(X, y)


def test_invalid_strategy_type():
    wrapper = MultiOutputRegressor(RecordingRegressor(), strategy=object())

    with pytest.raises(ValueError):
        wrapper.fit(np.zeros((4, 3)), np.zeros((4, 2)))
