import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from darts.utils.data.tabularization.features import LaggedFeatures
from darts.utils.multioutput import MultiOutputClassifier, MultiOutputRegressor


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


def test_fit_creates_one_estimator_per_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(RecordingRegressor())
    wrapper.fit(X, y)

    assert len(wrapper.estimators_) == 2
    for output_idx, estimator in enumerate(wrapper.estimators_):
        assert np.array_equal(estimator.fit_X_, X)
        assert np.array_equal(estimator.fit_y_, y[:, output_idx])

    predictions = wrapper.predict(X)
    assert predictions.shape == (4, 2)
    assert len(RecordingRegressor.predict_records) == 2
    for record in RecordingRegressor.predict_records:
        assert np.array_equal(record, X)


def test_array_container_is_accepted():
    array = np.arange(12, dtype=float).reshape(4, 3)
    X = LaggedFeatures(array, n_horizons=2)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(RecordingRegressor())
    wrapper.fit(X, y)

    for estimator in wrapper.estimators_:
        assert type(estimator.fit_X_) is np.ndarray
        assert np.array_equal(estimator.fit_X_, array)


def test_sample_weight_is_sliced_per_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    sample_weight = np.column_stack([
        np.arange(4) + 0.5,
        np.arange(4) + 10.5,
    ])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(RecordingRegressor())
    wrapper.fit(X, y, sample_weight=sample_weight)

    for output_idx in range(2):
        assert np.array_equal(
            wrapper.estimators_[output_idx].fit_sample_weight_,
            sample_weight[:, output_idx],
        )


def test_eval_set_is_routed_to_each_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    eval_set = [(X, y[:, 0]), (X, y[:, 1])]
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        eval_set_name="eval_set",
    )
    wrapper.fit(X, y, eval_set=eval_set)

    for output_idx, estimator in enumerate(wrapper.estimators_):
        samples, labels = estimator.fit_eval_set_[0]
        assert np.array_equal(samples, X)
        assert np.array_equal(labels, y[:, output_idx])


def test_eval_samples_are_routed_to_each_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.arange(4), np.arange(4) + 10])
    RecordingRegressor.reset()

    wrapper = MultiOutputRegressor(
        RecordingRegressor(),
        eval_set_name=("eval_samples", "eval_labels"),
    )
    wrapper.fit(
        X,
        y,
        eval_samples=[X, X],
        eval_labels=[y[:, 0], y[:, 1]],
    )

    for output_idx, estimator in enumerate(wrapper.estimators_):
        assert np.array_equal(estimator.fit_eval_samples_, X)
        assert np.array_equal(estimator.fit_eval_labels_, y[:, output_idx])


def test_classifier_predict_proba_returns_one_array_per_output():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.column_stack([np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])])
    RecordingClassifier.reset()

    wrapper = MultiOutputClassifier(RecordingClassifier())
    wrapper.fit(X, y)
    RecordingClassifier.reset()

    probabilities = wrapper.predict_proba(X)

    assert len(probabilities) == 2
    assert all(probability.shape == (4, 2) for probability in probabilities)
    assert len(RecordingClassifier.predict_proba_records) == 2
    for record in RecordingClassifier.predict_proba_records:
        assert np.array_equal(record, X)
