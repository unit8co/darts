import itertools

import numpy as np
import pytest
import sklearn.metrics as sklearn_metrics

import darts.metrics.metrics as metrics
from darts import TimeSeries


class TestClassificationMetrics:
    np.random.seed(42)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [0, 1, 2],  # deterministic labels, class probabilities, sampled labels
            [None, 4, [4, 1], [1, 2, 3, 4]],  # selected labels
            [
                (
                    (metrics.recall_score, {"label_reduction": "weighted"}),
                    (
                        sklearn_metrics.recall_score,
                        {"average": "weighted", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.recall_score, {"label_reduction": "micro"}),
                    (
                        sklearn_metrics.recall_score,
                        {"average": "micro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.recall_score, {}),
                    (
                        sklearn_metrics.recall_score,
                        {"average": "macro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.precision_score, {"label_reduction": "weighted"}),
                    (
                        sklearn_metrics.precision_score,
                        {"average": "weighted", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.precision_score, {"label_reduction": "micro"}),
                    (
                        sklearn_metrics.precision_score,
                        {"average": "micro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.precision_score, {}),
                    (
                        sklearn_metrics.precision_score,
                        {"average": "macro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.f1_score, {"label_reduction": "weighted"}),
                    (
                        sklearn_metrics.f1_score,
                        {"average": "weighted", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.f1_score, {"label_reduction": "micro"}),
                    (
                        sklearn_metrics.f1_score,
                        {"average": "micro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.f1_score, {}),
                    (
                        sklearn_metrics.f1_score,
                        {"average": "macro", "zero_division": np.nan},
                    ),
                    np.nanmean,
                ),
                (
                    (metrics.confusion_matrix, {}),
                    (sklearn_metrics.confusion_matrix, {"labels": [1, 2, 3, 4]}),
                    np.nansum,
                ),
                # (
                #     (metrics.macc, {}),
                #     (sklearn_metrics.accuracy_score, {}),
                #     np.nanmean,
                # ),
            ],
        ),
    )
    def test_classification_metric(self, config):
        method, labels, ((metric, kwargs), (skl_metric, skl_kwargs), comp_reduction) = (
            config
        )
        comp1_labels = np.array([2, 4])
        comp1_probas = np.array([
            [0.1, 0.9],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.7, 0.3],
        ])
        comp2_labels = np.array([1, 3, 4])
        comp2_probas = np.array([
            [0.3, 0.2, 0.5],
            [0.6, 0.1, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.1, 0.6],
        ])
        kwargs["labels"] = labels

        if not (metric == metrics.confusion_matrix and labels is None):
            skl_kwargs["labels"] = [labels] if isinstance(labels, int) else labels

        if method == 0:
            # deterministic class labels
            names = ["comp1", "comp2"]
            vals = np.concatenate(
                [
                    comp1_labels[np.argmax(comp1_probas, axis=1), None],
                    comp2_labels[np.argmax(comp2_probas, axis=1), None],
                ],
                axis=1,
            )
        elif method == 1:
            # class probabilities
            names = ["comp1_p2", "comp1_p4", "comp2_p1", "comp2_p3", "comp2_p4"]
            vals = np.concatenate([comp1_probas, comp2_probas], axis=1)
        else:
            # sampled class labels
            names = ["comp1", "comp2"]
            num_samples = 1000
            vals = np.zeros((4, 2, num_samples))
            for idx in range(len(comp1_probas)):
                vals[idx, 0, :] = np.random.choice(
                    comp1_labels, size=num_samples, p=comp1_probas[idx]
                )
                vals[idx, 1, :] = np.random.choice(
                    comp2_labels, size=num_samples, p=comp2_probas[idx]
                )

        # predicted labels are comp1: [4, 4, 2, 2], comp2: [4, 1, 3, 4]
        y_pred = TimeSeries.from_values(vals, names)
        y_true = TimeSeries.from_values(
            np.array([
                [4, 4, 2, 2],
                [4, 1, 4, 1],
            ]).T[:, :, np.newaxis],
            ["comp1", "comp2"],
        )

        # get predicted labels with highest probability / frequency
        y_true_vals, y_pred_vals = metrics._get_values_or_raise(
            y_true, y_pred, True, remove_nan_union=False, is_classification=True
        )

        # compare with sklearn metrics
        scores_expected = np.array([
            skl_metric(y_true_vals[:, i], y_pred_vals[:, i], **skl_kwargs)
            for i in range(y_true_vals.shape[1])
        ])
        # without component reduction
        scores_actual = metric(
            y_true,
            y_pred,
            component_reduction=None,
            **kwargs,
        )
        # replace NaNs with 0.0 to compare with sklearn metrics
        scores_actual[np.isnan(scores_actual)] = 0.0
        np.testing.assert_array_almost_equal(
            scores_actual,
            scores_expected,
        )

        # with mean component reduction
        np.testing.assert_array_almost_equal(
            metric(
                y_true,
                y_pred,
                **kwargs,
            ),
            comp_reduction(scores_expected, axis=0),
        )

    def test_wrong_pred_component_names(self):
        y_pred = TimeSeries.from_values(
            values=np.array([[0.1, 0.9, 0.3, 0.7]]),
            columns=["comp0_p0", "comp0_p4", "comp1_p0", "comp1_p3"],
        )
        y_true = TimeSeries.from_values(
            values=np.array([[4, 3]]),
            columns=["comp0", "comp1"],
        )
        assert metrics.macc(y_true, y_pred) == 1.0

        # pred components should have names ["comp0*", "comp1*"]
        y_pred._components = y_pred._components.str.replace("comp0", "comp3")
        with pytest.raises(ValueError) as err:
            metrics.macc(y_true, y_pred)
        assert str(err.value).startswith(
            "Could not resolve the predicted components for the classification metric"
        )

    def test_wrong_number_of_pred_components(self):
        y_pred = TimeSeries.from_values(
            values=np.array([[0.1, 0.9, 0.3, 0.7]]),
            columns=["comp1_p0", "comp1_p1", "comp2_p0", "comp2_p1"],
        )
        y_true = TimeSeries.from_values(
            values=np.array([[4, 4, 4]]),
            columns=["comp0", "comp1", "comp2"],
        )

        # pred should have 3 components
        with pytest.raises(ValueError) as err:
            metrics.macc(y_true, y_pred)
        assert str(err.value).startswith(
            "Could not resolve the predicted components for the classification metric"
        )

    def test_non_integer_label(self):
        y_pred = TimeSeries.from_values(
            values=np.array([[0.1, 0.9]]),
            columns=["comp0_p2.5", "comp0_p4"],
        )
        y_true = TimeSeries.from_values(
            values=np.array([[4]]),
            columns=["comp0"],
        )

        # pred should have 3 components
        with pytest.raises(ValueError) as err:
            metrics.macc(y_true, y_pred)
        assert str(err.value).startswith(
            "Could not parse class label from name: comp0_p2.5"
        )

    def test_mode(self):
        # Test with a simple list of integers
        data = np.array([2, 1, 2, 3, 4])
        assert metrics._mode(data) == 2

        # Test with a list of strings
        data = np.array(["apple", "banana", "apple", "orange"])
        assert metrics._mode(data) == "apple"

        # Test with a list where all elements are unique; gives first element with highest count
        data = np.array([1, 2, 3, 4])
        assert metrics._mode(data) == 1

    def test_get_highest_count_label_single_step_single_component(self):
        # Test with a simple list of integers
        data = np.array([2, 1, 2, 3, 4]).reshape((1, 1, 5))
        res = metrics._get_highest_count_label(data)
        assert res.shape == (1, 1, 1)
        assert np.all(res == 2)

        # Test with a list of integers with NaN values
        data = np.array([2, 1, 2, np.nan, 4]).reshape((1, 1, 5))
        res = metrics._get_highest_count_label(data)
        assert res.shape == (1, 1, 1)
        assert np.all(res == 2)

        # Test with a list of integers where NaN values have highest count
        data = np.array([2, 1, np.nan, np.nan, 4]).reshape((1, 1, 5))
        res = metrics._get_highest_count_label(data)
        assert res.shape == (1, 1, 1)
        assert np.isnan(res)

        # Test with a list of strings
        data = np.array(["apple", "banana", "apple", "orange"]).reshape((1, 1, 4))
        res = metrics._get_highest_count_label(data)
        assert res.shape == (1, 1, 1)
        assert np.all(res == "apple")

        # Test with a list where all elements are unique; gives lowest value element with highest count
        data = np.array([4, 1, 3, 2]).reshape((1, 1, 4))
        res = metrics._get_highest_count_label(data)
        assert res.shape == (1, 1, 1)
        assert np.all(res == 1)

    def test_get_highest_count_label_multi_step_multi_component(self):
        # Test with a simple list of integers
        data = np.array([
            [[2, 1, 2, 3, 4], [1, 2, 3, 3, 5]],
            [[1, 2, 3, 4, 5], [2, 5, 5, 4, 4]],
            [[-1, 0, 0, -1, 5], [0, 0, 0, 0, 0]],
        ])
        res = metrics._get_highest_count_label(data)
        assert res.shape == (3, 2, 1)
        assert np.all(
            res
            == np.array([
                [[2], [3]],
                [[1], [4]],
                [[-1], [0]],
            ])
        )

    def test_get_highest_probability_label_single_step_single_component(self):
        # Test with a binary labels
        data = np.array([0.1, 0.9]).reshape((1, 2, 1))
        res = metrics._get_highest_probability_label(data, ["a_p0", "a_p1"], ["a"])
        assert res.shape == (1, 1, 1)
        assert np.all(res == 1)

        # Test with three labels and unsorted label names
        data = np.array([0.1, 0.8, 0.1]).reshape((1, 3, 1))
        res = metrics._get_highest_probability_label(
            data, ["a_p0", "a_p2", "a_p1"], ["a"]
        )
        assert res.shape == (1, 1, 1)
        assert np.all(res == 2)

        # Test with three labels where two labels have the highest probability (returns the first one)
        data = np.array([0.4, 0.2, 0.4]).reshape((1, 3, 1))
        res = metrics._get_highest_probability_label(
            data, ["a_p3", "a_p2", "a_p1"], ["a"]
        )
        assert res.shape == (1, 1, 1)
        assert np.all(res == 3)

        with pytest.raises(ValueError) as err:
            # label cannot be converted to integer
            metrics._get_highest_probability_label(
                data, ["a_p3", "a_p2.0", "a_p1"], ["a"]
            )
        assert str(err.value).startswith(
            "Could not parse class label from name: a_p2.0"
        )

    def test_get_highest_probability_label_multi_step_multi_component(self):
        # Test with a binary labels
        data = np.array([
            [[0.1], [0.9], [0.6], [0.2], [0.2]],
            [[0.3], [0.7], [0.4], [0.5], [0.1]],
            [[0.5], [0.5], [0.2], [0.0], [0.8]],
        ])
        res = metrics._get_highest_probability_label(
            data, ["a_p0", "a_p1", "b_p5", "b_p4", "b_p1"], ["a", "b"]
        )
        assert res.shape == (3, 2, 1)
        assert np.all(
            res
            == np.array([
                [[1], [5]],
                [[1], [4]],
                [[0], [1]],
            ])
        )
