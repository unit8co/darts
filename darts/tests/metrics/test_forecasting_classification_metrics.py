import numpy as np
import pytest

from darts import TimeSeries
from darts.metrics import macc


class TestForecastingClassificationMetrics:
    np.random.seed(42)

    @pytest.mark.parametrize("is_sampled", [True, False])
    def test_probabilistic_metric(self, is_sampled):
        comp1_labels = [2, 4]
        comp1_probas = np.array([
            [0.1, 0.9],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.7, 0.3],
        ])
        comp2_labels = [1, 3, 4]
        comp2_probas = np.array([
            [0.3, 0.2, 0.5],
            [0.6, 0.1, 0.3],
            [0.2, 0.6, 0.2],
            [0.3, 0.1, 0.6],
        ])

        if not is_sampled:
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

        # without component reduction
        np.testing.assert_array_equal(
            macc(
                y_true,
                y_pred,
                component_reduction=None,
            ),
            np.array([1.0, 0.5]),  # comp1: 1.0, comp2: 0.5
        )

        # with mean component reduction
        np.testing.assert_array_equal(
            macc(
                y_true,
                y_pred,
                component_reduction=np.mean,
            ),
            np.array(0.75),
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
        assert macc(y_true, y_pred) == 1.0

        # pred components should have names ["comp0*", "comp1*"]
        y_pred._components = y_pred._components.str.replace("comp0", "comp3")
        with pytest.raises(ValueError) as err:
            macc(y_true, y_pred)
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
            macc(y_true, y_pred)
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
            macc(y_true, y_pred)
        assert str(err.value).startswith(
            "Could not parse class label from name: comp0_p2.5"
        )
