import numpy as np

from darts import TimeSeries
from darts.metrics import forecasting_classification


class TestForecastingClassificationMetrics:
    np.random.seed(42)

    def test_probabilistic_metric(self):
        names = ["comp1_p_2", "comp1_p_4", "comp2_p_1", "comp2_p_3", "comp2_p_4"]

        # random_num = np.random.random(len(names) * 10).reshape(-1, len(names), 1)
        # probability_data = random_num / random_num.sum(axis=SMPL_AX, keepdims=True)
        # y_pred = TimeSeries.from_values(probability_data, columns=names)

        y_pred = TimeSeries.from_values(
            np.array([
                [0.1, 0.4, 0.9, 0.5],  # comp1_p_2
                [0.9, 0.6, 0.1, 0.5],  # comp1_p_4
                [0.3, 0.5, 0.3, 0.4],  # comp2_p_1
                [0.2, 0.1, 0.4, 0.1],  # comp2_p_3
                [0.5, 0.4, 0.3, 0.5],  # comp2_p_4
            ]).T[:, :, np.newaxis],
            names,
        )
        y_true = TimeSeries.from_values(
            np.array([
                [4, 4, 2, 2],
                [4, 1, 3, 4],
            ]).T[:, :, np.newaxis],
            ["comp1", "comp2"],
        )

        assert forecasting_classification.macc(y_true, y_pred) == np.ones((4, 2))
