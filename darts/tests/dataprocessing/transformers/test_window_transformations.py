import itertools
import unittest

import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers.window_transformer import (
    ForecastingWindowTransformer,
)


class TimeSeriesWindowTransformTestCase(unittest.TestCase):

    times = pd.date_range("20130101", "20130110")
    target = TimeSeries.from_times_and_values(times, range(1, 11))

    series_multi_prob = (
        (target + 10)
        .stack(target + 20)
        .stack((target + 100).stack(target + 200), axis=2)
    )  # 2 comps, 2 samples
    series_multi_det = (
        (target + 10).stack(target + 20).stack((target + 30).stack(target + 40), axis=1)
    )  # 4 comps, 1 sample
    series_univ_det = target + 50  # 1 comp, 1 sample
    series_univ_prob = (target + 50).stack(target + 500, axis=2)  # 1 comp, 2 samples

    def test_ts_windowtransf_input_dictionary(self):
        """
        Test that the forecasting window transformer dictionary input parameter is correctly formatted
        """

        with self.assertRaises(ValueError):
            window_transformations = None  # None input
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = []  # empty list
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {}  # empty dictionary
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = [1, 2, 3]  # list of not dictionaries
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {"random_fn_name": "mean"}  # no 'function' key
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "wild_fn"
            }  # not valid built-in function for provided string
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {"function": None}  # None function value
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "quantile",
                "window": [3],
            }  # not enough mandatory arguments for quantile
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": -3,
            }  # negative window
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {"function": "mean", "window": None}  # None window
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {"function": "mean", "window": [5]}  # window list
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "step": -2,
            }  # Negative step
            self.series_univ_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "components": "2",
            }  # Negative comp_id integer
            self.series_multi_det.window_transform(
                window_transformations=window_transformations
            )

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "components": ["2"],
            }  # Negative comp_id integer list
            self.series_multi_det.window_transform(
                window_transformations=window_transformations
            )

        # validate final format when all is correct:
        # checks that window_transformations is a list of dictionaries
        # window, series_id and comp_id are lists of integers
        window_transformations = {
            "function": "quantile",
            "window": 3,
            "quantile": 0.5,
            "components": "0",
        }
        self.series_multi_det.window_transform(
            window_transformations=window_transformations,
            store_window_transformation=True,
        )

        self.assertEqual(
            self.series_multi_det.window_transformations,
            [
                {
                    "function": "quantile",
                    "window": 3,
                    "quantile": 0.5,
                    "components": ["0"],
                    "closed": "left",
                }
            ],
        )

        window_transformations = {
            "function": "quantile",
            "window": 3,
            "quantile": 0.5,
            "components": None,
            "series_id": [0, 1],
        }
        self.series_multi_det.window_transform(
            window_transformations=window_transformations,
            store_window_transformation=True,
        )
        self.assertEqual(
            self.series_multi_det.window_transformations,
            [{"function": "quantile", "window": 3, "quantile": 0.5, "closed": "left"}],
        )

    def test_ts_windowtransf_output(self):
        # univarite deterministic input
        window_transformations = {"function": "sum", "window": 1}
        transformed_ts = self.series_univ_det.window_transform(
            window_transformations=window_transformations
        )

        self.assertEqual(
            list(itertools.chain(*transformed_ts.values().tolist())),
            [51, 52, 53, 54, 55, 56, 57, 58, 59],
        )
        self.assertEqual(len(transformed_ts.components), 1)

        # multivariate deterministic input
        # transform one component
        window_transformations = {"function": "sum", "window": 1, "components": "0"}

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(len(transformed_ts.components), 1)

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations, keep_non_transformed=True
        )
        self.assertEqual(len(transformed_ts.components), 4)

        # transform multiple components
        window_transformations = {
            "function": "sum",
            "window": 1,
            "components": ["0", "0_1"],
        }

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(len(transformed_ts.components), 2)

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations, keep_non_transformed=True
        )
        self.assertEqual(len(transformed_ts.components), 4)

        # multiple transformations
        window_transformations = [
            {"function": "sum", "window": 1, "components": ["0", "0_1"]},
            {"function": "mean", "window": 1, "components": ["0", "0_1"]},
        ]

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(len(transformed_ts.components), 4)

        transformed_ts = self.series_multi_det.window_transform(
            window_transformations=window_transformations, keep_non_transformed=True
        )
        self.assertEqual(len(transformed_ts.components), 6)

        # multivariate probabilistic input
        window_transformations = [
            {"function": "sum", "window": 1, "components": ["0", "0_1"]},
            {"function": "mean", "window": 1, "components": ["0", "0_1"]},
        ]

        transformed_ts = self.series_multi_prob.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(len(transformed_ts.components), 4)
        self.assertEqual(transformed_ts.n_samples, 2)

    def test_ts_windowtransf_output_nabehavior(self):
        pass

    def test_ts_windowtransf_truncate_target(self):
        pass

    def test_ts_windowtransf_forecasting_safe(self):

        # built-in functions
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))

        expected_transformed_series = TimeSeries.from_times_and_values(
            times1[1:], [1, 2, 3, 4, 5, 6, 7, 8, 9], columns=["sum1_0"]
        )

        window_transformations = {
            "function": "sum",
            "window": 1,
            "closed": "left",
        }  # this is equivalent to a shift
        # if user specifies closed = 'left'
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        window_transformations = [
            {"function": "sum", "window": 1}
        ]  # if user doesn't specify closed; we default to closed = 'left'
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        window_transformations = {"function": "sum", "window": 1, "closed": "right"}
        # if user specifies closed != 'left' and forecasting_safe left to default = True
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        # User provided function
        expected_transformed_series_userFn = TimeSeries.from_times_and_values(
            times1[1:], [1, 2, 3, 4, 5, 6, 7, 8, 9], columns=["userFn1_0"]
        )

        def user_fn(x):
            return (
                x.sum()
            )  # instead of user_fn = lambda x: x.sum() to avoid linting error

        window_transformations = [
            {"function": user_fn, "window": 1, "closed": "left"}
        ]  # if user specifies closed = 'left'
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series_userFn)

        window_transformations = [
            {
                "function": user_fn,
                "window": 1,
            }  # default closed = 'left'
        ]
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series_userFn)

        window_transformations = [
            {
                "function": user_fn,
                "window": 1,
                "closed": "right",
            }
        ]
        transformed_ts = series_1.window_transform(
            window_transformations=window_transformations
        )
        self.assertEqual(transformed_ts, expected_transformed_series_userFn)


class WindowTransformerTestCase(unittest.TestCase):

    times = pd.date_range("20130101", "20130110")
    target = TimeSeries.from_times_and_values(times, range(1, 11))

    series_multi_prob = (
        (target + 10)
        .stack(target + 20)
        .stack((target + 100).stack(target + 200), axis=2)
    )  # 2 comps, 2 samples
    series_multi_det = (
        (target + 10).stack(target + 20).stack((target + 30).stack(target + 40), axis=1)
    )  # 4 comps, 1 sample
    series_univ_det = target + 50  # 1 comp, 1 sample
    series_univ_prob = (target + 50).stack(target + 500, axis=2)  # 1 comp, 2 samples

    sequence_det = [series_univ_det, series_multi_det]
    sequence_prob = [series_univ_prob, series_multi_prob]

    def test_window_transformer_input_dictionary(self):
        """
        Test that the forecasting window transformer dictionary input parameter is correctly formatted
        """

        with self.assertRaises(ValueError):
            window_transformations = None  # None input
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = []  # empty list
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {}  # empty dictionary
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = [1, 2, 3]  # list of not dictionaries
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "series_id": -2,
            }  # Negative series_id integer
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "series_id": [-2],
            }  # Negative series_id integer list
            ForecastingWindowTransformer(window_transformations=window_transformations)

    def test_window_transformer_iterator(self):
        # no series_id, no components : all series and all there components should receive the same transformation
        window_transformations = {"function": "mean", "window": 3}
        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        associations_list = list(transformer._transform_iterator(self.sequence_det))

        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], window_transformations)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], window_transformations)

        # series_id, no components : all series and all there components should receive the same transformation
        window_transformations = {"function": "mean", "window": 3, "series_id": 0}
        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 1)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], window_transformations)

        # series_id and specific components per series
        window_transformations = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "components": [["0"], ["0", "1"]],
        }
        series_0_transformation = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "components": ["0"],
        }
        series_1_transformation = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "components": ["0", "1"],
        }

        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], series_0_transformation)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], series_1_transformation)

        # series_id and same selected components for all selected series
        window_transformations = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "components": ["0"],
        }
        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], window_transformations)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], window_transformations)

        # no series_id and components : all series will receive the same transformation on those components
        window_transformations = {"function": "mean", "window": 3, "components": ["0"]}
        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], window_transformations)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], window_transformations)

    def test_window_transformer_output(self):
        window_transformations = {
            "function": "sum",
            "window": 1,
            "series_id": [0, 1],
            "components": [["0"], ["0", "0_1"]],
        }
        transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_ts_list = transformer.transform(
            self.sequence_det,
            treat_na=100,
            keep_non_transformed=True,
            forecasting_safe=True,
        )

        self.assertEqual(len(transformed_ts_list), 2)
        self.assertEqual(transformed_ts_list[1].n_components, 4)
        self.assertEqual(transformed_ts_list[1].n_timesteps, 10)

    def test_transformers_pipline(self):
        """
        Test that the forecasting window transformer can be used in a pipeline with other transformers

        """

        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))

        expected_transformed_series = TimeSeries.from_times_and_values(
            times1[1:], [1, 2, 3, 4, 5, 6, 7, 8, 9], columns=["sum1_0"]
        )

        window_transformations = [{"function": "sum", "window": 1}]

        pipeline = Pipeline([ForecastingWindowTransformer(window_transformations)])

        transformed_series = pipeline.fit_transform(series_1)

        self.assertEqual(transformed_series, expected_transformed_series)
