import unittest

from darts.dataprocessing.transformers.window_transformer import (
    ForecastingWindowTransformer,
)
from darts.utils.timeseries_generation import linear_timeseries as lt
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline


class WindowTransformationsTestCase(unittest.TestCase):
    def test_window_tranformer_input_dictionary_format(self):
        """
        Test that the forecasting window transformer dictionary input parameter is correctly formatted
        :return:
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
            window_transformations = {"random_fn_name": "mean"}  # no 'function' key
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "wild_fn"
            }  # not valid built-in function for provided string
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {"function": None}  # None function value
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "quantile",
                "window": [3],
            }  # not enough mandatory arguments for quantile
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": -3,
            }  # negative window
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {"function": "mean", "window": None}  # None window
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "step": -2,
            }  # Negative step
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

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "comp_id": -2,
            }  # Negative comp_id integer
            ForecastingWindowTransformer(window_transformations=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "comp_id": [-2],
            }  # Negative comp_id integer list
            ForecastingWindowTransformer(window_transformations=window_transformations)

        # validate final format when all is correct:
        # cheks that window_transformations is a list of dictionaries
        # window, series_id and comp_id are lists of integers
        window_transformations = {
            "function": "quantile",
            "window": 3,
            "quantile": 0.5,
            "comp_id": 0,
            "series_id": 0,
        }
        self.assertEqual(
            ForecastingWindowTransformer(
                window_transformations=window_transformations
            ).window_transformations,
            [
                {
                    "function": "quantile",
                    "window": [3],
                    "quantile": 0.5,
                    "comp_id": [0],
                    "series_id": [0],
                }
            ],
        )

        window_transformations = {
            "function": "quantile",
            "window": 3,
            "quantile": 0.5,
            "comp_id": None,
            "series_id": None,
        }
        self.assertEqual(
            ForecastingWindowTransformer(
                window_transformations=window_transformations
            ).window_transformations,
            [{"function": "quantile", "window": [3], "quantile": 0.5}],
        )

    def test_output_series_width(self):
        """
        Test that the forecasting window transformer output series width is correct
        :return:
        """

        mult_series_1 = lt(length=10).stack(
            lt(length=10) + 4
        )  # 2 components , width = 2
        mult_series_2 = (lt(length=10) + 3).stack(
            lt(length=10) + 5
        )  # 2 components , width = 2
        series_3 = lt(length=10) + 2  # 1 component , width = 1

        all_series = [mult_series_1, mult_series_2, series_3]  # 3 series

        # all series and all components
        window_transformations = {"function": "mean", "window": 3}
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 5)

        # all series and one component
        window_transformations = {"function": "mean", "window": 3, "comp_id": 0}
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 3)

        # particular series and two components
        window_transformations = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "comp_id": [0, 1],
        }
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 4)

        # one series and all components
        window_transformations = {"function": "mean", "window": 3, "series_id": 0}
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 2)

        # one series and one component
        window_transformations = {
            "function": "mean",
            "window": 3,
            "series_id": 0,
            "comp_id": 0,
        }
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 1)

        # specific series and specific component per series
        window_transformations = {
            "function": "mean",
            "window": 3,
            "series_id": [0, 1],
            "comp_id": [[0, 1], [0]],
        }
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 3)

        # two transformations, all series and all components
        window_transformations = [
            {"function": "mean", "window": 3},
            {"function": "std", "window": 3},
        ]
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 10)

        # two transformations, specific series and all components
        window_transformations = [
            {"function": "mean", "window": 3, "series_id": 0},
            {"function": "std", "window": 3, "series_id": 1},
        ]
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 4)

        # two transformations, specific series and specific components
        window_transformations = [
            {"function": "mean", "window": 3, "series_id": [0, 2], "comp_id": 0},
            {
                "function": "std",
                "window": 3,
                "series_id": [0, 1],
                "comp_id": [[0], [1]],
            },
        ]
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 4)

        # multiple window sizes for same function
        window_transformations = [
            {"function": "mean", "window": [3, 5]},
            {"function": "std", "window": 3},
        ]
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(all_series)
        self.assertEqual(len(transformed_series), 10)
        # self.assertEqual(len(transformed_series), 15)
        # TODO: validate how to return multiple window sizes for same function

        # series out of range
        window_transformations = {"function": "mean", "window": 3, "series_id": 3}
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        with self.assertRaises(ValueError):
            window_transformer.transform(all_series)

        # component out of range
        window_transformations = {"function": "mean", "window": 3, "comp_id": 2}
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        with self.assertRaises(ValueError):
            window_transformer.transform(all_series)

    def test_forecasting_safe_builtin_rolling(self):
        """
        Test that the forecasting window transformer, based on built_in rolling fucntions,
        is safe to use in forecasting pipelines

        """
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))
        window_transformations = [
            {"function": "sum", "window": 1, "closed": "left"}
        ]  # if user specifies closed = 'left'
        expected_transformed_series = TimeSeries.from_times_and_values(
            times1, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )  # closed = 'left' would basically not account for the value at the current
        # timestep for the computation of the window transformation
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(series_1)
        self.assertEqual(transformed_series, expected_transformed_series)

        window_transformations = [
            {"function": "sum", "window": 1}
        ]  # if user doesn't specify closed = 'left'; we default to closed = 'left'
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(series_1)
        self.assertEqual(transformed_series, expected_transformed_series)

    def test_forecasting_safe_user_defined_rolling(self):
        """
        Test that the forecasting window transformer, based on user defined rolling function,
        is safe to use in forecasting pipelines

        """
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))
        user_fn = lambda x: x.sum()

        window_transformations = [
            {"function": user_fn, "window": 1, "rolling": True, "closed": "left"}
        ]  # if user specifies closed = 'left'
        expected_transformed_series = TimeSeries.from_times_and_values(
            times1, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )  # closed = 'left' would basically not account for the value at the current
        # timestep for the computation of the window transformation
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(series_1)
        self.assertEqual(transformed_series, expected_transformed_series)

        window_transformations = [
            {
                "function": user_fn,
                "window": 1,
                "rolling": True,
            }  # default closed = 'left'
        ]
        window_transformer = ForecastingWindowTransformer(
            window_transformations=window_transformations
        )
        transformed_series = window_transformer.transform(series_1)
        self.assertEqual(transformed_series, expected_transformed_series)

    def test_user_defined_norolling_output_formatting(self):
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))
        user_fn = lambda x: x.sum()

        window_transformations = [{"function": user_fn, "window": 1, "closed": "left"}]
        window_transformer = ForecastingWindowTransformer(window_transformations)

        with self.assertRaises(
            AttributeError
        ):  # user defined function should set column name (if we don't set it)
            window_transformer.transform(series_1)

    def test_transformers_pipline(self):
        """
        Test that the forecasting window transformer can be used in a pipeline with other transformers

        """
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))

        expected_transformed_series = TimeSeries.from_times_and_values(
            times1, [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        window_transformations = [{"function": "sum", "window": 1}]

        pipeline = Pipeline([ForecastingWindowTransformer(window_transformations)])

        transformed_series = pipeline.fit_transform(series_1)

        self.assertEqual(transformed_series, expected_transformed_series)
