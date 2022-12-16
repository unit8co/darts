import itertools
import unittest

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Mapper, WindowTransformer


class TimeSeriesWindowTransformTestCase(unittest.TestCase):

    times = pd.date_range("20130101", "20130110")
    series_from_values = TimeSeries.from_values(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
    target = TimeSeries.from_times_and_values(times, range(1, 11))

    series_multi_prob = (
        (target + 10)
        .stack(target + 20)
        .concatenate((target + 100).stack(target + 200), axis=2)
    )  # 2 comps, 2 samples
    series_multi_det = (
        (target + 10).stack(target + 20).stack((target + 30).stack(target + 40))
    )  # 4 comps, 1 sample
    series_univ_det = target + 50  # 1 comp, 1 sample
    series_univ_prob = (target + 50).concatenate(
        target + 500, axis=2
    )  # 1 comp, 2 samples

    def test_ts_windowtransf_input_dictionary(self):
        """
        Test that the forecasting window transformer dictionary input parameter is correctly formatted
        """

        with self.assertRaises(TypeError):
            window_transformations = None  # None input
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = []  # empty list
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(KeyError):
            window_transformations = {}  # empty dictionary
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = [1, 2, 3]  # list of not dictionaries
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(KeyError):
            window_transformations = {"random_fn_name": "mean"}  # no 'function' key
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(AttributeError):
            window_transformations = {
                "function": "wild_fn"
            }  # not valid pandas built-in function
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": 1
            }  # not valid pandas built-in function nore callable
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {"function": None}  # None function value
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(TypeError):
            window_transformations = {
                "function": "quantile",
                "window": [3],
            }  # not enough mandatory arguments for quantile
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": -3,
            }  # negative window
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": None,
            }  # None window
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": [5],
            }  # window list
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "mode": "rolling",
                "step": -2,
            }  # Negative step
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "mode": "rnd",
            }  # invalid mode
            self.series_univ_det.window_transform(transforms=window_transformations)

        with self.assertRaises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": 3,
                "center": "True",
            }  # forecating_safe=True vs center=True
            self.series_univ_det.window_transform(transforms=window_transformations)

    def test_ts_windowtransf_output_series(self):
        # univariate deterministic input
        transforms = {"function": "sum", "mode": "rolling", "window": 1}
        transformed_ts = self.series_univ_det.window_transform(transforms=transforms)

        self.assertEqual(
            list(itertools.chain(*transformed_ts.values().tolist())),
            list(itertools.chain(*self.series_univ_det.values().tolist())),
        )
        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in self.series_univ_det.components
            ],
        )

        # multivariate deterministic input
        # transform one component
        transforms.update({"components": "0"})

        transformed_ts = self.series_multi_det.window_transform(transforms=transforms)
        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ],
        )

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )

        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ]
            + self.series_multi_det.components.to_list(),
        )

        # transform multiple components
        transforms = {
            "function": "sum",
            "mode": "rolling",
            "window": 1,
            "components": ["0", "0_1"],
        }

        transformed_ts = self.series_multi_det.window_transform(transforms=transforms)
        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ],
        )

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )

        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ]
            + self.series_multi_det.components.to_list(),
        )

        # multiple transformations

        transforms = [transforms] + [
            {
                "function": "mean",
                "mode": "rolling",
                "window": 1,
                "components": ["0", "0_1"],
            }
        ]

        transformed_ts = self.series_multi_det.window_transform(transforms=transforms)
        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transformation['mode']}_{transformation['function']}_{str(transformation['window'])}_{comp}"
                for transformation in transforms
                for comp in transformation["components"]
            ],
        )

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )
        self.assertEqual(
            transformed_ts.components.to_list(),
            [
                f"{transformation['mode']}_{transformation['function']}_{str(transformation['window'])}_{comp}"
                for transformation in transforms
                for comp in transformation["components"]
            ]
            + self.series_multi_det.components.to_list(),
        )

        # multivariate probabilistic input
        transformed_ts = self.series_multi_prob.window_transform(transforms=transforms)
        self.assertEqual(transformed_ts.n_samples, 2)

    def test_ts_windowtransf_output_nabehavior(self):
        window_transformations = {
            "function": "sum",
            "mode": "rolling",
            "window": 3,
            "min_periods": 2,
        }

        # fill na with a specific value
        transformed_ts = self.target.window_transform(
            window_transformations, treat_na=100
        )
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            [100, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            columns=["rolling_sum_3_2_0"],
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        # dropna
        transformed_ts = self.target.window_transform(
            window_transformations, treat_na="dropna"
        )
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times[1:],
            [3, 6, 9, 12, 15, 18, 21, 24, 27],
            columns=["rolling_sum_3_2_0"],
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        # backfill na
        transformed_ts = self.target.window_transform(
            window_transformations, treat_na="bfill", forecasting_safe=False
        )
        # backfill works only with forecasting_safe=False
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            [3, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            columns=["rolling_sum_3_2_0"],
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        with self.assertRaises(ValueError):
            # uknonwn treat_na
            self.target.window_transform(
                window_transformations, treat_na="fillrnd", forecasting_safe=False
            )

        with self.assertRaises(ValueError):
            # unauhtorized treat_na=bfill with forecasting_safe=True
            self.target.window_transform(window_transformations, treat_na="bfill")

    def test_tranformed_ts_index(self):

        # DateTimeIndex
        transformed_series = self.target.window_transform({"function": "sum"})
        self.assertEqual(
            self.target._time_index.__class__, transformed_series._time_index.__class__
        )
        # length index should not change for default transformation configurations
        self.assertEqual(
            len(self.target._time_index), len(transformed_series._time_index)
        )
        # RangeIndex
        transformed_series = self.series_from_values.window_transform(
            {"function": "sum"}
        )
        self.assertEqual(
            self.series_from_values._time_index.__class__,
            transformed_series._time_index.__class__,
        )
        self.assertEqual(
            len(self.series_from_values._time_index),
            len(transformed_series._time_index),
        )

    def test_include_current(self):
        # if "closed"="left" should not shift the index
        transformation = {
            "function": "sum",
            "mode": "rolling",
            "window": 1,
            "closed": "left",
        }
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            ["NaN", 1, 2, 3, 4, 5, 6, 7, 8, 9],
            columns=["rolling_sum_1_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        # shift the index
        transformation = {"function": "sum", "mode": "rolling", "window": 1}
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            ["NaN", 1, 2, 3, 4, 5, 6, 7, 8, 9],
            columns=["rolling_sum_1_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        transformation = [
            {"function": "sum", "mode": "rolling", "window": 1, "closed": "left"},
            {"function": "sum", "mode": "ewm", "span": 1},
        ]
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            [
                ["NaN", "NaN"],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
                [8, 8],
                [9, 9],
            ],
            columns=["rolling_sum_1_0", "ewm_sum_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            [
                [1, 1],
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
                [5, 5],
                [6, 6],
                [7, 7],
                [8, 8],
                [9, 9],
            ],
            columns=["rolling_sum_1_0", "ewm_sum_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation,
            include_current=False,
            forecasting_safe=False,
            treat_na="bfill",
        )
        self.assertEqual(transformed_ts, expected_transformed_series)

        transformation = [
            {
                "function": "sum",
                "mode": "rolling",
                "window": 2,
                "closed": "left",
                "min_periods": 2,
            },
            {"function": "sum", "mode": "ewm", "span": 1, "min_periods": 2},
        ]

        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            [
                ["NaN", "NaN"],
                ["NaN", "NaN"],
                [3, 2],
                [5, 3],
                [7, 4],
                [9, 5],
                [11, 6],
                [13, 7],
                [15, 8],
                [17, 9],
            ],
            columns=["rolling_sum_2_2_0", "ewm_sum_2_0"],
        )

        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        self.assertEqual(transformed_ts, expected_transformed_series)


class WindowTransformerTestCase(unittest.TestCase):

    times = pd.date_range("20130101", "20130110")
    target = TimeSeries.from_times_and_values(times, range(1, 11))

    series_multi_prob = (
        (target + 10)
        .stack(target + 20)
        .concatenate((target + 100).stack(target + 200), axis=2)
    )  # 2 comps, 2 samples
    series_multi_det = (
        (target + 10).stack(target + 20).stack((target + 30).stack(target + 40))
    )  # 4 comps, 1 sample
    series_univ_det = target + 50  # 1 comp, 1 sample
    series_univ_prob = (target + 50).concatenate(
        target + 500, axis=2
    )  # 1 comp, 2 samples

    sequence_det = [series_univ_det, series_multi_det]
    sequence_prob = [series_univ_prob, series_multi_prob]

    def test_window_transformer_iterator(self):
        # no series_id, no components : all series and all there components should receive the same transformation
        window_transformations = {"function": "mean"}
        transformer = WindowTransformer(transforms=window_transformations)
        expected_kwargs_dict = {
            "transforms": window_transformations,
            "keep_non_transformed": False,
            "treat_na": None,
            "forecasting_safe": True,
            "include_current": True,
        }
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], expected_kwargs_dict)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], expected_kwargs_dict)

        # no series_id and components : all series will receive the same transformation on those components
        window_transformations = {"function": "mean", "components": ["0"]}
        transformer = WindowTransformer(transforms=window_transformations)
        expected_kwargs_dict = {
            "transforms": window_transformations,
            "keep_non_transformed": False,
            "treat_na": None,
            "forecasting_safe": True,
            "include_current": True,
        }
        associations_list = list(transformer._transform_iterator(self.sequence_det))
        self.assertEqual(len(associations_list), 2)
        self.assertEqual(associations_list[0][0], self.series_univ_det)
        self.assertEqual(associations_list[0][1], expected_kwargs_dict)
        self.assertEqual(associations_list[1][0], self.series_multi_det)
        self.assertEqual(associations_list[1][1], expected_kwargs_dict)

    def test_window_transformer_output(self):
        window_transformations = {
            "function": "sum",
            "components": ["0"],
        }
        transformer = WindowTransformer(
            transforms=window_transformations,
            treat_na=100,
            keep_non_transformed=True,
            forecasting_safe=True,
        )
        transformed_ts_list = transformer.transform(self.sequence_det)

        self.assertEqual(len(transformed_ts_list), 2)
        self.assertEqual(transformed_ts_list[0].n_components, 2)
        self.assertEqual(
            transformed_ts_list[0].n_timesteps, self.series_multi_det.n_timesteps
        )
        self.assertEqual(transformed_ts_list[1].n_components, 5)
        self.assertEqual(
            transformed_ts_list[1].n_timesteps, self.series_multi_det.n_timesteps
        )

    def test_transformers_pipline(self):
        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, range(1, 11))

        expected_transformed_series = TimeSeries.from_times_and_values(
            times1,
            [100, 15, 30, 45, 60, 75, 90, 105, 120, 135],
            columns=["rolling_sum_3_2_0"],
        )

        # adds NaNs
        window_transformations = [
            {"function": "sum", "mode": "rolling", "window": 3, "min_periods": 2}
        ]

        def times_five(x):
            return x * 5

        mapper = Mapper(fn=times_five)

        window_transformer = WindowTransformer(
            transforms=window_transformations, treat_na=100
        )

        pipeline = Pipeline([mapper, window_transformer])

        transformed_series = pipeline.fit_transform(series_1)

        self.assertEqual(transformed_series, expected_transformed_series)
