import itertools

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Mapper, WindowTransformer
from darts.utils.utils import freqs


def helper_generate_ts_hierarchy(length: int):
    values = np.stack(
        [
            np.ones(
                length,
            )
            * 5,
            np.ones(
                length,
            )
            * 3,
            np.ones(
                length,
            )
            * 2,
        ],
        axis=1,
    )
    hierarchy = {"B": "A", "C": "A"}
    return TimeSeries.from_values(
        values=values, columns=["A", "B", "C"], hierarchy=hierarchy
    )


class TestTimeSeriesWindowTransform:
    times = pd.date_range("20130101", "20130110")
    series_from_values = TimeSeries.from_values(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
    target = TimeSeries.from_times_and_values(times, np.array(range(1, 11)))

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

        with pytest.raises(TypeError):
            window_transformations = None  # None input
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = []  # empty list
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(KeyError):
            window_transformations = {}  # empty dictionary
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = [1, 2, 3]  # list of not dictionaries
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(KeyError):
            window_transformations = {"random_fn_name": "mean"}  # no 'function' key
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(AttributeError):
            window_transformations = {
                "function": "wild_fn"
            }  # not valid pandas built-in function
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": 1
            }  # not valid pandas built-in function nore callable
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {"function": None}  # None function value
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(TypeError):
            window_transformations = {
                "function": "quantile",
                "window": [3],
            }  # not enough mandatory arguments for quantile
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": -3,
            }  # negative window
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": None,
            }  # None window
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": [5],
            }  # window list
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "mode": "rolling",
                "step": -2,
            }  # Negative step
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "window": 3,
                "mode": "rnd",
            }  # invalid mode
            self.series_univ_det.window_transform(transforms=window_transformations)

        with pytest.raises(ValueError):
            window_transformations = {
                "function": "mean",
                "mode": "rolling",
                "window": 3,
                "center": "True",
            }  # forecating_safe=True vs center=True
            self.series_univ_det.window_transform(transforms=window_transformations)

        # keep_names and overlapping transforms
        with pytest.raises(ValueError) as err:
            window_transformations = [
                {
                    "function": "mean",
                    "mode": "rolling",
                    "window": 3,
                    "components": self.series_multi_det.components[:1],
                },
                {
                    "function": "median",
                    "mode": "rolling",
                    "window": 3,
                    "components": self.series_multi_det.components,
                },
            ]
            self.series_multi_det.window_transform(
                transforms=window_transformations, keep_names=True
            )
        assert str(err.value) == (
            "Cannot keep the original component names as some transforms are overlapping "
            "(applied to the same components). Set `keep_names` to `False`."
        )

        # keep_names and keep_non_transformed
        with pytest.raises(ValueError) as err:
            window_transformations = [
                {
                    "function": "mean",
                    "mode": "rolling",
                    "window": 3,
                    "components": self.series_multi_det.components[:1],
                },
            ]
            self.series_multi_det.window_transform(
                transforms=window_transformations,
                keep_names=True,
                keep_non_transformed=True,
            )
        assert str(err.value) == (
            "`keep_names = True` and `keep_non_transformed = True` cannot be used together."
        )

    def test_ts_windowtransf_output_series(self):
        # univariate deterministic input
        transforms = {"function": "sum", "mode": "rolling", "window": 1}
        transformed_ts = self.series_univ_det.window_transform(transforms=transforms)

        assert list(itertools.chain(*transformed_ts.values().tolist())) == list(
            itertools.chain(*self.series_univ_det.values().tolist())
        )
        assert transformed_ts.components.to_list() == [
            f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
            for comp in self.series_univ_det.components
        ]

        # test customized function name that overwrites the pandas builtin transformation
        transforms = {
            "function": "sum",
            "mode": "rolling",
            "window": 1,
            "function_name": "customized_name",
        }
        transformed_ts = self.series_univ_det.window_transform(transforms=transforms)
        assert transformed_ts.components.to_list() == [
            f"{transforms['mode']}_{transforms['function_name']}_{str(transforms['window'])}_{comp}"
            for comp in self.series_univ_det.components
        ]
        del transforms["function_name"]

        # multivariate deterministic input
        # transform one component
        transforms.update({"components": "0"})

        transformed_ts = self.series_multi_det.window_transform(transforms=transforms)
        assert transformed_ts.components.to_list() == [
            f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
            for comp in transforms["components"]
        ]

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )

        assert (
            transformed_ts.components.to_list()
            == [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ]
            + self.series_multi_det.components.to_list()
        )

        # transform multiple components
        transforms = {
            "function": "sum",
            "mode": "rolling",
            "window": 1,
            "components": ["0", "0_1"],
        }

        transformed_ts = self.series_multi_det.window_transform(transforms=transforms)
        assert transformed_ts.components.to_list() == [
            f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
            for comp in transforms["components"]
        ]

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )

        assert (
            transformed_ts.components.to_list()
            == [
                f"{transforms['mode']}_{transforms['function']}_{str(transforms['window'])}_{comp}"
                for comp in transforms["components"]
            ]
            + self.series_multi_det.components.to_list()
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
        assert transformed_ts.components.to_list() == [
            f"{transformation['mode']}_{transformation['function']}_{str(transformation['window'])}_{comp}"
            for transformation in transforms
            for comp in transformation["components"]
        ]

        transformed_ts = self.series_multi_det.window_transform(
            transforms=transforms, keep_non_transformed=True
        )
        assert (
            transformed_ts.components.to_list()
            == [
                f"{transformation['mode']}_{transformation['function']}_{str(transformation['window'])}_{comp}"
                for transformation in transforms
                for comp in transformation["components"]
            ]
            + self.series_multi_det.components.to_list()
        )

        # multivariate probabilistic input
        transformed_ts = self.series_multi_prob.window_transform(transforms=transforms)
        assert transformed_ts.n_samples == 2

    def test_user_defined_function_behavior(self):
        def count_above_mean(array):
            mean = np.mean(array)
            return np.where(array > mean)[0].size

        transformation = {
            "function": count_above_mean,
            "mode": "rolling",
            "window": 5,
        }
        transformed_ts = self.target.window_transform(
            transformation,
        )
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            np.array([0, 1, 1, 2, 2, 2, 2, 2, 2, 2]),
            columns=["rolling_udf_5_0"],
        )
        assert transformed_ts == expected_transformed_series

        # test if a customized function name is provided
        transformation.update({"function_name": "count_above_mean"})
        transformed_ts = self.target.window_transform(
            transformation,
        )
        assert transformed_ts.components.to_list() == [
            f"{transformation['mode']}_{transformation['function_name']}_{str(transformation['window'])}_{comp}"
            for comp in self.target.components
        ]

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
            np.array([100, 3, 6, 9, 12, 15, 18, 21, 24, 27]),
            columns=["rolling_sum_3_2_0"],
        )
        assert transformed_ts == expected_transformed_series

        # dropna
        transformed_ts = self.target.window_transform(
            window_transformations, treat_na="dropna"
        )
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times[1:],
            np.array([3, 6, 9, 12, 15, 18, 21, 24, 27]),
            columns=["rolling_sum_3_2_0"],
        )
        assert transformed_ts == expected_transformed_series

        # backfill na
        transformed_ts = self.target.window_transform(
            window_transformations, treat_na="bfill", forecasting_safe=False
        )
        # backfill works only with forecasting_safe=False
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            np.array([3, 3, 6, 9, 12, 15, 18, 21, 24, 27]),
            columns=["rolling_sum_3_2_0"],
        )
        assert transformed_ts == expected_transformed_series

        with pytest.raises(ValueError):
            # uknonwn treat_na
            self.target.window_transform(
                window_transformations, treat_na="fillrnd", forecasting_safe=False
            )

        with pytest.raises(ValueError):
            # unauhtorized treat_na=bfill with forecasting_safe=True
            self.target.window_transform(window_transformations, treat_na="bfill")

    def test_tranformed_ts_index(self):
        # DateTimeIndex
        transformed_series = self.target.window_transform({"function": "sum"})
        assert (
            self.target._time_index.__class__
            == transformed_series._time_index.__class__
        )
        # length index should not change for default transformation configurations
        assert len(self.target._time_index) == len(transformed_series._time_index)
        # RangeIndex
        transformed_series = self.series_from_values.window_transform({
            "function": "sum"
        })
        assert (
            self.series_from_values._time_index.__class__
            == transformed_series._time_index.__class__
        )
        assert len(self.series_from_values._time_index) == len(
            transformed_series._time_index
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
            np.array(["NaN", 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns=["rolling_sum_1_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        assert transformed_ts == expected_transformed_series

        # shift the index
        transformation = {"function": "sum", "mode": "rolling", "window": 1}
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            np.array(["NaN", 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            columns=["rolling_sum_1_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        assert transformed_ts == expected_transformed_series

        transformation = [
            {"function": "sum", "mode": "rolling", "window": 1, "closed": "left"},
            {"function": "sum", "mode": "ewm", "span": 1},
        ]
        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            np.array([
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
            ]),
            columns=["rolling_sum_1_0", "ewm_sum_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        assert transformed_ts == expected_transformed_series

        expected_transformed_series = TimeSeries.from_times_and_values(
            self.times,
            np.array([
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
            ]),
            columns=["rolling_sum_1_0", "ewm_sum_0"],
        )
        transformed_ts = self.target.window_transform(
            transformation,
            include_current=False,
            forecasting_safe=False,
            treat_na="bfill",
        )
        assert transformed_ts == expected_transformed_series

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
            np.array([
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
            ]),
            columns=["rolling_sum_2_2_0", "ewm_sum_2_0"],
        )

        transformed_ts = self.target.window_transform(
            transformation, include_current=False
        )
        assert transformed_ts == expected_transformed_series

    @pytest.mark.parametrize(
        "transforms",
        [
            {
                "function": "median",
                "mode": "rolling",
                "window": 3,
            },
            {
                "function": "mean",
                "mode": "expanding",
                "window": 2,
                "components": ["A", "B", "C"],
            },
        ],
    )
    def test_ts_windowtransf_hierarchy(self, transforms):
        """Checking that supported transforms behave as expected:
        - implicitly applied to all components
        - passing explicitly all components
        """
        ts = helper_generate_ts_hierarchy(10)

        # renaming components based on transform parameters
        ts_tr = ts.window_transform(transforms=transforms)
        tr_prefix = (
            f"{transforms['mode']}_{transforms['function']}_{transforms['window']}_"
        )
        assert ts_tr.hierarchy == {
            tr_prefix + comp: [tr_prefix + "A"] for comp in ["B", "C"]
        }

        # keeping original components name
        ts_tr = ts.window_transform(transforms=transforms, keep_names=True)
        assert ts_tr.hierarchy == ts.hierarchy == {"C": ["A"], "B": ["A"]}

    @pytest.mark.parametrize(
        "transforms",
        [
            {"function": "median", "mode": "rolling", "window": 3, "components": ["B"]},
            [
                {
                    "function": "mean",
                    "mode": "expanding",
                    "window": 2,
                },
                {
                    "function": "median",
                    "mode": "rolling",
                    "window": 3,
                },
            ],
            [
                {
                    "function": "median",
                    "mode": "rolling",
                    "window": 3,
                    "components": ["B", "C"],
                },
                {
                    "function": "sum",
                    "mode": "rolling",
                    "window": 5,
                    "components": ["A", "C"],
                },
            ],
        ],
    )
    def test_ts_windowtransf_drop_hierarchy(self, transforms):
        """Checking that hierarchy is correctly removed when
        - transform is not applied to all the components
        - several transforms applied to all the components
        - two transforms with overlapping components
        """
        ts = helper_generate_ts_hierarchy(10)
        ts_tr = ts.window_transform(transforms=transforms)
        assert ts_tr.hierarchy is None

    def test_ts_windowtransf_hierarchy_wrong_args(self):
        ts = helper_generate_ts_hierarchy(10)

        # hierarchy + keep_non_transformed = ambiguity for hierarchy
        with pytest.raises(ValueError):
            ts.window_transform(
                transforms={
                    "function": "sum",
                    "mode": "rolling",
                    "window": 3,
                },
                keep_non_transformed=True,
            )


class TestWindowTransformer:
    times = pd.date_range("20130101", "20130110")
    target = TimeSeries.from_times_and_values(times, np.array(range(1, 11)))
    times_hourly = pd.date_range(start="20130101", freq="1" + freqs["h"], periods=10)
    target_hourly = TimeSeries.from_times_and_values(
        times_hourly, np.array(range(1, 11))
    )

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

        assert len(transformed_ts_list) == 2
        assert transformed_ts_list[0].n_components == 2
        assert transformed_ts_list[0].n_timesteps == self.series_multi_det.n_timesteps
        assert transformed_ts_list[1].n_components == 5
        assert transformed_ts_list[1].n_timesteps == self.series_multi_det.n_timesteps

    def test_window_transformer_offset_parameter(self):
        """
        Test that the window parameter can support offset of pandas.Timedelta
        """
        base_parameters = {
            "function": "mean",
            "components": ["0"],
            "mode": "rolling",
        }

        offset_parameters = base_parameters.copy()
        offset_parameters.update({"window": pd.Timedelta(hours=4)})
        offset_transformer = WindowTransformer(
            transforms=offset_parameters,
        )
        offset_transformed = offset_transformer.transform(self.target_hourly)

        integer_parameters = base_parameters.copy()
        integer_parameters.update({"window": 4})
        integer_transformer = WindowTransformer(
            transforms=integer_parameters,
        )
        integer_transformed = integer_transformer.transform(self.target_hourly)
        np.testing.assert_equal(
            integer_transformed.values(), offset_transformed.values()
        )
        assert offset_transformed.components[0] == "rolling_mean_0 days 04:00:00_0"
        assert integer_transformed.components[0] == "rolling_mean_4_0"

        invalid_parameters = base_parameters.copy()
        invalid_parameters.update({"window": pd.DateOffset(hours=4)})
        invalid_transformer = WindowTransformer(
            transforms=invalid_parameters,
        )
        # if pd.DateOffset, raise ValueError of non-fixed frequency
        with pytest.raises(ValueError):
            invalid_transformer.transform(self.target_hourly)

    def test_transformers_pipeline(self):
        """
        Test that the forecasting window transformer can be used in a pipeline

        """

        times1 = pd.date_range("20130101", "20130110")
        series_1 = TimeSeries.from_times_and_values(times1, np.array(range(1, 11)))

        expected_transformed_series = TimeSeries.from_times_and_values(
            times1,
            np.array([100, 15, 30, 45, 60, 75, 90, 105, 120, 135]),
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

        assert transformed_series == expected_transformed_series

    def test_transformer_hierarchy(self):
        ts = helper_generate_ts_hierarchy(10)
        transform = {
            "function": "median",
            "mode": "rolling",
            "window": 3,
        }

        # renaming components
        window_transformer = WindowTransformer(
            transforms=[transform],
        )
        ts_tr = window_transformer.transform(ts)
        tr_prefix = (
            f"{transform['mode']}_{transform['function']}_{transform['window']}_"
        )
        assert ts_tr.hierarchy == {
            tr_prefix + comp: [tr_prefix + "A"] for comp in ["B", "C"]
        }
        # keeping old components
        window_transformer = WindowTransformer(
            transforms=transform,
            keep_names=True,
        )
        ts_tr = window_transformer.transform(ts)
        assert ts_tr.hierarchy == ts.hierarchy == {"C": ["A"], "B": ["A"]}
