import logging
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from darts import TimeSeries
from darts.config import option_context
from darts.tests.conftest import PLOTLY_AVAILABLE
from darts.utils.utils import generate_index

if not PLOTLY_AVAILABLE:
    pytest.skip(
        "Plotly not available. test_timeseries_plotly tests will be skipped.",
        allow_module_level=True,
    )

import plotly.graph_objects as go


class TestTimeSeriesPlotly:
    n_comps = 2
    n_samples = 5

    # datetime index, deterministic
    series_dt_d = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, n_comps, 1)),
    )
    # datetime index, probabilistic
    series_dt_p = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, n_comps, n_samples)),
    )
    # range index, deterministic
    series_ri_d = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, n_comps, 1)),
    )
    # range index, probabilistic
    series_ri_p = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, n_comps, n_samples)),
    )

    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
            [True, False],
            [True, False],
        ),
    )
    def test_plotly_single_series(self, config):
        index_type, stoch_type, use_fig, use_darts_style = config
        with option_context("plotting.use_darts_style", use_darts_style):
            series = getattr(self, f"series_{index_type}_{stoch_type}")

            fig = series.plotly(fig=go.Figure() if use_fig else None)

            # Total Trace Count
            # Deterministic: 1 per comp
            # Probabilistic: 3 per comp (Low, Central, High)
            expected_total = self.n_comps * (3 if series.is_stochastic else 1)
            assert len(fig.data) == expected_total

            # Count Central Lines
            central_lines = [
                t
                for t in fig.data
                if (t.fill is None or t.fill == "none") and t.showlegend is not False
            ]
            assert len(central_lines) == self.n_comps

            # Count Confidence Interval Components
            if series.is_stochastic:
                # Low bounds (no fill, but hidden from legend)
                low_bounds = [
                    t
                    for t in fig.data
                    if (t.fill is None or t.fill == "none") and t.showlegend is False
                ]
                # High bounds (fill='tonexty', hidden from legend)
                high_bounds = [
                    t for t in fig.data if t.fill == "tonexty" and t.showlegend is False
                ]

                assert len(low_bounds) == self.n_comps
                assert len(high_bounds) == self.n_comps

    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
        ),
    )
    def test_plotly_point_series(self, config):
        index_type, stoch_type = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        # Slice to length 1
        series = series[:1]

        fig = series.plotly()

        # We expect exactly 1 trace per component
        assert len(fig.data) == self.n_comps

        for i, trace in enumerate(fig.data):
            # Check that it's a single point
            assert len(trace.x) == 1

            if series.is_stochastic:
                # Verify error_y exists and is configured correctly
                assert "error_y" in trace
                assert trace.error_y.type == "data"
                assert trace.error_y.symmetric is False

                # Extract actual values from series for comparison
                vals = series.all_values()
                central_val = np.median(vals[0, i, :])
                low_val = np.quantile(vals[0, i, :], 0.05)
                high_val = np.quantile(vals[0, i, :], 0.95)

                expected_upper_delta = high_val - central_val
                expected_lower_delta = central_val - low_val

                assert np.isclose(trace.error_y.array[0], expected_upper_delta)
                assert np.isclose(trace.error_y.arrayminus[0], expected_lower_delta)
            else:
                # Deterministic series should not have error bars
                if hasattr(trace, "error_y") and trace.error_y:
                    assert trace.error_y.visible is False or trace.error_y.array is None

    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
        ),
    )
    def test_plotly_empty_series(self, config):
        index_type, stoch_type = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        # Slice to length 0
        series = series[:0]

        fig = series.plotly()

        # If there are traces, verify they are empty
        for trace in fig.data:
            # Verify x and y are empty (can be empty list, tuple, or numpy array)
            assert len(trace.x) == 0
            assert len(trace.y) == 0

            # If stochastic, check that error bars aren't populated either
            if hasattr(trace, "error_y") and trace.error_y:
                if trace.error_y.array is not None:
                    assert len(trace.error_y.array) == 0

    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
            [
                {"title": "my title"},
                {"label": "comps"},
                {"label": ["comps_1", "comps_2"]},
                {"alpha": 0.1, "color": "blue"},
                {"opacity": 0.4, "color": "blue"},
                {"color": ["blue", "red"]},
                {"line_width": 3},
            ],
        ),
    )
    def test_plotly_params(self, config):
        index_type, stoch_type, kwargs = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")

        fig = series.plotly(**kwargs)

        # Filter for the main component traces
        central_traces = [t for t in fig.data if t.showlegend is not False]

        for i, trace in enumerate(central_traces):
            # Labels
            if "label" in kwargs:
                label = kwargs["label"]
                if isinstance(label, list):
                    assert trace.name == label[i]
                else:
                    assert label in trace.name

            # Colors
            if "color" in kwargs:
                color = kwargs["color"]
                actual_color = trace.line.color
                if isinstance(color, list):
                    assert actual_color == color[i]
                else:
                    assert actual_color == color

            # Opacity
            if "opacity" in kwargs:
                assert trace.opacity == kwargs["opacity"]

            # Line Width
            if "line_width" in kwargs:
                assert trace.line.width == kwargs["line_width"]

        # Alpha
        if series.is_stochastic and "alpha" in kwargs:
            # Find the fill traces
            fill_traces = [t for t in fig.data if t.fill == "tonexty"]
            for t in fill_traces:
                # Check if the alpha is present in the fillcolor string
                assert str(kwargs["alpha"]) in t.fillcolor

    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            [
                {"central_quantile": "mean"},
                {"central_quantile": 0.5},
                {
                    "low_quantile": 0.2,
                    "central_quantile": 0.6,
                    "high_quantile": 0.7,
                    "alpha": 0.1,
                },
            ],
        ),
    )
    def test_plotly_stochastic_params(self, config):
        index_type, kwargs = config
        stoch_type = "p"
        series = getattr(self, f"series_{index_type}_{stoch_type}")

        fig = series.plotly(**kwargs)

        assert isinstance(fig, go.Figure)

        # Basic trace structure verification for stochastic series:
        # Each component should have 3 traces: [Low, High, Central]
        expected_total_traces = self.n_comps * 3
        assert len(fig.data) == expected_total_traces

        low_trace = fig.data[0]
        high_trace = fig.data[0]
        low_trace = fig.data[1]
        central_trace = fig.data[2]

        # Check functional Plotly attributes
        assert high_trace.showlegend is False
        assert low_trace.fill == "tonexty"
        assert central_trace.showlegend is not False

    @pytest.mark.parametrize("config", ["dt", "ri"])
    def test_plotly_multiple_series(self, config):
        index_type = config
        series1 = getattr(self, f"series_{index_type}_d")
        series2 = getattr(self, f"series_{index_type}_p")

        # Call 1: Create new figure
        fig = series1.plotly(label="first")

        # Call 2: Pass existing figure to overlay
        fig = series2.plotly(fig=fig, label="second")

        assert isinstance(fig, go.Figure)

        # Trace Count Breakdown:
        # series1 (deterministic): 1 trace per component = 2 traces
        # series2 (probabilistic): 3 traces per component (Low, High, Central) = 6 traces
        # Total expected = 2 + 6 = 8 traces
        expected_total_traces = (self.n_comps * 1) + (self.n_comps * 3)
        assert len(fig.data) == expected_total_traces

        # Verify that labels are applied correctly to identify the different series
        trace_names = [t.name for t in fig.data if t.name is not None]

        # Check that labels from both calls exist
        assert any("first" in name for name in trace_names)
        assert any("second" in name for name in trace_names)

    @pytest.mark.parametrize(
        "n_points, threshold, expected_stride",
        [
            (100, 200, 1),  # Case 1: Under threshold, no downsampling
            (100, 75, 2),  # Case 2: Slightly over (100/2 = 50 <= 75), stride=2
            (100, 20, 8),  # Case 3: Way over (100/8 = 12.5 <= 20), next power of 2 is 8
            (100, -1, 1),  # Case 4: Downsampling disabled via -1
        ],
    )
    def test_plotly_downsampling_logic(self, n_points, threshold, expected_stride):
        # Create a univariate deterministic series
        series = TimeSeries.from_times_and_values(
            times=generate_index(start="2000-01-01", length=n_points),
            values=np.random.random((n_points, 1)),
        )

        # Execute plotly with the specific threshold
        fig = series.plotly(downsample_threshold=threshold)

        # Verify the number of points in the trace
        # The expected length is ceil(n_points / expected_stride)
        expected_len = int(np.ceil(n_points / expected_stride))

        actual_len = len(fig.data[0].x)
        assert actual_len == expected_len, (
            f"Downsampling failed for threshold {threshold}. "
            f"Expected {expected_len} points (stride {expected_stride}), got {actual_len}."
        )

    def test_plotly_downsampling_multivariate_stochastic(self):
        """
        Verify that downsampling accounts for total points:
        points = time_steps * components * traces_per_component
        """
        n_points = 100
        n_components = 2
        # Stochastic series has 3 traces per component (Low, High, Central)
        series = TimeSeries.from_times_and_values(
            times=generate_index(start="2000-01-01", length=n_points),
            values=np.random.random((n_points, n_components, 5)),
        )

        # Total points = 100 * 2 * 3 = 600
        # To get below 100, we need 600 / n <= 100 -> n >= 6.
        # Lowest power of two is 8.
        fig = series.plotly(downsample_threshold=100)

        # Check the length of any trace (they should all be downsampled by the same stride)
        expected_len = int(np.ceil(100 / 8))
        assert len(fig.data[0].x) == expected_len

    @patch("builtins.__import__")
    def test_plotly_import_error(self, mock_import):
        """
        Tests that a helpful ImportError is raised when plotly is not installed.
        """

        # Define a side effect for the mock import
        def side_effect(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("mocked plotly import error")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        # The TimeSeries.plotly() method should catch the internal ImportError
        # and re-raise the specific Darts error message.
        with pytest.raises(
            ImportError, match="Plotly is not installed. Please install it with:"
        ):
            self.series_dt_d.plotly()

    def test_plotly_param_validation(self):
        """
        Final coverage targets: color type validation, sequence length validation,
        stochastic ribbons, opacity modification, and univariate labeling.
        """
        # 1. Trigger: Invalid color type (integer instead of string/sequence)
        # Hits: raise_log(ValueError("`color` and `c` must be a string..."))
        with pytest.raises(ValueError, match="`color` and `c` must be a string"):
            self.series_dt_d.plotly(color=123)

        # 2. Trigger: Color sequence length mismatch
        # (Assuming series has 2 components, we pass 3 colors)
        with pytest.raises(ValueError, match="The `color` sequence length.*is invalid"):
            self.series_dt_d.plotly(color=["red", "blue", "green"])

        # 3. Trigger: Partial quantiles (low_quantile=None) -> No ribbon logic
        fig_no_ribbon = self.series_dt_p.plotly(low_quantile=None)
        assert len(fig_no_ribbon.data) == self.n_comps * 1
        assert not any(t.fill == "tonexty" for t in fig_no_ribbon.data)

        # 4. Trigger: Successful ribbon (both quantiles) + _modify_color_opacity (rgb string)
        # Hits the regex logic: if '(' in color: ...
        fig_ribbon = self.series_dt_p.plotly(
            low_quantile=0.2, high_quantile=0.8, color="rgb(255, 0, 0)"
        )
        assert len(fig_ribbon.data) == self.n_comps * 3
        # Robust check for rgba conversion
        fill_colors = [
            str(t.fillcolor).replace(" ", "") for t in fig_ribbon.data if t.fillcolor
        ]
        assert any("rgba(255,0,0," in c for c in fill_colors)

        # 5. Trigger: Univariate label branch (n_components == 1)
        series_univ = self.series_dt_d[self.series_dt_d.columns[0]]
        fig_univ = series_univ.plotly(label="univariate_test")
        assert fig_univ.data[0].name == "univariate_test"

    def test_plotly_prepare_plot_params(self, caplog):
        """
        Covers all branches in _prepare_plot_params including errors, warnings,
        and label/quantile logic.
        """
        # 1. Quantile Validation Errors
        with pytest.raises(ValueError, match='central_quantile must be either "mean"'):
            self.series_dt_d.plotly(central_quantile=1.5)

        with pytest.raises(
            ValueError, match="confidence interval low and high quantiles"
        ):
            self.series_dt_p.plotly(low_quantile=-0.1, high_quantile=0.5)

        # 2. Component Slicing & Warning (max_nr_components)
        with caplog.at_level(logging.WARNING):
            fig = self.series_dt_d.plotly(max_nr_components=1)
            assert "is larger than the maximum number of components" in caplog.text
            assert len(fig.data) == 1

        # 3. Label Resolution: Sequence Length Error
        # Passing 3 labels for a 2-component series
        with pytest.raises(
            ValueError, match="The `label` sequence must have the same length"
        ):
            self.series_dt_d.plotly(label=["A", "B", "C"])

        # 4. Label Resolution Branches (Loop logic)
        # Branch: custom_labels (Sequence of strings)
        fig_seq = self.series_dt_d.plotly(label=["custom1", "custom2"])
        assert fig_seq.data[0].name == "custom1"
        # Branch: label == "" (Empty string triggers component names)
        fig_empty = self.series_dt_d.plotly(label="")
        assert fig_empty.data[0].name == self.series_dt_d.components[0]
        # Branch: self.n_components == 1 (Univariate)
        # We slice to 1 component; label should be exactly the string passed
        series_uni = self.series_dt_d[self.series_dt_d.components[0]]
        fig_uni = series_uni.plotly(label="single_label")
        assert fig_uni.data[0].name == "single_label"
        # Branch: else (Multivariate default: {label}_{comp_name})
        fig_multi = self.series_dt_d.plotly(label="prefix")
        expected_name = f"prefix_{self.series_dt_d.components[0]}"
        assert fig_multi.data[0].name == expected_name

        # 5. Color resolution: Simultaneous use (Already in param_validation, but here for completeness)
        with pytest.raises(ValueError, match="must not be used simultaneously"):
            self.series_dt_d.plotly(color="red", c="blue")

        # 6. Color resolution: Using 'c' instead of 'color'
        fig_c = self.series_dt_d.plotly(c="green")
        # Check if the trace color is green (Plotly stores this in marker or line)
        assert fig_c.data[0].line.color == "green"
