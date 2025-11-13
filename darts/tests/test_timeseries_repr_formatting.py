import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries


class TestTimeSeriesReprFormatting:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures for TimeSeries representation tests."""
        self.times = pd.date_range("20130101", "20130110", freq="D")
        self.pd_series = pd.Series(range(10), index=self.times)
        self.simple_series = TimeSeries.from_series(self.pd_series)

    def test_str_simple_deterministic_series(self):
        """Test __str__ output for a simple deterministic time series."""
        result = str(self.simple_series)
        lines = result.split("\n")

        assert isinstance(result, str)

        props_idx = [i for i, line in enumerate(lines) if line == "Properties:"][0]
        assert (
            "Shape         (times: 10, components: 1, samples: 1)"
            in lines[props_idx + 1]
        )
        assert "Time frame" in lines[props_idx + 2]
        assert "2013-01-01" in lines[props_idx + 2]
        assert "2013-01-10" in lines[props_idx + 2]
        assert "Size" in lines[props_idx + 3] and "B" in lines[props_idx + 3]

        assert "Static covariates:" in result
        assert "Hierarchy:" in result
        assert "Metadata:" in result

    def test_str_stochastic_series(self):
        """Test __str__ for stochastic series displays median."""
        values = np.random.rand(10, 1, 3)
        series = TimeSeries(times=self.times, values=values, components=["value"])
        result = str(series)
        lines = result.split("\n")

        props_idx = [i for i, line in enumerate(lines) if line == "Properties:"][0]
        assert (
            "Shape         (times: 10, components: 1, samples: 3)"
            in lines[props_idx + 1]
        )

        assert "(displaying median of samples):" in result

    def test_str_multivariate_series(self):
        """Test __str__ for multivariate time series."""
        values = np.random.rand(10, 3, 1)
        series = TimeSeries(times=self.times, values=values, components=["a", "b", "c"])
        result = str(series)
        lines = result.split("\n")

        props_idx = [i for i, line in enumerate(lines) if line == "Properties:"][0]
        assert (
            "Shape         (times: 10, components: 3, samples: 1)"
            in lines[props_idx + 1]
        )

        df_header = [
            line for line in lines if "a" in line and "b" in line and "c" in line
        ]
        assert len(df_header) > 0

    def test_str_with_static_covariates(self):
        """Test __str__ output for a series with static covariates."""
        static_cov_df = pd.DataFrame({
            "sc1": [1.0],
            "sc2": ["example_value"],
        })
        series_with_sc = self.simple_series.with_static_covariates(static_cov_df)

        result = str(series_with_sc)
        lines = result.split("\n")

        assert "Static covariates:" in result
        static_cov_idx = [
            i for i, line in enumerate(lines) if line == "Static covariates:"
        ][0]
        assert "sc1" in lines[static_cov_idx + 1]
        assert "sc2" in lines[static_cov_idx + 1]
        assert "1.0" in lines[static_cov_idx + 2]
        assert "example_value" in lines[static_cov_idx + 2]

    def test_str_with_hierarchy(self):
        """Test __str__ output for a series with hierarchy."""
        # multivariate series with 2 components
        values = np.random.rand(10, 2, 1)
        series = TimeSeries(
            times=self.times,
            values=values,
            components=["total", "a"],
        )

        # hierarchy - a is child of total
        hierarchy = {"total": ["a"]}
        series_with_hierarchy = series.with_hierarchy(hierarchy)

        result = str(series_with_hierarchy)
        lines = result.split("\n")

        assert "Hierarchy:" in result
        hierarchy_idx = [i for i, line in enumerate(lines) if line == "Hierarchy:"][0]
        hierarchy_line = lines[hierarchy_idx + 1]
        assert "total" in hierarchy_line
        assert "['a']" in hierarchy_line

    def test_str_with_metadata(self):
        """Test __str__ output for a series with metadata."""
        metadata = {
            "source": "test_data",
            "version": "1.0",
            "created": "2024-01-15",
        }
        series_with_metadata = self.simple_series.with_metadata(metadata)

        result = str(series_with_metadata)
        lines = result.split("\n")

        assert "Metadata:" in result
        metadata_idx = [i for i, line in enumerate(lines) if line == "Metadata:"][0]

        metadata_lines = lines[metadata_idx + 1 : metadata_idx + 4]
        metadata_text = "\n".join(metadata_lines)

        assert "source" in metadata_text and "test_data" in metadata_text
        assert "version" in metadata_text and "1.0" in metadata_text
        assert "created" in metadata_text and "2024-01-15" in metadata_text

    def test_str_with_all_features(self):
        """Test __str__ output for a series with all optional features."""
        # multivariate series
        values = np.random.rand(10, 2, 1)
        series = TimeSeries(
            times=self.times,
            values=values,
            components=["total", "a"],
        )

        # add all optional features
        static_cov_df = pd.DataFrame({
            "region": ["Europe"],
        })
        hierarchy = {"total": ["a"]}
        metadata = {
            "source": "test",
            "region": "EU",
        }

        series = (
            series.with_static_covariates(static_cov_df)
            .with_hierarchy(hierarchy)
            .with_metadata(metadata)
        )

        result = str(series)
        lines = result.split("\n")

        section_order = []
        for line in lines:
            if line == "Properties:":
                section_order.append("properties")
            elif line == "Static covariates:":
                section_order.append("static")
            elif line == "Hierarchy:":
                section_order.append("hierarchy")
            elif line == "Metadata:":
                section_order.append("metadata")
        assert section_order == ["properties", "static", "hierarchy", "metadata"]

        assert "Shape         (times: 10, components: 2, samples: 1)" in result
        assert "region" in result and "Europe" in result
        assert "total" in result and "['a']" in result
        assert "source" in result and "test" in result

    def test_repr_matches_str(self):
        """Test that __repr__ delegates to __str__."""
        assert str(self.simple_series) == repr(self.simple_series)

    def test_repr_is_string(self):
        """Test that __repr__ returns a string."""
        result = repr(self.simple_series)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_repr_html_basic(self):
        """Test _repr_html_ produces HTML output."""
        result = self.simple_series._repr_html_()
        assert isinstance(result, str)
        assert "<table" in result
        assert "<strong>" in result
        assert "Properties" in result
        assert "Shape" in result
        assert "(times: 10, components: 1, samples: 1)" in result
        assert "<th>" in result and "</th>" in result
        assert "<td>" in result and "</td>" in result

    def test_repr_html_with_stochastic_series(self):
        """Test _repr_html_ for stochastic series."""
        values = np.random.rand(10, 1, 3)
        series = TimeSeries(times=self.times, values=values, components=["value"])
        result = series._repr_html_()
        assert isinstance(result, str)
        assert "(times: 10, components: 1, samples: 3)" in result
        assert "displaying median of samples" in result

    def test_repr_html_includes_sections(self):
        """Test _repr_html_ includes all required sections."""
        result = self.simple_series._repr_html_()
        assert "Properties" in result
        assert "Static covariates" in result
        assert "Hierarchy" in result
        assert "Metadata" in result

    def test_str_does_not_modify_series(self):
        """Test that __str__ does not modify the series."""
        original_values = self.simple_series.values()
        _ = str(self.simple_series)
        assert np.array_equal(original_values, self.simple_series.values())

    def test_repr_html_does_not_modify_series(self):
        """Test that _repr_html_ does not modify the series."""
        original_values = self.simple_series.values()
        _ = self.simple_series._repr_html_()
        assert np.array_equal(original_values, self.simple_series.values())


class TestTimeSeriesFormattingEdgeCases:
    """Test edge cases for TimeSeries formatting."""

    def test_str_with_nan_values(self):
        """Test __str__ with NaN values."""
        times = pd.date_range("20130101", "20130105", freq="D")
        values = np.array([[1.0], [np.nan], [3.0], [4.0], [np.nan]]).reshape(5, 1, 1)
        series = TimeSeries(times=times, values=values, components=["value"])
        result = str(series)
        assert isinstance(result, str)
        assert "Properties:" in result

    def test_str_with_very_large_series(self):
        """Test __str__ with large time series."""
        times = pd.date_range("20130101", periods=1000, freq="D")
        values = np.random.rand(1000, 5, 1)
        series = TimeSeries(
            times=times, values=values, components=[f"c{i}" for i in range(5)]
        )
        result = str(series)
        assert isinstance(result, str)
        assert "times: 1000" in result
        assert "components: 5" in result
