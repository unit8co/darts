import textwrap

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries

MAX_ROWS = 10
MAX_COLS = 10
MAX_ITEMS = 5
MARGIN = 2


def helper_generate_values(shape: tuple[int, int, int]):
    values = np.arange(shape[0])[:, np.newaxis, np.newaxis]
    if shape[1] > 1:
        values = np.concatenate([values + 5 * i for i in range(shape[1])], axis=1)
    if shape[2] > 1:
        values = np.concatenate([values + 5 * i for i in range(shape[2])], axis=2)
    return values


class TestTimeSeriesReprFormatting:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures for TimeSeries representation tests."""
        self.times = pd.date_range("20130101", "20130110", freq="D")
        self.pd_series = pd.Series(range(MAX_ROWS), index=self.times)
        self.simple_series = TimeSeries.from_series(self.pd_series)

    def test_str_simple_deterministic_series(self):
        """Test __str__ output for a simple deterministic time series."""
        expected = textwrap.dedent(
            """\
                          0
            2013-01-01  0.0
            2013-01-02  1.0
            2013-01-03  2.0
            2013-01-04  3.0
            2013-01-05  4.0
            2013-01-06  5.0
            2013-01-07  6.0
            2013-01-08  7.0
            2013-01-09  8.0
            2013-01-10  9.0

            shape: (10, 1, 1), freq: D, size: 80.00 B
            """
        ).rstrip()
        assert str(self.simple_series) == expected

    def test_str_stochastic_series(self):
        """Test __str__ for stochastic series displays median."""
        values = helper_generate_values((10, 1, 3))
        series = TimeSeries(times=self.times, values=values, components=["value"])

        expected = textwrap.dedent(
            """\
                        value
            2013-01-01    5.0
            2013-01-02    6.0
            2013-01-03    7.0
            2013-01-04    8.0
            2013-01-05    9.0
            2013-01-06   10.0
            2013-01-07   11.0
            2013-01-08   12.0
            2013-01-09   13.0
            2013-01-10   14.0

            shape: (10, 1, 3), freq: D, size: 240.00 B
            info: only sample median was displayed
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_multivariate_series(self):
        """Test __str__ for multivariate time series."""
        values = helper_generate_values((MAX_ROWS, 3, 1))
        series = TimeSeries(times=self.times, values=values, components=["a", "b", "c"])

        expected = textwrap.dedent(
            """\
                          a     b     c
            2013-01-01  0.0   5.0  10.0
            2013-01-02  1.0   6.0  11.0
            2013-01-03  2.0   7.0  12.0
            2013-01-04  3.0   8.0  13.0
            2013-01-05  4.0   9.0  14.0
            2013-01-06  5.0  10.0  15.0
            2013-01-07  6.0  11.0  16.0
            2013-01-08  7.0  12.0  17.0
            2013-01-09  8.0  13.0  18.0
            2013-01-10  9.0  14.0  19.0

            shape: (10, 3, 1), freq: D, size: 240.00 B
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_with_large_series(self):
        """Test __str__ with time series larger than max rows and columns."""
        times = pd.date_range("20130101", periods=MAX_ROWS + 1, freq="D")

        values = np.arange(len(times))[:, np.newaxis]
        values = np.concatenate([values + 5 * i for i in range(MAX_COLS + 1)], axis=1)
        series = TimeSeries(
            times=times,
            values=values,
            components=[f"c{i}" for i in range(values.shape[1])],
        )

        expected = textwrap.dedent(
            """\
                          c0    c1    c2    c3    c4  ...    c6    c7    c8    c9   c10
            2013-01-01   0.0   5.0  10.0  15.0  20.0  ...  30.0  35.0  40.0  45.0  50.0
            2013-01-02   1.0   6.0  11.0  16.0  21.0  ...  31.0  36.0  41.0  46.0  51.0
            2013-01-03   2.0   7.0  12.0  17.0  22.0  ...  32.0  37.0  42.0  47.0  52.0
            2013-01-04   3.0   8.0  13.0  18.0  23.0  ...  33.0  38.0  43.0  48.0  53.0
            2013-01-05   4.0   9.0  14.0  19.0  24.0  ...  34.0  39.0  44.0  49.0  54.0
            ...          ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...
            2013-01-07   6.0  11.0  16.0  21.0  26.0  ...  36.0  41.0  46.0  51.0  56.0
            2013-01-08   7.0  12.0  17.0  22.0  27.0  ...  37.0  42.0  47.0  52.0  57.0
            2013-01-09   8.0  13.0  18.0  23.0  28.0  ...  38.0  43.0  48.0  53.0  58.0
            2013-01-10   9.0  14.0  19.0  24.0  29.0  ...  39.0  44.0  49.0  54.0  59.0
            2013-01-11  10.0  15.0  20.0  25.0  30.0  ...  40.0  45.0  50.0  55.0  60.0

            shape: (11, 11, 1), freq: D, size: 968.00 B
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_with_large_series_efficient(self):
        """Test __str__ with time series larger than (max rows + 2 * margin) which activates a more
        efficient routine."""
        times = pd.date_range("20130101", periods=MAX_ROWS + 2 * MARGIN + 1, freq="D")

        values = np.arange(len(times))[:, np.newaxis]
        values = np.concatenate(
            [values + 5 * i for i in range(MAX_COLS + 2 * MARGIN + 1)], axis=1
        )
        series = TimeSeries(
            times=times,
            values=values,
            components=[f"c{i}" for i in range(values.shape[1])],
        )

        expected = textwrap.dedent(
            """\
                          c0    c1    c2    c3    c4  ...   c10   c11   c12   c13   c14
            2013-01-01   0.0   5.0  10.0  15.0  20.0  ...  50.0  55.0  60.0  65.0  70.0
            2013-01-02   1.0   6.0  11.0  16.0  21.0  ...  51.0  56.0  61.0  66.0  71.0
            2013-01-03   2.0   7.0  12.0  17.0  22.0  ...  52.0  57.0  62.0  67.0  72.0
            2013-01-04   3.0   8.0  13.0  18.0  23.0  ...  53.0  58.0  63.0  68.0  73.0
            2013-01-05   4.0   9.0  14.0  19.0  24.0  ...  54.0  59.0  64.0  69.0  74.0
            ...          ...   ...   ...   ...   ...  ...   ...   ...   ...   ...   ...
            2013-01-11  10.0  15.0  20.0  25.0  30.0  ...  60.0  65.0  70.0  75.0  80.0
            2013-01-12  11.0  16.0  21.0  26.0  31.0  ...  61.0  66.0  71.0  76.0  81.0
            2013-01-13  12.0  17.0  22.0  27.0  32.0  ...  62.0  67.0  72.0  77.0  82.0
            2013-01-14  13.0  18.0  23.0  28.0  33.0  ...  63.0  68.0  73.0  78.0  83.0
            2013-01-15  14.0  19.0  24.0  29.0  34.0  ...  64.0  69.0  74.0  79.0  84.0

            shape: (15, 15, 1), freq: D, size: 1.76 KB
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_with_nan_values(self):
        """Test __str__ with NaN values."""
        times = pd.date_range("20130101", "20130105", freq="D")
        values = np.array([[1.0], [np.nan], [3.0], [4.0], [np.nan]]).reshape(5, 1, 1)
        series = TimeSeries(times=times, values=values, components=["value"])

        expected = textwrap.dedent(
            """\
                        value
            2013-01-01    1.0
            2013-01-02    NaN
            2013-01-03    3.0
            2013-01-04    4.0
            2013-01-05    NaN

            shape: (5, 1, 1), freq: D, size: 40.00 B
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_with_static_covariates(self):
        """Test __str__ output for a series with static covariates."""
        static_cov_df = pd.DataFrame({
            "sc1": [1.0],
            "sc2": ["example_value"],
        })
        series_with_sc = self.simple_series.with_static_covariates(static_cov_df)

        expected = textwrap.dedent(
            """\
                          0
            2013-01-01  0.0
            2013-01-02  1.0
            2013-01-03  2.0
            2013-01-04  3.0
            2013-01-05  4.0
            2013-01-06  5.0
            2013-01-07  6.0
            2013-01-08  7.0
            2013-01-09  8.0
            2013-01-10  9.0

            shape: (10, 1, 1), freq: D, size: 80.00 B

            Static covariates:
                static_covariates  sc1            sc2
                0                  1.0  example_value
            """
        ).rstrip()
        assert str(series_with_sc) == expected

    def test_str_with_hierarchy(self):
        """Test __str__ output for a series with hierarchy."""
        # multivariate series with 2 components
        values = helper_generate_values((10, 2, 1))
        series = TimeSeries(
            times=self.times,
            values=values,
            components=["total", "a"],
        )

        # hierarchy - "a" is child of total
        hierarchy = {"total": ["a"]}
        series_with_hierarchy = series.with_hierarchy(hierarchy)

        expected = textwrap.dedent(
            """\
                        total     a
            2013-01-01    0.0   5.0
            2013-01-02    1.0   6.0
            2013-01-03    2.0   7.0
            2013-01-04    3.0   8.0
            2013-01-05    4.0   9.0
            2013-01-06    5.0  10.0
            2013-01-07    6.0  11.0
            2013-01-08    7.0  12.0
            2013-01-09    8.0  13.0
            2013-01-10    9.0  14.0

            shape: (10, 2, 1), freq: D, size: 160.00 B

            Hierarchy:
                total         ['a']
            """
        ).rstrip()
        assert str(series_with_hierarchy) == expected

    def test_str_with_metadata(self):
        """Test __str__ output for a series with metadata."""
        metadata = {
            "source": "test_data",
            "version": "1.0",
            "created": "2024-01-15",
        }
        series_with_metadata = self.simple_series.with_metadata(metadata)

        expected = textwrap.dedent(
            """\
                          0
            2013-01-01  0.0
            2013-01-02  1.0
            2013-01-03  2.0
            2013-01-04  3.0
            2013-01-05  4.0
            2013-01-06  5.0
            2013-01-07  6.0
            2013-01-08  7.0
            2013-01-09  8.0
            2013-01-10  9.0

            shape: (10, 1, 1), freq: D, size: 80.00 B

            Metadata:
                source        test_data
                version       1.0
                created       2024-01-15
            """
        ).rstrip()
        assert str(series_with_metadata) == expected

    def test_str_with_all_features(self):
        """Test __str__ output for a series with all optional features."""
        # multivariate series
        values = helper_generate_values((10, 2, 1))
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

        expected = textwrap.dedent(
            """\
                        total     a
            2013-01-01    0.0   5.0
            2013-01-02    1.0   6.0
            2013-01-03    2.0   7.0
            2013-01-04    3.0   8.0
            2013-01-05    4.0   9.0
            2013-01-06    5.0  10.0
            2013-01-07    6.0  11.0
            2013-01-08    7.0  12.0
            2013-01-09    8.0  13.0
            2013-01-10    9.0  14.0

            shape: (10, 2, 1), freq: D, size: 160.00 B

            Static covariates:
                static_covariates  region
                global_components  Europe
            Hierarchy:
                total         ['a']
            Metadata:
                source        test
                region        EU
            """
        ).rstrip()
        assert str(series) == expected

    def test_str_with_many_metadata_keys(self):
        """Test __str__ output properly truncates metadata keys."""
        metadata = {str(i): str(i + 1) for i in range(MAX_ITEMS + 1)}
        series_with_metadata = self.simple_series.with_metadata(metadata)

        expected = textwrap.dedent(
            """\
                          0
            2013-01-01  0.0
            2013-01-02  1.0
            2013-01-03  2.0
            2013-01-04  3.0
            2013-01-05  4.0
            2013-01-06  5.0
            2013-01-07  6.0
            2013-01-08  7.0
            2013-01-09  8.0
            2013-01-10  9.0

            shape: (10, 1, 1), freq: D, size: 80.00 B

            Metadata:
                0             1
                1             2
                2             3
                3             4
                ...           ...
                5             6
            """
        ).rstrip()
        assert str(series_with_metadata) == expected

    def test_str_with_large_metadata_names(self):
        """Test __str__ output properly truncates metadata."""
        metadata = {"this is a very long name": "1"}
        series_with_metadata = self.simple_series.with_metadata(metadata)

        expected = textwrap.dedent(
            """\
                          0
            2013-01-01  0.0
            2013-01-02  1.0
            2013-01-03  2.0
            2013-01-04  3.0
            2013-01-05  4.0
            2013-01-06  5.0
            2013-01-07  6.0
            2013-01-08  7.0
            2013-01-09  8.0
            2013-01-10  9.0

            shape: (10, 1, 1), freq: D, size: 80.00 B

            Metadata:
                this is a...  1
            """
        ).rstrip()
        assert str(series_with_metadata) == expected

    def test_repr_matches_str(self):
        """Test that __repr__ delegates to __str__."""
        assert str(self.simple_series) == repr(self.simple_series)

    def test_repr_html_basic(self):
        """Test _repr_html_ produces HTML output."""
        result = self.simple_series._repr_html_()
        assert isinstance(result, str)
        assert "<table" in result
        assert "shape: (10, 1, 1), freq: D, size: 80.00 B" in result
        assert "<th>" in result and "</th>" in result
        assert "<td>" in result and "</td>" in result

    def test_repr_html_with_stochastic_series(self):
        """Test _repr_html_ for stochastic series."""
        values = helper_generate_values((10, 1, 3))
        series = TimeSeries(times=self.times, values=values, components=["value"])
        result = series._repr_html_()
        assert isinstance(result, str)
        assert "shape: (10, 1, 3), freq: D, size: 240.00 B" in result
        assert "info: only sample median was displayed" in result

    def test_repr_html_no_additional_data(self):
        """Test _repr_html_ does not include sections if no additional data is given (static covs, ...)."""
        result = self.simple_series._repr_html_()
        assert "Static covariates" not in result
        assert "Hierarchy" not in result
        assert "Metadata" not in result

    def test_repr_html_includes_subsections(self):
        """Test _repr_html_ includes sections if additional data is given (static covs, ...)."""
        # multivariate series
        values = helper_generate_values((10, 2, 1))
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
        result = series._repr_html_()
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
