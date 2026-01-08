from itertools import product
from unittest.mock import patch

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, option_context
from darts.utils.utils import generate_index


class TestTimeSeriesPlot:
    # datetime index, deterministic
    n_comps = 2
    series_dt_d = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, n_comps, 1)),
    )
    # datetime index, probabilistic
    series_dt_p = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, n_comps, 5)),
    )
    # range index, deterministic
    series_ri_d = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, n_comps, 1)),
    )
    # range index, probabilistic
    series_ri_p = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, n_comps, 5)),
    )

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
            [True, False],
            [True, False],
        ),
    )
    def test_plot_single_series(self, mock_show, config):
        index_type, stoch_type, use_ax, use_darts_style = config
        with option_context("plotting.use_darts_style", use_darts_style):
            series = getattr(self, f"series_{index_type}_{stoch_type}")
            if use_ax:
                _, ax = plt.subplots()
            else:
                ax = None
            series.plot(ax=ax)

            # For deterministic series with len > 1: one line per component
            # For probabilistic series with len > 1: one line per component + one area per component
            ax = ax if use_ax else plt.gca()

            # Count lines (Line2D objects with multiple data points representing actual lines)
            lines = [line for line in ax.lines if len(line.get_xdata()) > 1]
            assert len(lines) == self.n_comps

            # For probabilistic: count filled areas (PolyCollection from fill_between)
            if series.is_stochastic:
                areas = [
                    coll
                    for coll in ax.collections
                    if isinstance(coll, mcollections.PolyCollection)
                ]
                assert len(areas) == self.n_comps

            plt.show()
            plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
        ),
    )
    def test_plot_point_series(self, mock_show, config):
        index_type, stoch_type = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        series = series[:1]
        series.plot()

        # For deterministic series with len == 1: one point per component
        # For probabilistic series with len == 1: one point per component + one vertical line per component
        ax = plt.gca()

        # Count points (Line2D objects with markers representing single points)
        points = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 1 and line.get_marker() != "None"
        ]
        assert len(points) == self.n_comps

        # For probabilistic: count vertical lines for confidence intervals
        if series.is_stochastic:
            # The confidence interval is plotted as a line with "-+" marker
            # It's a vertical line where x-coordinates are the same
            vert_lines = []
            for line in ax.lines:
                xdata = np.asarray(line.get_xdata())
                ydata = np.asarray(line.get_ydata())
                if len(xdata) == 2 and len(ydata) == 2:
                    # check if x-coords are the same (vertical line)
                    xdiff = xdata[0] - xdata[1]

                    if isinstance(xdiff, pd.Timedelta):
                        xdiff = xdiff.total_seconds()

                    if abs(xdiff) < 1e-10:
                        vert_lines.append(line)
            assert len(vert_lines) == self.n_comps

        plt.show()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
        ),
    )
    def test_plot_empty_series(self, mock_show, config):
        index_type, stoch_type = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        series = series[:0]
        series.plot()

        # For len == 0: no points or lines should be plotted
        ax = plt.gca()
        # empty plot creates a line with empty data, but we want to check for actual plotted content
        # no points
        points = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 1 and line.get_marker() != "None"
        ]
        assert len(points) == 0

        # no lines
        lines_meaningful = [line for line in ax.lines if len(line.get_xdata()) > 1]
        assert len(lines_meaningful) == 0

        # no areas
        areas = [
            coll
            for coll in ax.collections
            if isinstance(coll, mcollections.PolyCollection)
        ]
        assert len(areas) == 0

        plt.show()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
            [
                {"new_plot": True},
                {"default_formatting": False},
                {"title": "my title"},
                {"label": "comps"},
                {"label": ["comps_1", "comps_2"]},
                {"alpha": 0.1, "color": "blue"},
                {"color": ["blue", "red"]},
                {"lw": 2},
            ],
        ),
    )
    def test_plot_params(self, mock_show, config):
        index_type, stoch_type, kwargs = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        series.plot(**kwargs)
        plt.show()
        plt.close()

    @patch("matplotlib.pyplot.show")
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
    def test_plot_stochastic_params(self, mock_show, config):
        (index_type, kwargs), stoch_type = config, "p"
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        series.plot(**kwargs)
        plt.show()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize("config", ["dt", "ri"])
    def test_plot_multiple_series(self, mock_show, config):
        index_type = config
        series1 = getattr(self, f"series_{index_type}_d")
        series2 = getattr(self, f"series_{index_type}_p")
        series1.plot()
        series2.plot()
        plt.show()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize("config", ["d", "p"])
    def test_cannot_plot_different_index_types(self, mock_show, config):
        stoch_type = config
        series1 = getattr(self, f"series_dt_{stoch_type}")
        series2 = getattr(self, f"series_ri_{stoch_type}")
        # datetime index plot changes x-axis to use datetime index
        series1.plot()
        # cannot plot a range index on datetime index
        with pytest.raises(TypeError):
            series2.plot()
        plt.show()
        plt.close()
