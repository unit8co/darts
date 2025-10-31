from itertools import product
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from darts import TimeSeries
from darts.utils.utils import generate_index


class TestTimeSeriesPlot:
    # datetime index, deterministic
    series_dt_d = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, 2, 1)),
    )
    # datetime index, probabilistic
    series_dt_p = TimeSeries.from_times_and_values(
        times=generate_index(start="2000-01-01", length=10, freq="D"),
        values=np.random.random((10, 2, 5)),
    )
    # range index, deterministic
    series_ri_d = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, 2, 1)),
    )
    # range index, probabilistic
    series_ri_p = TimeSeries.from_times_and_values(
        times=generate_index(start=0, length=10, freq=1),
        values=np.random.random((10, 2, 5)),
    )

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize(
        "config",
        product(
            ["dt", "ri"],
            ["d", "p"],
        ),
    )
    def test_plot_single(self, mock_show, config):
        index_type, stoch_type = config
        series = getattr(self, f"series_{index_type}_{stoch_type}")
        series.plot()
        plt.show()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize("config", ["dt", "ri"])
    def test_plot_multiple(self, mock_show, config):
        index_type = config
        series1 = getattr(self, f"series_{index_type}_d")
        series2 = getattr(self, f"series_{index_type}_p")
        series1.plot()
        series2.plot()
        plt.show()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    @pytest.mark.parametrize("config", ["d", "p"])
    def test_cannot_plot_different_index_types(self, mock_show, config):
        stoch_type = config
        series1 = getattr(self, f"series_dt_{stoch_type}")
        series2 = getattr(self, f"series_ri_{stoch_type}")
        series1.plot()
        series2.plot()
        plt.show()
        mock_show.assert_called_once()
        plt.close()
