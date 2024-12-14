import itertools

import pandas as pd
import pytest

import darts.utils.historical_forecasts.utils as hfc_utils
from darts.models import LinearRegressionModel
from darts.utils.timeseries_generation import linear_timeseries


class TestHistoricalForecastsUtils:
    model = LinearRegressionModel(lags=1)

    def test_historical_forecasts_check_kwargs(self):
        # `hfc_args` not part of `dict_kwargs` works
        hfc_args = {"a", "b"}
        dict_kwargs = {"c": 0, "d": 0}
        out = hfc_utils._historical_forecasts_check_kwargs(
            hfc_args=hfc_args,
            name_kwargs="some_name",
            dict_kwargs=dict_kwargs,
        )
        assert out == dict_kwargs

        # `hfc_args` is part of `dict_kwargs` fails
        with pytest.raises(ValueError):
            _ = hfc_utils._historical_forecasts_check_kwargs(
                hfc_args={"c"},
                name_kwargs="some_name",
                dict_kwargs=dict_kwargs,
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],  # retrain
            [True, False],  # show warnings
            [{}, {"some_fit_param": 0}],  # fit kwargs
            [{}, {"some_predict_param": 0}],  # predict kwargs
        ),
    )
    def test_historical_forecasts_sanitize_kwargs(self, config):
        retrain, show_warnings, fit_kwargs, pred_kwargs = config
        fit_kwargs_out, pred_kwargs_out = (
            hfc_utils._historical_forecasts_sanitize_kwargs(
                self.model,
                fit_kwargs=fit_kwargs,
                predict_kwargs=pred_kwargs,
                retrain=retrain,
                show_warnings=show_warnings,
            )
        )
        assert fit_kwargs_out == fit_kwargs
        assert pred_kwargs_out == pred_kwargs

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "fit_kwargs": {"series": 0},
                "predict_kwargs": None,
                "retrain": True,
                "show_warnings": False,
            },
            {
                "fit_kwargs": None,
                "predict_kwargs": {"series": 0},
                "retrain": True,
                "show_warnings": False,
            },
        ],
    )
    def test_historical_forecasts_sanitize_kwargs_invalid(self, kwargs):
        with pytest.raises(ValueError):
            _ = hfc_utils._historical_forecasts_sanitize_kwargs(self.model, **kwargs)

    def test_historical_forecasts_check_start(self):
        """"""
        series = linear_timeseries(start=0, length=1)
        kwargs = {
            "start": 0,
            "start_format": "value",
            "series_start": 0,
            "ref_start": 0,
            "ref_end": 0,
            "stride": 0,
            "series_idx": 0,
            "is_historical_forecast": False,
        }
        # low enough start idx works with any kwargs
        hfc_utils._check_start(series, start_idx=0, **kwargs)

        # start idx >= len(series) raises error
        with pytest.raises(ValueError):
            hfc_utils._check_start(series, start_idx=1, **kwargs)

    @pytest.mark.parametrize(
        "config",
        [
            (True, pd.Timestamp("2000-01-01"), "value"),
            (True, 0.9, "value"),
            (True, 0.9, "position"),
            (True, 0, "position"),
            (True, 0, "value"),
            (True, -1, "position"),
            (False, pd.Timestamp("2000-01-01"), "value"),
            (False, 0.9, "value"),
            (False, 0.9, "position"),
            (False, 0, "position"),
            (False, -1, "position"),
        ],
    )
    def test_historical_forecasts_check_start_invalid(self, config):
        """"""
        is_dt, start, start_format = config
        series = linear_timeseries(start="2000-01-01" if is_dt else 0, length=1)
        series_start = series.start_time()
        kwargs = {
            "start": start,
            "start_format": start_format,
            "series_start": series_start,
            "ref_start": 0,
            "ref_end": 0,
            "stride": 0,
            "series_idx": 0,
            "is_historical_forecast": False,
        }

        # low enough start idx works with any kwargs
        with pytest.raises(ValueError) as err:
            hfc_utils._check_start(series, start_idx=1, **kwargs)

        # make sure we reach the expected error message and message is specific to input
        position_msg = f"position `{start}` corresponding to time "
        if start_format == "position" or is_dt and not isinstance(start, pd.Timestamp):
            assert position_msg in str(err.value)
        else:
            assert position_msg not in str(err.value)

    @pytest.mark.parametrize(
        "config",
        [
            (0, 0, 0),
            (1, 1, 1),
            (1, 10, 1),
            (-1, 1, 0),
            (-3, 1, 0),
            (-1, 2, 1),
            (-2, 2, 0),
            (-3, 2, 1),
        ],
    )
    def test_adjust_start(self, config):
        """Check relative start position adjustment."""
        start_rel, stride, start_expected = config
        assert hfc_utils._adjust_start(start_rel, stride) == start_expected
