import itertools
import logging
import random
from itertools import product

import numpy as np
import pandas as pd
import pytest

import darts.metrics as metrics
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.logging import get_logger
from darts.models import (
    ARIMA,
    FFT,
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveDrift,
    NaiveSeasonal,
    RandomForest,
    Theta,
)
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.timeseries_generation import constant_timeseries as ct
from darts.utils.timeseries_generation import gaussian_timeseries as gt
from darts.utils.timeseries_generation import linear_timeseries as lt
from darts.utils.timeseries_generation import random_walk_timeseries as rt
from darts.utils.timeseries_generation import sine_timeseries as st
from darts.utils.utils import generate_index

logger = get_logger(__name__)


if TORCH_AVAILABLE:
    from darts.models import BlockRNNModel, TCNModel


def get_dummy_series(
    ts_length: int, lt_end_value: int = 10, st_value_offset: int = 10
) -> TimeSeries:
    return (
        lt(length=ts_length, end_value=lt_end_value)
        + st(length=ts_length, value_y_offset=st_value_offset)
        + rt(length=ts_length)
    )


def compare_best_against_random(model_class, params, series, stride=1):
    # instantiate best model in expanding window mode
    np.random.seed(1)
    best_model_1, _, _ = model_class.gridsearch(
        params,
        series,
        forecast_horizon=10,
        stride=stride,
        metric=metrics.mape,
        start=series.time_index[-21],
    )

    # instantiate best model in split mode
    train, val = series.split_before(series.time_index[-10])
    best_model_2, _, _ = model_class.gridsearch(
        params, train, val_series=val, metric=metrics.mape
    )

    # instantiate model with random parameters from 'params'
    random.seed(1)
    random_param_choice = {}
    for key in params.keys():
        random_param_choice[key] = random.choice(params[key])
    random_model = model_class(**random_param_choice)

    # perform backtest forecasting on both models
    best_score_1 = best_model_1.backtest(
        series, start=series.time_index[-21], forecast_horizon=10
    )
    random_score_1 = random_model.backtest(
        series, start=series.time_index[-21], forecast_horizon=10
    )

    # perform train/val evaluation on both models
    best_model_2.fit(train)
    best_score_2 = metrics.mape(best_model_2.predict(len(val)), series)
    random_model = model_class(**random_param_choice)
    random_model.fit(train)
    random_score_2 = metrics.mape(random_model.predict(len(val)), series)

    # check whether best models are at least as good as random models
    expanding_window_ok = best_score_1 <= random_score_1
    split_ok = best_score_2 <= random_score_2

    return expanding_window_ok and split_ok


class TestBacktesting:
    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [False, True],
            [[metrics.mape], [metrics.mape, metrics.mape]],
        ),
    )
    def test_output_single_series_hfc_lpo_true(self, config):
        """Tests backtest based on historical forecasts generated on a single `series` (or list of one `series`)
        with last_points_only=True"""
        is_univariate, series_as_list, metric = config
        is_multi_metric = len(metric) > 1
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        y = y if not series_as_list else [y]
        hfc = hfc if not series_as_list else [hfc]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                reduction=None,
                metric=metric,
                last_points_only=False,
            )
        if series_as_list:
            error_msg = "Expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]`"
        else:
            error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        # number of forecasts do not match number of `series`
        if series_as_list:
            with pytest.raises(ValueError) as err:
                _ = model.backtest(
                    series=y,
                    historical_forecasts=hfc + y,
                    reduction=None,
                    metric=metric,
                    last_points_only=True,
                )
            error_msg = f"expected `historical_forecasts` of type `Sequence[TimeSeries]` with length n={len(y)}."
            assert str(err.value).endswith(error_msg)

        # no reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=None,
            metric=metric,
            last_points_only=True,
        )
        bt = bt if series_as_list else [bt]
        assert isinstance(bt, list) and len(bt) == 1
        bt = bt[0]
        if not is_multi_metric:
            # inner type expected: 1 float
            assert isinstance(bt, float) and bt == 100.0
        else:
            # inner shape expected: (n metrics = 2,)
            assert isinstance(bt, np.ndarray)
            np.testing.assert_array_almost_equal(bt, np.array([100.0, 100.0]))

        # with reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=np.mean,
            metric=metric,
            last_points_only=True,
        )
        bt = bt if series_as_list else [bt]
        assert isinstance(bt, list) and len(bt) == 1
        bt = bt[0]
        if not is_multi_metric:
            # inner type expected: 1 float
            assert isinstance(bt, float) and bt == 100.0
        else:
            # inner shape expected: (n metrics = 2,)
            assert isinstance(bt, np.ndarray)
            np.testing.assert_array_almost_equal(bt, np.array([100.0, 100.0]))

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [False, True],
            [[metrics.mape], [metrics.mape, metrics.mape]],
            [1, 2],
        ),
    )
    def test_output_single_series_hfc_lpo_false(self, config):
        """Tests backtest based on historical forecasts generated on a single `series` (or list of one `series`)
        with last_points_only=False"""
        is_univariate, series_as_list, metric, n_forecasts = config
        is_multi_metric = len(metric) > 1
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        hfc = [y, hfc]
        hfc = hfc[:n_forecasts]

        y = y if not series_as_list else [y]
        hfc = hfc if not series_as_list else [hfc]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=True``
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                reduction=None,
                metric=metric,
                last_points_only=True,
            )
        if series_as_list:
            error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        else:
            error_msg = "Expected `historical_forecasts` of type `TimeSeries`"
        assert str(err.value).startswith(error_msg)

        # number of forecasts do not match number of `series`
        if series_as_list:
            with pytest.raises(ValueError) as err:
                _ = model.backtest(
                    series=y,
                    historical_forecasts=hfc + [y],
                    reduction=None,
                    metric=metric,
                    last_points_only=False,
                )
            error_msg = (
                f"expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]`"
                f" with length n={len(y)}."
            )
            assert str(err.value).endswith(error_msg)

        # no reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=None,
            metric=metric,
            last_points_only=False,
        )
        bt = bt if series_as_list else [bt]
        assert isinstance(bt, list) and len(bt) == 1
        bt = bt[0]
        assert isinstance(bt, np.ndarray)
        if not is_multi_metric:
            # inner shape expected: (n hist forecasts = 2,)
            np.testing.assert_array_almost_equal(
                bt, np.array([0.0, 100.0])[:n_forecasts]
            )
        else:
            # inner shape expected: (n hist forecasts = 2, n metrics = 2)
            np.testing.assert_array_almost_equal(
                bt, np.array([[0.0, 0.0], [100.0, 100.0]])[:n_forecasts]
            )

        # with reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=np.mean,
            metric=metric,
            last_points_only=False,
        )
        bt = bt if series_as_list else [bt]
        assert isinstance(bt, list) and len(bt) == 1
        bt = bt[0]
        score_exp = 0.0 if n_forecasts == 1 else 50.0
        if not is_multi_metric:
            # inner shape expected: 1 float
            assert isinstance(bt, float) and bt == score_exp
        else:
            # inner shape expected: (n metrics = 2,)
            assert isinstance(bt, np.ndarray)
            np.testing.assert_array_almost_equal(bt, np.array([score_exp, score_exp]))

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [[metrics.mape], [metrics.mape, metrics.mape]],
        ),
    )
    def test_output_multi_series_hfc_lpo_true(self, config):
        """Tests backtest based on historical forecasts generated on multiple `series` with last_points_only=True"""
        is_univariate, metric = config
        is_multi_metric = len(metric) > 1
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        hfc = [y, hfc]
        y = [y, y]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                reduction=None,
                metric=metric,
                last_points_only=False,
            )
        error_msg = (
            "Expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]`"
        )
        assert str(err.value).startswith(error_msg)

        # number of forecasts do not match number of `series`
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc + [y[0]],
                reduction=None,
                metric=metric,
                last_points_only=True,
            )
        error_msg = f"expected `historical_forecasts` of type `Sequence[TimeSeries]` with length n={len(y)}."
        assert str(err.value).endswith(error_msg)

        # no reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=None,
            last_points_only=True,
            metric=metric,
        )
        assert isinstance(bt, list) and len(bt) == 2
        if not is_multi_metric:
            # per series, inner type expected: 1 float
            assert bt == [0.0, 100.0]
        else:
            # per series, inner shape expected: (n metrics = 2,)
            assert all(isinstance(bt_, np.ndarray) for bt_ in bt)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0, 0.0]))
            np.testing.assert_array_almost_equal(bt[1], np.array([100.0, 100.0]))

        # with reduction
        bt = model.backtest(
            series=y,
            historical_forecasts=hfc,
            reduction=np.mean,
            last_points_only=True,
            metric=metric,
        )
        assert isinstance(bt, list) and len(bt) == 2
        if not is_multi_metric:
            # per series, inner type expected: 1 float
            assert bt == [0.0, 100.0]
        else:
            # per series, inner shape expected: (n metrics = 2,)
            assert all(isinstance(bt_, np.ndarray) for bt_ in bt)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0, 0.0]))
            np.testing.assert_array_almost_equal(bt[1], np.array([100.0, 100.0]))

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [[metrics.mape], [metrics.mape, metrics.mape]],
        ),
    )
    def test_output_multi_series_hfc_lpo_false(self, config):
        """Tests backtest based on historical forecasts generated on multiple `series` with
        last_points_only=False.
        """
        is_univariate, metric = config
        is_multi_metric = len(metric) > 1
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        hfc = [[y], [hfc]]
        y = [y, y]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                reduction=None,
                metric=metric,
                last_points_only=True,
            )
        error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        # number of forecasts do not match number of `series`
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc + [[y[0]]],
                reduction=None,
                metric=metric,
                last_points_only=False,
            )
        error_msg = f"expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]` with length n={len(y)}."
        assert str(err.value).endswith(error_msg)

        # no reduction
        bt = model.backtest(
            series=y, historical_forecasts=hfc, reduction=None, metric=metric
        )
        assert isinstance(bt, list) and len(bt) == 2
        assert isinstance(bt[0], np.ndarray)
        assert isinstance(bt[1], np.ndarray)
        if not is_multi_metric:
            # inner shape expected: (n hist forecasts = 1,)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0]))
            np.testing.assert_array_almost_equal(bt[1], np.array([100.0]))
        else:
            # inner shape expected: (n metrics = 2, n hist forecasts = 1)
            np.testing.assert_array_almost_equal(bt[0], np.array([[0.0, 0.0]]))
            np.testing.assert_array_almost_equal(bt[1], np.array([[100.0, 100.0]]))

        # with reduction
        bt = model.backtest(
            series=y, historical_forecasts=hfc, reduction=np.mean, metric=metric
        )
        assert isinstance(bt, list) and len(bt) == 2
        if not is_multi_metric:
            # inner type expected: 1 float
            assert bt == [0.0, 100.0]
        else:
            # inner shape expected: (n metrics = 2,)
            assert isinstance(bt[0], np.ndarray)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0, 0.0]))
            assert isinstance(bt[1], np.ndarray)
            np.testing.assert_array_almost_equal(bt[1], np.array([100.0, 100.0]))

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [[metrics.mape], [metrics.mape, metrics.mape]],
        ),
    )
    def test_output_multi_series_hfc_lpo_false_different_n_fcs(self, config):
        """Tests backtest based on historical forecasts generated on multiple `series` with
        last_points_only=False, and the historical forecasts have different lengths
        """
        is_univariate, metric = config
        is_multi_metric = len(metric) > 1
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        hfc = [[y], [hfc, hfc]]
        y = [y, y]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                reduction=None,
                metric=metric,
                last_points_only=True,
            )
        error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        # no reduction
        bt = model.backtest(
            series=y, historical_forecasts=hfc, reduction=None, metric=metric
        )
        assert isinstance(bt, list) and len(bt) == 2
        assert isinstance(bt[0], np.ndarray)
        assert isinstance(bt[1], np.ndarray)
        if not is_multi_metric:
            # inner shape expected: (n hist forecasts = 1,)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0]))
            # inner shape expected: (n hist forecasts = 2,)
            np.testing.assert_array_almost_equal(bt[1], np.array([100.0, 100.0]))
        else:
            # inner shape expected: (n metrics = 2, n hist forecasts = 1)
            np.testing.assert_array_almost_equal(bt[0], np.array([[0.0, 0.0]]))
            # inner shape expected: (n metrics = 2, n hist forecasts = 2)
            np.testing.assert_array_almost_equal(
                bt[1], np.array([[100.0, 100.0], [100.0, 100.0]])
            )

        # with reduction
        bt = model.backtest(
            series=y, historical_forecasts=hfc, reduction=np.mean, metric=metric
        )
        assert isinstance(bt, list) and len(bt) == 2
        if not is_multi_metric:
            # inner type expected: 1 float
            assert bt == [0.0, 100.0]
        else:
            # inner shape expected: (n metrics = 2,)
            assert isinstance(bt[0], np.ndarray)
            np.testing.assert_array_almost_equal(bt[0], np.array([0.0, 0.0]))
            assert isinstance(bt[1], np.ndarray)

    def test_backtest_forecasting(self):
        linear_series = lt(length=50)
        linear_series_int = TimeSeries.from_values(linear_series.values())
        linear_series_multi = linear_series.stack(linear_series)

        # univariate model + univariate series
        score = NaiveDrift().backtest(
            linear_series,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score == 1.0

        # univariate model + univariate series + historical_forecasts precalculated
        forecasts = NaiveDrift().historical_forecasts(
            linear_series,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            last_points_only=False,
        )
        precalculated_forecasts_score = NaiveDrift().backtest(
            linear_series,
            historical_forecasts=forecasts,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score == precalculated_forecasts_score

        # very large train length should not affect the backtest
        score = NaiveDrift().backtest(
            linear_series,
            train_length=10000,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score == 1.0

        # using several metric function should not affect the backtest
        score = NaiveDrift().backtest(
            linear_series,
            train_length=10000,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=[metrics.r2_score, metrics.mape],
        )
        np.testing.assert_almost_equal(score, np.array([1.0, 0.0]))

        # window of size 2 is too small for naive drift
        with pytest.raises(ValueError):
            NaiveDrift().backtest(
                linear_series,
                train_length=2,
                start=pd.Timestamp("20000201"),
                forecast_horizon=3,
                metric=metrics.r2_score,
            )

        # test that it also works for time series that are not Datetime-indexed
        score = NaiveDrift().backtest(
            linear_series_int, start=0.7, forecast_horizon=3, metric=metrics.r2_score
        )
        assert score == 1.0

        with pytest.raises(ValueError):
            NaiveDrift().backtest(
                linear_series,
                start=pd.Timestamp("20000218"),
                forecast_horizon=3,
                overlap_end=False,
            )
        NaiveDrift().backtest(
            linear_series, start=pd.Timestamp("20000217"), forecast_horizon=3
        )
        NaiveDrift().backtest(
            linear_series,
            start=pd.Timestamp("20000218"),
            forecast_horizon=3,
            overlap_end=True,
        )

        # Using forecast_horizon default value
        NaiveDrift().backtest(linear_series, start=pd.Timestamp("20000216"))
        NaiveDrift().backtest(
            linear_series, start=pd.Timestamp("20000217"), overlap_end=True
        )

        # Using an int or float value for start
        NaiveDrift().backtest(linear_series, start=30)
        NaiveDrift().backtest(linear_series, start=0.7, overlap_end=True)

        # Set custom train window length
        NaiveDrift().backtest(linear_series, train_length=10, start=30)

        # Using invalid start and/or forecast_horizon values
        with pytest.raises(ValueError):
            NaiveDrift().backtest(linear_series, start=0.7, forecast_horizon=-1)
        with pytest.raises(ValueError):
            NaiveDrift().backtest(linear_series, start=-0.7, forecast_horizon=1)

        with pytest.raises(ValueError):
            NaiveDrift().backtest(linear_series, start=100)
        with pytest.raises(ValueError):
            NaiveDrift().backtest(linear_series, start=1.2)
        with pytest.raises(TypeError):
            NaiveDrift().backtest(linear_series, start="wrong type")
        with pytest.raises(ValueError):
            NaiveDrift().backtest(linear_series, train_length=0, start=0.5)
        with pytest.raises(TypeError):
            NaiveDrift().backtest(linear_series, train_length=1.2, start=0.5)
        with pytest.raises(TypeError):
            NaiveDrift().backtest(linear_series, train_length="wrong type", start=0.5)

        with pytest.raises(ValueError):
            NaiveDrift().backtest(
                linear_series, start=49, forecast_horizon=2, overlap_end=False
            )

        # univariate model + multivariate series
        with pytest.raises(ValueError):
            FFT().backtest(
                linear_series_multi, start=pd.Timestamp("20000201"), forecast_horizon=3
            )

        # multivariate model + univariate series
        if TORCH_AVAILABLE:
            tcn_model = TCNModel(
                input_chunk_length=12,
                output_chunk_length=1,
                batch_size=1,
                n_epochs=1,
                **tfm_kwargs,
            )
            # cannot perform historical forecasts with `retrain=False` and untrained model
            with pytest.raises(ValueError):
                _ = tcn_model.historical_forecasts(
                    linear_series,
                    start=pd.Timestamp("20000125"),
                    forecast_horizon=3,
                    verbose=False,
                    last_points_only=True,
                    retrain=False,
                )

            pred = tcn_model.historical_forecasts(
                linear_series,
                start=pd.Timestamp("20000125"),
                forecast_horizon=3,
                verbose=False,
                last_points_only=True,
            )
            assert pred.width == 1
            assert pred.end_time() == linear_series.end_time()

            # multivariate model + multivariate series
            # historical forecasts doesn't overwrite model object -> we can use different input dimensions
            tcn_model.backtest(
                linear_series_multi,
                start=pd.Timestamp("20000125"),
                forecast_horizon=3,
                verbose=False,
            )

            # univariate model
            tcn_model = TCNModel(
                input_chunk_length=12,
                output_chunk_length=1,
                batch_size=1,
                n_epochs=1,
                **tfm_kwargs,
            )
            tcn_model.fit(linear_series, verbose=False)
            # univariate fitted model + multivariate series
            with pytest.raises(ValueError):
                tcn_model.backtest(
                    linear_series_multi,
                    start=pd.Timestamp("20000125"),
                    forecast_horizon=3,
                    verbose=False,
                    retrain=False,
                )

            tcn_model = TCNModel(
                input_chunk_length=12,
                output_chunk_length=3,
                batch_size=1,
                n_epochs=1,
                **tfm_kwargs,
            )
            pred = tcn_model.historical_forecasts(
                linear_series_multi,
                start=pd.Timestamp("20000125"),
                forecast_horizon=3,
                verbose=False,
                last_points_only=True,
            )
            assert pred.width == 2
            assert pred.end_time() == linear_series.end_time()

    def test_backtest_multiple_series(self):
        series = [AirPassengersDataset().load(), MonthlyMilkDataset().load()]
        model = NaiveSeasonal(K=1)

        error = model.backtest(
            series,
            train_length=30,
            forecast_horizon=2,
            stride=1,
            retrain=True,
            last_points_only=False,
            verbose=False,
        )

        expected = [11.63104, 6.09458]
        assert len(error) == 2
        assert round(abs(error[0] - expected[0]), 4) == 0
        assert round(abs(error[1] - expected[1]), 4) == 0

    def test_backtest_regression(self, caplog):
        np.random.seed(4)

        gaussian_series = gt(mean=2, length=50)
        sine_series = st(length=50)
        features = gaussian_series.stack(sine_series)
        features_multivariate = (
            (gaussian_series + sine_series).stack(gaussian_series).stack(sine_series)
        )
        target = sine_series

        features = features.with_columns_renamed(
            features.components, ["Value0", "Value1"]
        )

        features_multivariate = features_multivariate.with_columns_renamed(
            features_multivariate.components, ["Value0", "Value1", "Value2"]
        )

        # univariate feature test
        score = LinearRegressionModel(
            lags=None, lags_future_covariates=[0, -1]
        ).backtest(
            series=target,
            future_covariates=features,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
            last_points_only=True,
        )
        assert score > 0.9

        # univariate feature test + train length
        score = LinearRegressionModel(
            lags=None, lags_future_covariates=[0, -1]
        ).backtest(
            series=target,
            future_covariates=features,
            start=pd.Timestamp("20000201"),
            train_length=20,
            forecast_horizon=3,
            metric=metrics.r2_score,
            last_points_only=True,
        )
        assert score > 0.9

        # Using an int or float value for start
        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features,
            start=30,
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score > 0.9

        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features,
            start=0.5,
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score > 0.9

        # Using a too small start value
        warning_expected = (
            "`start` position `{0}` corresponding to time `{1}` is before the first "
            "predictable/trainable historical forecasting point for series at index: 0. Using the first historical "
            "forecasting point `2000-01-15 00:00:00` that lies a round-multiple of `stride=1` ahead of `start`. "
            "To hide these warnings, set `show_warnings=False`."
        )
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _ = RandomForest(lags=12).backtest(
                series=target, start=0, forecast_horizon=3
            )
            assert warning_expected.format(0, target.start_time()) in caplog.text
        caplog.clear()

        with caplog.at_level(logging.WARNING):
            _ = RandomForest(lags=12).backtest(
                series=target, start=0.01, forecast_horizon=3
            )
            assert warning_expected.format(0.01, target.start_time()) in caplog.text
        caplog.clear()

        # Using RandomForest's start default value
        score = RandomForest(lags=12, random_state=0).backtest(
            series=target, forecast_horizon=3, start=0.5, metric=metrics.r2_score
        )
        assert score > 0.95

        # multivariate feature test
        score = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        assert score > 0.94

        # multivariate feature test with train window 35
        score_35 = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            train_length=35,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        logger.info(
            "Score for multivariate feature test with train window 35 is: ", score_35
        )
        assert score_35 > 0.92

        # multivariate feature test with train window 45
        score_45 = RandomForest(
            lags=12, lags_future_covariates=[0, -1], random_state=0
        ).backtest(
            series=target,
            train_length=45,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
        )
        logger.info(
            "Score for multivariate feature test with train window 45 is: ", score_45
        )
        assert score_45 > 0.94
        assert score_45 > score_35

        # multivariate with stride
        score = RandomForest(
            lags=12, lags_future_covariates=[0], random_state=0
        ).backtest(
            series=target,
            future_covariates=features_multivariate,
            start=pd.Timestamp("20000201"),
            forecast_horizon=3,
            metric=metrics.r2_score,
            last_points_only=True,
            stride=3,
        )
        assert score > 0.9

    @pytest.mark.parametrize("model_cls", [Theta, ARIMA])
    def test_backtest_bad_covariates(self, model_cls):
        """Passing unsupported covariate should raise an exception"""
        series = lt(start_value=1, end_value=10, length=31, dtype="float32")
        model = model_cls()
        bt_kwargs = {"start": -1, "start_format": "position", "show_warnings": False}
        model.backtest(series=series, **bt_kwargs)

        with pytest.raises(ValueError) as msg:
            model.backtest(series=series, past_covariates=series, **bt_kwargs)
        assert str(msg.value).startswith(
            "Model cannot be fit/trained with `past_covariates`."
        )
        if not model.supports_future_covariates:
            with pytest.raises(ValueError) as msg:
                model.backtest(series=series, future_covariates=series, **bt_kwargs)
            assert str(msg.value).startswith(
                "Model cannot be fit/trained with `future_covariates`."
            )

    def test_gridsearch(self):
        np.random.seed(1)

        dummy_series = get_dummy_series(ts_length=50)
        dummy_series_int_index = TimeSeries.from_values(dummy_series.values())

        theta_params = {"theta": list(range(3, 10))}
        assert compare_best_against_random(Theta, theta_params, dummy_series)
        assert compare_best_against_random(Theta, theta_params, dummy_series_int_index)
        assert compare_best_against_random(Theta, theta_params, dummy_series, stride=2)

        fft_params = {"nr_freqs_to_keep": [10, 50, 100], "trend": [None, "poly", "exp"]}
        assert compare_best_against_random(FFT, fft_params, dummy_series)

        es_params = {"seasonal_periods": list(range(5, 10))}
        assert compare_best_against_random(
            ExponentialSmoothing, es_params, dummy_series
        )

    def test_gridsearch_metric_score(self):
        np.random.seed(1)

        model_class = Theta
        params = {"theta": list(range(3, 6))}
        dummy_series = get_dummy_series(ts_length=50)

        best_model, _, score = model_class.gridsearch(
            params,
            series=dummy_series,
            forecast_horizon=10,
            stride=1,
            start=dummy_series.time_index[-21],
        )
        recalculated_score = best_model.backtest(
            series=dummy_series,
            start=dummy_series.time_index[-21],
            forecast_horizon=10,
            stride=1,
        )

        assert score == recalculated_score, "The metric scores should match"

    def test_gridsearch_random_search(self):
        np.random.seed(1)

        dummy_series = get_dummy_series(ts_length=50)

        param_range = list(range(10, 20))
        params = {"lags": param_range}

        model = RandomForest(lags=1)
        result = model.gridsearch(
            params, dummy_series, forecast_horizon=1, n_random_samples=5
        )

        assert isinstance(result[0], RandomForest)
        assert isinstance(result[1]["lags"], int)
        assert isinstance(result[2], float)
        assert min(param_range) <= result[1]["lags"] <= max(param_range)

    def test_gridsearch_n_random_samples_bad_arguments(self):
        dummy_series = get_dummy_series(ts_length=50)

        params = {"lags": list(range(1, 11)), "past_covariates": list(range(1, 11))}

        with pytest.raises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=-5
            )
        with pytest.raises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=105
            )
        with pytest.raises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=-24.56
            )
        with pytest.raises(ValueError):
            RandomForest.gridsearch(
                params, dummy_series, forecast_horizon=1, n_random_samples=1.5
            )

    def test_gridsearch_n_random_samples(self):
        np.random.seed(1)

        params = {"lags": list(range(1, 11)), "past_covariates": list(range(1, 11))}

        params_cross_product = list(product(*params.values()))

        # Test absolute sample
        absolute_sampled_result = RandomForest._sample_params(params_cross_product, 10)
        assert len(absolute_sampled_result) == 10

        # Test percentage sample
        percentage_sampled_result = RandomForest._sample_params(
            params_cross_product, 0.37
        )
        assert len(percentage_sampled_result) == 37

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_gridsearch_n_jobs(self):
        """
        Testing that running gridsearch with multiple workers returns the same
        best_parameters as the single worker run.
        """

        rng_seed = 1

        np.random.seed(rng_seed)

        dummy_series = get_dummy_series(
            ts_length=100, lt_end_value=1, st_value_offset=0
        ).astype(np.float32)
        ts_train, ts_val = dummy_series.split_before(split_point=0.8)

        test_cases = [
            {
                "model": ARIMA,  # ExtendedForecastingModel
                "parameters": {"p": [18, 4], "q": [2, 3], "random_state": [rng_seed]},
            },
            {
                "model": BlockRNNModel,  # TorchForecastingModel
                "parameters": {
                    "input_chunk_length": [5, 10],
                    "output_chunk_length": [1, 3],
                    "n_epochs": [1, 5],
                    "random_state": [rng_seed],
                    "pl_trainer_kwargs": [tfm_kwargs["pl_trainer_kwargs"]],
                },
            },
        ]

        for test in test_cases:
            model = test["model"]
            parameters = test["parameters"]

            np.random.seed(rng_seed)
            _, best_params1, _ = model.gridsearch(
                parameters=parameters, series=ts_train, val_series=ts_val, n_jobs=1
            )

            np.random.seed(rng_seed)
            _, best_params2, _ = model.gridsearch(
                parameters=parameters, series=ts_train, val_series=ts_val, n_jobs=2
            )

            assert best_params1 == best_params2

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    def test_gridsearch_multi(self):
        dummy_series = st(length=40, value_y_offset=10).stack(
            lt(length=40, end_value=20)
        )
        tcn_params = {
            "input_chunk_length": [12],
            "output_chunk_length": [3],
            "n_epochs": [1],
            "batch_size": [1],
            "kernel_size": [2, 3, 4],
            "pl_trainer_kwargs": [tfm_kwargs["pl_trainer_kwargs"]],
        }
        TCNModel.gridsearch(
            tcn_params, dummy_series, forecast_horizon=3, metric=metrics.mape
        )

    @pytest.mark.parametrize(
        "model_cls,parameters",
        zip([NaiveSeasonal, ARIMA], [{"K": [1, 2]}, {"p": [18, 4]}]),
    )
    def test_gridsearch_bad_covariates(self, model_cls, parameters):
        """Passing unsupported covariate should raise an exception"""
        dummy_series = get_dummy_series(
            ts_length=100, lt_end_value=1, st_value_offset=0
        ).astype(np.float32)

        ts_train, ts_val = dummy_series.split_before(split_point=0.8)

        bt_kwargs = {"start": -1, "start_format": "position", "show_warnings": False}

        model = model_cls()
        model_cls.gridsearch(
            parameters=parameters, series=ts_train, val_series=ts_val, **bt_kwargs
        )

        with pytest.raises(ValueError) as msg:
            model_cls.gridsearch(
                parameters=parameters,
                series=ts_train,
                past_covariates=dummy_series,
                val_series=ts_val,
                **bt_kwargs,
            )
        assert str(msg.value).startswith(
            "Model cannot be fit/trained with `past_covariates`."
        )
        if not model.supports_future_covariates:
            with pytest.raises(ValueError) as msg:
                model_cls.gridsearch(
                    parameters=parameters,
                    series=ts_train,
                    future_covariates=dummy_series,
                    val_series=ts_val,
                    **bt_kwargs,
                )
            assert str(msg.value).startswith(
                "Model cannot be fit/trained with `future_covariates`."
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product([True, False], [True, False]),
    )
    def test_gridsearch_sample_weight(self, config):
        """check that passing sample weights work and that it yields different results than without sample weights."""
        manual_weight, use_val_series = config
        ts = AirPassengersDataset().load()
        if manual_weight:
            sample_weight = np.linspace(0, 1, len(ts))
            sample_weight = ts.with_values(np.expand_dims(sample_weight, -1))
        else:
            sample_weight = "linear"

        parameters = {"lags": [3], "output_chunk_length": [1]}
        start_kwargs = {"start": -1, "start_format": "position"}
        gs_kwargs = {"val_series": ts} if use_val_series else {"forecast_horizon": 1}
        gs_non_weighted = LinearRegressionModel.gridsearch(
            parameters, series=ts[:-1], **start_kwargs, **gs_kwargs
        )[-1]

        gs_weighted = LinearRegressionModel.gridsearch(
            parameters,
            series=ts[:-1],
            sample_weight=sample_weight,
            **start_kwargs,
            **gs_kwargs,
        )[-1]

        # check that the predictions are different
        assert gs_weighted != gs_non_weighted

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                metrics.ase,
                metrics.mase,
            ],
            [1, 2],
        ),
    )
    def test_scaled_metrics(self, config):
        """Tests backtest for scaled metrics based on historical forecasts generated on a sequence
        `series` with last_points_only=False"""
        metric, m = config
        y = lt(length=20)
        hfc = lt(length=10, start=y.start_time() + 10 * y.freq)
        y = [y, y]
        hfc = [[hfc, hfc], [hfc]]

        model = NaiveDrift()
        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=False,
            reduction=None,
            metric_kwargs={"m": m},
        )
        assert isinstance(bts, list) and len(bts) == 2

        bt_expected = metric(y[0], hfc[0][0], insample=y[0], m=m)
        for bt_list in bts:
            for bt in bt_list:
                np.testing.assert_array_almost_equal(bt, bt_expected)

    @pytest.mark.parametrize(
        "metric",
        [
            [metrics.mae],  # mae does not support time_reduction
            [metrics.mae, metrics.ae],  # ae supports time_reduction
        ],
    )
    def test_metric_kwargs(self, metric):
        """Tests backtest with different metric_kwargs based on historical forecasts generated on a sequence
        `series` with last_points_only=False"""
        y = lt(length=20)
        y = y.stack(y + 1.0)
        hfc = lt(length=10, start=y.start_time() + 10 * y.freq)
        hfc = hfc.stack(hfc + 1.0)
        y = [y, y]
        hfc = [[hfc, hfc], [hfc]]

        metric_kwargs = [{"component_reduction": np.median}]
        if len(metric) > 1:
            # give metric specific kwargs
            metric_kwargs.append({
                "component_reduction": np.median,
                "time_reduction": np.mean,
            })

        model = NaiveDrift()
        # backtest should fail with invalid metric kwargs (mae does not support time reduction)
        with pytest.raises(TypeError) as err:
            _ = model.backtest(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
                reduction=None,
                metric_kwargs={
                    "component_reduction": np.median,
                    "time_reduction": np.mean,
                },
            )
        assert str(err.value).endswith("unexpected keyword argument 'time_reduction'")

        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=False,
            reduction=None,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(bts, list) and len(bts) == 2

        # `ae` with time and component reduction is equal to `mae` with component reduction
        bt_expected = metrics.mae(y[0], hfc[0][0], component_reduction=np.median)
        for bt_list in bts:
            for bt in bt_list:
                np.testing.assert_array_almost_equal(bt, bt_expected)

        def time_reduced_metric(*args, **kwargs):
            return metrics.ae(*args, **kwargs, time_reduction=np.mean)

        # check that single kwargs can be used for all metrics if params are supported
        metric = [metric[0], time_reduced_metric]
        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=False,
            reduction=None,
            metric_kwargs=metric_kwargs[0],
        )
        assert isinstance(bts, list) and len(bts) == 2

        # `ae` with time and component reduction is equal to `mae` with component reduction
        bt_expected = metrics.mae(y[0], hfc[0][0], component_reduction=np.median)
        for bt_list in bts:
            for bt in bt_list:
                np.testing.assert_array_almost_equal(bt, bt_expected)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                [metrics.mae],  # mae does not support time_reduction
                [metrics.mae, metrics.ae],  # ae supports time_reduction
                [metrics.miw],  # quantile interval metric
                [metrics.miw, metrics.iw],
            ],
            [True, False],  # last_points_only
        ),
    )
    def test_metric_quantiles_lpo(self, config):
        """Tests backtest with quantile and quantile interval metrics from expected probabilistic or quantile
        historical forecasts."""
        metric, lpo = config
        is_interval_metric = metric[0].__name__ == "miw"

        q = [0.05, 0.5, 0.60, 0.95]
        q_interval = [(0.05, 0.50), (0.50, 0.60), (0.60, 0.95), (0.05, 0.60)]

        y = lt(length=20)
        y = y.stack(y + 1.0)
        hfc = TimeSeries.from_times_and_values(
            times=generate_index(start=y.start_time() + 10 * y.freq, length=10),
            values=np.random.random((10, 1, 100)),
        )
        hfc = hfc.stack(hfc + 1.0)
        y = [y, y]
        if lpo:
            hfc = [hfc, hfc]
        else:
            hfc = [[hfc, hfc], [hfc]]

        metric_kwargs = [{"component_reduction": np.median}]
        if not is_interval_metric:
            metric_kwargs[0]["q"] = q
        else:
            metric_kwargs[0]["q_interval"] = q_interval
        if len(metric) > 1:
            # give metric specific kwargs
            metric_kwargs2 = {
                "component_reduction": np.median,
                "time_reduction": np.mean,
            }
            if not is_interval_metric:
                metric_kwargs2["q"] = q
            else:
                metric_kwargs2["q_interval"] = q_interval
            metric_kwargs.append(metric_kwargs2)

        model = NaiveDrift()

        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=None,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(bts, list) and len(bts) == 2
        if lpo:
            bts = [[bt] for bt in bts]
        # `ae` with time and component reduction is equal to `mae` with component reduction
        hfc_single = hfc[0][0] if not lpo else hfc[0]
        q_kwargs = {"q": q} if not is_interval_metric else {"q_interval": q_interval}
        bt_expected = metric[0](
            y[0], hfc_single, component_reduction=np.median, **q_kwargs
        )
        shape_expected = (len(q),)
        if len(metric) > 1:
            bt_expected = np.concatenate([bt_expected[:, None]] * 2, axis=1)
            shape_expected += (len(metric),)
        for bt_list in bts:
            for bt in bt_list:
                assert bt.shape == shape_expected
                np.testing.assert_array_almost_equal(bt, bt_expected)

        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=np.mean,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(bts, list) and len(bts) == 2
        for bt in bts:
            assert bt.shape == shape_expected
            np.testing.assert_array_almost_equal(bt, bt_expected)

        def time_reduced_metric(*args, **kwargs):
            metric_f = metrics.iw if is_interval_metric else metrics.ae
            return metric_f(*args, **kwargs, time_reduction=np.mean)

        # check that single kwargs can be used for all metrics if params are supported
        metric = [metric[0], time_reduced_metric]
        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=None,
            metric_kwargs=metric_kwargs[0],
        )
        assert isinstance(bts, list) and len(bts) == 2
        if lpo:
            bts = [[bt] for bt in bts]
        # `ae` / `miw` with time and component reduction is equal to `mae` / `miw` with component reduction
        bt_expected = metric[0](
            y[0], hfc_single, component_reduction=np.median, **q_kwargs
        )
        bt_expected = np.concatenate([bt_expected[:, None]] * 2, axis=1)
        shape_expected = (len(q), len(metric))
        for bt_list in bts:
            for bt in bt_list:
                assert bt.shape == shape_expected
                np.testing.assert_array_almost_equal(bt, bt_expected)

        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=np.mean,
            metric_kwargs=metric_kwargs[0],
        )
        assert isinstance(bts, list) and len(bts) == 2
        for bt in bts:
            assert bt.shape == shape_expected
            np.testing.assert_array_almost_equal(bt, bt_expected)

        # without component reduction
        metric_kwargs = {"component_reduction": None}
        if not is_interval_metric:
            metric_kwargs["q"] = q
        else:
            metric_kwargs["q_interval"] = q_interval
        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=None,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(bts, list) and len(bts) == 2
        if lpo:
            bts = [[bt] for bt in bts]

        # `ae` / `iw` with time and no component reduction is equal to `mae` / `miw` without component reduction
        bt_expected = metric[0](y[0], hfc_single, **metric_kwargs)
        bt_expected = np.concatenate([bt_expected[:, None]] * 2, axis=1)
        shape_expected = (len(q) * y[0].width, len(metric))
        for bt_list in bts:
            for bt in bt_list:
                assert bt.shape == shape_expected
                np.testing.assert_array_almost_equal(bt, bt_expected)

        bts = model.backtest(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=lpo,
            reduction=np.mean,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(bts, list) and len(bts) == 2
        for bt in bts:
            assert bt.shape == shape_expected
            np.testing.assert_array_almost_equal(bt, bt_expected)

    @pytest.mark.parametrize(
        "config",
        product([True, False], [True, False]),
    )
    def test_backtest_sample_weight(self, config):
        """check that passing sample weights work and that it yields different results than without sample weights."""
        manual_weight, multi_series = config
        ts = AirPassengersDataset().load()
        if manual_weight:
            sample_weight = np.linspace(0, 1, len(ts))
            sample_weight = ts.with_values(np.expand_dims(sample_weight, -1))
        else:
            sample_weight = "linear"

        if multi_series:
            ts = [ts] * 2
            sample_weight = [sample_weight] * 2 if manual_weight else sample_weight

        model = LinearRegressionModel(lags=3, output_chunk_length=1)
        start_kwargs = {"start": -1, "start_format": "position"}
        bt_non_weighted = model.backtest(series=ts, **start_kwargs)

        model = LinearRegressionModel(lags=3, output_chunk_length=1)
        bt_weighted = model.backtest(
            series=ts, sample_weight=sample_weight, **start_kwargs
        )

        if not multi_series:
            bt_weighted = [bt_weighted]
            bt_non_weighted = [bt_non_weighted]

        # check that the predictions are different
        for bt_nw, bt_w in zip(bt_non_weighted, bt_weighted):
            assert bt_w != bt_nw
