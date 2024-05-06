import itertools

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.ad.utils import eval_metric_from_binary_prediction, eval_metric_from_scores


class TestAnomalyDetectionModel:
    np.random.seed(42)

    # univariate series
    ts_uv = TimeSeries.from_times_and_values(
        values=np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        times=pd.date_range("2000-01-01", freq="D", periods=6),
    )
    # multivariate series
    ts_mv = ts_uv.stack(
        TimeSeries.from_times_and_values(
            values=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            times=pd.date_range("2000-01-01", freq="D", periods=6),
        )
    )
    # series with integer index
    ts_uv_idx = TimeSeries.from_values(ts_uv.values(copy=False))
    ts_mv_idx = TimeSeries.from_values(ts_mv.values(copy=False))

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                ("AUC_ROC", (1.0, 0.0, 0.5)),
                ("AUC_PR", (1.0, 0.5, 0.5)),
            ],
            [
                # ts_uv,
                ts_mv,
                # ts_uv_idx,
                ts_mv_idx,
            ],
            [False, True],
        ),
    )
    def test_eval_pred_scores(self, config):
        (metric, scores_exp), series, series_as_list = config
        is_multivariate = series.width > 1

        # the inverse of the binary anomalies will have 0. accuracy
        inv_series = TimeSeries.from_times_and_values(
            values=~series.values().astype(bool), times=series.time_index
        )

        # average (0.5) scores
        med_vals = inv_series.values(copy=True)
        med_vals[:] = 0.5
        med_series = TimeSeries.from_times_and_values(
            values=med_vals, times=series.time_index
        )

        series = [series] if series_as_list else series
        inv_series = [inv_series] if series_as_list else inv_series
        med_series = [med_series] if series_as_list else med_series

        def check_metric(series, pred_series, metric, sc_exp):
            score = eval_metric_from_scores(
                anomalies=series, pred_scores=pred_series, metric=metric
            )
            score = score if series_as_list else [score]
            assert isinstance(score, list) and len(score) == 1
            score = score[0]
            if not is_multivariate:
                assert isinstance(score, float)
                assert score == sc_exp
            else:
                assert isinstance(score, list) and score == [sc_exp] * 2

        # perfect predictions
        check_metric(series, series, metric, scores_exp[0])

        # worst predictions
        check_metric(series, inv_series, metric, scores_exp[1])

        # 0.5 predictions
        check_metric(series, med_series, metric, scores_exp[2])

        # actual series must be binary
        with pytest.raises(ValueError) as err:
            check_metric(med_series, series, metric, scores_exp[2])
        assert str(err.value).startswith(
            "Input series `anomalies` must have binary values only."
        )

        # wrong metric
        with pytest.raises(ValueError) as err:
            check_metric(series, med_series, "recall", scores_exp[2])
        assert str(err.value).startswith("Argument `metric` must be one of ")

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                ("precision", (1.0, 0.0, 0.5)),
                ("recall", (1.0, 0.0, 0.5)),
                ("f1", (1.0, 0.0, 0.5)),
                ("accuracy", (1.0, 0.0, 0.5)),
            ],
            [ts_uv, ts_mv, ts_uv_idx, ts_mv_idx],
            [False, True],
        ),
    )
    def test_eval_pred_binary(self, config):
        (metric, scores_exp), series, series_as_list = config
        is_multivariate = series.width > 1

        # the inverse of the binary anomalies will have 0. accuracy
        inv_series = TimeSeries.from_times_and_values(
            values=~series.values().astype(bool), times=series.time_index
        )

        # average (0.5) scores
        med_vals = inv_series.values(copy=True)
        med_vals[:] = 0.5
        med_series = TimeSeries.from_times_and_values(
            values=med_vals, times=series.time_index
        )

        series = [series] if series_as_list else series
        inv_series = [inv_series] if series_as_list else inv_series
        med_series = [med_series] if series_as_list else med_series

        def check_metric(series, pred_series, metric, sc_exp):
            score = eval_metric_from_binary_prediction(
                anomalies=series,
                pred_anomalies=pred_series,
                metric=metric,
            )
            score = score if series_as_list else [score]
            assert isinstance(score, list) and len(score) == 1
            score = score[0]
            if not is_multivariate:
                assert isinstance(score, float)
                assert score == sc_exp
            else:
                assert isinstance(score, list) and score == [sc_exp] * 2

        # perfect predictions
        check_metric(series, series, metric, scores_exp[0])

        # worst predictions
        check_metric(series, inv_series, metric, scores_exp[1])

        # actual series must be binary
        with pytest.raises(ValueError) as err:
            check_metric(med_series, series, metric, scores_exp[2])
        assert str(err.value).startswith(
            "Input series `anomalies` must have binary values only."
        )

        # pred must be binary
        with pytest.raises(ValueError) as err:
            check_metric(series, med_series, metric, scores_exp[2])
        assert str(err.value).startswith(
            "Input series `pred_anomalies` must have binary values only."
        )

        # wrong metric
        with pytest.raises(ValueError) as err:
            check_metric(series, med_series, "AUC_ROC", scores_exp[2])
        assert str(err.value).startswith("Argument `metric` must be one of ")
