import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..metrics import metrics


class MetricsTestCase(DartsBaseTestClass):

    pd_train = pd.Series(np.sin(np.pi * np.arange(31) / 4) + 1, index=pd.date_range('20121201', '20121231'))
    pd_train_not_periodic = pd.Series(range(31), index=pd.date_range('20121201', '20121231'))
    pd_series1 = pd.Series(range(10), index=pd.date_range('20130101', '20130110'))
    pd_series2 = pd.Series(np.random.rand(10) * 10 + 1, index=pd.date_range('20130101', '20130110'))
    pd_series3 = pd.Series(np.sin(np.pi * np.arange(20) / 4) + 1, index=pd.date_range('20130101', '20130120'))
    series_train = TimeSeries.from_series(pd_train)
    series_train_not_periodic = TimeSeries.from_series(pd_train_not_periodic)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    pd_series1[:] = pd_series1.mean()
    series0: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series3)
    series12: TimeSeries = series1.stack(series2)
    series21: TimeSeries = series2.stack(series1)
    series1b = TimeSeries.from_times_and_values(pd.date_range('20130111', '20130120'), series1.values())
    series2b = TimeSeries.from_times_and_values(pd.date_range('20130111', '20130120'), series2.values())

    def test_zero(self):
        with self.assertRaises(ValueError):
            metrics.mape(self.series1, self.series1)

        with self.assertRaises(ValueError):
            metrics.smape(self.series1, self.series1)

        with self.assertRaises(ValueError):
            metrics.mape(self.series12, self.series12)

        with self.assertRaises(ValueError):
            metrics.smape(self.series12, self.series12)

        with self.assertRaises(ValueError):
            metrics.ope(self.series1 - self.series1.pd_series().mean(), self.series1 - self.series1.pd_series().mean())

    def test_same(self):
        self.assertEqual(metrics.mape(self.series1 + 1, self.series1 + 1), 0)
        self.assertEqual(metrics.smape(self.series1 + 1, self.series1 + 1), 0)
        self.assertEqual(metrics.mase(self.series1 + 1, self.series1 + 1, self.series_train, 1), 0)
        self.assertEqual(metrics.marre(self.series1 + 1, self.series1 + 1), 0)
        self.assertEqual(metrics.r2_score(self.series1 + 1, self.series1 + 1), 1)
        self.assertEqual(metrics.ope(self.series1 + 1, self.series1 + 1), 0)

    def helper_test_shape_equality(self, metric):
        self.assertAlmostEqual(metric(self.series12, self.series21),
                               metric(self.series1.append(self.series2b), self.series2.append(self.series1b)))

    def helper_test_multivariate_duplication_equality(self, metric, **kwargs):

        test_cases = [
            (self.series1 + 1, self.series2),
            (self.series1 + 1, self.series3),
            (self.series2, self.series3)
        ]

        for s1, s2 in test_cases:
            s11 = s1.stack(s1)
            s22 = s2.stack(s2)
            # default intra
            self.assertAlmostEqual(metric(s1, s2, **kwargs),
                                   metric(s11, s22, **kwargs))
            # custom intra
            self.assertAlmostEqual(metric(s1, s2, **kwargs, reduction=(lambda x: x[0])),
                                   metric(s11, s22, **kwargs, reduction=(lambda x: x[0])))

    def helper_test_multiple_ts_duplication_equality(self, metric, **kwargs):

        test_cases = [
            (self.series1 + 1, self.series2),
            (self.series1 + 1, self.series3),
            (self.series2, self.series3)
        ]

        for s1, s2 in test_cases:
            s11 = [s1.stack(s1)] * 2
            s22 = [s2.stack(s2)] * 2
            # default intra and inter
            self.assertAlmostEqual([metric(s1, s2, **kwargs)] * 2,
                                   metric(s11, s22, **kwargs))
            # custom intra and inter
            self.assertAlmostEqual(metric(s1, s2, **kwargs, reduction=np.mean, inter_reduction=np.max),
                                   metric(s11, s22, **kwargs, reduction=np.mean, inter_reduction=np.max))

    def helper_test_nan(self, metric):
        # univariate
        non_nan_metric = metric(self.series1[:9] + 1, self.series2[:9])
        nan_series1 = self.series1.copy()
        nan_series1._xa.values[-1,:,:] = np.nan
        nan_metric = metric(nan_series1 + 1, self.series2)
        self.assertEqual(non_nan_metric, nan_metric)
        # multivariate (TODO)

    def test_r2(self):
        from sklearn.metrics import r2_score
        self.assertEqual(metrics.r2_score(self.series1, self.series0), 0)
        self.assertEqual(metrics.r2_score(self.series1, self.series2),
                         r2_score(self.series1.values(), self.series2.values()))

        self.helper_test_multivariate_duplication_equality(metrics.r2_score)
        self.helper_test_multiple_ts_duplication_equality(metrics.r2_score)
        self.helper_test_nan(metrics.r2_score)

    def test_marre(self):
        self.assertAlmostEqual(metrics.marre(self.series1, self.series2),
                               metrics.marre(self.series1 + 100, self.series2 + 100))
        self.helper_test_multivariate_duplication_equality(metrics.marre)
        self.helper_test_multiple_ts_duplication_equality(metrics.marre)
        self.helper_test_nan(metrics.marre)

    def test_season(self):
        with self.assertRaises(ValueError):
            metrics.mase(self.series3, self.series3 * 1.3, self.series_train, 8)

    def test_mse(self):
        self.helper_test_shape_equality(metrics.mse)
        self.helper_test_nan(metrics.mse)

    def test_mae(self):
        self.helper_test_shape_equality(metrics.mae)
        self.helper_test_nan(metrics.mae)

    def test_rmse(self):
        self.helper_test_multivariate_duplication_equality(metrics.rmse)
        self.helper_test_multiple_ts_duplication_equality(metrics.rmse)

        self.assertAlmostEqual(metrics.rmse(self.series1.append(self.series2b), self.series2.append(self.series1b)),
                               metrics.mse(self.series12,
                                           self.series21,
                                           reduction=(lambda x: np.sqrt(np.mean(x)))))
        self.helper_test_nan(metrics.rmse)

    def test_rmsle(self):
        self.helper_test_multivariate_duplication_equality(metrics.rmsle)
        self.helper_test_multiple_ts_duplication_equality(metrics.rmsle)
        self.helper_test_nan(metrics.rmsle)

    def test_coefficient_of_variation(self):
        self.helper_test_multivariate_duplication_equality(metrics.coefficient_of_variation)
        self.helper_test_multiple_ts_duplication_equality(metrics.coefficient_of_variation)
        self.helper_test_nan(metrics.coefficient_of_variation)

    def test_mape(self):
        self.helper_test_multivariate_duplication_equality(metrics.mape)
        self.helper_test_multiple_ts_duplication_equality(metrics.mape)
        self.helper_test_nan(metrics.mape)

    def test_smape(self):
        self.helper_test_multivariate_duplication_equality(metrics.smape)
        self.helper_test_multiple_ts_duplication_equality(metrics.smape)
        self.helper_test_nan(metrics.smape)

    def test_mase(self):

        insample = self.series_train
        test_cases = [
            (self.series1 + 1, self.series2, insample),
            (self.series1 + 1, self.series3, insample),
            (self.series2, self.series3, insample)
        ]

        for s1, s2, insample in test_cases:

            # multivariate, series as args
            self.assertAlmostEqual(metrics.mase(s1.stack(s1), s2.stack(s2), insample.stack(insample),
                                                reduction=(lambda x: x[0])),
                                   metrics.mase(s1, s2, insample))
            # multi-ts, series as kwargs
            self.assertAlmostEqual(metrics.mase(actual_series=[s1] * 2, pred_series=[s2] * 2, insample=[insample] * 2,
                                                reduction=(lambda x: x[0]), inter_reduction=(lambda x: x[0])),
                                   metrics.mase(s1, s2, insample))
            # checking with n_jobs and verbose
            self.assertAlmostEqual(metrics.mase([s1] * 5, pred_series=[s2] * 5, insample=[insample] * 5,
                                                reduction=(lambda x: x[0]),
                                                inter_reduction=(lambda x: x[0])),
                                   metrics.mase([s1] * 5, [s2] * 5, insample=[insample] * 5,
                                                reduction=(lambda x: x[0]), inter_reduction=(lambda x: x[0]),
                                                n_jobs=-1, verbose=True))
        # checking with m=None
        self.assertAlmostEqual(metrics.mase(self.series2, self.series2, self.series_train_not_periodic, m=None),
                               metrics.mase([self.series2] * 2, [self.series2] * 2,
                                            [self.series_train_not_periodic] * 2, m=None, inter_reduction=np.mean))

        # fails because of wrong indexes (series1/2 indexes should be the continuation of series3)
        with self.assertRaises(ValueError):
            metrics.mase(self.series1, self.series2, self.series3, 1)
        # multi-ts, second series is not a TimeSeries
        with self.assertRaises(ValueError):
            metrics.mase([self.series1] * 2, self.series2, [insample] * 2)
        # multi-ts, insample series is not a TimeSeries
        with self.assertRaises(ValueError):
            metrics.mase([self.series1] * 2, [self.series2] * 2, insample)
        # multi-ts one array has different length
        with self.assertRaises(ValueError):
            metrics.mase([self.series1] * 2, [self.series2] * 2, [insample] * 3)
        # not supported input
        with self.assertRaises(ValueError):
            metrics.mase(1, 2, 3)

    def test_ope(self):
        self.helper_test_multivariate_duplication_equality(metrics.ope)
        self.helper_test_multiple_ts_duplication_equality(metrics.ope)
        self.helper_test_nan(metrics.ope)

    def test_metrics_arguments(self):
        series00 = self.series0.stack(self.series0)
        series11 = self.series1.stack(self.series1)
        self.assertEqual(metrics.r2_score(series11, series00, True, reduction=np.mean), 0)
        self.assertEqual(metrics.r2_score(series11, series00, reduction=np.mean), 0)
        self.assertEqual(metrics.r2_score(series11, pred_series=series00, reduction=np.mean), 0)
        self.assertEqual(metrics.r2_score(series00, actual_series=series11, reduction=np.mean), 0)
        self.assertEqual(metrics.r2_score(True, reduction=np.mean, pred_series=series00, actual_series=series11), 0)
        self.assertEqual(metrics.r2_score(series00, True, reduction=np.mean, actual_series=series11), 0)
        self.assertEqual(metrics.r2_score(series11, True, reduction=np.mean, pred_series=series00), 0)

        # should fail if kwargs are passed as args, because of the "*"
        with self.assertRaises(TypeError):
            metrics.r2_score(series00, series11, False, np.mean)

    def test_multiple_ts(self):

        dim = 2

        # simple test
        multi_ts_1 = [self.series1 + 1, self.series1 + 1]
        multi_ts_2 = [self.series1 + 2, self.series1 + 1]
        self.assertEqual(metrics.rmse(multi_ts_1, multi_ts_2, reduction=np.mean, inter_reduction=np.mean), 0.5)

        # checking univariate, multivariate and multi-ts gives same metrics with same values
        series11 = self.series1.stack(self.series1) + 1
        series22 = self.series2.stack(self.series2)
        multi_1 = [series11] * dim
        multi_2 = [series22] * dim

        test_metric = [metrics.r2_score, metrics.rmse, metrics.mape, metrics.smape, metrics.mae,
                       metrics.coefficient_of_variation, metrics.ope, metrics.marre, metrics.mse, metrics.rmsle]

        for metric in test_metric:
            self.assertEqual(metric(self.series1 + 1, self.series2), metric(series11, series22))
            self.assertEqual([metric(series11, series22)] * 2, metric(multi_1, multi_2))

        # trying different functions
        shifted_1 = self.series1 + 1
        shifted_2 = self.series1 + 2
        shifted_3 = self.series1 + 3

        self.assertEqual(metrics.rmse([shifted_1, shifted_1], [shifted_2, shifted_3],
                                      reduction=np.mean, inter_reduction=np.max),
                         metrics.rmse(shifted_1, shifted_3))

        # checking if the result is the same with different n_jobs and verbose True
        self.assertEqual(metrics.rmse([shifted_1, shifted_1], [shifted_2, shifted_3],
                                      reduction=np.mean, inter_reduction=np.max),
                         metrics.rmse([shifted_1, shifted_1], [shifted_2, shifted_3],
                                      reduction=np.mean, inter_reduction=np.max,
                                      n_jobs=-1,
                                      verbose=True))
