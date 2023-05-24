import unittest

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import (
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveDrift,
    NaiveEnsembleModel,
    NaiveSeasonal,
    RegressionEnsembleModel,
    StatsForecastAutoARIMA,
    Theta,
)
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from darts.models import NBEATSModel, RNNModel, TCNModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - Some ensemble models tests will be skipped.")
    TORCH_AVAILABLE = False


def _make_ts(start_value=0, n=100):
    times = pd.date_range(start="1/1/2013", periods=n, freq="D")
    pd_series = pd.Series(range(start_value, start_value + n), index=times)
    return TimeSeries.from_series(pd_series)


class EnsembleModelsTestCase(DartsBaseTestClass):
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    seq1 = [_make_ts(0), _make_ts(10), _make_ts(20)]
    cov1 = [_make_ts(5), _make_ts(15), _make_ts(25)]

    def test_untrained_models(self):
        model = NaiveDrift()
        _ = NaiveEnsembleModel([model])

        # trained models should raise error
        model.fit(self.series1)
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([model])

        # an untrained ensemble model should also give untrained underlying models
        model_ens = NaiveEnsembleModel([NaiveDrift()])
        model_ens.fit(self.series1)
        assert model_ens.models[0]._fit_called
        new_model = model_ens.untrained_model()
        assert not new_model.models[0]._fit_called

    def test_extreme_lag_inference(self):
        ensemble = NaiveEnsembleModel([NaiveDrift()])
        assert ensemble.extreme_lags == (
            -3,
            -1,
            None,
            None,
            None,
            None,
        )  # test if default is okay

        model1 = LinearRegressionModel(
            lags=3, lags_past_covariates=[-3, -5], lags_future_covariates=[7, 8]
        )
        model2 = LinearRegressionModel(
            lags=5, lags_past_covariates=6, lags_future_covariates=[6, 9]
        )

        ensemble = NaiveEnsembleModel(
            [model1, model2]
        )  # test if infers extreme lags is okay
        expected = (-5, 0, -6, -1, 6, 9)
        assert expected == ensemble.extreme_lags

    def test_input_models_local_models(self):
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([])
        # models are not instantiated
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        # one model is not instantiated
        with self.assertRaises(ValueError):
            NaiveEnsembleModel(
                [NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()]
            )
        NaiveEnsembleModel(
            [NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()]
        )

    def test_call_predict_local_models(self):
        naive_ensemble = NaiveEnsembleModel([NaiveSeasonal(), Theta()])
        with self.assertRaises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        pred1 = naive_ensemble.predict(5)
        assert self.series1.components == pred1.components

    def test_call_backtest_naive_ensemble_local_models(self):
        ensemble = NaiveEnsembleModel([NaiveSeasonal(5), Theta(2, 5)])
        ensemble.fit(self.series1)
        assert ensemble.extreme_lags == (-10, 0, None, None, None, None)
        ensemble.backtest(self.series1)

    def test_predict_ensemble_local_models(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_ensemble = NaiveEnsembleModel([naive, theta])
        naive_ensemble.fit(self.series1 + self.series2)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)

        self.assertTrue(
            np.array_equal(forecast_naive_ensemble.values(), forecast_mean.values())
        )

    def test_stochastic_ensemble(self):
        model1 = LinearRegressionModel(lags=1, likelihood="quantile")
        model2 = LinearRegressionModel(lags=2, likelihood="quantile")

        naive_ensemble = NaiveEnsembleModel([model1, model2])
        self.assertTrue(naive_ensemble._is_probabilistic())

        regression_ensemble = RegressionEnsembleModel(
            [model1, model2], regression_train_n_points=1
        )
        self.assertTrue(regression_ensemble._is_probabilistic())

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_input_models_global_models(self):
        # one model is not instantiated
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel])
        NaiveEnsembleModel([RNNModel(12), TCNModel(10, 2), NBEATSModel(10, 2)])

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_univariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel(
            [
                RNNModel(12, n_epochs=1),
                TCNModel(10, 2, n_epochs=1),
                NBEATSModel(10, 2, n_epochs=1),
            ]
        )
        with self.assertRaises(Exception):
            naive_ensemble.predict(5)

        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_multivariate_input_no_covariates(self):
        naive_ensemble = NaiveEnsembleModel(
            [
                RNNModel(12, n_epochs=1),
                TCNModel(10, 2, n_epochs=1),
                NBEATSModel(10, 2, n_epochs=1),
            ]
        )
        naive_ensemble.fit(self.seq1)
        naive_ensemble.predict(n=5, series=self.seq1)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_global_models_multivariate_input_with_covariates(self):
        naive_ensemble = NaiveEnsembleModel(
            [
                RNNModel(12, n_epochs=1),
                TCNModel(10, 2, n_epochs=1),
                NBEATSModel(10, 2, n_epochs=1),
            ]
        )
        naive_ensemble.fit(self.seq1, self.cov1)
        predict_series = [s[:12] for s in self.seq1]
        predict_covariates = [c[:14] for c in self.cov1]
        naive_ensemble.predict(
            n=2, series=predict_series, past_covariates=predict_covariates
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_input_models_mixed(self):
        # NaiveDrift is local, RNNModel is global
        naive_ensemble = NaiveEnsembleModel([NaiveDrift(), RNNModel(12, n_epochs=1)])
        # ensemble is neither local, nor global
        self.assertFalse(naive_ensemble.is_local_ensemble)
        self.assertFalse(naive_ensemble.is_global_ensemble)

        # ensemble contains one local model, no support for multiple ts fit
        with self.assertRaises(ValueError):
            naive_ensemble.fit([self.series1, self.series2])

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_call_predict_different_covariates_support(self):
        # AutoARIMA support future covariates only
        local_ensemble_one_covs = NaiveEnsembleModel(
            [NaiveDrift(), StatsForecastAutoARIMA()]
        )
        with self.assertRaises(ValueError):
            local_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        local_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)

        # RNN support future covariates only
        mixed_ensemble_one_covs = NaiveEnsembleModel(
            [NaiveDrift(), RNNModel(12, n_epochs=1)]
        )
        with self.assertRaises(ValueError):
            mixed_ensemble_one_covs.fit(self.series1, past_covariates=self.series2)
        mixed_ensemble_one_covs.fit(self.series1, future_covariates=self.series2)

        # both models support future covariates only
        mixed_ensemble_future_covs = NaiveEnsembleModel(
            [StatsForecastAutoARIMA(), RNNModel(12, n_epochs=1)]
        )
        mixed_ensemble_future_covs.fit(self.series1, future_covariates=self.series2)
        with self.assertRaises(ValueError):
            mixed_ensemble_future_covs.fit(self.series1, past_covariates=self.series2)

        # RegressionModels with different covariates
        global_ensemble_both_covs = NaiveEnsembleModel(
            [
                LinearRegressionModel(lags=1, lags_past_covariates=[-1]),
                LinearRegressionModel(lags=1, lags_future_covariates=[1]),
            ]
        )
        # missing future covariates
        with self.assertRaises(ValueError):
            global_ensemble_both_covs.fit(self.series1, past_covariates=self.series2)
        # missing past covariates
        with self.assertRaises(ValueError):
            global_ensemble_both_covs.fit(self.series1, future_covariates=self.series2)
        global_ensemble_both_covs.fit(
            self.series1, past_covariates=self.series2, future_covariates=self.series2
        )

    def test_fit_multivar_ts_with_local_models(self):
        naive = NaiveEnsembleModel(
            [NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()]
        )
        with self.assertRaises(ValueError):
            naive.fit(self.seq1)

    def test_fit_univar_ts_with_covariates_for_local_models(self):
        naive = NaiveEnsembleModel(
            [NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()]
        )
        with self.assertRaises(ValueError):
            naive.fit(self.series1, self.series2)

    def test_predict_with_target(self):
        series_long = self.series1
        series_short = series_long[:25]

        # train with a single series
        ensemble_model = self.get_global_ensembe_model()
        ensemble_model.fit(series_short, past_covariates=series_long)
        # predict after end of train series
        preds = ensemble_model.predict(n=5, past_covariates=series_long)
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 2)
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 1)

        # train with multiple series
        ensemble_model = self.get_global_ensembe_model()
        ensemble_model.fit([series_short] * 2, past_covariates=[series_long] * 2)
        with self.assertRaises(ValueError):
            # predict without passing series should raise an error
            ensemble_model.predict(n=5, past_covariates=series_long)
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 2)
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 1)

    @staticmethod
    def get_global_ensembe_model(output_chunk_length=5):
        lags = [-1, -2, -5]
        return NaiveEnsembleModel(
            models=[
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
            ],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
